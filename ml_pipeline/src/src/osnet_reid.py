import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── Load Models ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# YOLOv8 - person detector
yolo_model = YOLO('yolov8n.pt')

# DeepSORT - multi-person tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.3
)

# OSNet - full body re-identification
osnet_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    pretrained=True
)
osnet_model = osnet_model.to(device)
osnet_model.eval()

# ─── Config ───────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.6   # from SRS spec
FRAME_SKIP = 2

# ─── OSNet Preprocessing ─────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── OSNet Embedding ──────────────────────────────────────────────────────────

def get_embedding(image):
    """
    Extract OSNet 512-d body embedding from a PIL image or numpy array.
    Returns normalized embedding or None if crop is too small.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if image is None:
        return None

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = osnet_model(tensor)
    emb = emb.squeeze().cpu().numpy()
    # Normalize for cosine similarity
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def cosine_similarity(a, b):
    return float(np.dot(a, b))

# ─── YOLOv8 Person Detection ──────────────────────────────────────────────────

def detect_persons(frame):
    """
    Run YOLOv8 on frame, return list of (x1, y1, x2, y2, confidence)
    for all detected persons (class 0).
    """
    results = yolo_model(frame, classes=[0], verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append((x1, y1, x2, y2, conf))
    return detections

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def find_person_in_video(image_path, video_path, output_path="output.avi"):
    """
    Full OSNet Re-ID pipeline:
      1. Extract OSNet body embedding from reference image
      2. For each video frame:
           a. YOLOv8   → detect all persons
           b. DeepSORT → assign stable track IDs
           c. OSNet    → compare each tracked person to reference
      3. Annotate matched person in green, others in grey
      4. Print match report at the end
    """

    # Step 1: Reference embedding
    print(f"[1/3] Loading reference image: {image_path}")
    ref_img = Image.open(image_path).convert("RGB")
    ref_embedding = get_embedding(np.array(ref_img))
    if ref_embedding is None:
        print("ERROR: Could not extract embedding from reference image.")
        return

    # Step 2: Open video
    print(f"[2/3] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"    Video: {total_frms} frames @ {fps:.1f} FPS  ({width}x{height})")
    print(f"    Scanning every {FRAME_SKIP} frame(s)...\n")

    # Output video writer
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    # Step 3: Frame loop
    print("[3/3] Processing frames...")
    frame_idx     = 0
    match_records = []
    found         = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:

            # ── A. YOLOv8 detection ──────────────────────────────────────────
            detections = detect_persons(frame)

            # ── B. DeepSORT tracking ─────────────────────────────────────────
            ds_input = [
                ([x1, y1, x2 - x1, y2 - y1], conf, 'person')
                for x1, y1, x2, y2, conf in detections
            ]
            tracks = tracker.update_tracks(ds_input, frame=frame)

            # ── C. OSNet matching ────────────────────────────────────────────
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                # Clamp to frame bounds
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(width, x2);  y2 = min(height, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Skip crops too small — from SRS (min 50x100 pixels)
                crop_h, crop_w = crop.shape[:2]
                if crop_h < 100 or crop_w < 50:
                    continue

                embedding = get_embedding(crop)

                if embedding is not None:
                    similarity = cosine_similarity(ref_embedding, embedding)

                    if similarity >= SIMILARITY_THRESHOLD:
                        # ── TARGET found ── draw GREEN box ──────────────────
                        timestamp = frame_idx / fps
                        match_records.append((frame_idx, timestamp, track_id, similarity))
                        found = True

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"TARGET  id:{track_id}  sim:{similarity:.2f}",
                            (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                        )
                        print(f"  Match  frame={frame_idx:5d}  "
                              f"t={timestamp:6.2f}s  "
                              f"track={track_id}  "
                              f"sim={similarity:.4f}")
                    else:
                        # ── other person ── draw GREY box ───────────────────
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                        cv2.putText(
                            frame,
                            f"id:{track_id}",
                            (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1
                        )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # ── HUD overlay ──────────────────────────────────────────────────
            cv2.putText(
                frame,
                f"frame {frame_idx}  |  t={frame_idx/fps:.1f}s  |  tracks={len(tracks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ─── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULT")
    print("=" * 55)

    if found:
        print(f"  Person WAS found in the video.")
        print(f"  Total match frames  : {len(match_records)}")

        best = max(match_records, key=lambda r: r[3])
        print(f"  Best match          : frame {best[0]} "
              f"@ {best[1]:.2f}s  (sim={best[3]:.4f})")

        first = match_records[0]
        last  = match_records[-1]
        print(f"  First appearance    : {first[1]:.2f}s")
        print(f"  Last appearance     : {last[1]:.2f}s")

        unique_tracks = set(r[2] for r in match_records)
        print(f"  Matched track IDs   : {sorted(unique_tracks)}")
    else:
        print("  Person was NOT found in the video.")
        print(f"  (threshold = {SIMILARITY_THRESHOLD}  —  try lowering if missed)")

    print(f"\n  Annotated video saved → {output_path}")
    print("=" * 55)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    find_person_in_video(
        image_path="image1.jpg",
        video_path="video.mp4",
        output_path="output.avi"
    )
