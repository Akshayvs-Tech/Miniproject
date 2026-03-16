import cv2
import numpy as np
import torch
import torchreid
import insightface
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── Load Models ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

yolo_model = YOLO('yolov8n.pt')

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.3
)

osnet_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    pretrained=True
)
osnet_model = osnet_model.to(device)
osnet_model.eval()

arcface_model = insightface.app.FaceAnalysis(name='buffalo_l')
arcface_model.prepare(ctx_id=0 if device == "cuda" else -1)

# ─── Config ───────────────────────────────────────────────────────────────────

OSNET_THRESHOLD   = 0.6
ARCFACE_THRESHOLD = 0.55
FRAME_SKIP        = 2

# ─── OSNet Preprocessing ─────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_osnet_embedding(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if image is None:
        return None
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = osnet_model(tensor)
    emb = emb.squeeze().cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def get_arcface_embedding(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    if img is None or img.size == 0:
        return None
    faces = arcface_model.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def cosine_similarity(a, b):
    return float(np.dot(a, b))

# ─── YOLOv8 Detection ────────────────────────────────────────────────────────

def detect_persons(frame):
    results = yolo_model(frame, classes=[0], verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append((x1, y1, x2, y2, conf))
    return detections

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def find_person_in_video(image_path, video_path, output_path="output.avi"):

    print(f"[1/3] Loading reference image: {image_path}")
    ref_img     = cv2.imread(image_path)
    ref_osnet   = get_osnet_embedding(ref_img)
    ref_arcface = get_arcface_embedding(ref_img)

    if ref_osnet is None:
        print("ERROR: Could not extract OSNet embedding from reference image.")
        return
    if ref_arcface is None:
        print("WARNING: No face detected in reference image — running body-only matching.")

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

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    print("[3/3] Processing frames...")
    frame_idx     = 0
    match_records = []
    found         = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:

            detections = detect_persons(frame)

            ds_input = [
                ([x1, y1, x2 - x1, y2 - y1], conf, 'person')
                for x1, y1, x2, y2, conf in detections
            ]
            tracks = tracker.update_tracks(ds_input, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())

                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(width, x2);  y2 = min(height, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]
                if crop_h < 100 or crop_w < 50:
                    continue

                osnet_emb  = get_osnet_embedding(crop)
                body_sim   = 0.0
                body_match = False
                if osnet_emb is not None:
                    body_sim   = cosine_similarity(ref_osnet, osnet_emb)
                    body_match = body_sim >= OSNET_THRESHOLD

                face_sim   = 0.0
                face_match = False
                if ref_arcface is not None:
                    arc_emb = get_arcface_embedding(crop)
                    if arc_emb is not None:
                        face_sim   = cosine_similarity(ref_arcface, arc_emb)
                        face_match = face_sim >= ARCFACE_THRESHOLD

                matched = body_match or face_match

                if matched:
                    timestamp = frame_idx / fps
                    match_records.append((frame_idx, timestamp, track_id, max(body_sim, face_sim)))
                    found = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"Person ID: {track_id}",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )
                    print(f"  Match  frame={frame_idx:5d}  t={timestamp:6.2f}s  "
                          f"track={track_id}  body={body_sim:.4f}  face={face_sim:.4f}")
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                    cv2.putText(
                        frame, f"id:{track_id}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1
                    )

            cv2.putText(
                frame,
                f"frame {frame_idx}  |  t={frame_idx/fps:.1f}s  |  tracks={len(tracks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print("\n" + "=" * 55)
    print("  RESULT")
    print("=" * 55)

    if found:
        print(f"  Person WAS found in the video.")
        print(f"  Total match frames  : {len(match_records)}")
        best  = max(match_records, key=lambda r: r[3])
        first = match_records[0]
        last  = match_records[-1]
        print(f"  Best match          : frame {best[0]} @ {best[1]:.2f}s  (sim={best[3]:.4f})")
        print(f"  First appearance    : {first[1]:.2f}s")
        print(f"  Last appearance     : {last[1]:.2f}s")
        print(f"  Matched track IDs   : {sorted(set(r[2] for r in match_records))}")
    else:
        print("  Person was NOT found in the video.")
        print(f"  (body threshold={OSNET_THRESHOLD}, face threshold={ARCFACE_THRESHOLD})")

    print(f"\n  Annotated video saved → {output_path}")
    print("=" * 55)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    find_person_in_video(
        image_path="image1.jpg",
        video_path="video.mp4",
        output_path="output.avi"
    )
