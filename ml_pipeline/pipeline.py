"""
ml_pipeline/pipeline.py
Connects all models: YOLOv8 + DeepSORT + ArcFace + OSNet.
Called by backend/services/ml_service.py — contains NO FastAPI code.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

from ml_pipeline import arcface, osnet_reid
from ml_pipeline.config import (
    OUTPUT_DIR,
    FRAME_SKIP,
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_NN_BUDGET,
    DEEPSORT_MAX_COSINE_DIST,
)

# ─── Result Schema ────────────────────────────────────────────────────────────

@dataclass
class MatchRecord:
    frame:         int
    timestamp_sec: float
    track_id:      int
    similarity:    float
    match_type:    str

@dataclass
class PipelineResult:
    found:                bool
    total_match_frames:   int
    first_appearance_sec: float | None
    last_appearance_sec:  float | None
    best_match:           MatchRecord | None
    matched_track_ids:    list[int]
    output_video_path:    str | None
    match_records:        list[MatchRecord] = field(default_factory=list)

# ─── Core Pipeline ────────────────────────────────────────────────────────────

def run(img_path: str, vid_path: str, video_id: str) -> PipelineResult:
    """
    Full person-finding pipeline:
      1. Extract ArcFace + OSNet embeddings from reference image
      2. For each video frame:
           a. YOLOv8   → detect all persons
           b. DeepSORT → assign stable track IDs
           c. ArcFace  → face-based match
           d. OSNet    → body-based match
      3. Annotate and write output video
      4. Return structured PipelineResult
    """

    # ── Step 1: Reference embeddings ────────────────────────────────────────
    ref_arcface = arcface.get_embedding(img_path)
    ref_osnet   = osnet_reid.get_embedding(img_path)

    if ref_arcface is None and ref_osnet is None:
        raise ValueError(
            "Could not extract any features (face or body) from the reference image."
        )

    # ── Step 2: Open video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[pipeline] Video opened — {width}x{height} @ {fps:.1f} FPS")

    out_path = str(OUTPUT_DIR / f"{video_id}.avi")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    # Fresh tracker per request — avoids ID collisions between calls
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nn_budget=DEEPSORT_NN_BUDGET,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DIST,
    )

    # ── Step 3: Frame loop ──────────────────────────────────────────────────
    frame_idx    = 0
    match_records: list[MatchRecord] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            detections = osnet_reid.detect_persons(frame)

            ds_input = [
                ([x1, y1, x2 - x1, y2 - y1], conf, "person")
                for x1, y1, x2, y2, conf in detections
            ]
            tracks = tracker.update_tracks(ds_input, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                is_match   = False
                best_sim   = 0.0
                match_type = ""

                # ArcFace — face match
                if ref_arcface is not None and crop.shape[0] >= arcface.MIN_CROP_SIZE and crop.shape[1] >= arcface.MIN_CROP_SIZE:
                    emb_arc = arcface.get_embedding(crop)
                    if emb_arc is not None:
                        sim_arc = float(np.dot(ref_arcface, emb_arc))
                        if sim_arc >= arcface.SIMILARITY_THRESHOLD:
                            is_match   = True
                            best_sim   = max(best_sim, sim_arc)
                            match_type += "[Face]"

                # OSNet — body match
                if ref_osnet is not None and crop.shape[0] >= osnet_reid.MIN_CROP_H and crop.shape[1] >= osnet_reid.MIN_CROP_W:
                    emb_osnet = osnet_reid.get_embedding(crop)
                    if emb_osnet is not None:
                        sim_osnet = float(np.dot(ref_osnet, emb_osnet))
                        if sim_osnet >= osnet_reid.SIMILARITY_THRESHOLD:
                            is_match   = True
                            best_sim   = max(best_sim, sim_osnet)
                            match_type += "[Body]"

                if is_match:
                    ts = round(frame_idx / fps, 3)
                    record = MatchRecord(
                        frame=frame_idx,
                        timestamp_sec=ts,
                        track_id=track_id,
                        similarity=round(best_sim, 4),
                        match_type=match_type,
                    )
                    match_records.append(record)
                    print(f"  [MATCH] frame={frame_idx}  t={ts}s  track={track_id}  sim={best_sim:.4f} {match_type}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"TARGET {match_type} id:{track_id} sim:{best_sim:.2f}",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                    cv2.putText(
                        frame, f"id:{track_id}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
                    )

            cv2.putText(
                frame,
                f"frame {frame_idx}  |  t={frame_idx/fps:.1f}s  |  tracks={len(tracks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[pipeline] Done — {frame_idx} frames processed.")

    # ── Step 4: Build result ──────────────────────────────────────────────────
    found = len(match_records) > 0
    best  = max(match_records, key=lambda r: r.similarity) if found else None

    return PipelineResult(
        found                = found,
        total_match_frames   = len(match_records),
        first_appearance_sec = match_records[0].timestamp_sec  if found else None,
        last_appearance_sec  = match_records[-1].timestamp_sec if found else None,
        best_match           = best,
        matched_track_ids    = sorted({r.track_id for r in match_records}),
        output_video_path    = out_path if found else None,
        match_records        = match_records,
    )
