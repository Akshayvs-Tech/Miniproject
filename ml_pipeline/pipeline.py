"""
ml_pipeline/pipeline.py
Connects all models: YOLOv8 + DeepSORT + ArcFace + OSNet.
Called by backend/services/ml_service.py — contains NO FastAPI code.

Can also be run directly as a script from the project root:
    python -m ml_pipeline.pipeline        (preferred)
    python ml_pipeline/pipeline.py        (also works via the path fix below)
"""

import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# When run as a standalone script (python ml_pipeline/pipeline.py), Python sets
# __package__ to None and the project root is NOT on sys.path.  This guard
# inserts it so all `from ml_pipeline import …` imports resolve correctly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # Miniproject/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Also add ml_pipeline/ itself so local torchreid is importable
_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from dataclasses import dataclass, field
from deep_sort_realtime.deepsort_tracker import DeepSort

from ml_pipeline import arcface, osnet_reid, action_recognition
from ml_pipeline.config import (
    OUTPUT_DIR,
    FRAME_SKIP,
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_NN_BUDGET,
    DEEPSORT_MAX_COSINE_DIST,
    TRACK_ID_REMAP_IOU,
    TRACK_ID_REMAP_MAX_GAP,
)
from collections import deque

# ─── Result Schema ────────────────────────────────────────────────────────────

@dataclass
class MatchRecord:
    frame:         int
    timestamp_sec: float
    track_id:      int
    similarity:    float
    match_type:    str
    action:        str = "analyzing..."

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

    # Init Action Models
    action_recognition.init_models()

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

    # Track-ID remap state to reduce ID switches
    id_alias: dict[int, int] = {}
    recent_tracks: dict[int, tuple[tuple[int, int, int, int], int]] = {}

    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return float(inter_area / (area_a + area_b - inter_area + 1e-6))

    def _resolve_track_id(raw_id: int, bbox, frame_idx: int, used_ids: set[int]) -> int:
        # If we already mapped this raw ID, reuse it unless it conflicts this frame
        if raw_id in id_alias:
            canonical_id = id_alias[raw_id]
            if canonical_id in used_ids:
                canonical_id = raw_id
                id_alias[raw_id] = canonical_id
            recent_tracks[canonical_id] = (bbox, frame_idx)
            return canonical_id

        best_id = None
        best_iou = 0.0
        for cand_id, (cand_bbox, last_seen) in list(recent_tracks.items()):
            if frame_idx - last_seen > TRACK_ID_REMAP_MAX_GAP:
                continue
            if cand_id in used_ids:
                continue
            iou = _bbox_iou(bbox, cand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = cand_id

        if best_id is not None and best_iou >= TRACK_ID_REMAP_IOU:
            id_alias[raw_id] = best_id
            recent_tracks[best_id] = (bbox, frame_idx)
            return best_id

        id_alias[raw_id] = raw_id
        recent_tracks[raw_id] = (bbox, frame_idx)
        return raw_id

    # ── Step 3: Frame loop ──────────────────────────────────────────────────
    # ── Step 3: Global state for action recognition ─────────────────────────
    frame_idx    = 0
    match_records: list[MatchRecord] = []

    # Per-track action state (pose buffers + stable labels)
    track_actions: dict[int, action_recognition.TrackActionState] = {}

    # Canonical target ID once the person is matched at least once
    target_id: int | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            detections = osnet_reid.detect_persons(frame)

            # Filter detections by confidence and minimum crop size to reduce
            # spurious detections and improve tracking / action recognition.
            from ml_pipeline.osnet_reid import MIN_CROP_H, MIN_CROP_W

            ds_input = []
            for x1, y1, x2, y2, conf in detections:
                w = int(x2 - x1)
                h = int(y2 - y1)
                # Skip very low-confidence detections
                if conf < 0.25:
                    continue
                # Skip small crops that will harm re-id and pose estimation
                if h < MIN_CROP_H or w < MIN_CROP_W:
                    continue
                ds_input.append(([x1, y1, w, h], conf, "person"))
            tracks = tracker.update_tracks(ds_input, frame=frame)

            used_ids: set[int] = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                raw_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                canonical_id = _resolve_track_id(raw_id, (x1, y1, x2, y2), frame_idx, used_ids)
                used_ids.add(canonical_id)

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

                # ── Action Recognition ──
                if canonical_id not in track_actions:
                    track_actions[canonical_id] = action_recognition.create_track_state(canonical_id)

                act_state = track_actions[canonical_id]
                act_state.update(crop, fps)
                display_action = act_state.display_action()

                # If this is a match and we already have a target ID, remap
                # the current track to the target ID to avoid ID switches.
                effective_id = canonical_id
                if is_match:
                    if target_id is None:
                        target_id = canonical_id
                    elif canonical_id != target_id:
                        id_alias[raw_id] = target_id
                        recent_tracks[target_id] = ((x1, y1, x2, y2), frame_idx)
                        effective_id = target_id
                        # Keep action history under the canonical target ID
                        track_actions[target_id] = act_state
                        if canonical_id in track_actions and canonical_id != target_id:
                            del track_actions[canonical_id]

                if is_match:
                    ts = round(frame_idx / fps, 3)
                    record = MatchRecord(
                        frame=frame_idx,
                        timestamp_sec=ts,
                        track_id=effective_id,
                        similarity=round(best_sim, 4),
                        match_type=match_type,
                        action=display_action,
                    )
                    match_records.append(record)
                    print(f"  [MATCH] frame={frame_idx}  t={ts}s  track={effective_id}  sim={best_sim:.4f} {match_type} action={record.action}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"TARGET {match_type} id:{effective_id} sim:{best_sim:.2f}",
                        (x1, max(y1 - 25, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        frame,
                        f"Action: {record.action}",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2,
                    )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                    cv2.putText(
                        frame, f"id:{canonical_id} | {display_action}",
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

    # Cleanup
    action_recognition.cleanup()
    cap.release()
    writer.release()
    print(f"[pipeline] Done — {frame_idx} frames processed.")

    # ── Step 4: Backfill generic actions with per-track stable labels ─────────
    for rec in match_records:
        state = track_actions.get(rec.track_id)
        if not state:
            continue
        summary = action_recognition.summarize_track_action(state)
        if rec.action in action_recognition.GENERIC_LABELS or rec.action.startswith("analyzing"):
            rec.action = summary

    # ── Step 5: Build result ──────────────────────────────────────────────────
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
