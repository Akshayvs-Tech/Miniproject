"""
backend/controllers/process_controller.py
Business logic for the /process endpoint.
Calls ml_service, saves result to DB, returns response schema.
"""

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from backend.database import results_collection
from backend.services import ml_service
from backend.schemas import MatchRecord, ProcessResult

# ─── Controller ───────────────────────────────────────────────────────────────

async def handle_process(
    image_bytes:    bytes,
    image_suffix:   str,
    video_bytes:    bytes,
    video_suffix:   str,
    image_filename: str,
    video_filename: str,
) -> ProcessResult:
    """
    Orchestrates a full /process request:
      1. Delegate file handling + ML inference to ml_service
      2. Persist result in MongoDB
      3. Return ProcessResult response schema
    """
    try:
        video_id, result = ml_service.process_request(
            image_bytes    = image_bytes,
            image_suffix   = image_suffix,
            video_bytes    = video_bytes,
            video_suffix   = video_suffix,
            image_filename = image_filename,
            video_filename = video_filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    found = result.found
    best  = result.best_match

    # Persist to MongoDB
    await results_collection.insert_one({
        "video_id":             video_id,
        "image_filename":       image_filename,
        "video_filename":       video_filename,
        "found":                found,
        "total_match_frames":   result.total_match_frames,
        "first_appearance_sec": result.first_appearance_sec,
        "last_appearance_sec":  result.last_appearance_sec,
        "best_match": {
            "frame":         best.frame,
            "timestamp_sec": best.timestamp_sec,
            "track_id":      best.track_id,
            "similarity":    best.similarity,
        } if best else None,
        "matched_track_ids": result.matched_track_ids,
        "created_at":        datetime.utcnow(),
    })

    return ProcessResult(
        found                = found,
        total_match_frames   = result.total_match_frames,
        first_appearance_sec = result.first_appearance_sec,
        last_appearance_sec  = result.last_appearance_sec,
        best_match           = MatchRecord(
            frame         = best.frame,
            timestamp_sec = best.timestamp_sec,
            track_id      = best.track_id,
            similarity    = best.similarity,
        ) if best else None,
        matched_track_ids = result.matched_track_ids,
        output_video_id   = video_id if found else None,
    )
