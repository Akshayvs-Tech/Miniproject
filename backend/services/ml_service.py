"""
backend/services/ml_service.py
Bridge between FastAPI and the ML pipeline.
Handles file I/O (temp files, cleanup) and calls ml_pipeline.pipeline.run().
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from ml_pipeline import pipeline
from ml_pipeline.pipeline import PipelineResult

# ─── Public API ───────────────────────────────────────────────────────────────

def process_request(
    image_bytes: bytes,
    image_suffix: str,
    video_bytes: bytes,
    video_suffix: str,
    image_filename: str,
    video_filename: str,
) -> tuple[str, PipelineResult]:
    """
    1. Saves uploaded bytes to temp files
    2. Runs ml_pipeline.pipeline.run()
    3. Cleans up temp files
    Returns (video_id, PipelineResult)
    """
    img_path = vid_path = None
    video_id = str(uuid.uuid4())

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_img:
            tmp_img.write(image_bytes)
            img_path = tmp_img.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as tmp_vid:
            tmp_vid.write(video_bytes)
            vid_path = tmp_vid.name

        result = pipeline.run(
            img_path=img_path,
            vid_path=vid_path,
            video_id=video_id,
        )
        return video_id, result

    finally:
        for p in filter(None, [img_path, vid_path]):
            try:
                os.remove(p)
            except Exception:
                pass
