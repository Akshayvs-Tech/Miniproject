"""
backend/routes/process.py
Routes: POST /process   — run person-finding pipeline
        GET  /video/{id} — download annotated output video
"""

from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from backend.controllers.process_controller import handle_process
from backend.schemas import ProcessResult
from ml_pipeline.config import OUTPUT_DIR

router = APIRouter()

# ─── POST /process ────────────────────────────────────────────────────────────

@router.post("/process", response_model=ProcessResult)
async def process(
    image: UploadFile = File(..., description="Reference photo of the person (JPEG/PNG)"),
    video: UploadFile = File(..., description="Video to search through (MP4/AVI)"),
):
    # Validate content types
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Image must be JPEG or PNG.")
    if video.content_type not in ("video/mp4", "video/x-msvideo", "video/avi"):
        raise HTTPException(status_code=400, detail="Video must be MP4 or AVI.")

    image_bytes = await image.read()
    video_bytes = await video.read()

    img_suffix = Path(image.filename).suffix or ".jpg"
    vid_suffix = Path(video.filename).suffix or ".mp4"

    return await handle_process(
        image_bytes    = image_bytes,
        image_suffix   = img_suffix,
        video_bytes    = video_bytes,
        video_suffix   = vid_suffix,
        image_filename = image.filename,
        video_filename = video.filename,
    )

# ─── GET /video/{video_id} ────────────────────────────────────────────────────

@router.get("/video/{video_id}")
def download_video(video_id: str):
    out_path = OUTPUT_DIR / f"{video_id}.avi"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(str(out_path), media_type="video/x-msvideo", filename="result.avi")
