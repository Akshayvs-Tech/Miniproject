"""
backend/app.py
FastAPI application entry point.
Run with: uvicorn backend.app:app --reload  (from project root Miniproject/)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Support running from the backend directory with: uvicorn app:app --reload
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend.database import client
from backend.routes import process, health, auth

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await client.admin.command("ping")
        print("✅ MongoDB Atlas connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
    yield

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Person Finder API",
    description = "Find a person in a video using ArcFace + OSNet + YOLOv8 + DeepSORT",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(process.router)
app.include_router(health.router)
