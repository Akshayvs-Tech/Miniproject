"""
backend/routes/health.py
Routes: GET /health    — server liveness check
        GET /db-health — MongoDB connection check
"""

from fastapi import APIRouter
from backend.database import client, DB_NAME

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/db-health")
async def db_health():
    try:
        await client.admin.command("ping")
        return {"mongodb": "✅ connected", "database": DB_NAME}
    except Exception as e:
        return {"mongodb": "❌ disconnected", "error": str(e)}
