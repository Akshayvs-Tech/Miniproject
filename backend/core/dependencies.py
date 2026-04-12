"""
backend/core/dependencies.py
FastAPI dependency: get_current_user — validates Bearer JWT on protected routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo.errors import PyMongoError

from backend.core.security import decode_access_token
from backend.database import users_collection

# ─── Bearer scheme ────────────────────────────────────────────────────────────

_bearer = HTTPBearer()

# ─── Dependency ───────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """
    Validates the Bearer JWT from the Authorization header.
    Returns the user document from MongoDB.
    Raises 401 if the token is missing, invalid, expired, or the user is gone.
    Raises 503 if the database cannot be reached.
    """
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or expired token.",
        headers     = {"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    try:
        user = await users_collection.find_one({"_id": user_id})
    except PyMongoError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach the database. Check MONGO_URI, network access, and Atlas IP allowlist.",
        )
    if user is None:
        raise credentials_exception

    return user
