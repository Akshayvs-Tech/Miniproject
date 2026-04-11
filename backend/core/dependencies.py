"""
backend/core/dependencies.py
FastAPI dependency: get_current_user — validates Bearer JWT on protected routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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
    Raises 401 if the token is missing, invalid, or expired.
    Raises 404 if the user no longer exists in DB.
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

    user = await users_collection.find_one({"_id": user_id})
    if user is None:
        raise credentials_exception

    return user
