"""
backend/controllers/auth_controller.py
Business logic for user registration and login.

Not used by FastAPI: routes live in backend/routes/auth.py with different fields
(UserOut has id/full_name; this module expects username / TokenResponse). Do not
swap the router to this module without aligning schemas and user documents.
"""

from datetime import datetime

from fastapi import HTTPException, status

from backend.database import users_collection
from backend.core.security import hash_password, verify_password, create_access_token
from backend.schemas import UserCreate, UserLogin, TokenResponse, UserOut

# ─── Register ─────────────────────────────────────────────────────────────────

async def handle_register(data: UserCreate) -> UserOut:
    """
    Register a new user.
    - Checks email uniqueness.
    - Stores hashed password.
    - Returns the created user (no password).
    """
    existing = await users_collection.find_one({"email": data.email.lower()})
    if existing:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "An account with this email already exists.",
        )

    user_doc = {
        "_id":        data.email.lower(),   # use email as _id for uniqueness
        "email":      data.email.lower(),
        "username":   data.username,
        "password":   hash_password(data.password),
        "created_at": datetime.utcnow(),
    }
    await users_collection.insert_one(user_doc)

    return UserOut(email=user_doc["email"], username=user_doc["username"])

# ─── Login ────────────────────────────────────────────────────────────────────

async def handle_login(data: UserLogin) -> TokenResponse:
    """
    Authenticate a user and return a JWT access token.
    """
    user = await users_collection.find_one({"email": data.email.lower()})

    if user is None or not verify_password(data.password, user["password"]):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Incorrect email or password.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user["_id"]})

    return TokenResponse(
        access_token = token,
        token_type   = "bearer",
        username     = user["username"],
        email        = user["email"],
    )
