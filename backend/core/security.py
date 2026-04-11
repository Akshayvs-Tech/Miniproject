"""
backend/core/security.py
Password hashing and JWT encode/decode utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

JWT_SECRET      = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ─── Password Hashing ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain text password using bcrypt."""
    pwd_bytes = plain.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain text password against a hashed one."""
    try:
        pwd_bytes = plain.encode('utf-8')
        hashed_bytes = hashed.encode('utf-8')
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)
    except Exception:
        return False

# ─── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token.
    `data` should contain at minimum {"sub": user_id_string}.
    """
    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload.update({"exp": expire})
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.
    Returns the payload dict, or None if invalid/expired.
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None
