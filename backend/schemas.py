"""
backend/schemas.py
Pydantic response/request schemas shared across routes and controllers.
"""

from typing import Optional
from pydantic import BaseModel

class MatchRecord(BaseModel):
    frame:         int
    timestamp_sec: float
    track_id:      int
    similarity:    float

class ProcessResult(BaseModel):
    found:                bool
    total_match_frames:   int
    first_appearance_sec: Optional[float] = None
    last_appearance_sec:  Optional[float] = None
    best_match:           Optional[MatchRecord] = None
    matched_track_ids:    list[int]
    output_video_id:      Optional[str] = None

# ─── Auth Schemas ─────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str
    full_name: Optional[str] = None

class UserLogin(UserBase):
    password: str

class UserOut(UserBase):
    id: str
    full_name: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
