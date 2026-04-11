"""
backend/routes/auth.py
Authentication routes for User Registration and Login.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from backend.database import users_collection
from backend.schemas import UserCreate, UserLogin, UserOut, Token
from backend.core.security import hash_password, verify_password, create_access_token
from backend.core.dependencies import get_current_user
import uuid

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_pass = hash_password(user_data.password)
    
    new_user = {
        "_id": user_id,
        "email": user_data.email,
        "password": hashed_pass,
        "full_name": user_data.full_name
    }
    
    await users_collection.insert_one(new_user)
    
    return {
        "id": user_id,
        "email": user_data.email,
        "full_name": user_data.full_name
    }

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = await users_collection.find_one({"email": credentials.email})
    
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["_id"]})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["_id"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name")
    }
