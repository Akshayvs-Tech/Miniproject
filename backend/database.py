"""
backend/database.py
MongoDB async connection — shared across all backend modules.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()   # loads .env from project root

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "person_finder")

client             = AsyncIOMotorClient(MONGO_URI)
db                 = client[DB_NAME]
results_collection = db["results"]
users_collection   = db["users"]
