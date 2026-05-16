"""
backend/database.py
MongoDB async connection — shared across all backend modules.

IMPORTANT: pymongo>=4 + motor>=3 with mongodb+srv:// URIs resolve DNS
synchronously during MongoClient() construction (inside parse_uri).
This means even AsyncIOMotorClient crashes at constructor time if Atlas DNS
is unreachable.  We use a true lazy factory — no client object is created at
module import time.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

MONGO_URI: str = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or "").strip()
DB_NAME:   str = os.getenv("DB_NAME", "person_finder")

_LOCAL_MONGO = "mongodb://127.0.0.1:27017"

if not MONGO_URI:
    MONGO_URI = _LOCAL_MONGO
    print(
        f"⚠️  MONGO_URI not set — using local MongoDB at {_LOCAL_MONGO}. "
        "Set MONGO_URI in Miniproject/.env for Atlas or another host."
    )

# ── Lazy factory ──────────────────────────────────────────────────────────────
# Nothing below creates a MongoClient at import time.

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return (and lazily create) the shared Motor client."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client


def get_db():
    return get_client()[DB_NAME]


def get_results_collection():
    return get_client()[DB_NAME]["results"]


def get_users_collection():
    return get_client()[DB_NAME]["users"]


# ── Backward-compat shims ─────────────────────────────────────────────────────
# Existing routes import `results_collection` and `users_collection` as module
# globals.  We expose async-aware proxy objects so they keep working.

class _AsyncCollectionProxy:
    """Proxy to a Motor collection; defers client creation to first call."""

    def __init__(self, name: str):
        self._name = name

    def _col(self):
        return get_client()[DB_NAME][self._name]

    # ── async helpers used directly by routes ──
    async def find_one(self, *args, **kwargs):
        return await self._col().find_one(*args, **kwargs)

    async def insert_one(self, *args, **kwargs):
        return await self._col().insert_one(*args, **kwargs)

    def find(self, *args, **kwargs):
        return self._col().find(*args, **kwargs)

    # Forward everything else (count_documents, aggregate, …)
    def __getattr__(self, item):
        return getattr(self._col(), item)


results_collection = _AsyncCollectionProxy("results")
users_collection   = _AsyncCollectionProxy("users")

# `client` and `db` are accessed only in health.py and app.py lifespan — both
# now use get_client() directly after the fixes applied there.  Keep these as
# thin aliases just in case any other import needs them.
class _ClientProxy:
    def __getattr__(self, item):
        return getattr(get_client(), item)
    def __getitem__(self, key):
        return get_client()[key]

client = _ClientProxy()   # do NOT call get_client() here — stays lazy
db     = None              # not used directly; routes use the collection proxies
