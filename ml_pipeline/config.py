"""
ml_pipeline/config.py
Central configuration for all ML models and pipeline settings.
"""

import sys
from pathlib import Path

# Ensure local torchreid source is importable from project root
_TORCHREID_SRC = Path(__file__).resolve().parent
if str(_TORCHREID_SRC) not in sys.path:
    sys.path.insert(0, str(_TORCHREID_SRC))

# ─── Project Paths ────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent.parent          # Miniproject/
DATA_DIR    = ROOT_DIR / "data"
INPUT_DIR   = DATA_DIR / "inputs"
OUTPUT_DIR  = DATA_DIR / "outputs"

MODELS_DIR  = Path(__file__).resolve().parent / "models"

# ─── Model Weights ────────────────────────────────────────────────────────────

YOLO_WEIGHTS = ROOT_DIR / "yolov8n.pt"    # yolov8n.pt lives at project root

# ─── ArcFace Config ───────────────────────────────────────────────────────────

ARCFACE_MODEL_NAME       = "buffalo_l"
ARCFACE_SIMILARITY_THRESHOLD = 0.55       # raised from 0.4 to reduce false matches
ARCFACE_MIN_CROP_SIZE    = 60             # minimum crop dimension (px) to attempt face match

# ─── OSNet Config ─────────────────────────────────────────────────────────────

OSNET_MODEL_NAME         = "osnet_x1_0"
OSNET_NUM_CLASSES        = 751
OSNET_SIMILARITY_THRESHOLD = 0.6          # from SRS spec
OSNET_MIN_CROP_H         = 100            # minimum crop height (px) for body match
OSNET_MIN_CROP_W         = 50             # minimum crop width  (px) for body match
OSNET_INPUT_SIZE         = (256, 128)     # (H, W) expected by OSNet
OSNET_MEAN               = [0.485, 0.456, 0.406]
OSNET_STD                = [0.229, 0.224, 0.225]

# ─── Pipeline Config ──────────────────────────────────────────────────────────

FRAME_SKIP               = 2              # process every N-th frame
DEEPSORT_MAX_AGE         = 30
DEEPSORT_N_INIT          = 3
DEEPSORT_NN_BUDGET       = 100
DEEPSORT_MAX_COSINE_DIST = 0.3

# ─── Ensure data dirs exist ───────────────────────────────────────────────────

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
