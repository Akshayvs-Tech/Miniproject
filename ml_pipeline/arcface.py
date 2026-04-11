"""
ml_pipeline/arcface.py
ArcFace face recognition — embedding extraction only.
All config imported from config.py.
"""

import cv2
import numpy as np
import insightface

from ml_pipeline.config import (
    ARCFACE_MODEL_NAME,
    ARCFACE_SIMILARITY_THRESHOLD,
    ARCFACE_MIN_CROP_SIZE,
)

# ─── Load Model (once at import) ─────────────────────────────────────────────

arcface_model = insightface.app.FaceAnalysis(name=ARCFACE_MODEL_NAME)
arcface_model.prepare(ctx_id=-1)   # ctx_id=-1 = CPU; set 0 for GPU

SIMILARITY_THRESHOLD = ARCFACE_SIMILARITY_THRESHOLD
MIN_CROP_SIZE        = ARCFACE_MIN_CROP_SIZE

# ─── Public API ───────────────────────────────────────────────────────────────

def get_embedding(image) -> np.ndarray | None:
    """
    Extract ArcFace face embedding from a file path or numpy BGR array.
    Returns a normalized 512-d embedding, or None if no face detected.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    if img is None or img.size == 0:
        return None

    faces = arcface_model.get(img)
    if not faces:
        return None

    return faces[0].normed_embedding
