"""
ml_pipeline/osnet_reid.py
OSNet body re-identification — embedding extraction + YOLO person detection.
All config imported from config.py.
"""

import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from ml_pipeline.config import (
    YOLO_WEIGHTS,
    OSNET_MODEL_NAME,
    OSNET_NUM_CLASSES,
    OSNET_SIMILARITY_THRESHOLD,
    OSNET_MIN_CROP_H,
    OSNET_MIN_CROP_W,
    OSNET_INPUT_SIZE,
    OSNET_MEAN,
    OSNET_STD,
    FRAME_SKIP,
)

# ─── Device ───────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OSNet] Running on: {device}")

# ─── Load Models (once at import) ────────────────────────────────────────────

yolo_model = YOLO(str(YOLO_WEIGHTS))

osnet_model = torchreid.models.build_model(
    name=OSNET_MODEL_NAME,
    num_classes=OSNET_NUM_CLASSES,
    pretrained=True,
)
osnet_model = osnet_model.to(device)
osnet_model.eval()

# ─── Preprocessing ────────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize(OSNET_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=OSNET_MEAN, std=OSNET_STD),
])

# ─── Public Constants ─────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = OSNET_SIMILARITY_THRESHOLD
MIN_CROP_H           = OSNET_MIN_CROP_H
MIN_CROP_W           = OSNET_MIN_CROP_W

# ─── Public API ───────────────────────────────────────────────────────────────

def get_embedding(image) -> np.ndarray | None:
    """
    Extract OSNet 512-d body embedding from a file path or numpy BGR array.
    Returns a normalized embedding, or None on failure.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return None
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        return None

    tensor = _transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = osnet_model(tensor)

    emb = emb.squeeze().cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def detect_persons(frame: np.ndarray) -> list[tuple]:
    """
    Run YOLOv8 on a BGR frame.
    Returns list of (x1, y1, x2, y2, confidence) for all detected persons.
    """
    results = yolo_model(frame, classes=[0], verbose=False)[0]
    return [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
         int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0]))
        for b in results.boxes
    ]
