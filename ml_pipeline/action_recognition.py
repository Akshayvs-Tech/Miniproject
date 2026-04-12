"""
ml_pipeline/action_recognition.py
Integrates LSTM-based person action recognition and optional SlowFast scene analysis.
Adapted from user-provided demo.py.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from collections import deque
from pathlib import Path

# --- Constants & Config (Sync with slowfast_model/demo.py where possible) ---
SLOWFAST_DIR      = Path(__file__).resolve().parent / "slowfast_model"
LSTM_CHECKPOINT   = SLOWFAST_DIR / "action_lstm.pth"
CRIME_CHECKPOINT  = SLOWFAST_DIR / "slowfast_ucfcrime_final.pth"

SEQ_LEN           = 30
NUM_FRAMES        = 32
ALPHA             = 4

# Thresholds
ACTION_THRESHOLD  = 0.60
FIGHT_THRESHOLD   = 0.85
FIGHT_VOTE_NEEDED = 5
VOTE_WINDOW       = 8
FIGHT_MIN_SECONDS = 2.0

# Rule thresholds
RULE_HOLD_FRAMES  = 6
STAND_MOTION_MAX  = 0.025
STAND_KNEE_MIN    = 145
BACK_VISIBILITY_MAX = 0.35
MIN_PERSON_H      = 60

# MediaPipe landmark indices
LM_MAP = {
    "nose":            0,
    "left_eye":        2,  "right_eye":        5,
    "left_ear":        7,  "right_ear":        8,
    "left_shoulder":  11,  "right_shoulder":  12,
    "left_elbow":     13,  "right_elbow":     14,
    "left_wrist":     15,  "right_wrist":     16,
    "left_hip":       23,  "right_hip":       24,
    "left_knee":      25,  "right_knee":      26,
    "left_ankle":     27,  "right_ankle":     28,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Model Class ──────────────────────────────────────────────────────────────

class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.classifier(out)

# ─── Internal State ────────────────────────────────────────────────────────────

_lstm_model    = None
_action_classes = []
_mp_pose       = None

# ─── Initialization ───────────────────────────────────────────────────────────

def init_models():
    global _lstm_model, _action_classes, _mp_pose
    
    # 1. Load LSTM
    if not LSTM_CHECKPOINT.exists():
        print(f"⚠️ Action LSTM checkpoint not found at {LSTM_CHECKPOINT}")
    else:
        try:
            ckpt = torch.load(LSTM_CHECKPOINT, map_location=device)
            _action_classes = ckpt["classes"]
            _lstm_model = ActionLSTM(
                input_size  = ckpt["input_size"],
                hidden_size = ckpt["hidden_size"],
                num_layers  = ckpt["num_layers"],
                num_classes = ckpt["num_classes"],
            ).to(device)
            _lstm_model.load_state_dict(ckpt["model_state_dict"])
            _lstm_model.eval()
            print(f"✅ Action Recognition LSTM loaded: {_action_classes}")
        except Exception as e:
            print(f"❌ Failed to load Action LSTM: {e}")

    # 2. Setup MediaPipe
    try:
        mp_p = mp.solutions.pose
        _mp_pose = mp_p.Pose(
            static_image_mode        = False,
            model_complexity         = 1,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5
        )
        print("✅ MediaPipe Pose initialized")
    except Exception as e:
        print(f"❌ Failed to init MediaPipe: {e}")

# ─── Helper Geometry ──────────────────────────────────────────────────────────

def _get_lm(kp, name):
    idx = LM_MAP[name]
    return kp[idx*3 : idx*3 + 2]

def _angle_between(a, b, c):
    ba    = np.array(a) - np.array(b)
    bc    = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def _motion_magnitude(kp_history):
    if len(kp_history) < 2:
        return 0.0
    recent = list(kp_history)[-6:]
    hips   = [kp[LM_MAP["left_hip"]*3 : LM_MAP["left_hip"]*3 + 2] for kp in recent]
    diffs  = [np.linalg.norm(np.array(hips[i]) - np.array(hips[i-1]))
              for i in range(1, len(hips))]
    return float(np.mean(diffs)) if diffs else 0.0

# ─── Action Logic ─────────────────────────────────────────────────────────────

def extract_keypoints(crop):
    if _mp_pose is None or crop is None or crop.size == 0:
        return np.zeros(99), 0.0, None
    
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = _mp_pose.process(rgb)
    
    kp = np.zeros(99)
    if result.pose_landmarks:
        kp = np.array([
            [lm.x, lm.y, lm.z]
            for lm in result.pose_landmarks.landmark
        ]).flatten()

    # Visibility
    vis = 0.0
    if result.pose_landmarks:
        key_ids = [11, 12, 13, 14, 15, 16, 23, 24]
        vis_scores = [result.pose_landmarks.landmark[i].visibility for i in key_ids]
        vis = float(np.mean(vis_scores))
        
    return kp, vis, result

def rule_based_action(kp, kp_history, mp_results):
    if np.all(kp == 0) or mp_results is None:
        return None, 0.0

    # Facing away check
    nose_vis = mp_results.pose_landmarks.landmark[0].visibility
    nose_z   = kp[LM_MAP["nose"]*3 + 2]
    l_ear_z  = kp[LM_MAP["left_ear"]*3 + 2]
    r_ear_z  = kp[LM_MAP["right_ear"]*3 + 2]
    avg_ear_z = (l_ear_z + r_ear_z) / 2
    if nose_vis < BACK_VISIBILITY_MAX and nose_z < -0.05 and avg_ear_z > nose_z + 0.04:
        return "turned_away", 0.85

    # Extract landmarks for pose
    l_hip   = _get_lm(kp, "left_hip")
    r_hip   = _get_lm(kp, "right_hip")
    l_knee  = _get_lm(kp, "left_knee")
    r_knee  = _get_lm(kp, "right_knee")
    l_ankle = _get_lm(kp, "left_ankle")
    r_ankle = _get_lm(kp, "right_ankle")
    l_sh    = _get_lm(kp, "left_shoulder")
    r_sh    = _get_lm(kp, "right_shoulder")

    l_ka = _angle_between(l_hip, l_knee, l_ankle)
    r_ka = _angle_between(r_hip, r_knee, r_ankle)
    avg_ka = (l_ka + r_ka) / 2

    # Y values
    avg_hip_y   = (l_hip[1]   + r_hip[1])   / 2
    avg_knee_y  = (l_knee[1]  + r_knee[1])  / 2
    avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    avg_sh_y    = (l_sh[1]    + r_sh[1])    / 2

    torso_ratio  = (avg_hip_y - avg_sh_y) / (avg_ankle_y - avg_sh_y + 1e-6)
    hip_knee_gap = avg_knee_y - avg_hip_y
    motion       = _motion_magnitude(kp_history)

    if avg_ka < 120 or (avg_ka < 145 and hip_knee_gap < 0.08):
        return "sit", 0.88
    
    if avg_ka > STAND_KNEE_MIN and torso_ratio > 0.35 and motion < STAND_MOTION_MAX:
        return "stand", 0.90

    return None, 0.0

def predict_action(tid, kp, kp_history, kp_buffer, mp_results, visibility, vote_deque, fight_timer_ref, fps):
    global _lstm_model, _action_classes

    # 1. Rules first
    label, conf = rule_based_action(kp, kp_history, mp_results)
    if label:
        return label, conf, "rule"

    # 2. LSTM second
    if _lstm_model and len(kp_buffer) == SEQ_LEN:
        x = torch.FloatTensor(np.array(kp_buffer)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs     = F.softmax(_lstm_model(x), dim=1)
            raw_conf, idx = probs.max(1)
        
        raw_label = _action_classes[idx.item()]
        motion = _motion_magnitude(kp_history)

        # Apply filtering
        final_label, final_conf = raw_label, raw_conf.item()
        
        if raw_label in ("walk", "run") and motion < 0.012:
            final_label, final_conf = "stand", 0.85
        
        if raw_label == "fight" and visibility < 0.55:
            final_label, final_conf = "interacting", 0.65
        
        # Fight voting logic
        vote_deque.append((final_label, final_conf))
        if final_label == "fight":
            if final_conf < FIGHT_THRESHOLD or [l for l,c in vote_deque].count("fight") < FIGHT_VOTE_NEEDED:
                final_label, final_conf = "interacting", 0.60
            else:
                fight_timer_ref[0] += (1.0 / fps)
                if fight_timer_ref[0] < FIGHT_MIN_SECONDS:
                    return "interacting", 0.60, "lstm"
        else:
            fight_timer_ref[0] = 0.0

        if final_conf >= ACTION_THRESHOLD:
            return final_label, final_conf, "lstm"
        
        return "interacting", 0.55, "lstm"

    # 3. Fallback/Buffer filling
    return "analyzing...", 0.3, "none"

def cleanup():
    if _mp_pose:
        _mp_pose.close()
