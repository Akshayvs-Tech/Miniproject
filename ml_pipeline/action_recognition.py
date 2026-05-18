"""
ml_pipeline/action_recognition.py
LSTM + rule-based person action recognition (MediaPipe pose → ActionLSTM).
"""

from __future__ import annotations

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ---
SLOWFAST_DIR    = Path(__file__).resolve().parent / "slowfast_model"
LSTM_CHECKPOINT = SLOWFAST_DIR / "action_lstm.pth"

# --- Sequence / thresholds ---
SEQ_LEN              = 30
MIN_SEQ_LEN          = 12          # partial LSTM (padded to SEQ_LEN)
ACTION_THRESHOLD     = 0.52
LOW_CONF_THRESHOLD   = 0.38        # prefer real class over "interacting"
FIGHT_THRESHOLD      = 0.85
FIGHT_VOTE_NEEDED    = 4
VOTE_WINDOW          = 10
FIGHT_MIN_SECONDS    = 1.5
STABLE_VOTE_MIN      = 3           # frames needed for stable label

STAND_MOTION_MAX     = 0.017
SIT_MOTION_MAX       = 0.020
WALK_MOTION_MIN      = 0.014
RUN_MOTION_MIN       = 0.035
STAND_KNEE_MIN       = 145
BACK_VISIBILITY_MAX  = 0.35
MIN_PERSON_H         = 80          # upscale crop below this height

GENERIC_LABELS = frozenset({"analyzing...", "interacting", "unknown"})

LM_MAP = {
    "nose": 0, "left_eye": 2, "right_eye": 5,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

_lstm_model: ActionLSTM | None = None
_action_classes: list[str] = []
_mp_pose = None                    # legacy solutions API
_pose_landmarker = None            # mediapipe.tasks (0.10.30+)
_pose_backend: str = "none"        # "legacy" | "tasks" | "none"

POSE_MODEL_PATH = SLOWFAST_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out)


@dataclass
class TrackActionState:
    """Per DeepSORT track — buffers pose history and stable action label."""
    track_id: int
    kp_buffer: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    kp_history: deque = field(default_factory=lambda: deque(maxlen=12))
    vote_deque: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    fight_timer: list = field(default_factory=lambda: [0.0])
    label_history: deque = field(default_factory=lambda: deque(maxlen=40))
    last_label: str = "analyzing..."
    last_conf: float = 0.0
    last_src: str = "none"
    stable_label: str = "analyzing..."
    stable_conf: float = 0.0

    def update(self, crop, fps: float) -> tuple[str, float, str]:
        kp_raw, vis, nose_vis = extract_keypoints(crop)

        # Temporal smoothing: average last few keypoints including current frame
        hist_preview = list(self.kp_history) + [kp_raw]
        if len(hist_preview) >= 3:
            smoothed = np.mean(hist_preview[-3:], axis=0)
        else:
            smoothed = np.mean(hist_preview, axis=0)

        kp = smoothed.astype(float)

        # Append smoothed keypoints to buffers
        self.kp_buffer.append(kp)
        self.kp_history.append(kp)

        label, conf, src = predict_action(
            kp, self.kp_history, self.kp_buffer, nose_vis, vis,
            self.vote_deque, self.fight_timer, fps,
        )
        self.last_label, self.last_conf, self.last_src = label, conf, src
        if label not in GENERIC_LABELS:
            self.label_history.append((label, conf))

        self._refresh_stable()
        return self.last_label, self.last_conf, self.last_src

    def _refresh_stable(self):
        if not self.label_history:
            if self.last_label not in GENERIC_LABELS:
                self.stable_label, self.stable_conf = self.last_label, self.last_conf
            return

        counts = Counter(lbl for lbl, _ in self.label_history)
        top, n = counts.most_common(1)[0]
        if n >= STABLE_VOTE_MIN:
            confs = [c for lbl, c in self.label_history if lbl == top]
            self.stable_label = top
            self.stable_conf = float(np.mean(confs))

    def display_action(self) -> str:
        """Best label for UI / match records — prefers stable history over transient states."""
        if self.stable_label not in GENERIC_LABELS:
            return self.stable_label
        if self.last_label not in GENERIC_LABELS:
            return self.last_label
        buf = len(self.kp_buffer)
        if buf < MIN_SEQ_LEN:
            return f"analyzing ({buf}/{SEQ_LEN})"
        return self.last_label


# ─── Init / cleanup ───────────────────────────────────────────────────────────

def _init_pose_backend():
    """MediaPipe 0.10.30+ removed solutions — use Tasks API with auto model download."""
    global _mp_pose, _pose_landmarker, _pose_backend

    if hasattr(mp, "solutions"):
        try:
            _mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.50,
                min_tracking_confidence=0.50,
            )
            _pose_backend = "legacy"
            print("[Action] MediaPipe Pose (legacy solutions) initialized")
            return
        except Exception as e:
            print(f"[Action] Legacy MediaPipe init failed: {e}")

    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python as mp_tasks

        if not POSE_MODEL_PATH.exists():
            print(f"[Action] Downloading pose model to {POSE_MODEL_PATH} ...")
            SLOWFAST_DIR.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        _pose_backend = "tasks"
        print("[Action] MediaPipe Pose Landmarker (tasks API) initialized")
    except Exception as e:
        _pose_backend = "none"
        print(f"[Action] MediaPipe Pose unavailable: {e}")


def init_models():
    global _lstm_model, _action_classes

    if not LSTM_CHECKPOINT.exists():
        print(f"[Action] LSTM checkpoint not found at {LSTM_CHECKPOINT}")
    else:
        try:
            ckpt = torch.load(LSTM_CHECKPOINT, map_location=device, weights_only=False)
            _action_classes = list(ckpt["classes"])
            _lstm_model = ActionLSTM(
                input_size=ckpt["input_size"],
                hidden_size=ckpt["hidden_size"],
                num_layers=ckpt["num_layers"],
                num_classes=ckpt["num_classes"],
            ).to(device)
            _lstm_model.load_state_dict(ckpt["model_state_dict"])
            _lstm_model.eval()
            print(f"[Action] LSTM loaded: {_action_classes}")
        except Exception as e:
            print(f"[Action] Failed to load LSTM: {e}")

    _init_pose_backend()


def cleanup():
    global _mp_pose, _pose_landmarker, _pose_backend
    if _mp_pose:
        _mp_pose.close()
        _mp_pose = None
    if _pose_landmarker:
        _pose_landmarker.close()
        _pose_landmarker = None
    _pose_backend = "none"


def create_track_state(track_id: int) -> TrackActionState:
    return TrackActionState(track_id=track_id)


def summarize_track_action(state: TrackActionState) -> str:
    """Final label after video ends — for backfilling match rows."""
    return state.display_action()


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _get_lm(kp, name):
    idx = LM_MAP[name]
    return kp[idx * 3: idx * 3 + 2]


def _angle_between(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _motion_magnitude(kp_history) -> float:
    if len(kp_history) < 2:
        return 0.0
    recent = list(kp_history)[-6:]
    hips = [kp[LM_MAP["left_hip"] * 3: LM_MAP["left_hip"] * 3 + 2] for kp in recent]
    diffs = [
        float(np.linalg.norm(np.array(hips[i]) - np.array(hips[i - 1])))
        for i in range(1, len(hips))
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def _prepare_lstm_sequence(kp_buffer) -> list | None:
    buf = list(kp_buffer)
    if len(buf) < MIN_SEQ_LEN:
        return None
    if len(buf) < SEQ_LEN:
        pad = [buf[0]] * (SEQ_LEN - len(buf))
        buf = pad + buf
    # Normalize sequence to be translation- and scale-invariant.
    def _normalize_frame(kp):
        # kp is (99,) => 33 landmarks of (x,y,z)
        kp = np.array(kp).reshape(-1, 3)
        # center on hips midpoint
        left_hip = kp[LM_MAP['left_hip']]
        right_hip = kp[LM_MAP['right_hip']]
        center = (left_hip + right_hip) / 2.0

        # scale by torso height (shoulder midpoint to ankle midpoint)
        left_sh = kp[LM_MAP['left_shoulder']]
        right_sh = kp[LM_MAP['right_shoulder']]
        sh_mid = (left_sh + right_sh) / 2.0
        left_ank = kp[LM_MAP['left_ankle']]
        right_ank = kp[LM_MAP['right_ankle']]
        an_mid = (left_ank + right_ank) / 2.0
        torso_h = np.linalg.norm(sh_mid - an_mid) + 1e-6

        norm = (kp - center) / torso_h
        return norm.flatten()

    norm_buf = [_normalize_frame(f) for f in buf]
    return norm_buf


# ─── Keypoints ────────────────────────────────────────────────────────────────

def extract_keypoints(crop):
    """Returns (kp[99], mean_visibility, nose_visibility)."""
    if _pose_backend == "none" or crop is None or crop.size == 0:
        return np.zeros(99), 0.0, 0.0

    h, w = crop.shape[:2]
    if min(h, w) < MIN_PERSON_H:
        scale = MIN_PERSON_H / max(min(h, w), 1)
        crop = cv2.resize(
            crop,
            (max(int(w * scale), 1), max(int(h * scale), 1)),
            interpolation=cv2.INTER_LINEAR,
        )

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    kp = np.zeros(99)
    vis = 0.0
    nose_vis = 0.0

    if _pose_backend == "legacy" and _mp_pose is not None:
        result = _mp_pose.process(rgb)
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            kp = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
            key_ids = [11, 12, 13, 14, 15, 16, 23, 24]
            vis = float(np.mean([lms[i].visibility for i in key_ids]))
            nose_vis = float(lms[0].visibility)
        return kp, vis, nose_vis

    if _pose_backend == "tasks" and _pose_landmarker is not None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _pose_landmarker.detect(mp_image)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            kp = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
            key_ids = [11, 12, 13, 14, 15, 16, 23, 24]
            vis = float(np.mean([lms[i].visibility for i in key_ids]))
            nose_vis = float(lms[0].visibility)
        return kp, vis, nose_vis

    return kp, vis, nose_vis


# ─── Rules ────────────────────────────────────────────────────────────────────

def instant_posture_guess(kp, kp_history) -> tuple[str, float]:
    """Quick label while the LSTM buffer is still filling."""
    if np.all(kp == 0):
        return "stand", 0.35

    motion = _motion_magnitude(kp_history)
    if motion >= RUN_MOTION_MIN:
        return "run", 0.55
    if motion >= WALK_MOTION_MIN:
        return "walk", 0.50

    l_hip, r_hip = _get_lm(kp, "left_hip"), _get_lm(kp, "right_hip")
    l_knee, r_knee = _get_lm(kp, "left_knee"), _get_lm(kp, "right_knee")
    l_ankle, r_ankle = _get_lm(kp, "left_ankle"), _get_lm(kp, "right_ankle")
    l_sh, r_sh = _get_lm(kp, "left_shoulder"), _get_lm(kp, "right_shoulder")

    avg_ka = (_angle_between(l_hip, l_knee, l_ankle) + _angle_between(r_hip, r_knee, r_ankle)) / 2
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    avg_knee_y = (l_knee[1] + r_knee[1]) / 2
    avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    avg_sh_y = (l_sh[1] + r_sh[1]) / 2
    torso_ratio = (avg_hip_y - avg_sh_y) / (avg_ankle_y - avg_sh_y + 1e-6)
    hip_knee_gap = avg_knee_y - avg_hip_y

    if avg_ka < 120 or (avg_ka < 145 and hip_knee_gap < 0.08):
        if motion < SIT_MOTION_MAX:
            return "sit", 0.72
    if avg_ka > STAND_KNEE_MIN and torso_ratio > 0.30 and motion < STAND_MOTION_MAX:
        return "stand", 0.70
    return "stand", 0.45


def motion_heuristic(kp_history) -> tuple[str, float] | tuple[None, float]:
    motion = _motion_magnitude(kp_history)
    if motion >= RUN_MOTION_MIN:
        return "run", min(0.75, 0.45 + motion * 4)
    if motion >= WALK_MOTION_MIN:
        return "walk", min(0.72, 0.42 + motion * 5)
    if motion < STAND_MOTION_MAX:
        return "stand", 0.65
    return None, 0.0


def rule_based_action(kp, kp_history, nose_vis: float):
    if np.all(kp == 0):
        return None, 0.0

    nose_z = kp[LM_MAP["nose"] * 3 + 2]
    l_ear_z = kp[LM_MAP["left_ear"] * 3 + 2]
    r_ear_z = kp[LM_MAP["right_ear"] * 3 + 2]
    if (
        nose_vis < BACK_VISIBILITY_MAX
        and nose_z < -0.05
        and (l_ear_z + r_ear_z) / 2 > nose_z + 0.04
    ):
        return "turned_away", 0.85

    l_hip, r_hip = _get_lm(kp, "left_hip"), _get_lm(kp, "right_hip")
    l_knee, r_knee = _get_lm(kp, "left_knee"), _get_lm(kp, "right_knee")
    l_ankle, r_ankle = _get_lm(kp, "left_ankle"), _get_lm(kp, "right_ankle")
    l_sh, r_sh = _get_lm(kp, "left_shoulder"), _get_lm(kp, "right_shoulder")
    l_wrist, r_wrist = _get_lm(kp, "left_wrist"), _get_lm(kp, "right_wrist")
    nose = _get_lm(kp, "nose")

    l_ka = _angle_between(l_hip, l_knee, l_ankle)
    r_ka = _angle_between(r_hip, r_knee, r_ankle)
    avg_ka = (l_ka + r_ka) / 2

    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    avg_knee_y = (l_knee[1] + r_knee[1]) / 2
    avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    avg_sh_y = (l_sh[1] + r_sh[1]) / 2
    torso_ratio = (avg_hip_y - avg_sh_y) / (avg_ankle_y - avg_sh_y + 1e-6)
    hip_knee_gap = avg_knee_y - avg_hip_y
    motion = _motion_magnitude(kp_history)

    if avg_ka < 120 or (avg_ka < 145 and hip_knee_gap < 0.08):
        if motion < SIT_MOTION_MAX:
            return "sit", 0.88

    knee_diff = abs(l_ka - r_ka)
    if knee_diff > 28 and avg_ka < 145 and torso_ratio < 0.50 and motion > 0.01:
        return "riding_lmv", 0.80

    face_y = nose[1]
    wrist_near = (
        (abs(l_wrist[1] - face_y) < 0.15 and abs(l_wrist[0] - nose[0]) < 0.20)
        or (abs(r_wrist[1] - face_y) < 0.15 and abs(r_wrist[0] - nose[0]) < 0.20)
    )
    if wrist_near and motion < STAND_MOTION_MAX:
        return "stand", 0.78

    if avg_ka > STAND_KNEE_MIN and torso_ratio > 0.35 and motion < STAND_MOTION_MAX:
        return "stand", 0.90

    mh, mc = motion_heuristic(kp_history)
    if mh and mh != "stand":
        return mh, mc

    return None, 0.0


def _run_lstm(kp_buffer) -> tuple[str, float, np.ndarray]:
    seq = _prepare_lstm_sequence(kp_buffer)
    if not _lstm_model or not seq:
        return "", 0.0, np.array([])

    x = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(_lstm_model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return _action_classes[idx], float(probs[idx]), probs


def _filter_lstm_label(
    raw_label: str,
    raw_conf: float,
    probs: np.ndarray,
    motion: float,
    visibility: float,
    vote_deque: deque,
    fight_timer_ref: list,
    fps: float,
    kp_history,
) -> tuple[str, float]:
    label, conf = raw_label, raw_conf

    if label in ("walk", "run") and motion < (WALK_MOTION_MIN * 0.8):
        label, conf = "stand", 0.85

    if label == "fight" and visibility < 0.50:
        # downgrade fight — pick next-best non-fight class
        order = np.argsort(probs)[::-1]
        for i in order:
            cand = _action_classes[int(i)]
            if cand != "fight" and probs[i] >= LOW_CONF_THRESHOLD:
                return cand, float(probs[i])
        label, conf = "stand", 0.55

    vote_deque.append((label, conf))

    if label == "fight":
        recent = [l for l, _ in vote_deque]
        if conf < FIGHT_THRESHOLD or recent.count("fight") < FIGHT_VOTE_NEEDED:
            order = np.argsort(probs)[::-1]
            for i in order:
                cand = _action_classes[int(i)]
                if cand != "fight":
                    return cand, float(probs[i])
        else:
            fight_timer_ref[0] += 1.0 / max(fps, 1.0)
            if fight_timer_ref[0] < FIGHT_MIN_SECONDS:
                return "fight", conf * 0.9
        return "fight", conf
    else:
        fight_timer_ref[0] = 0.0

    if conf >= ACTION_THRESHOLD:
        return label, conf

    if conf >= LOW_CONF_THRESHOLD:
        return label, conf

    # second-best class if close
    order = np.argsort(probs)[::-1]
    if len(order) >= 2:
        i2 = int(order[1])
        if probs[i2] >= LOW_CONF_THRESHOLD and (probs[order[0]] - probs[i2]) < 0.12:
            return _action_classes[i2], float(probs[i2])

    mh, mc = motion_heuristic(kp_history)
    if mh:
        return mh, mc
    return label, max(conf, LOW_CONF_THRESHOLD * 0.9)


def predict_action(
    kp,
    kp_history,
    kp_buffer,
    nose_vis,
    visibility,
    vote_deque,
    fight_timer_ref,
    fps,
):
    motion = _motion_magnitude(kp_history)

    label, conf = rule_based_action(kp, kp_history, nose_vis)
    if label:
        return label, conf, "rule"

    buf_len = len(kp_buffer)
    if buf_len < MIN_SEQ_LEN:
        guess = instant_posture_guess(kp, kp_history)
        return guess[0], guess[1], "guess"

    if _lstm_model:
        raw_label, raw_conf, probs = _run_lstm(kp_buffer)
        if raw_label:
            final_label, final_conf = _filter_lstm_label(
                raw_label, raw_conf, probs, motion, visibility,
                vote_deque, fight_timer_ref, fps, kp_history,
            )
            if final_label not in GENERIC_LABELS:
                return final_label, final_conf, "lstm"

    mh, mc = motion_heuristic(kp_history)
    if mh:
        return mh, mc, "motion"

    return instant_posture_guess(kp, kp_history)[0], 0.45, "guess"
