# =====================================================
# SCRIPT 3 — demo.py
#
# Full CCTV Action Recognition Pipeline:
#   YOLOv8n       → detect each person
#   MediaPipe     → extract body keypoints per person
#   Rule-based    → stand, sit, talk, ride_bike, turned_away (instant)
#   LSTM          → walk, run, fight, ride (needs 30 frames)
#   Fallback      → "interacting" for unrecognized actions
#   SlowFast UCF  → scene-level crime detection
#
# Usage:
#   python demo.py --source video.mp4
#   python demo.py --source 0
# =====================================================

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import argparse
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r50


# =====================================================
# Config
# =====================================================

LSTM_CHECKPOINT   = "action_lstm.pth"
CRIME_CHECKPOINT  = "slowfast_ucfcrime_final.pth"
SEQ_LEN           = 30
NUM_FRAMES        = 32
ALPHA             = 4

# SlowFast
CRIME_THRESHOLD    = 0.30
SKIP_NORMAL_VIDEOS = True

# LSTM thresholds
ACTION_THRESHOLD  = 0.60
FIGHT_THRESHOLD   = 0.85
FIGHT_VOTE_NEEDED = 5
VOTE_WINDOW       = 8
FIGHT_MIN_SECONDS = 2.0

# Rule thresholds
RULE_HOLD_FRAMES  = 6
STAND_MOTION_MAX  = 0.025
STAND_KNEE_MIN    = 145
BACK_VISIBILITY_MAX = 0.35  # nose visibility below this = facing away
MIN_PERSON_H      = 60

CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "NormalVideos", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

CRIME_COLORS = {
    "Abuse":         (0, 0, 255),
    "Arrest":        (0, 165, 255),
    "Arson":         (0, 0, 200),
    "Assault":       (0, 0, 255),
    "Burglary":      (0, 255, 255),
    "Explosion":     (0, 0, 180),
    "Fighting":      (0, 0, 255),
    "NormalVideos":  (0, 200, 0),
    "RoadAccidents": (0, 165, 255),
    "Robbery":       (0, 0, 255),
    "Shooting":      (0, 0, 255),
    "Shoplifting":   (0, 255, 255),
    "Stealing":      (0, 255, 255),
    "Vandalism":     (0, 165, 255),
}

ALERT_ACTIONS = {"fight", "fighting", "assault", "abuse", "robbery",
                 "shooting", "shoot", "punch", "kick", "hit", "attack"}

# MediaPipe landmark indices
LM = {
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
print(f"✅ Device: {device}")


# =====================================================
# LSTM Model
# =====================================================

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


# =====================================================
# Load models
# =====================================================

print("Loading LSTM action classifier...")
lstm_ckpt      = torch.load(LSTM_CHECKPOINT, map_location=device)
ACTION_CLASSES = lstm_ckpt["classes"]
lstm_model     = ActionLSTM(
    input_size  = lstm_ckpt["input_size"],
    hidden_size = lstm_ckpt["hidden_size"],
    num_layers  = lstm_ckpt["num_layers"],
    num_classes = lstm_ckpt["num_classes"],
).to(device)
lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
lstm_model.eval()
print(f"✅ LSTM loaded — classes: {ACTION_CLASSES}")

print("Loading SlowFast UCF Crime...")
crime_model = slowfast_r50(pretrained=False)
crime_model.blocks[-1].proj = nn.Linear(
    crime_model.blocks[-1].proj.in_features, len(CRIME_CLASSES)
)
crime_ckpt = torch.load(CRIME_CHECKPOINT, map_location=device)
crime_model.load_state_dict(
    crime_ckpt["model_state_dict"] if "model_state_dict" in crime_ckpt
    else crime_ckpt
)
crime_model = crime_model.to(device)
crime_model.eval()
print("✅ SlowFast UCF Crime loaded")

print("Loading YOLOv8n...")
yolo = YOLO("yolov8n.pt")
print("✅ YOLOv8n loaded")

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode        = False,
    model_complexity         = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5
)
print("✅ MediaPipe Pose loaded")


# =====================================================
# Helper geometry functions
# =====================================================

def get_lm(kp, name):
    idx = LM[name]
    return kp[idx*3 : idx*3 + 2]   # x, y only


def angle_between(a, b, c):
    ba    = np.array(a) - np.array(b)
    bc    = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def motion_magnitude(kp_history):
    """Average hip displacement over recent frames."""
    if len(kp_history) < 2:
        return 0.0
    recent = list(kp_history)[-6:]
    hips   = [kp[LM["left_hip"]*3 : LM["left_hip"]*3 + 2] for kp in recent]
    diffs  = [np.linalg.norm(np.array(hips[i]) - np.array(hips[i-1]))
              for i in range(1, len(hips))]
    return float(np.mean(diffs)) if diffs else 0.0


# =====================================================
# Keypoint extraction + visibility
# =====================================================

def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    return np.zeros(99)


def pose_visibility(results):
    """
    Returns average visibility of key upper body landmarks.
    Low value = person facing away or occluded.
    """
    if not results.pose_landmarks:
        return 0.0
    key_ids = [11, 12, 13, 14, 15, 16, 23, 24]
    vis = [results.pose_landmarks.landmark[i].visibility for i in key_ids]
    return float(np.mean(vis))


def nose_visibility(results):
    """Returns MediaPipe visibility score for the nose landmark specifically."""
    if not results.pose_landmarks:
        return 0.0
    return results.pose_landmarks.landmark[0].visibility


# =====================================================
# Rule-based action detection
# =====================================================

def rule_based_action(kp, kp_history, results):
    """
    Checks rules in priority order:
      1. turned_away  (facing backward)
      2. sit
      3. ride_bike
      4. talking
      5. stand
    Returns (label, confidence) or (None, 0) to fall to LSTM.
    """
    if np.all(kp == 0):
        return None, 0.0

    # ---- TURNED AWAY rule ----
    # Nose visibility is low when person faces away
    # Ear z values become more prominent than nose z
    nose_vis = nose_visibility(results)
    nose_z   = kp[LM["nose"]*3 + 2]
    l_ear_z  = kp[LM["left_ear"]*3 + 2]
    r_ear_z  = kp[LM["right_ear"]*3 + 2]
    avg_ear_z = (l_ear_z + r_ear_z) / 2
    if nose_vis < BACK_VISIBILITY_MAX and nose_z < -0.05 and avg_ear_z > nose_z + 0.04:
        return "turned_away", 0.85

    # Extract landmarks
    nose    = get_lm(kp, "nose")
    l_sh    = get_lm(kp, "left_shoulder")
    r_sh    = get_lm(kp, "right_shoulder")
    l_wrist = get_lm(kp, "left_wrist")
    r_wrist = get_lm(kp, "right_wrist")
    l_hip   = get_lm(kp, "left_hip")
    r_hip   = get_lm(kp, "right_hip")
    l_knee  = get_lm(kp, "left_knee")
    r_knee  = get_lm(kp, "right_knee")
    l_ankle = get_lm(kp, "left_ankle")
    r_ankle = get_lm(kp, "right_ankle")

    l_knee_angle = angle_between(l_hip, l_knee, l_ankle)
    r_knee_angle = angle_between(r_hip, r_knee, r_ankle)
    avg_knee     = (l_knee_angle + r_knee_angle) / 2

    avg_hip_y    = (l_hip[1]   + r_hip[1])   / 2
    avg_knee_y   = (l_knee[1]  + r_knee[1])  / 2
    avg_ankle_y  = (l_ankle[1] + r_ankle[1]) / 2
    avg_sh_y     = (l_sh[1]    + r_sh[1])    / 2

    torso_ratio  = (avg_hip_y - avg_sh_y) / (avg_ankle_y - avg_sh_y + 1e-6)
    hip_knee_gap = avg_knee_y - avg_hip_y
    motion       = motion_magnitude(kp_history)

    # ---- SIT rule ----
    if avg_knee < 120 or (avg_knee < 145 and hip_knee_gap < 0.08):
        return "sit", 0.88

    # ---- RIDE BIKE rule ----
    knee_diff = abs(l_knee_angle - r_knee_angle)
    if knee_diff > 30 and avg_knee < 145 and torso_ratio < 0.50 and motion > 0.01:
        return "ride_bike", 0.82

    # ---- TALKING rule ----
    face_y          = nose[1]
    l_wrist_ydist   = abs(l_wrist[1] - face_y)
    r_wrist_ydist   = abs(r_wrist[1] - face_y)
    l_wrist_xdist   = abs(l_wrist[0] - nose[0])
    r_wrist_xdist   = abs(r_wrist[0] - nose[0])
    wrist_near_face = (
        (l_wrist_ydist < 0.15 and l_wrist_xdist < 0.20) or
        (r_wrist_ydist < 0.15 and r_wrist_xdist < 0.20)
    )
    if wrist_near_face and motion < 0.03:
        return "talking", 0.80

    # ---- STAND rule ----
    if avg_knee > STAND_KNEE_MIN and torso_ratio > 0.35 and motion < STAND_MOTION_MAX:
        return "stand", 0.90

    return None, 0.0


def instant_posture_guess(kp, kp_history):
    """Quick single-frame guess to fill empty boxes while LSTM buffer fills."""
    if np.all(kp == 0):
        return "stand", 0.35

    l_hip   = get_lm(kp, "left_hip")
    r_hip   = get_lm(kp, "right_hip")
    l_knee  = get_lm(kp, "left_knee")
    r_knee  = get_lm(kp, "right_knee")
    l_ankle = get_lm(kp, "left_ankle")
    r_ankle = get_lm(kp, "right_ankle")
    l_sh    = get_lm(kp, "left_shoulder")
    r_sh    = get_lm(kp, "right_shoulder")

    l_ka        = angle_between(l_hip, l_knee, l_ankle)
    r_ka        = angle_between(r_hip, r_knee, r_ankle)
    avg_ka      = (l_ka + r_ka) / 2
    avg_hip_y   = (l_hip[1]   + r_hip[1])   / 2
    avg_knee_y  = (l_knee[1]  + r_knee[1])  / 2
    avg_ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    avg_sh_y    = (l_sh[1]    + r_sh[1])    / 2
    torso_ratio = (avg_hip_y - avg_sh_y) / (avg_ankle_y - avg_sh_y + 1e-6)
    hip_knee_gap = avg_knee_y - avg_hip_y
    motion      = motion_magnitude(kp_history)

    if avg_ka < 120 or (avg_ka < 145 and hip_knee_gap < 0.08):
        return "sit", 0.70
    if avg_ka > 145 and motion > 0.02:
        return "walk", 0.50
    if avg_ka > 140 and torso_ratio > 0.30:
        return "stand", 0.70
    return "stand", 0.45


# =====================================================
# LSTM filter with fight vote + interacting fallback
# =====================================================

def filter_lstm_prediction(label, conf, vote_deque, motion, visibility):
    """
    - Motion gate: block walk/run if person is stationary
    - Visibility gate: block fight if pose is unreliable
    - Fight: needs high confidence + sustained vote window
    - Low confidence → interacting
    """
    # Motion gate
    if label in ("walk", "run") and motion < 0.012:
        return "stand", 0.85

    # Visibility gate — unreliable pose = not fight
    if label == "fight" and visibility < 0.55:
        return "interacting", 0.65

    vote_deque.append((label, conf))

    if label == "fight":
        if conf < FIGHT_THRESHOLD:
            return "interacting", 0.60
        recent = [l for l, c in vote_deque]
        if recent.count("fight") < FIGHT_VOTE_NEEDED:
            return "interacting", 0.60
        return "fight", conf

    if conf >= ACTION_THRESHOLD:
        return label, conf

    return "interacting", 0.55


# =====================================================
# LSTM runner
# =====================================================

def run_lstm(keypoint_seq):
    x = torch.FloatTensor(keypoint_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        probs     = F.softmax(lstm_model(x), dim=1)
        conf, idx = probs.max(1)
    return ACTION_CLASSES[idx.item()], conf.item()


# =====================================================
# SlowFast runner
# =====================================================

def preprocess_frame_slowfast(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def run_slowfast(model, frame_buffer):
    if len(frame_buffer) < NUM_FRAMES:
        return None, 0.0
    video    = torch.stack(list(frame_buffer), dim=1).unsqueeze(0)
    slow_idx = torch.linspace(0, video.shape[2]-1, video.shape[2]//ALPHA).long()
    slow     = torch.index_select(video, 2, slow_idx).to(device)
    fast     = video.to(device)
    with torch.no_grad():
        probs     = F.softmax(model([slow, fast]), dim=1)
        conf, idx = probs.max(1)
    return idx.item(), conf.item()


# =====================================================
# Drawing helpers
# =====================================================

def draw_scene_banner(frame, crime_label, crime_conf):
    h, w  = frame.shape[:2]
    color = CRIME_COLORS.get(crime_label, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"SCENE: {crime_label}  {crime_conf*100:.1f}%",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.rectangle(frame, (0, 50), (int(w * crime_conf), 55), color, -1)
    return frame


def draw_person(frame, x1, y1, x2, y2, action_label, action_conf, tid, src):
    is_alert    = any(a in action_label.lower() for a in ALERT_ACTIONS)
    is_interact = action_label == "interacting"
    is_away     = action_label == "turned_away"

    if is_alert:
        color = (0, 0, 255)       # red
    elif is_interact:
        color = (0, 165, 255)     # orange
    elif is_away:
        color = (180, 180, 180)   # grey
    else:
        color = (255, 255, 0)     # cyan

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag         = "[R]" if src == "rule" else "[L]"
    label       = f"P{tid}: {action_label} {action_conf*100:.0f}% {tag}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y     = max(y1 - 5, th + 5)
    cv2.rectangle(frame,
                  (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                  color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


# =====================================================
# Per-person state
# =====================================================

person_kp_buffers = {}
person_kp_history = {}
person_actions    = {}
person_action_src = {}
person_rule_count = {}
person_vote_deque = {}
person_fight_secs = {}


# =====================================================
# Main loop
# =====================================================

def run_demo(source, save_output=True):
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ Video: {width}x{height} @ {fps:.1f}fps  ({total} frames)")

    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = f"demo_output_{timestamp}.mp4"
        writer    = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"✅ Saving to: {out_path}")

    scene_buffer       = deque(maxlen=NUM_FRAMES)
    current_crime      = "Analyzing..."
    current_crime_conf = 0.0
    frame_count        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # --------------------------------------------------
        # 1. Scene-level crime detection (SlowFast UCF)
        # --------------------------------------------------
        scene_buffer.append(preprocess_frame_slowfast(frame))
        if len(scene_buffer) == NUM_FRAMES and frame_count % NUM_FRAMES == 0:
            idx, conf = run_slowfast(crime_model, scene_buffer)
            if idx is not None:
                label = CRIME_CLASSES[idx]
                if SKIP_NORMAL_VIDEOS and label == "NormalVideos":
                    pass
                elif conf >= CRIME_THRESHOLD:
                    current_crime      = label
                    current_crime_conf = conf
                    print(f"  🚨 Crime detected: {label} ({conf*100:.1f}%)")

        # --------------------------------------------------
        # 2. Person detection (YOLOv8)
        # --------------------------------------------------
        results    = yolo.track(frame, persist=True, classes=[0],
                                conf=0.4, verbose=False)
        active_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids   = results[0].boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                if (y2 - y1) < MIN_PERSON_H:
                    continue

                active_ids.add(tid)

                # Init buffers for new person
                if tid not in person_kp_buffers:
                    person_kp_buffers[tid] = deque(maxlen=SEQ_LEN)
                    person_kp_history[tid] = deque(maxlen=10)
                    person_vote_deque[tid] = deque(maxlen=VOTE_WINDOW)
                    person_rule_count[tid] = 0
                    person_actions[tid]    = ("stand", 0.40)
                    person_action_src[tid] = "rule"
                    person_fight_secs[tid] = 0.0

                # ------------------------------------------
                # 3. Extract keypoints
                # ------------------------------------------
                pad  = 20
                crop = frame[
                    max(0, y1-pad) : min(height, y2+pad),
                    max(0, x1-pad) : min(width,  x2+pad)
                ]
                kp         = np.zeros(99)
                visibility = 0.0
                result     = None
                if crop.size > 0:
                    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    kp     = extract_keypoints(result)
                    visibility = pose_visibility(result)

                person_kp_buffers[tid].append(kp)
                person_kp_history[tid].append(kp)

                # ------------------------------------------
                # 4. Rule-based check (every frame)
                # ------------------------------------------
                rule_label, rule_conf = rule_based_action(
                    kp, person_kp_history[tid], result
                )

                if rule_label is not None:
                    person_rule_count[tid] += 1
                    if person_rule_count[tid] >= RULE_HOLD_FRAMES:
                        person_actions[tid]    = (rule_label, rule_conf)
                        person_action_src[tid] = "rule"
                else:
                    person_rule_count[tid] = 0

                    # ------------------------------------------
                    # 5. LSTM (buffer full + no rule matched)
                    # ------------------------------------------
                    if len(person_kp_buffers[tid]) == SEQ_LEN:
                        raw_label, raw_conf = run_lstm(
                            np.array(person_kp_buffers[tid])
                        )
                        curr_motion = motion_magnitude(person_kp_history[tid])
                        f_label, f_conf = filter_lstm_prediction(
                            raw_label, raw_conf, person_vote_deque[tid],
                            curr_motion, visibility
                        )
                        if f_label == "fight":
                            person_fight_secs[tid] += 1.0 / fps
                            if person_fight_secs[tid] >= FIGHT_MIN_SECONDS:
                                person_actions[tid]    = ("fight", f_conf)
                                person_action_src[tid] = "lstm"
                            # else: keep previous label while timer builds
                        else:
                            person_fight_secs[tid] = 0.0
                            person_actions[tid]    = (f_label, f_conf)
                            person_action_src[tid] = "lstm"
                    else:
                        # Buffer still filling — instant posture guess
                        if person_actions[tid][1] < 0.50:
                            g_label, g_conf = instant_posture_guess(
                                kp, person_kp_history[tid]
                            )
                            person_actions[tid]    = (g_label, g_conf)
                            person_action_src[tid] = "rule"

                # Draw bounding box + label
                action_label, action_conf = person_actions[tid]
                frame = draw_person(
                    frame, x1, y1, x2, y2,
                    action_label, action_conf, tid,
                    person_action_src[tid]
                )

        # Clean up people no longer in frame
        for tid in list(person_kp_buffers.keys()):
            if tid not in active_ids:
                del person_kp_buffers[tid]
                del person_kp_history[tid]
                del person_actions[tid]
                del person_action_src[tid]
                del person_rule_count[tid]
                del person_vote_deque[tid]
                del person_fight_secs[tid]

        # --------------------------------------------------
        # 6. Draw scene banner + frame counter
        # --------------------------------------------------
        frame = draw_scene_banner(frame, current_crime, current_crime_conf)
        cv2.putText(frame, f"{frame_count}/{total}",
                    (width - 160, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if save_output:
            writer.write(frame)

        cv2.imshow("CCTV Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user")
            break

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total} — "
                  f"Scene: {current_crime} ({current_crime_conf*100:.1f}%)")

    cap.release()
    if save_output:
        writer.release()
        print(f"✅ Saved: {out_path}")
    cv2.destroyAllWindows()
    pose.close()
    print(f"✅ Done — {frame_count} frames processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  type=str, default="0")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run_demo(source=args.source, save_output=not args.no_save)