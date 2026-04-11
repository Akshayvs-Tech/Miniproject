# =====================================================
# SCRIPT 1 — collect_keypoints.py
#
# Extracts MediaPipe keypoints from video clips and
# saves them as training data for the LSTM classifier.
#
# Folder structure needed:
#   data/
#     walk/
#       clip1.mp4
#       clip2.mp4
#     run/
#       clip1.mp4
#     sit/
#       clip1.mp4
#     stand/
#       clip1.mp4
#     fight/
#       clip1.mp4
#     ... (any action classes you want)
#
# Output:
#   keypoints.npy   — shape (N, SEQ_LEN, 99)
#   labels.npy      — shape (N,)
#   classes.txt     — one class name per line
#
# Usage:
#   python collect_keypoints.py --data_dir data/
# =====================================================

import cv2
import numpy as np
import mediapipe as mp
import os
import argparse
from collections import defaultdict

SEQ_LEN    = 30    # number of frames per sequence (1 second at 30fps)
STEP       = 15    # sliding window step (50% overlap for more samples)
MIN_FRAMES = SEQ_LEN

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """
    Extract 33 landmarks x (x, y, z) = 99 values from MediaPipe results.
    Returns zeros if no person detected.
    """
    if results.pose_landmarks:
        return np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ]).flatten()   # shape (99,)
    else:
        return np.zeros(99)


def process_video(video_path, pose):
    """
    Extract keypoint sequences from a single video.
    Returns list of sequences, each shape (SEQ_LEN, 99).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  Could not open: {video_path}")
        return []

    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        all_keypoints.append(extract_keypoints(result))

    cap.release()

    if len(all_keypoints) < MIN_FRAMES:
        print(f"  ⚠️  Too short ({len(all_keypoints)} frames): {video_path}")
        return []

    # Sliding window to create multiple sequences from one video
    sequences = []
    for start in range(0, len(all_keypoints) - SEQ_LEN + 1, STEP):
        seq = all_keypoints[start : start + SEQ_LEN]
        sequences.append(np.array(seq))   # (SEQ_LEN, 99)

    return sequences


def collect(data_dir):
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print(f"✅ Classes found ({len(classes)}): {classes}")

    all_sequences = []
    all_labels    = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for label_idx, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            videos  = [
                f for f in os.listdir(cls_dir)
                if f.endswith((".mp4", ".avi", ".mov", ".dav"))
            ]
            print(f"\n  [{cls}] — {len(videos)} videos")

            for vid in videos:
                vid_path  = os.path.join(cls_dir, vid)
                sequences = process_video(vid_path, pose)
                print(f"    {vid} → {len(sequences)} sequences")
                for seq in sequences:
                    all_sequences.append(seq)
                    all_labels.append(label_idx)

    X = np.array(all_sequences)   # (N, SEQ_LEN, 99)
    y = np.array(all_labels)      # (N,)

    np.save("keypoints.npy", X)
    np.save("labels.npy",    y)

    with open("classes.txt", "w") as f:
        f.write("\n".join(classes))

    print(f"\n✅ Saved {len(X)} sequences from {len(classes)} classes")
    print(f"   keypoints.npy → {X.shape}")
    print(f"   labels.npy    → {y.shape}")
    print(f"   classes.txt   → {classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Folder containing class subfolders with videos")
    args = parser.parse_args()
    collect(args.data_dir)