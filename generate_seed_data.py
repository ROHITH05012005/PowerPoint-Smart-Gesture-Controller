"""
Biometric Synthetic Data & Seed Generator for PowerPoint Gesture Assistant.
Generates mathematically accurate anatomical 3D hand poses for classes 0 to 5:
0: Fist (Neutral)
1: One Finger (Index) -> Open PowerPoint
2: Two Fingers (Index + Middle) -> Next Slide
3: Three Fingers (Index + Middle + Ring) -> Previous Slide
4: Four Fingers (Index + Middle + Ring + Pinky) -> Exit PPT
5: Five Fingers (Open Hand) -> Zoom Text
"""
import os
import csv
import numpy as np
import math
from config import GESTURE_CLASSES, DATASET_DIR, DATASET_FILE
from src.dataset import extract_features_from_landmarks

def create_base_hand_landmarks() -> np.ndarray:
    """Returns canonical anatomical upright open palm (21 landmarks)."""
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.50, 0.82, 0.00]  # Wrist

    # Thumb
    coords[1] = [0.44, 0.76, -0.01]
    coords[2] = [0.40, 0.69, -0.02]
    coords[3] = [0.37, 0.63, -0.03]
    coords[4] = [0.34, 0.58, -0.04]

    # Index
    coords[5] = [0.46, 0.61, -0.01]
    coords[6] = [0.45, 0.51, -0.02]
    coords[7] = [0.44, 0.43, -0.03]
    coords[8] = [0.44, 0.35, -0.04]

    # Middle
    coords[9] =  [0.50, 0.60, -0.01]
    coords[10] = [0.50, 0.49, -0.02]
    coords[11] = [0.50, 0.40, -0.03]
    coords[12] = [0.50, 0.32, -0.04]

    # Ring
    coords[13] = [0.54, 0.61, -0.01]
    coords[14] = [0.55, 0.52, -0.02]
    coords[15] = [0.56, 0.44, -0.03]
    coords[16] = [0.56, 0.36, -0.04]

    # Pinky
    coords[17] = [0.58, 0.64, -0.01]
    coords[18] = [0.60, 0.57, -0.02]
    coords[19] = [0.61, 0.50, -0.03]
    coords[20] = [0.62, 0.43, -0.04]

    return coords

def curl_finger(coords: np.ndarray, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int, curl_factor: float = 1.0):
    """Curls a finger inward towards the MCP base."""
    mcp = coords[mcp_idx]
    target_pip = mcp + np.array([0.0, 0.05, 0.03])
    target_dip = mcp + np.array([0.0, 0.09, 0.05])
    target_tip = mcp + np.array([0.0, 0.12, 0.06])

    coords[pip_idx] = coords[pip_idx] * (1.0 - curl_factor) + target_pip * curl_factor
    coords[dip_idx] = coords[dip_idx] * (1.0 - curl_factor) + target_dip * curl_factor
    coords[tip_idx] = coords[tip_idx] * (1.0 - curl_factor) + target_tip * curl_factor

def curl_thumb(coords: np.ndarray, curl_factor: float = 1.0):
    """Curls thumb inward towards palm."""
    target_ip = coords[2] + np.array([0.04, 0.02, 0.03])
    target_tip = coords[2] + np.array([0.06, 0.04, 0.05])
    coords[3] = coords[3] * (1.0 - curl_factor) + target_ip * curl_factor
    coords[4] = coords[4] * (1.0 - curl_factor) + target_tip * curl_factor

def generate_gesture_pose(gesture_id: int) -> np.ndarray:
    """Generates canonical 3D pose for finger counting gestures 0 to 5."""
    coords = create_base_hand_landmarks()

    if gesture_id == 0:  # 0 Fingers: Closed Fist
        curl_thumb(coords, 1.0)
        curl_finger(coords, 5, 6, 7, 8, 1.0)
        curl_finger(coords, 9, 10, 11, 12, 1.0)
        curl_finger(coords, 13, 14, 15, 16, 1.0)
        curl_finger(coords, 17, 18, 19, 20, 1.0)

    elif gesture_id == 1:  # 1 Finger: Index finger only
        curl_thumb(coords, 0.90)
        # Index is extended (8)
        curl_finger(coords, 9, 10, 11, 12, 0.95)
        curl_finger(coords, 13, 14, 15, 16, 0.95)
        curl_finger(coords, 17, 18, 19, 20, 0.95)

    elif gesture_id == 2:  # 2 Fingers: Index + Middle
        curl_thumb(coords, 0.90)
        # Index (8) & Middle (12) extended
        coords[8] += np.array([-0.015, 0.0, 0.0])
        coords[12] += np.array([0.015, 0.0, 0.0])
        curl_finger(coords, 13, 14, 15, 16, 0.95)
        curl_finger(coords, 17, 18, 19, 20, 0.95)

    elif gesture_id == 3:  # 3 Fingers: Index + Middle + Ring
        curl_thumb(coords, 0.90)
        # Index (8), Middle (12), Ring (16) extended
        curl_finger(coords, 17, 18, 19, 20, 0.95)

    elif gesture_id == 4:  # 4 Fingers: Index + Middle + Ring + Pinky
        curl_thumb(coords, 0.95)
        # 4 fingers extended

    elif gesture_id == 5:  # 5 Fingers: All extended (Open Hand)
        pass  # Base hand is all 5 extended

    return coords

def apply_random_transform(coords: np.ndarray) -> np.ndarray:
    """Applies realistic 3D rotations, scaling, translation, and noise."""
    c = coords.copy()

    roll = np.random.uniform(-0.35, 0.35)
    pitch = np.random.uniform(-0.25, 0.25)
    yaw = np.random.uniform(-0.25, 0.25)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    wrist = c[0].copy()
    c = (c - wrist) @ R.T + wrist

    scale = np.random.uniform(0.75, 1.30)
    c = (c - wrist) * scale + wrist

    c += np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.05, 0.05)])
    c += np.random.normal(0, 0.007, c.shape)

    return c

def generate_seed_dataset(samples_per_class: int = 500):
    """Generates synthetic dataset and writes to CSV."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    rows = []

    print(f"Generating synthetic training dataset ({samples_per_class} samples/class)...")
    for gesture_id in sorted(GESTURE_CLASSES.keys()):
        class_name = GESTURE_CLASSES[gesture_id]
        print(f"  [+] Synthesizing Class {gesture_id}: {class_name}")

        for _ in range(samples_per_class):
            base_pose = generate_gesture_pose(gesture_id)
            augmented_pose = apply_random_transform(base_pose)
            lm_list = [(augmented_pose[i, 0], augmented_pose[i, 1], augmented_pose[i, 2]) for i in range(21)]
            features = extract_features_from_landmarks(lm_list)
            rows.append(list(features) + [gesture_id])

    with open(DATASET_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"feat_{i}" for i in range(len(rows[0]) - 1)] + ["label"]
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[SUCCESS] Generated {len(rows)} samples across {len(GESTURE_CLASSES)} gesture classes!")
    print(f"Dataset saved to: {DATASET_FILE}")

if __name__ == "__main__":
    generate_seed_dataset(samples_per_class=500)
