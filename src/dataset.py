"""
Dataset handling, biometric feature extraction, and PyTorch dataset loader.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from typing import List, Tuple

def extract_features_from_landmarks(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Transforms 21 raw (x, y, z) MediaPipe landmarks into 73 invariant features:
    - 63 normalized coordinates (centered at wrist, scaled by palm size)
    - 10 inter-finger Euclidean distance ratios
    
    This ensures complete translation and scale invariance.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, received {len(landmarks)}")

    coords = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

    # 1. Translation Invariance: Shift origin to Wrist (landmark 0)
    wrist = coords[0]
    relative_coords = coords - wrist  # wrist becomes (0, 0, 0)

    # 2. Scale Invariance: Divide by palm reference distance (Wrist -> Middle MCP #9)
    palm_size = np.linalg.norm(coords[9] - coords[0])
    if palm_size < 1e-4:
        palm_size = 1.0  # Prevent division by zero

    normalized_coords = relative_coords / palm_size  # shape (21, 3)
    flat_coords = normalized_coords.flatten()         # 63 features

    # 3. Geometric Invariant Ratios (Distances between key fingertips & wrist)
    # Fingertip indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20, Wrist=0
    d_thumb_index   = np.linalg.norm(normalized_coords[4] - normalized_coords[8])
    d_index_middle  = np.linalg.norm(normalized_coords[8] - normalized_coords[12])
    d_middle_ring   = np.linalg.norm(normalized_coords[12] - normalized_coords[16])
    d_ring_pinky    = np.linalg.norm(normalized_coords[16] - normalized_coords[20])
    d_thumb_middle  = np.linalg.norm(normalized_coords[4] - normalized_coords[12])

    d_thumb_wrist   = np.linalg.norm(normalized_coords[4])
    d_index_wrist   = np.linalg.norm(normalized_coords[8])
    d_middle_wrist  = np.linalg.norm(normalized_coords[12])
    d_ring_wrist    = np.linalg.norm(normalized_coords[16])
    d_pinky_wrist   = np.linalg.norm(normalized_coords[20])

    geometric_features = np.array([
        d_thumb_index,
        d_index_middle,
        d_middle_ring,
        d_ring_pinky,
        d_thumb_middle,
        d_thumb_wrist,
        d_index_wrist,
        d_middle_wrist,
        d_ring_wrist,
        d_pinky_wrist
    ], dtype=np.float32)

    # Combine: 63 + 10 = 73 features
    full_feature_vector = np.concatenate([flat_coords, geometric_features])
    return full_feature_vector


class GestureDataset(Dataset):
    """
    PyTorch Dataset with on-the-fly data augmentation (noise, scaling, small rotation).
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx].clone()
        label = self.labels[idx]

        if self.augment:
            # Add subtle Gaussian noise to simulate camera sensor jitter
            noise = torch.randn_like(feat) * 0.015
            feat = feat + noise
            # Subtle random scaling (0.95x - 1.05x)
            scale = np.random.uniform(0.95, 1.05)
            feat = feat * scale

        return feat, label
