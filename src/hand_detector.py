"""
MediaPipe Hand Tracking and Biometric Landmark Extraction Module.
Compatible with both modern MediaPipe Tasks API (v1.0+) and legacy Solutions API (v0.10-).
Provides ultra-reliable finger counting for 0 (Fist), 1, 2, 3, 4, and 5 fingers.
"""
import os
import cv2
import numpy as np
import urllib.request
from typing import List, Tuple, Optional, Dict

class HandDetector:
    """
    High-performance hand tracker with universal MediaPipe compatibility.
    """
    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6
    ):
        import mediapipe as mp
        self.mp = mp
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.results = None
        self.cached_landmarks = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.use_tasks_api = False
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=mode,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.use_tasks_api = True
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            from config import MODEL_DIR
            model_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
            if not os.path.exists(model_path):
                print(f"[INFO] Downloading official MediaPipe HandLandmarker model to {model_path}...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Model downloaded successfully.")

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> bool:
        """Processes BGR frame and caches tracking results."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_bgr.shape

        if not self.use_tasks_api:
            self.results = self.hands.process(frame_rgb)
            if self.results and self.results.multi_hand_landmarks:
                hand_landmarks = self.results.multi_hand_landmarks[0]
                raw_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                pixel_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                handedness = "Right"
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[0].classification[0].label
                self.cached_landmarks = {
                    "raw_norm": raw_norm,
                    "pixel_coords": pixel_coords,
                    "handedness": handedness
                }
                return True
            else:
                self.cached_landmarks = None
                return False
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            self.results = self.hand_landmarker.detect(mp_image)

            if self.results and self.results.hand_landmarks and len(self.results.hand_landmarks) > 0:
                hand_landmarks = self.results.hand_landmarks[0]
                raw_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                pixel_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                handedness = "Right"
                if self.results.handedness and len(self.results.handedness) > 0:
                    categories = self.results.handedness[0]
                    if categories and len(categories) > 0:
                        handedness = getattr(categories[0], "category_name", "Right")
                self.cached_landmarks = {
                    "raw_norm": raw_norm,
                    "pixel_coords": pixel_coords,
                    "handedness": handedness
                }
                return True
            else:
                self.cached_landmarks = None
                return False

    def get_landmarks(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Returns cached landmark dictionary."""
        return self.cached_landmarks

    def get_fingers_up(self, landmarks_dict: Dict) -> List[int]:
        """
        Returns list [thumb, index, middle, ring, pinky] with 1 for extended, 0 for curled.
        Uses biometric joint angles & distance ratios to eliminate false thumb detection on fists.
        """
        pts = landmarks_dict["pixel_coords"]

        # 4 Main fingers: Compare Tip Y with PIP joint Y (in image coordinates, smaller Y is higher up)
        index_up = 1 if pts[8][1] < pts[6][1] else 0
        middle_up = 1 if pts[12][1] < pts[10][1] else 0
        ring_up = 1 if pts[16][1] < pts[14][1] else 0
        pinky_up = 1 if pts[20][1] < pts[18][1] else 0

        # Thumb detection: Euclidean distance from Thumb Tip (4) to Pinky Base MCP (17)
        # When thumb is folded over a fist, it sits close to the palm
        # When thumb is extended outwards, it spreads away from knuckle 17 and knuckle 5
        d_tip_pinky = np.linalg.norm(np.array(pts[4]) - np.array(pts[17]))
        d_ip_pinky = np.linalg.norm(np.array(pts[3]) - np.array(pts[17]))
        d_tip_index = np.linalg.norm(np.array(pts[4]) - np.array(pts[5]))
        palm_width = np.linalg.norm(np.array(pts[5]) - np.array(pts[17]))

        # Thumb is only extended if it extends past IP and spreads away from index base
        thumb_up = 1 if (d_tip_pinky > d_ip_pinky * 1.12) and (d_tip_index > palm_width * 0.52) else 0

        # Safety rule: If all 4 fingers are down, thumb cannot be counting as 1, it's a closed fist!
        if (index_up + middle_up + ring_up + pinky_up) == 0:
            thumb_up = 0

        return [thumb_up, index_up, middle_up, ring_up, pinky_up]

    def count_fingers(self, landmarks_dict: Dict) -> int:
        """
        Translates finger states into intuitive 0..5 count:
        0: Fist (Neutral Rest)
        1: 1 Finger (Index extended)
        2: 2 Fingers (Index + Middle)
        3: 3 Fingers (Index + Middle + Ring or Thumb + Index + Middle)
        4: 4 Fingers (Index + Middle + Ring + Pinky)
        5: 5 Fingers (Open Hand / All 5 fingers extended)
        """
        fingers = self.get_fingers_up(landmarks_dict)
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
        four_sum = index_up + middle_up + ring_up + pinky_up

        if four_sum == 0:
            return 0  # Fist is ALWAYS 0
        elif four_sum == 1 and index_up:
            return 1  # 1 Finger: Index
        elif four_sum == 2 and index_up and middle_up:
            return 2  # 2 Fingers: Index + Middle
        elif four_sum == 3:
            return 3  # 3 Fingers
        elif four_sum == 2 and thumb_up and index_up:
            return 3  # Alternative 3: Thumb + Index + Middle
        elif four_sum == 4:
            return 5 if thumb_up else 4
        else:
            return min(5, four_sum + thumb_up)

    def get_pinch_distance(self, landmarks_dict: Dict, idx1: int = 4, idx2: int = 8) -> float:
        """Returns normalized Euclidean distance between two landmark indices."""
        raw = landmarks_dict["raw_norm"]
        p1 = np.array([raw[idx1][0], raw[idx1][1]])
        p2 = np.array([raw[idx2][0], raw[idx2][1]])
        return float(np.linalg.norm(p1 - p2))
