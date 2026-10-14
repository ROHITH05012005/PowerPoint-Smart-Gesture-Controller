"""
Dynamic Motion, Approximate Circle Fitting, and Thumb Posture Engine.
Features:
1. ApproximateCircleEstimator: Least-Squares Algebraic Circle Fitting.
   Records air drawing trajectory, fits a circle to the rough hand-drawn path,
   estimates percentage match, and determines Clockwise (Open PPT) vs Anti-Clockwise (Close PPT).
2. Progressive Thumb Posture: Thumbs Up (Next Slide) & Thumbs Down (Exit Slideshow).
3. Intentional Swipe Engine: Swipe Left (Prev Slide) & Swipe Right (Next Slide).
"""
import time
import math
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict

class ApproximateCircleEstimator:
    """
    Records hand trajectory in free air and uses Least-Squares Algebraic Circle Fitting
    to approximately estimate if the air-drawn motion matches a circle.
    Determines Clockwise (Open PPT) vs Anti-Clockwise (Close PPT) via unwrapped angular sweep.
    """
    def __init__(self, max_points: int = 65, cooldown: float = 1.1):
        self.buffer = deque(maxlen=max_points)  # stores (x, y, timestamp)
        self.cooldown = cooldown
        self.last_trigger_time = 0.0
        # Cache for visual lock-on rendering: (cx, cy, radius, match_score, is_cw, expire_time)
        self.fitted_display = None

    def add_point(self, px: int, py: int, now: float):
        """Adds moving fingertip coordinate to trajectory."""
        # Prune points older than 2.0s
        while len(self.buffer) > 0 and (now - self.buffer[0][2]) > 2.0:
            self.buffer.popleft()

        if len(self.buffer) > 0:
            last = self.buffer[-1]
            dist = math.hypot(px - last[0], py - last[1])
            # Only record if hand moved at least 3 pixels (filters stationary tremor)
            if dist < 3:
                return
        self.buffer.append((px, py, now))

    def clear(self):
        """Clears trajectory buffer."""
        self.buffer.clear()

    def get_trail_points(self) -> List[Tuple[int, int]]:
        """Returns list of (x, y) coordinates for rendering."""
        return [(int(p[0]), int(p[1])) for p in self.buffer]

    def get_fitted_display(self, now: float) -> Optional[Dict]:
        """Returns active fitted circle details if within display window."""
        if self.fitted_display and now < self.fitted_display["expire"]:
            return self.fitted_display
        return None

    def estimate_circle(self, now: float) -> Tuple[str, float, Optional[Tuple[int, int, int]]]:
        """
        Processes recorded air points using Least Squares Circle Fitting and angular sweep.
        Returns:
            (status, match_score, (cx, cy, radius))
            status: 'NONE', 'CW_CIRCLE', 'CCW_CIRCLE', or 'COOLDOWN'
        """
        if now - self.last_trigger_time < self.cooldown:
            return "COOLDOWN", 0.0, None

        if len(self.buffer) < 14:
            return "NONE", 0.0, None

        # Filter points within the last 1.8 seconds
        recent = [p for p in self.buffer if (now - p[2]) < 1.8]
        if len(recent) < 14:
            return "NONE", 0.0, None

        pts = np.array([(p[0], p[1]) for p in recent], dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]

        # 1. Bounding box check
        w = float(np.max(x) - np.min(x))
        h = float(np.max(y) - np.min(y))
        if w < 45 or h < 45:
            return "NONE", 0.0, None  # Motion too small

        # Aspect ratio (must be reasonably 2D, not a flat line)
        aspect = min(w, h) / max(w, h)
        if aspect < 0.38:
            return "NONE", 0.0, None  # Too flattened/linear

        # 2. Algebraic Least Squares Circle Fit:
        # Equation: (x - a)^2 + (y - b)^2 = R^2
        # Matrix form: 2ax + 2by + c = x^2 + y^2, where c = R^2 - a^2 - b^2
        try:
            A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
            f = x**2 + y**2
            C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
            a, b, c = float(C[0]), float(C[1]), float(C[2])
            r_sq = a**2 + b**2 + c
            if r_sq <= 0:
                return "NONE", 0.0, None
            R = math.sqrt(r_sq)

            # Check fitted circle size (reasonable hand motion: 25px to 380px radius)
            if R < 25 or R > 380:
                return "NONE", 0.0, None

            # 3. Calculate radial distance standard deviation error (RMSE)
            distances = np.hypot(x - a, y - b)
            radial_std = float(np.std(distances))
            rmse = radial_std / R

            # 4. Angular Sweep: Unwrapped atan2 around fitted center
            # Image space has Y pointing down, so CW increases angle, CCW decreases angle
            angles = np.unwrap(np.arctan2(y - b, x - a))
            delta_angle = float(angles[-1] - angles[0])
            abs_delta_angle = abs(delta_angle)

            # Must have swept at least ~260 degrees (4.5 radians) of an arc
            if abs_delta_angle < 4.5:
                return "NONE", 0.0, None

            # 5. Loop closure check: start point & end point proximity relative to R
            start_end_dist = math.hypot(x[-1] - x[0], y[-1] - y[0])
            closure_ratio = start_end_dist / R

            # Match Score (0 to 100%)
            match_score = max(0.0, min(100.0, (1.0 - rmse) * 100))

            # Permissive threshold: RMSE < 0.38, closure ratio < 1.6, and match score >= 62%
            if rmse < 0.38 and closure_ratio < 1.60 and match_score >= 62.0:
                self.last_trigger_time = now
                is_cw = (delta_angle > 0)
                self.fitted_display = {
                    "cx": int(a),
                    "cy": int(b),
                    "radius": int(R),
                    "score": match_score,
                    "is_cw": is_cw,
                    "expire": now + 1.2
                }
                self.clear()

                if is_cw:
                    return "CW_CIRCLE", match_score, (int(a), int(b), int(R))
                else:
                    return "CCW_CIRCLE", match_score, (int(a), int(b), int(R))

        except Exception:
            pass

        return "NONE", 0.0, None


class MotionAndPostureTracker:
    """
    Unified manager handling:
    - Approximate Circle Estimator (CW -> Open PPT, CCW -> Close PPT)
    - Intentional Swipes (Left -> Prev Slide, Right -> Next Slide)
    - Thumbs Up (👍) and Thumbs Down (👎)
    """
    def __init__(self, cx: int = 640, cy: int = 360, radius: int = 115):
        self.circle_estimator = ApproximateCircleEstimator(max_points=65, cooldown=1.1)
        self.swipe_history = deque(maxlen=30)
        self.last_swipe_time = 0.0
        self.last_posture_time = 0.0
        self.swipe_cooldown = 0.65
        self.posture_cooldown = 1.0

    def get_swipe_trail(self) -> List[Tuple[int, int]]:
        """Returns list of recent coordinates for visual swipe streak rendering."""
        return [(int(p[0]), int(p[1])) for p in self.swipe_history]

    def is_hand_moving(self, speed_threshold: float = 170.0) -> bool:
        """
        Returns True if the hand is actively moving (e.g., swiping or waving).
        Used to prevent static postures from charging up during intentional dynamic motion.
        """
        if len(self.swipe_history) < 3:
            return False
        pts = list(self.swipe_history)
        recent = [p for p in pts if (pts[-1][2] - p[2]) <= 0.25]
        if len(recent) < 2:
            return False
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.04:
            return False
        dist = math.hypot(recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1])
        speed = dist / dt
        return speed > speed_threshold

    def detect_posture(self, landmarks_dict: Dict, now: float) -> str:
        """
        Detects primary static hand postures with clear joint hierarchy:
        - 'OPEN_PALM'   : All 5 fingers extended (Open PPT)
        - 'THUMBS_UP'   : 4 fingers curled + thumb pointing straight up (Next Slide)
        - 'PEACE'       : Index + Middle extended, Ring + Pinky curled (Prev Slide)
        - 'POINTING'    : Index extended alone (Exit Slideshow)
        - 'FIST'        : All 4 fingers curled + thumb tucked (Close PPT)
        - 'THUMBS_DOWN' : 4 fingers curled + thumb pointing straight down (Exit Slideshow)
        - 'NONE'        : Neutral / transition state
        """
        if now - self.last_posture_time < self.posture_cooldown:
            return "NONE"

        pts = landmarks_dict["pixel_coords"]
        wrist = np.array(pts[0], dtype=np.float32)

        def is_extended(tip_idx: int, pip_idx: int) -> bool:
            tip = np.array(pts[tip_idx], dtype=np.float32)
            pip = np.array(pts[pip_idx], dtype=np.float32)
            d_tip = np.linalg.norm(tip - wrist)
            d_pip = np.linalg.norm(pip - wrist)
            return (tip[1] < pip[1] - 10) and (d_tip > d_pip * 1.08)

        index_ext = is_extended(8, 6)
        middle_ext = is_extended(12, 10)
        ring_ext = is_extended(16, 14)
        pinky_ext = is_extended(20, 18)

        extended_count = int(index_ext) + int(middle_ext) + int(ring_ext) + int(pinky_ext)
        curled_count = 4 - extended_count

        t_tip = pts[4]
        t_ip = pts[3]
        t_mcp = pts[2]
        index_mcp = pts[5]

        # Thumb extended check
        d_ttip_wrist = np.linalg.norm(np.array(t_tip) - wrist)
        d_ttip_idx = np.linalg.norm(np.array(t_tip) - np.array(index_mcp))
        thumb_extended = (d_ttip_idx > 45) and (d_ttip_wrist > np.linalg.norm(np.array(t_mcp) - wrist) * 1.10)

        # 1. OPEN PALM (✋): All 4 main fingers extended + thumb extended
        if extended_count >= 4 and thumb_extended:
            return "OPEN_PALM"

        # 2. PEACE SIGN (✌️): Index + Middle extended, Ring + Pinky curled (Prev Slide)
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "PEACE"

        # 3. POINTING (☝️): Index extended alone (Exit Slideshow)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "POINTING"

        # If 4 main fingers are curled:
        if curled_count == 4:
            # 4. THUMBS UP (👍): progressive upward joint hierarchy (Tip < IP < MCP)
            if (t_tip[1] < t_ip[1] - 12) and (t_ip[1] < t_mcp[1] - 6) and (t_tip[1] < wrist[1] - 25):
                return "THUMBS_UP"

            # 5. THUMBS DOWN (👎): progressive downward joint hierarchy (Tip > IP > MCP)
            if (t_tip[1] > t_ip[1] + 12) and (t_ip[1] > t_mcp[1] + 6) and (t_tip[1] > index_mcp[1] + 25):
                return "THUMBS_DOWN"

            # 6. FIST (✊): thumb folded/tucked across fingers, not sticking out up/down
            if not thumb_extended and (t_tip[1] >= index_mcp[1] - 25):
                return "FIST"

        # Tolerant fist (at least 3 curled, no fingers extended, thumb tucked)
        if curled_count >= 3 and extended_count == 0:
            if not thumb_extended and (t_tip[1] >= index_mcp[1] - 25):
                return "FIST"

        return "NONE"

    def check_thumb_posture(self, landmarks_dict: Dict, now: float) -> str:
        """Backwards compatibility alias for thumb detection."""
        posture = self.detect_posture(landmarks_dict, now)
        if posture in ["THUMBS_UP", "THUMBS_DOWN"]:
            return posture
        return "NONE"

    def register_posture_action(self, now: float):
        """Registers execution of a static posture action."""
        self.last_posture_time = now

    def register_thumb_action(self, now: float):
        self.register_posture_action(now)

    def check_intentional_swipe(self, px: int, py: int, now: float) -> str:
        """
        High-precision peak-to-peak swipe detection.
        Accurately identifies rapid horizontal hand motions (Left -> Prev Slide, Right -> Next Slide)
        across natural human arm arcs, rejecting slow hand drifts and stationary tremors.
        """
        self.swipe_history.append((px, py, now))

        if now - self.last_swipe_time < self.swipe_cooldown:
            return "NONE"

        # Filter recent points within the last 0.45s window
        recent = [p for p in self.swipe_history if (now - p[2]) <= 0.45]
        if len(recent) < 4:
            return "NONE"

        p_min_x = min(recent, key=lambda p: p[0])
        p_max_x = max(recent, key=lambda p: p[0])

        # Left-to-Right swipe: min_x occurred before max_x in time
        if p_min_x[2] < p_max_x[2]:
            dt = p_max_x[2] - p_min_x[2]
            dx = p_max_x[0] - p_min_x[0]
            dy = p_max_x[1] - p_min_x[1]
            time_since_peak = now - p_max_x[2]
            direction = "SWIPE_RIGHT"
        # Right-to-Left swipe: max_x occurred before min_x in time
        else:
            dt = p_min_x[2] - p_max_x[2]
            dx = -(p_max_x[0] - p_min_x[0])
            dy = p_min_x[1] - p_max_x[1]
            time_since_peak = now - p_min_x[2]
            direction = "SWIPE_LEFT"

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Swipe constraints:
        # 1. Minimum displacement of 75px
        # 2. Duration of swipe between 0.06s and 0.42s
        # 3. Peak reached recently (within 0.22s)
        # 4. Horizontal dominance: abs(dx) >= abs(dy) * 1.2 (allows natural arm arcs)
        # 5. Fast velocity >= 220 px/s
        if (
            abs_dx >= 75
            and 0.06 <= dt <= 0.42
            and time_since_peak <= 0.22
            and abs_dx >= abs_dy * 1.2
        ):
            velocity = abs_dx / dt
            if velocity >= 220.0:
                self.last_swipe_time = now
                self.swipe_history.clear()
                self.circle_estimator.clear()
                return direction

        return "NONE"

