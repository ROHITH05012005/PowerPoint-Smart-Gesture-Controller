"""
Futuristic Cyberpunk / Iron-Man Sci-Fi HUD Renderer for Assisted Guided Circle & Thumb Engine.
Renders assisted circular track with start marker, magnetic tracking bead, and thumb hold meters.
"""
import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Optional

class CyberpunkHUD:
    """
    Renders an assisted Iron-Man HUD with magnetic guided circle track and thumb meters.
    """
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.w = frame_width
        self.h = frame_height

        # Sci-Fi Color Palette (BGR)
        self.NEON_CYAN = (255, 243, 0)       # #00F3FF
        self.NEON_MAGENTA = (255, 0, 255)    # #FF00FF
        self.ARC_ORANGE = (0, 165, 255)      # Stark Reactor Gold/Orange
        self.LASER_GREEN = (50, 255, 120)    # Laser Green
        self.ALERT_RED = (40, 40, 255)       # Alert Red
        self.ELECTRIC_YELLOW = (0, 230, 255) # Electric Yellow
        self.DARK_PANEL = (20, 18, 24)       # Tech Dark
        self.TEXT_WHITE = (245, 245, 245)

        # Skeleton connections
        self.BONES = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        self.reticle_angle = 0.0
        self.flash_alpha = 0.0
        self.action_banner_text = ""
        self.action_banner_sub = ""
        self.action_banner_timer = 0.0
        self.last_time = time.time()

    def update_animation(self):
        """Advances rotational and flash animations."""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.reticle_angle = (self.reticle_angle + dt * 110.0) % 360.0
        if self.flash_alpha > 0.0:
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 2.5)

    def trigger_action_flash(self, action_name: str, sub_text: str = ""):
        """Triggers screen flash and pop-up banner."""
        self.flash_alpha = 0.45
        self.action_banner_text = action_name
        self.action_banner_sub = sub_text
        self.action_banner_timer = time.time() + 1.4

    def draw_air_drawing_trail(
        self,
        frame: np.ndarray,
        trail_points: List[Tuple[int, int]],
        active_point: Optional[Tuple[int, int]] = None
    ):
        """
        Renders a glowing neon trail of the user's recorded air-drawing trajectory.
        Older points fade into translucent cyan, leading points glow bright yellow/white.
        """
        num_pts = len(trail_points)
        if num_pts < 2:
            if active_point:
                # Idle targeting reticle at fingertip
                px, py = active_point
                cv2.circle(frame, (px, py), 12, self.NEON_CYAN, 1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 4, self.ARC_ORANGE, -1, cv2.LINE_AA)
            return

        # Multi-segment glowing ribbon
        overlay = frame.copy()
        for i in range(1, num_pts):
            p1 = trail_points[i - 1]
            p2 = trail_points[i]
            # Ratio from 0.0 (oldest) to 1.0 (newest)
            ratio = i / float(num_pts)
            thickness = int(2 + ratio * 4)  # 2px to 6px wide

            # Gradient from deep cyan (older) to vivid electric yellow (newest)
            b = int(255 * (1.0 - 0.5 * ratio))
            g = int(200 + 55 * ratio)
            r = int(255 * ratio)
            color = (b, g, r)

            cv2.line(overlay, p1, p2, color, thickness + 4, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (255, 255, 255) if ratio > 0.8 else color, max(1, thickness - 1), cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Leading point indicator
        lead_pt = trail_points[-1]
        cv2.circle(frame, lead_pt, 8, self.ARC_ORANGE, 2, cv2.LINE_AA)
        cv2.circle(frame, lead_pt, 4, self.TEXT_WHITE, -1, cv2.LINE_AA)

        # Display recording badge near finger if actively drawing
        if num_pts >= 8:
            lx, ly = lead_pt
            badge_text = f"RECORDING MOTION: {num_pts} pts"
            cv2.rectangle(frame, (lx + 15, ly - 22), (lx + 230, ly + 2), (25, 20, 35), -1)
            cv2.rectangle(frame, (lx + 15, ly - 22), (lx + 230, ly + 2), self.NEON_CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, badge_text, (lx + 22, ly - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.NEON_CYAN, 1, cv2.LINE_AA)

    def draw_fitted_circle_overlay(self, frame: np.ndarray, fitted_info: Optional[Dict]):
        """
        Renders the estimated circle with holographic lock-on rings and match score badge.
        """
        if not fitted_info:
            return

        cx = fitted_info["cx"]
        cy = fitted_info["cy"]
        radius = fitted_info["radius"]
        score = fitted_info["score"]
        is_cw = fitted_info["is_cw"]

        glow_color = self.LASER_GREEN if is_cw else self.ARC_ORANGE
        action_name = "CLOCKWISE -> OPEN PPT" if is_cw else "ANTI-CLOCKWISE -> CLOSE PPT"

        # Glowing fitted circle
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius, glow_color, 8, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), radius + 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), max(5, radius - 8), glow_color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Center reticle
        cv2.circle(frame, (cx, cy), 5, self.TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), glow_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), glow_color, 1, cv2.LINE_AA)

        # Match Score Pill Badge
        badge_w, badge_h = 320, 48
        bx = max(10, min(self.w - badge_w - 10, cx - badge_w // 2))
        by = max(65, min(self.h - badge_h - 40, cy - radius - badge_h - 12))

        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (20, 22, 30), -1)
        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), glow_color, 2, cv2.LINE_AA)

        score_text = f"ESTIMATED CIRCLE: {score:.0f}% MATCH"
        cv2.putText(frame, score_text, (bx + 14, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, action_name, (bx + 14, by + 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, glow_color, 1, cv2.LINE_AA)

    def draw_guided_circle_track(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        progress: float,
        bead_pt: Optional[Tuple[int, int]] = None
    ):
        """Compatibility wrapper delegating to modern freehand HUD."""
        pass

    def draw_posture_hold_indicator(
        self,
        frame: np.ndarray,
        anchor_pt: Tuple[int, int],
        hold_ratio: float,
        posture_type: str
    ):
        """
        Draws glowing holographic circular progress ring when holding a posture:
        - OPEN_PALM: Laser green charging ring -> Open PowerPoint (F5)
        - FIST: Alert red charging ring -> Close PowerPoint
        - THUMBS_UP: Cyan charging ring -> Next Slide
        - THUMBS_DOWN: Red charging ring -> Exit Slideshow
        """
        if hold_ratio <= 0.0 or posture_type == "NONE":
            return

        ax, ay = anchor_pt
        radius = 34

        config = {
            "OPEN_PALM": (self.LASER_GREEN, "HOLD: OPEN POWERPOINT (F5)"),
            "THUMBS_UP": (self.NEON_CYAN, "HOLD: NEXT SLIDE >>"),
            "PEACE": (self.ARC_ORANGE, "HOLD: PREV SLIDE <<"),
            "POINTING": (self.ELECTRIC_YELLOW, "HOLD: EXIT SLIDESHOW (ESC)"),
            "FIST": (self.ALERT_RED, "HOLD: CLOSE POWERPOINT"),
            "THUMBS_DOWN": (self.ALERT_RED, "HOLD: EXIT SLIDESHOW (ESC)")
        }

        color, label = config.get(posture_type, (self.NEON_CYAN, posture_type))
        span = int(hold_ratio * 360)

        # Background Ring
        overlay = frame.copy()
        cv2.circle(overlay, (ax, ay), radius, (40, 35, 50), -1)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)
        cv2.circle(frame, (ax, ay), radius, (60, 60, 75), 2, cv2.LINE_AA)

        if span > 0:
            cv2.ellipse(frame, (ax, ay), (radius, radius), -90, 0, span, color, 5, cv2.LINE_AA)
            cv2.ellipse(frame, (ax, ay), (radius, radius), -90, 0, span, self.TEXT_WHITE, 1, cv2.LINE_AA)

        # Percentage inside ring
        pct_text = f"{int(hold_ratio * 100)}%"
        cv2.putText(frame, pct_text, (ax - 15, ay + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_WHITE, 1, cv2.LINE_AA)

        # Label Banner above ring
        lx = ax - 90
        ly = ay - radius - 12
        cv2.rectangle(frame, (lx, ly - 16), (lx + 210, ly + 6), (25, 20, 35), -1)
        cv2.rectangle(frame, (lx, ly - 16), (lx + 210, ly + 6), color, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (lx + 8, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def draw_thumb_hold_indicator(self, frame: np.ndarray, thumb_pt: Tuple[int, int], hold_ratio: float, thumb_type: str):
        """Compatibility wrapper delegating to draw_posture_hold_indicator."""
        self.draw_posture_hold_indicator(frame, thumb_pt, hold_ratio, thumb_type)

    def draw_glowing_skeleton(self, frame: np.ndarray, pixel_coords: List[Tuple[int, int]]):
        """Renders glowing multi-pass neon skeleton."""
        if not pixel_coords or len(pixel_coords) < 21:
            return

        glow_layer = frame.copy()
        for p1_idx, p2_idx in self.BONES:
            p1 = pixel_coords[p1_idx]
            p2 = pixel_coords[p2_idx]
            cv2.line(glow_layer, p1, p2, self.NEON_CYAN, 5, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.35, frame, 0.65, 0, frame)

        for p1_idx, p2_idx in self.BONES:
            p1 = pixel_coords[p1_idx]
            p2 = pixel_coords[p2_idx]
            cv2.line(frame, p1, p2, self.TEXT_WHITE, 1, cv2.LINE_AA)

        for i, (px, py) in enumerate(pixel_coords):
            if i == 8:  # Index Tip
                cv2.circle(frame, (px, py), 10, self.ARC_ORANGE, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, self.NEON_CYAN, -1, cv2.LINE_AA)
            elif i == 4:  # Thumb Tip
                cv2.circle(frame, (px, py), 10, self.ELECTRIC_YELLOW, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, self.TEXT_WHITE, -1, cv2.LINE_AA)
            elif i == 0:
                cv2.circle(frame, (px, py), 7, self.NEON_MAGENTA, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 3, self.NEON_CYAN, -1, cv2.LINE_AA)

    def draw_top_telemetry_bar(self, frame: np.ndarray, fps: float, active_status: str):
        """Draws sleek top bar."""
        panel_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, panel_h), self.DARK_PANEL, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, panel_h), (self.w, panel_h), self.NEON_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, "PPT GESTURE ASSISTANT", (20, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, self.NEON_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "EASY GESTURES: PALM (OPEN) | THUMB UP (NEXT) | PEACE (PREV) | POINT (EXIT) | FIST (CLOSE)", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        # Center Status Badge
        pill_w, pill_h = 420, 38
        pill_x = (self.w - pill_w) // 2
        pill_y = 9

        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (35, 30, 45), -1)
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), self.NEON_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, active_status, (pill_x + 16, pill_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (self.w - 160, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.NEON_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "STATUS: ONLINE", (self.w - 160, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.LASER_GREEN, 1, cv2.LINE_AA)

    def draw_action_pop_banner(self, frame: np.ndarray):
        """Draws bright pop-up notification on action trigger."""
        if time.time() < self.action_banner_timer and self.action_banner_text:
            banner_w = 520
            banner_h = 65
            bx = (self.w - banner_w) // 2
            by = self.h // 2 - 32

            overlay = frame.copy()
            cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h), (15, 60, 30), -1)
            cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
            cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), self.LASER_GREEN, 2, cv2.LINE_AA)

            cv2.putText(frame, self.action_banner_text, (bx + 20, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.TEXT_WHITE, 2, cv2.LINE_AA)
            if self.action_banner_sub:
                cv2.putText(frame, self.action_banner_sub, (bx + 22, by + 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.NEON_CYAN, 1, cv2.LINE_AA)

    def draw_bottom_controls_guide(self, frame: np.ndarray, status: str):
        """Draws bottom footer guide."""
        panel_h = 32
        y_top = self.h - panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (self.w, self.h), self.DARK_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, y_top), (self.w, y_top), self.NEON_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, f">> {status}", (20, y_top + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.ARC_ORANGE, 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q: QUIT]  [H: TOGGLE HUD]", (self.w - 240, y_top + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def apply_flash_effect(self, frame: np.ndarray):
        """Applies screen flash on action execution."""
        if self.flash_alpha > 0.01:
            overlay = np.full_like(frame, 255)
            cv2.addWeighted(overlay, self.flash_alpha, frame, 1.0 - self.flash_alpha, 0, frame)
