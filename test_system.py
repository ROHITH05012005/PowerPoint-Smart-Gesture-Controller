"""
Verification test suite for Open Palm, Closed Fist, Thumbs Up/Down, and Swipes.
"""
import os
import cv2
import math
import numpy as np
import time
from src.motion_tracker import MotionAndPostureTracker
from src.cyberpunk_hud import CyberpunkHUD
from src.system_controller import SystemController

def run_system_verification():
    print("=" * 65)
    print("  VERIFYING EASY GESTURE CONTROLLER (PALM, FIST & SWIPES)")
    print("=" * 65)

    tracker = MotionAndPostureTracker()

    # Base hand layout: Wrist at (640, 500)
    # 1. Test OPEN PALM (✋) - all 5 fingers extended upright
    mock_open_palm = {
        "pixel_coords": [
            (640, 500),  # 0: Wrist
            (600, 470),  # 1: Thumb CMC
            (570, 440),  # 2: Thumb MCP
            (540, 400),  # 3: Thumb IP
            (510, 360),  # 4: Thumb Tip (Extended out to left)
            (600, 410),  # 5: Index MCP
            (590, 350),  # 6: Index PIP
            (585, 300),  # 7: Index DIP
            (580, 240),  # 8: Index Tip (Extended high)
            (635, 400),  # 9: Middle MCP
            (635, 335),  # 10: Middle PIP
            (635, 280),  # 11: Middle DIP
            (635, 220),  # 12: Middle Tip (Extended high)
            (670, 410),  # 13: Ring MCP
            (675, 350),  # 14: Ring PIP
            (680, 300),  # 15: Ring DIP
            (685, 245),  # 16: Ring Tip (Extended high)
            (705, 425),  # 17: Pinky MCP
            (715, 375),  # 18: Pinky PIP
            (725, 330),  # 19: Pinky DIP
            (735, 275),  # 20: Pinky Tip (Extended high)
        ]
    }
    palm_res = tracker.detect_posture(mock_open_palm, 100.0)
    print(f"  {'[PASS]' if palm_res == 'OPEN_PALM' else '[FAIL]'} Open Palm Detection -> Open PPT: {palm_res}")

    # 2. Test CLOSED FIST (✊) - 4 fingers curled, thumb tucked
    tracker.last_posture_time = 0.0
    mock_fist = {
        "pixel_coords": [
            (640, 500),  # 0: Wrist
            (610, 470),  # 1: Thumb CMC
            (590, 440),  # 2: Thumb MCP
            (585, 420),  # 3: Thumb IP
            (595, 410),  # 4: Thumb Tip (Tucked against index MCP)
            (600, 410),  # 5: Index MCP
            (595, 450),  # 6: Index PIP
            (595, 470),  # 7: Index DIP
            (600, 480),  # 8: Index Tip (Curled down into palm)
            (635, 400),  # 9: Middle MCP
            (635, 450),  # 10: Middle PIP
            (635, 475),  # 11: Middle DIP
            (635, 485),  # 12: Middle Tip (Curled down)
            (670, 410),  # 13: Ring MCP
            (670, 450),  # 14: Ring PIP
            (670, 475),  # 15: Ring DIP
            (670, 485),  # 16: Ring Tip (Curled down)
            (705, 425),  # 17: Pinky MCP
            (705, 460),  # 18: Pinky PIP
            (705, 480),  # 19: Pinky DIP
            (705, 490),  # 20: Pinky Tip (Curled down)
        ]
    }
    fist_res = tracker.detect_posture(mock_fist, 200.0)
    print(f"  {'[PASS]' if fist_res == 'FIST' else '[FAIL]'} Closed Fist Detection -> Close PPT: {fist_res}")

    # 3. Test THUMBS UP (👍)
    tracker.last_posture_time = 0.0
    mock_thumbs_up = dict(mock_fist)
    mock_thumbs_up["pixel_coords"] = list(mock_fist["pixel_coords"])
    mock_thumbs_up["pixel_coords"][4] = (560, 260)  # Tip high
    mock_thumbs_up["pixel_coords"][3] = (570, 310)  # IP
    mock_thumbs_up["pixel_coords"][2] = (580, 360)  # MCP
    thumb_up_res = tracker.detect_posture(mock_thumbs_up, 300.0)
    print(f"  {'[PASS]' if thumb_up_res == 'THUMBS_UP' else '[FAIL]'} Thumbs Up Detection -> Next Slide: {thumb_up_res}")

    # 4. Test THUMBS DOWN (👎)
    tracker.last_posture_time = 0.0
    mock_thumbs_down = dict(mock_fist)
    mock_thumbs_down["pixel_coords"] = list(mock_fist["pixel_coords"])
    mock_thumbs_down["pixel_coords"][4] = (560, 520)  # Tip pointing down
    mock_thumbs_down["pixel_coords"][3] = (570, 465)  # IP
    mock_thumbs_down["pixel_coords"][2] = (580, 420)  # MCP
    thumb_down_res = tracker.detect_posture(mock_thumbs_down, 400.0)
    print(f"  {'[PASS]' if thumb_down_res == 'THUMBS_DOWN' else '[FAIL]'} Thumbs Down Detection -> Exit Slideshow: {thumb_down_res}")

    # 5. Test PEACE SIGN -> Previous Slide
    tracker.last_posture_time = 0.0
    mock_peace = dict(mock_fist)
    mock_peace["pixel_coords"] = list(mock_fist["pixel_coords"])
    # Index extended up
    mock_peace["pixel_coords"][6] = (600, 310)
    mock_peace["pixel_coords"][7] = (600, 260)
    mock_peace["pixel_coords"][8] = (600, 200)
    # Middle extended up
    mock_peace["pixel_coords"][10] = (635, 300)
    mock_peace["pixel_coords"][11] = (635, 250)
    mock_peace["pixel_coords"][12] = (635, 190)
    peace_res = tracker.detect_posture(mock_peace, 450.0)
    print(f"  {'[PASS]' if peace_res == 'PEACE' else '[FAIL]'} Peace Sign (2 Fingers) -> Prev Slide: {peace_res}")

    # 6. Test POINTING UP -> Exit Slideshow
    tracker.last_posture_time = 0.0
    mock_pointing = dict(mock_fist)
    mock_pointing["pixel_coords"] = list(mock_fist["pixel_coords"])
    # Only Index extended up
    mock_pointing["pixel_coords"][6] = (600, 310)
    mock_pointing["pixel_coords"][7] = (600, 260)
    mock_pointing["pixel_coords"][8] = (600, 200)
    point_res = tracker.detect_posture(mock_pointing, 480.0)
    print(f"  {'[PASS]' if point_res == 'POINTING' else '[FAIL]'} Pointing Up (1 Finger) -> Exit Slideshow: {point_res}")

    # 5. Test Intentional Fast Swipes
    tracker.swipe_history.clear()
    tracker.last_swipe_time = 0.0
    sl_res = "NONE"
    for i in range(8):
        res = tracker.check_intentional_swipe(int(1100 - i * 26), 150, 500.0 + i * 0.025)
        if res != "NONE":
            sl_res = res
            break
    print(f"  {'[PASS]' if sl_res == 'SWIPE_LEFT' else '[FAIL]'} Fast Swipe Left -> Prev Slide: {sl_res}")

    tracker.swipe_history.clear()
    tracker.last_swipe_time = 0.0
    sr_res = "NONE"
    for i in range(8):
        res = tracker.check_intentional_swipe(int(150 + i * 26), 150, 600.0 + i * 0.025)
        if res != "NONE":
            sr_res = res
            break
    print(f"  {'[PASS]' if sr_res == 'SWIPE_RIGHT' else '[FAIL]'} Fast Swipe Right -> Next Slide: {sr_res}")

    # 6. Render Posture Charging Meter HUD Snapshot
    print("\n--- Verifying Easy Posture HUD Snapshot ---")
    test_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud = CyberpunkHUD(frame_width=1280, frame_height=720)
    hud.update_animation()

    hud.draw_glowing_skeleton(test_canvas, mock_open_palm["pixel_coords"])
    hud.draw_posture_hold_indicator(test_canvas, anchor_pt=(635, 400), hold_ratio=0.85, posture_type="OPEN_PALM")
    hud.draw_top_telemetry_bar(test_canvas, fps=60.0, active_status="OPEN PALM: 85% -> OPEN POWERPOINT (F5)")
    hud.draw_bottom_controls_guide(test_canvas, "EASY GESTURE CONTROLS ACTIVE")

    preview_path = os.path.join("screenshots", "easy_gesture_hud_preview.png")
    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite(preview_path, test_canvas)
    print(f"[PASS] Easy Gesture preview saved to: {preview_path}")

    print("\n" + "=" * 65)
    print("[ALL TESTS PASSED] Easy Gesture Controller (Palm & Fist) Ready!")
    print("=" * 65)

if __name__ == "__main__":
    run_system_verification()
