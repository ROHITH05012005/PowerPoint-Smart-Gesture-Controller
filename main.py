"""
AI-Powered PowerPoint Smart Gesture Virtual Computer Controller.
Controls:
  ✋ Open Palm (Hold 0.35s)     : Open PowerPoint Presentation (F5)
  👍 Thumbs Up (Hold 0.35s)      : Next Slide (>>)
  ✌️ Peace Sign (Hold 0.35s)     : Previous Slide (<<)
  ☝️ Point Up (Hold 0.35s)       : Exit Slideshow (ESC)
  ✊ Closed Fist (Hold 0.35s)    : Close PowerPoint
  ➡️ Fast Swipe Right (Optional) : Next Slide (>>)
  ⬅️ Fast Swipe Left (Optional)  : Previous Slide (<<)
"""
import os
import sys

# Suppress MediaPipe C++ stderr logging
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import cv2
import numpy as np

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    DEMO_PPTX
)
from src.hand_detector import HandDetector
from src.system_controller import SystemController
from src.cyberpunk_hud import CyberpunkHUD
from src.motion_tracker import MotionAndPostureTracker

def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 65)
    print("  AI-POWERED PPT SMART GESTURE CONTROLLER")
    print("=" * 65)
    print("EASY GESTURE CONTROLS:")
    print("  [ ✋ PALM ]        Hold 0.35s : Open PowerPoint Presentation (F5)")
    print("  [ 👍 THUMBS UP ]   Hold 0.35s : Next Slide (>>)")
    print("  [ ✌️ PEACE / 2 ]   Hold 0.35s : Previous Slide (<<)")
    print("  [ ☝️ POINT / 1 ]   Hold 0.35s : Exit Slideshow (ESC)")
    print("  [ ✊ CLOSED FIST ]  Hold 0.35s : Close PowerPoint Application")
    print("  [ SWIPES ]         Optional   : Fast Swipe Right (>>) / Left (<<)")
    print("=" * 65 + "\n")

    # DirectShow backend avoids MSMF lock issues on Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Sensor exposure warmup
    test_frame = None
    for _ in range(12):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            break
        time.sleep(0.08)

    if test_frame is None:
        print("[ERROR] Cannot read frame from webcam.")
        return

    actual_h, actual_w, _ = test_frame.shape

    detector = HandDetector(mode=False, max_hands=1, detection_confidence=0.7)
    hud = CyberpunkHUD(frame_width=actual_w, frame_height=actual_h)
    controller = SystemController(demo_pptx_path=DEMO_PPTX)
    tracker = MotionAndPostureTracker()

    prev_time = time.time()
    fps_smooth = 30.0
    hud_visible = True
    last_status = "READY (Show Palm, Fist, or Swipe)"

    # Posture hold confirmation state (10 frames = ~0.35s)
    posture_hold_count = 0
    active_posture = "NONE"
    POSTURE_HOLD_FRAMES = 10

    print("[READY] System online! Position hand in webcam window.")
    print("CONTROLS: [H] Toggle HUD  |  [Q] Quit")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            cur_time = time.time()
            dt = cur_time - prev_time
            prev_time = cur_time
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            hud.update_animation()

            has_hand = detector.process(frame)
            lm_dict = detector.get_landmarks(frame) if has_hand else None

            posture_hold_ratio = 0.0
            anchor_pt = (0, 0)

            if lm_dict:
                pts = lm_dict["pixel_coords"]
                wrist_pt = pts[0]
                middle_mcp = pts[9]
                # Center of the palm: robust against finger curl, tilt, and motion blur
                palm_center = ((wrist_pt[0] + middle_mcp[0]) // 2, (wrist_pt[1] + middle_mcp[1]) // 2)

                # 1. Continuous Intentional Swipe Evaluation
                swipe_action = tracker.check_intentional_swipe(palm_center[0], palm_center[1], cur_time)

                if swipe_action == "SWIPE_LEFT":
                    controller.prev_slide()
                    last_status = "SWIPE LEFT -> PREV SLIDE <<"
                    hud.trigger_action_flash("<< SWIPE LEFT <<", "RETURNED TO PREVIOUS SLIDE")
                    posture_hold_count = 0
                    active_posture = "NONE"

                elif swipe_action == "SWIPE_RIGHT":
                    controller.next_slide()
                    last_status = "SWIPE RIGHT -> NEXT SLIDE >>"
                    hud.trigger_action_flash(">> SWIPE RIGHT >>", "ADVANCED TO NEXT SLIDE")
                    posture_hold_count = 0
                    active_posture = "NONE"

                else:
                    # 2. Check Static Postures only if hand is NOT actively moving (prevents open-hand swipe from triggering PPT open)
                    if tracker.is_hand_moving(speed_threshold=170.0):
                        posture_hold_count = 0
                        active_posture = "NONE"
                    else:
                        detected = tracker.detect_posture(lm_dict, cur_time)
                        if detected in ["OPEN_PALM", "FIST", "THUMBS_UP", "PEACE", "POINTING", "THUMBS_DOWN"]:
                            if detected == active_posture:
                                posture_hold_count += 1
                            else:
                                active_posture = detected
                                posture_hold_count = 1

                            posture_hold_ratio = min(1.0, posture_hold_count / float(POSTURE_HOLD_FRAMES))
                            if active_posture in ["THUMBS_UP", "THUMBS_DOWN"]:
                                anchor_pt = pts[4]
                            elif active_posture in ["PEACE", "POINTING"]:
                                anchor_pt = pts[8]
                            else:
                                anchor_pt = palm_center

                            # Trigger upon completing hold frames
                            if posture_hold_count == POSTURE_HOLD_FRAMES:
                                if active_posture == "OPEN_PALM":
                                    controller.open_powerpoint()
                                    last_status = "OPEN PALM -> OPENED POWERPOINT (F5)"
                                    hud.trigger_action_flash(">> OPEN PALM <<", "OPENED POWERPOINT PRESENTATION (F5)")
                                    tracker.register_posture_action(cur_time)

                                elif active_posture == "THUMBS_UP":
                                    controller.next_slide()
                                    last_status = "THUMBS UP -> NEXT SLIDE >>"
                                    hud.trigger_action_flash(">> THUMBS UP >>", "ADVANCED TO NEXT SLIDE")
                                    tracker.register_posture_action(cur_time)

                                elif active_posture == "PEACE":
                                    controller.prev_slide()
                                    last_status = "PEACE (2 FINGERS) -> PREV SLIDE <<"
                                    hud.trigger_action_flash("<< PEACE SIGN <<", "RETURNED TO PREVIOUS SLIDE")
                                    tracker.register_posture_action(cur_time)

                                elif active_posture == "POINTING":
                                    controller.exit_powerpoint()
                                    last_status = "POINT UP -> EXIT SLIDESHOW (ESC)"
                                    hud.trigger_action_flash(">> POINT UP >>", "EXITED SLIDESHOW (ESC)")
                                    tracker.register_posture_action(cur_time)

                                elif active_posture == "FIST":
                                    controller.close_powerpoint()
                                    last_status = "CLOSED FIST -> CLOSED POWERPOINT"
                                    hud.trigger_action_flash(">> CLOSED FIST <<", "CLOSED POWERPOINT APPLICATION")
                                    tracker.register_posture_action(cur_time)

                                elif active_posture == "THUMBS_DOWN":
                                    controller.exit_powerpoint()
                                    last_status = "THUMBS DOWN -> EXIT SLIDESHOW (ESC)"
                                    hud.trigger_action_flash(">> THUMBS DOWN >>", "EXITED SLIDESHOW (ESC)")
                                    tracker.register_posture_action(cur_time)
                        else:
                            posture_hold_count = 0
                            active_posture = "NONE"

            else:
                posture_hold_count = 0
                active_posture = "NONE"
                # If hand left the frame, clear stale swipe trajectory after brief grace period
                if tracker.swipe_history and (cur_time - tracker.swipe_history[-1][2] > 0.35):
                    tracker.swipe_history.clear()

            # Render Clean Sci-Fi Cyberpunk HUD
            if hud_visible:
                if lm_dict:
                    hud.draw_glowing_skeleton(frame, lm_dict["pixel_coords"])
                    # Visual charging ring with action label when holding steady
                    hud.draw_posture_hold_indicator(frame, anchor_pt, posture_hold_ratio, active_posture)
                    # Glowing motion streak trail during rapid hand gestures
                    trail = tracker.get_swipe_trail()
                    if len(trail) > 3 and tracker.is_hand_moving(speed_threshold=120.0):
                        hud.draw_air_drawing_trail(frame, trail, palm_center)

                hud.draw_top_telemetry_bar(frame, fps_smooth, last_status)
                hud.draw_bottom_controls_guide(frame, last_status)
                hud.draw_action_pop_banner(frame)
                hud.apply_flash_effect(frame)

            cv2.imshow("AI-Powered Guided Gesture Virtual Computer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('h'):
                hud_visible = not hud_visible

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Gesture Controller terminated cleanly.")

if __name__ == "__main__":
    main()
