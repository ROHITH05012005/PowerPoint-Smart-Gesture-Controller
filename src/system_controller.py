"""
System Controller for PowerPoint Presentation & Virtual Computer Navigation.
Executes actions corresponding to finger gestures:
1: Open PowerPoint
2: Next Slide
3: Previous Slide
4: Exit PPT / Slideshow
5: Zoom Text
"""
import os
import time
import subprocess
import pyautogui
from config import ACTION_COOLDOWNS, DEMO_PPTX

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

class SystemController:
    """
    Translates gesture commands into PowerPoint controls, slide navigation, and text zooming.
    """
    def __init__(self, demo_pptx_path: str = DEMO_PPTX):
        self.demo_pptx = os.path.abspath(demo_pptx_path)
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_action_time = {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0
        }
        self.ppt_opened = False
        self.zoom_level = 100

    def can_trigger(self, gesture_id: int) -> bool:
        """Checks if enough debounce time has elapsed for this gesture."""
        cooldown = ACTION_COOLDOWNS.get(gesture_id, 1.0)
        now = time.time()
        return (now - self.last_action_time.get(gesture_id, 0.0)) >= cooldown

    def register_trigger(self, gesture_id: int):
        """Updates timestamp of last triggered action."""
        self.last_action_time[gesture_id] = time.time()

    def open_powerpoint(self) -> str:
        """
        [Gesture 1]: Opens PowerPoint application.
        Launches presentation_demo.pptx and puts it in presentation mode.
        """
        if not self.can_trigger(1):
            return "COOLDOWN"

        self.register_trigger(1)
        print("\n[ACTION: 1] Opening PowerPoint Application...")

        # 1. Launch PowerPoint with the demo presentation
        try:
            if os.path.exists(self.demo_pptx):
                os.startfile(self.demo_pptx)
            else:
                # Direct executable launch
                ppt_exe = r"C:\Program Files\Microsoft Office\Root\Office16\POWERPNT.EXE"
                if os.path.exists(ppt_exe):
                    subprocess.Popen([ppt_exe])
                else:
                    subprocess.Popen(["start", "powerpnt"], shell=True)
            self.ppt_opened = True

            # Brief pause to allow window to focus, then start Fullscreen Slideshow (F5)
            time.sleep(1.2)
            pyautogui.press('f5')
            return "LAUNCHED_PPT"
        except Exception as e:
            print(f"[ERROR] Could not launch PowerPoint: {e}")
            return "ERROR"

    def next_slide(self) -> str:
        """
        [Gesture 2]: Advances to the Next Slide.
        """
        if not self.can_trigger(2):
            return "COOLDOWN"

        self.register_trigger(2)
        print("[ACTION: 2] Next Slide (>>)")
        try:
            pyautogui.press('right')
            return "NEXT_SLIDE"
        except Exception as e:
            print(f"[ERROR] Next slide failed: {e}")
            return "ERROR"

    def prev_slide(self) -> str:
        """
        [Gesture 3]: Returns to the Previous Slide.
        """
        if not self.can_trigger(3):
            return "COOLDOWN"

        self.register_trigger(3)
        print("[ACTION: 3] Previous Slide (<<)")
        try:
            pyautogui.press('left')
            return "PREV_SLIDE"
        except Exception as e:
            print(f"[ERROR] Previous slide failed: {e}")
            return "ERROR"

    def exit_powerpoint(self) -> str:
        """
        [Gesture 4]: Exits from PPT Slideshow / Closes presentation.
        """
        if not self.can_trigger(4):
            return "COOLDOWN"

        self.register_trigger(4)
        print("[ACTION: 4] Exiting Slideshow / PowerPoint (ESC)")
        try:
            # Press Escape to end slideshow
            pyautogui.press('esc')
            return "EXIT_PPT"
        except Exception as e:
            print(f"[ERROR] Exit PPT failed: {e}")
            return "ERROR"

    def close_powerpoint(self) -> str:
        """
        Closes PowerPoint application completely (ESC + Alt+F4).
        """
        if not self.can_trigger(4):
            return "COOLDOWN"

        self.register_trigger(4)
        print("[ACTION: CLOSE] Closing PowerPoint Application (Alt+F4)...")
        try:
            pyautogui.press('esc')
            time.sleep(0.12)
            pyautogui.hotkey('alt', 'f4')
            self.ppt_opened = False
            return "CLOSED_PPT"
        except Exception as e:
            print(f"[ERROR] Close PPT failed: {e}")
            return "ERROR"

    def zoom_text(self) -> str:
        """
        [Gesture 5]: Zooms in on text / slide content.
        Uses Ctrl + '+' (or PowerPoint slide zoom '+').
        """
        if not self.can_trigger(5):
            return "COOLDOWN"

        self.register_trigger(5)
        print("[ACTION: 5] Zooming Text / Slide (+)")
        try:
            # In PPT Slideshow, pressing '+' zooms in
            pyautogui.press('+')
            # Standard browser/editor text zoom hotkey Ctrl + '+'
            pyautogui.hotkey('ctrl', '+')
            self.zoom_level = min(300, self.zoom_level + 15)
            return "ZOOM_TEXT"
        except Exception as e:
            print(f"[ERROR] Zoom text failed: {e}")
            return "ERROR"
