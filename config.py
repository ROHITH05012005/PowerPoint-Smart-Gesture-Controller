"""
Configuration settings for AI-Powered Gesture-Controlled Virtual Computer.
Configured for PowerPoint Presentation and Text Navigation control.
"""
import os

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 60

# Screen & Interaction Box
FRAME_MARGIN_X = 120
FRAME_MARGIN_Y = 90

# Gesture Action Cooldowns (Debounce in seconds to avoid multi-triggering)
ACTION_COOLDOWNS = {
    1: 3.0,   # OPEN PPT: Wait 3 seconds before opening again
    2: 0.6,   # NEXT SLIDE: 0.6s between slide advances
    3: 0.6,   # PREV SLIDE: 0.6s between slide returns
    4: 1.5,   # EXIT PPT: 1.5s debounce
    5: 0.6    # ZOOM TEXT: 0.6s step zoom
}

# Number of consecutive frames gesture must be held before triggering action
# This ensures zero accidental triggers while moving hand
GESTURE_HOLD_FRAMES = 8

# Deep Learning Model Settings
NUM_LANDMARKS = 21
NUM_CLASSES = 6
INPUT_FEATURES = 73
CONFIDENCE_THRESHOLD = 0.70

# Gesture Class Labels for PowerPoint Presentation Control
GESTURE_CLASSES = {
    0: "NEUTRAL_FIST",     # Fist / 0 fingers: Standby, no action
    1: "ONE_OPEN_PPT",     # 1 Finger: Open PowerPoint application
    2: "TWO_NEXT_SLIDE",   # 2 Fingers: Next Slide
    3: "THREE_PREV_SLIDE", # 3 Fingers: Previous Slide
    4: "FOUR_EXIT_PPT",    # 4 Fingers: Exit Slideshow / Close PPT
    5: "FIVE_ZOOM_TEXT"    # 5 Fingers: Zoom the text / Magnify
}

GESTURE_DESCRIPTIONS = {
    "NEUTRAL_FIST": "Fist: Standby (Hand at Rest)",
    "ONE_OPEN_PPT": "[ 1 ] Open PowerPoint Presentation",
    "TWO_NEXT_SLIDE": "[ 2 ] Next Slide (>>)",
    "THREE_PREV_SLIDE": "[ 3 ] Previous Slide (<<)",
    "FOUR_EXIT_PPT": "[ 4 ] Exit from PPT / Slideshow",
    "FIVE_ZOOM_TEXT": "[ 5 ] Zoom the Text (+)"
}

# Paths - supports both standard Python and PyInstaller frozen bundle
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_FILE = os.path.join(DATASET_DIR, "gesture_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pth")
SCREENSHOTS_DIR = os.path.join(EXE_DIR, "screenshots")

# Presentation file: check EXE directory first (for custom user presentation), then bundled
if os.path.exists(os.path.join(EXE_DIR, "presentation_demo.pptx")):
    DEMO_PPTX = os.path.join(EXE_DIR, "presentation_demo.pptx")
else:
    DEMO_PPTX = os.path.join(BASE_DIR, "presentation_demo.pptx")

# Futuristic Cyberpunk / Iron-Man HUD Color Palette (BGR)
HUD_CYAN = (255, 243, 0)          # Cyber Cyan #00F3FF
HUD_MAGENTA = (255, 0, 255)       # Neon Magenta
HUD_ORANGE = (0, 165, 255)        # Stark Arc Reactor Orange
HUD_GREEN = (50, 255, 120)        # Laser Green
HUD_DARK_BG = (15, 15, 20)        # Deep Charcoal
HUD_WHITE = (255, 255, 255)       # Bright Pure White
HUD_RED = (40, 40, 255)           # Alert Red
HUD_YELLOW = (0, 240, 255)        # Electric Yellow
