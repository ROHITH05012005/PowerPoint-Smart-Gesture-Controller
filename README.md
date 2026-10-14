# 🌌 AI-Powered PowerPoint Smart Gesture Controller

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.1.0-00C0FF.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Operational-brightgreen.svg)]()

An interactive, touchless Presentation Controller featuring a **Holographic Guided Circle Portal** and **Biometric Thumb Posture Recognition** with an **Iron-Man Cyberpunk HUD**.

---

## 🎯 Gesture Control Mappings

| Gesture | Hand Pose (Hold 0.35s) | Action Executed |
| :---: | :--- | :--- |
| **✋ Open Palm** | All 5 fingers extended upwards | **Open PowerPoint Presentation** *(Fullscreen `F5`)* |
| **👍 Thumbs Up** | Fist with thumb pointing straight **UP** | **Next Slide (`>>`)** *(Advance presentation)* |
| **✌️ Peace Sign** | Index and Middle extended (**2 fingers**) | **Previous Slide (`<<`)** *(Return to prev slide)* |
| **☝️ Point Up** | Index finger pointing straight **UP** | **Exit Slideshow (`ESC`)** *(Leave fullscreen)* |
| **✊ Closed Fist** | Closed fist with fingers tucked | **Close PowerPoint Application** *(`Alt+F4`)* |
| **➡️ / ⬅️ Swipes** | Quick hand wave to Right / Left | *Optional dynamic Next / Previous slide* |

---

## 📸 Guided Holographic Portal HUD

The center of the screen features an interactive circular track:
- **Visual Progress Arc**: As you trace the circle with your index finger, an arc fills up dynamically (`0% -> 100%`).
- **Clockwise (Cyan/Green Arc)** $\rightarrow$ Triggers **Open PowerPoint**.
- **Anti-Clockwise (Orange/Red Arc)** $\rightarrow$ Triggers **Close PowerPoint**.
- **Live Command Card**: Sidebar updates in real-time, highlighting your active gesture.

---

## 🚀 How to Run

```powershell
python main.py
```

### Keyboard Shortcuts
- **`H`**: Toggle HUD overlay on / off.
- **`Q`** or **`ESC`**: Quit application cleanly.
