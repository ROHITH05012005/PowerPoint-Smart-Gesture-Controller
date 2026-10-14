"""
Interactive Webcam Dataset Recorder for Hand Gesture Recognition.
Allows users to record custom gesture samples directly from their webcam.
Saves biometric feature vectors to dataset/gesture_data.csv.
"""
import os
import csv
import time
import cv2
import numpy as np
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    GESTURE_CLASSES, GESTURE_DESCRIPTIONS,
    DATASET_DIR, DATASET_FILE
)
from src.hand_detector import HandDetector
from src.dataset import extract_features_from_landmarks

def count_existing_samples():
    """Counts recorded samples per class in the CSV file."""
    counts = {k: 0 for k in GESTURE_CLASSES.keys()}
    if not os.path.exists(DATASET_FILE):
        return counts

    try:
        with open(DATASET_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    label = int(row[-1])
                    if label in counts:
                        counts[label] += 1
    except Exception:
        pass
    return counts

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    sample_counts = count_existing_samples()

    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
        return

    detector = HandDetector(mode=False, max_hands=1, detection_confidence=0.7)

    current_class_id = 0
    is_recording = False
    record_delay = 0.08  # delay between frame captures
    last_record_time = time.time()

    # Check if CSV exists, if not write header
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"feat_{i}" for i in range(73)] + ["label"]
            writer.writerow(header)

    print("\n" + "=" * 65)
    print("      AI GESTURE RECORDER - WEBCAM DATASET COLLECTOR")
    print("=" * 65)
    print("KEYS:")
    print("  [0-7] : Select Gesture Class")
    print("  [SPACE] : Toggle continuous recording")
    print("  [C]     : Capture single sample")
    print("  [Q]     : Exit")
    print("=" * 65 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror viewing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        has_hand = detector.process(frame)
        lm_dict = detector.get_landmarks(frame) if has_hand else None

        # Handle recording
        now = time.time()
        captured_this_frame = False

        if (is_recording or cv2.waitKey(1) & 0xFF == ord('c')) and lm_dict and (now - last_record_time > record_delay):
            try:
                features = extract_features_from_landmarks(lm_dict["raw_norm"])
                with open(DATASET_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(features) + [current_class_id])
                sample_counts[current_class_id] += 1
                last_record_time = now
                captured_this_frame = True
            except Exception as e:
                print(f"[ERROR] Feature extraction failed: {e}")

        # Draw HUD Skeleton
        if lm_dict:
            pts = lm_dict["pixel_coords"]
            # Draw bones
            bones = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
            ]
            for p1, p2 in bones:
                cv2.line(frame, pts[p1], pts[p2], (255, 243, 0), 2, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (0, 165, 255), -1, cv2.LINE_AA)

        # UI Overlay: Top Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.line(frame, (0, 80), (w, 80), (255, 243, 0), 1, cv2.LINE_AA)

        # Active Class Info
        class_name = GESTURE_CLASSES[current_class_id]
        desc = GESTURE_DESCRIPTIONS.get(class_name, "")
        cv2.putText(frame, f"SELECT CLASS [{current_class_id}]: {class_name}", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 243, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"POSE: {desc}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Recording Status
        rec_status = "RECORDING ON" if is_recording else "STANDBY (PAUSED)"
        rec_color = (0, 0, 255) if is_recording else (120, 120, 120)
        cv2.rectangle(frame, (w - 240, 15), (w - 20, 65), (35, 30, 45), -1)
        cv2.rectangle(frame, (w - 240, 15), (w - 20, 65), rec_color, 2, cv2.LINE_AA)
        cv2.putText(frame, rec_status, (w - 225, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, rec_color, 2, cv2.LINE_AA)

        # Sidebar: Sample Counts per Class
        sb_w = 260
        sb_h = 240
        cv2.rectangle(overlay, (20, 100), (20 + sb_w, 100 + sb_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (20, 100), (20 + sb_w, 100 + sb_h), (255, 243, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "SAMPLES RECORDED", (32, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 243, 0), 1, cv2.LINE_AA)

        for cid in sorted(GESTURE_CLASSES.keys()):
            y_pos = 148 + cid * 22
            cname = GESTURE_CLASSES[cid]
            cnt = sample_counts.get(cid, 0)
            prefix = "> " if cid == current_class_id else "  "
            color = (0, 255, 128) if cid == current_class_id else (180, 180, 180)
            cv2.putText(frame, f"{prefix}[{cid}] {cname[:12]}: {cnt}", (32, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # Bottom Helper Bar
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "[0-7] Choose Gesture  |  [SPACE] Start/Stop Auto-Record  |  [C] Capture 1  |  [Q] Save & Quit",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 243, 0), 1, cv2.LINE_AA)

        cv2.imshow("Webcam Gesture Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Spacebar
            is_recording = not is_recording
        elif ord('0') <= key <= ord('7'):
            current_class_id = key - ord('0')

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Dataset collection closed. Samples saved to:", DATASET_FILE)

if __name__ == "__main__":
    main()
