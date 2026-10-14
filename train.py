"""
PyTorch Deep Learning Training Script for Hand Gesture Classification.
Trains GestureClassifierNN, evaluates accuracy, and exports model weights.
"""
import os
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Pure numpy implementation used for stratified splitting (zero extra dependency)

from config import (
    DATASET_FILE, MODEL_DIR, MODEL_PATH,
    INPUT_FEATURES, NUM_CLASSES, GESTURE_CLASSES
)
from src.model import GestureClassifierNN
from src.dataset import GestureDataset
from generate_seed_data import generate_seed_dataset

def load_data(csv_path: str):
    """
    Loads features and labels from CSV using pure numpy.
    """
    if not os.path.exists(csv_path):
        print(f"[INFO] Dataset file not found at {csv_path}. Generating seed data now...")
        generate_seed_dataset(samples_per_class=450)

    features = []
    labels = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if not row:
                continue
            feats = [float(val) for val in row[:-1]]
            label = int(row[-1])
            features.append(feats)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

def stratified_split(features: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    """
    Performs a deterministic stratified train/test split.
    """
    np.random.seed(seed)
    train_indices = []
    val_indices = []

    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        np.random.shuffle(c_indices)
        split_pt = int(len(c_indices) * (1.0 - val_ratio))
        train_indices.extend(c_indices[:split_pt])
        val_indices.extend(c_indices[split_pt:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return (
        features[train_indices], labels[train_indices],
        features[val_indices], labels[val_indices]
    )

def train_gesture_model(epochs: int = 35, batch_size: int = 32, lr: float = 0.001):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 65)
    print("      DEEP LEARNING TRAINING PIPELINE - GESTURE CLASSIFIER")
    print("=" * 65)

    # 1. Load dataset
    features, labels = load_data(DATASET_FILE)
    print(f"[+] Loaded {len(features)} total samples across {len(np.unique(labels))} classes.")
    print(f"[+] Input feature dimension: {features.shape[1]}")

    # 2. Train / Val split
    X_train, y_train, X_val, y_val = stratified_split(features, labels, val_ratio=0.2)
    print(f"[+] Training samples: {len(X_train)}  |  Validation samples: {len(X_val)}")

    # 3. Create PyTorch datasets & dataloaders
    train_dataset = GestureDataset(X_train, y_train, augment=True)
    val_dataset = GestureDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training device: {device}")

    model = GestureClassifierNN(input_dim=features.shape[1], num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    start_time = time.time()

    print("\n--- Starting Training Epochs ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        scheduler.step()
        epoch_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                outputs = model(feats)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        val_acc = (correct_val / total_val) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": features.shape[1],
                "num_classes": NUM_CLASSES,
                "best_val_acc": best_val_acc,
                "classes": GESTURE_CLASSES
            }, MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"[SUCCESS] Training completed in {elapsed:.2f}s!")
    print(f"[BEST VALIDATION ACCURACY]: {best_val_acc:.2f}%")
    print(f"[MODEL SAVED]: {MODEL_PATH}")

    # Save class dictionary JSON for easy inference inspection
    classes_json_path = os.path.join(MODEL_DIR, "gesture_classes.json")
    with open(classes_json_path, "w") as f:
        json.dump(GESTURE_CLASSES, f, indent=4)
    print(f"[CLASSES EXPORTED]: {classes_json_path}")
    print("=" * 65 + "\n")

    return best_val_acc

if __name__ == "__main__":
    train_gesture_model(epochs=35, batch_size=32, lr=0.001)
