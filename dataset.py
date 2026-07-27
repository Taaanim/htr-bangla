"""
dataset.py — Dataset loading, preprocessing, and augmentation for Bangla HTR

Loads 122-class Bangla handwriting dataset using cv2 (matching the proven approach).
Images: 32×32 RGB | Labels: sklearn LabelEncoder
Handles class imbalance with WeightedRandomSampler.
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ─── Constants ────────────────────────────────────────────────
IMG_SIZE = 32
BASE_DIR = "Bangla dataset"
DATASET_DIR = os.path.join(BASE_DIR, "dataset_filtered")
CSV_PATH = os.path.join(BASE_DIR, "metaData_img.csv")


def get_folder_to_char(csv_path: str = CSV_PATH) -> dict:
    """Reads metadata CSV and returns folder_id -> character_name mapping."""
    df = pd.read_csv(csv_path)
    return dict(zip(df.iloc[:, 0].astype(str).str.strip(), df.iloc[:, 1].str.strip()))


def load_images(root_dir: str = DATASET_DIR, csv_path: str = CSV_PATH,
                size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all images from dataset folders, returns (images, labels) arrays.

    Images are loaded as RGB, resized to (size, size).
    Labels are the character strings from metadata CSV.
    """
    folder_to_char = get_folder_to_char(csv_path)
    images, labels = [], []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        char_name = folder_to_char.get(folder.strip(), folder)

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            img = cv2.imread(fpath)
            if img is not None:
                img = cv2.resize(img, (size, size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(char_name)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    print(f"Loaded {len(X)} images across {len(set(y))} classes")
    return X, y


class CharDataset(Dataset):
    """PyTorch Dataset for character images."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 transform: Optional[transforms.Compose] = None):
        # features: (N, C, H, W) float32 normalized
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def prepare_data(test_size: float = 0.15, batch_size: int = 128,
                 random_state: int = 42) -> Tuple[DataLoader, DataLoader, LabelEncoder, int]:
    """Full data pipeline: load → encode → split → augment → DataLoaders.

    Returns: (train_loader, val_loader, label_encoder, num_classes)
    """
    X, y = load_images()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Save label encoder for inference
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(le, "checkpoints/label_encoder.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # Transpose to PyTorch format: (N, H, W, C) → (N, C, H, W)
    X_chw = np.transpose(X, (0, 3, 1, 2))

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_chw, y_encoded, test_size=test_size,
        random_state=random_state, stratify=y_encoded
    )
    print(f"Train: {len(X_train)} images | Val: {len(X_val)} images")

    # Training augmentation — realistic for handwriting
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    ])

    train_dataset = CharDataset(X_train, y_train, transform=train_transform)
    val_dataset = CharDataset(X_val, y_val, transform=None)

    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, le, num_classes


def process_character(char_img: np.ndarray, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """Preprocesses a character image for model inference.

    Handles both light background (drawing canvas) and dark background (dataset).
    Returns (size, size, 3) RGB float32 normalized or uint8 array.
    """
    if char_img is None or char_img.size == 0:
        return None

    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_img

    is_white_bg = gray.mean() > 127
    if is_white_bg:
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        if is_white_bg:
            resized = cv2.bitwise_not(resized)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    x, y, w, h = cv2.boundingRect(coords)
    ink_crop = gray[y:y+h, x:x+w]

    pad = int(max(w, h) * 0.18)
    side = max(w, h) + pad * 2
    bg_val = 255 if is_white_bg else 0
    square = np.full((side, side), bg_val, dtype=np.uint8)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = ink_crop

    final = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    final = cv2.GaussianBlur(final, (3, 3), 0)

    # Ensure final output is ALWAYS dark background (matching dataset training format)
    if final.mean() > 127:
        final = cv2.bitwise_not(final)

    return cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)


if __name__ == "__main__":
    X, y = load_images()
    print(f"Images shape: {X.shape}, Labels: {len(set(y))} unique classes")
    print(f"Sample labels: {list(set(y))[:10]}")

    counts = Counter(y)
    print(f"Min samples/class: {min(counts.values())}, Max: {max(counts.values())}")
