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
from PIL import Image, ImageOps, ImageEnhance
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
    """Contrast-Enhanced Inverted Aspect-Preserved Preprocessing for Bangla Characters.

    Inspired by banglaWrittenWordOCR pipeline:
    1. Boosts image contrast (2.0x) using PIL ImageEnhance to sharpen strokes.
    2. Adaptively inverts polarity so characters are white text on dark background.
    3. Resizes with LANCZOS preserving aspect ratio and pads into a (size x size) canvas.
    Returns (size, size, 3) RGB uint8 array.
    """
    if char_img is None or char_img.size == 0:
        return None

    h_raw, w_raw = char_img.shape[:2]
    if h_raw <= 0 or w_raw <= 0:
        return None

    # Convert OpenCV image (BGR/Gray) to PIL Image
    if len(char_img.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(char_img)

    # 1. Boost Contrast by 2.2x & Sharpness by 2.5x
    try:
        enhancer_c = ImageEnhance.Contrast(pil_img)
        img_enhanced = enhancer_c.enhance(2.2)
        enhancer_s = ImageEnhance.Sharpness(img_enhanced)
        img_enhanced = enhancer_s.enhance(2.5)
    except Exception:
        img_enhanced = pil_img

    # 2. Check Background Polarity & Adaptively Invert
    gray_np = np.array(img_enhanced.convert('L'))
    is_white_bg = gray_np.mean() > 127

    if is_white_bg:
        img_inv = ImageOps.invert(img_enhanced.convert('L'))
    else:
        img_inv = img_enhanced.convert('L')

    # 3. Aspect-Preserved Pad to Target Canvas using LANCZOS
    w, h = img_inv.size
    max_target = size - 4
    scale = max_target / float(max(w, h))

    new_w = max(1, min(max_target, int(w * scale)))
    new_h = max(1, min(max_target, int(h * scale)))

    resized = img_inv.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create dark background canvas
    canvas = Image.new('L', (size, size), color=0)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas.paste(resized, (x_off, y_off))

    # Convert to RGB numpy array
    rgb_canvas = canvas.convert('RGB')
    return np.array(rgb_canvas)


if __name__ == "__main__":
    X, y = load_images()
    print(f"Images shape: {X.shape}, Labels: {len(set(y))} unique classes")
    print(f"Sample labels: {list(set(y))[:10]}")

    counts = Counter(y)
    print(f"Min samples/class: {min(counts.values())}, Max: {max(counts.values())}")
