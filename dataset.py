"""
dataset.py - Dataset Handler, Augmentation Pipeline, & Synthetic Text Generator

Handles loading the 121-class local Bangla handwriting dataset, mapping labels from metaData_img.csv,
applying realistic handwriting augmentations (elastic transform, skew, blur, noise),
and generating synthetic text samples using system/custom Bangla TTF fonts.
"""

import os
import glob
import random
import math
from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates


class BanglaTokenizer:
    """Tokenizer mapping Bangla characters to token IDs and back."""

    BLANK_TOKEN = "<blank>"
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, metadata_csv_path: Optional[str] = None):
        self.char2idx: Dict[str, int] = {
            self.BLANK_TOKEN: 0,
            self.PAD_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.UNK_TOKEN: 4,
        }
        self.idx2char: Dict[int, str] = {v: k for k, v in self.char2idx.items()}
        self.folder2char: Dict[str, str] = {}

        if metadata_csv_path and os.path.exists(metadata_csv_path):
            self.load_from_metadata(metadata_csv_path)

    def load_from_metadata(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        # Expect columns: 'Folder Name', 'Char Name'
        for _, row in df.iterrows():
            folder_id = str(row['Folder Name']).strip()
            char = str(row['Char Name']).strip()
            self.folder2char[folder_id] = char
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def add_character(self, char: str) -> int:
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            return idx
        return self.char2idx[char]

    def encode(self, text: str) -> List[int]:
        """Converts Bangla text string into sequence of token IDs."""
        tokens = []
        for char in text:
            tokens.append(self.char2idx.get(char, self.char2idx[self.UNK_TOKEN]))
        return tokens

    def decode(self, tokens: List[int], ignore_special: bool = True) -> str:
        """Converts token IDs back into string, ignoring CTC blanks and special tokens if specified."""
        chars = []
        for idx in tokens:
            if ignore_special and idx in (0, 1, 2, 3, 4):
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def decode_ctc(self, tokens: List[int]) -> str:
        """Decodes CTC output sequence by collapsing consecutive duplicates and removing blanks."""
        collapsed = []
        prev = None
        for t in tokens:
            if t != prev:
                if t != 0:  # 0 is CTC blank
                    collapsed.append(t)
                prev = t
        return self.decode(collapsed)

    def __len__(self) -> int:
        return len(self.char2idx)


def apply_elastic_transform(image: np.ndarray, alpha: float = 34.0, sigma: float = 4.0) -> np.ndarray:
    """Applies elastic distortion to simulate organic handwriting variations."""
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    if image.ndim == 3:
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(y + dy[..., None], (-1, 1)), np.reshape(x + dx[..., None], (-1, 1)), np.reshape(z, (-1, 1))
    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted_image.reshape(shape)


def augment_handwriting(pil_img: Image.Image, is_training: bool = True) -> Image.Image:
    """Applies realistic optical and geometric augmentations for offline HTR."""
    if not is_training:
        return pil_img

    img = pil_img.copy()

    # 1. Random Rotation / Skewing (-12 to +12 deg)
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # 2. Random Contrast / Brightness
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    # 3. Gaussian Blur
    if random.random() < 0.3:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 4. Elastic Transform
    if random.random() < 0.3:
        arr = np.array(img)
        arr = apply_elastic_transform(arr, alpha=random.uniform(15, 30), sigma=random.uniform(3, 5))
        img = Image.fromarray(arr.astype(np.uint8))

    # 5. Gaussian Noise
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


class BanglaSyntheticTextGenerator:
    """Synthesizes text line/word images from Bangla strings using TTF fonts."""

    def __init__(self, font_paths: Optional[List[str]] = None, img_height: int = 64):
        self.img_height = img_height
        self.font_paths = font_paths or []

        # Find system fonts if none provided
        if not self.font_paths:
            system_fonts = [
                "/System/Library/Fonts/KohinoorBangla.ttc",
                "/System/Library/Fonts/Supplemental/Bangla MN.ttc",
                "/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc"
            ]
            self.font_paths = [f for f in system_fonts if os.path.exists(f)]

    def render_text(self, text: str, font_size: int = 36) -> Image.Image:
        """Renders a string into a grayscale handwriting-style image."""
        font = None
        if self.font_paths:
            font_path = random.choice(self.font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Measure text box size
        dummy_img = Image.new("L", (1, 1), 255)
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = max(bbox[2] - bbox[0] + 20, 32)
        text_h = max(bbox[3] - bbox[1] + 20, 32)

        # Create image and draw text
        img = Image.new("L", (text_w, text_h), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 5), text, font=font, fill=0)

        # Resize keeping height constant
        w, h = img.size
        aspect_ratio = w / float(h)
        target_w = max(int(self.img_height * aspect_ratio), 32)
        img = img.resize((target_w, self.img_height), Image.BICUBIC)

        return img


class LocalBanglaDataset(Dataset):
    """Dataset class for local 121-class Bangla handwriting images."""

    def __init__(self, root_dir: str, metadata_csv: str, tokenizer: BanglaTokenizer,
                 img_height: int = 64, img_width: int = 128, is_training: bool = True):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width
        self.is_training = is_training
        self.samples: List[Tuple[str, str]] = []

        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found at: {metadata_csv}")

        df = pd.read_csv(metadata_csv)

        # Iterate over folder paths
        for _, row in df.iterrows():
            folder_name = str(row['Folder Name']).strip()
            char_name = str(row['Char Name']).strip()
            folder_path = os.path.join(root_dir, folder_name)

            if os.path.isdir(folder_path):
                image_files = glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                              glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                              glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
                for img_p in image_files:
                    self.samples.append((img_p, char_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label_str = self.samples[idx]
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            img = Image.new("L", (self.img_width, self.img_height), 255)

        # Augmentation
        img = augment_handwriting(img, is_training=self.is_training)

        # Resize keeping aspect ratio & pad
        w, h = img.size
        ratio = self.img_height / float(h)
        new_w = min(int(w * ratio), self.img_width)
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        padded_img = Image.new("L", (self.img_width, self.img_height), 255)
        padded_img.paste(img, (0, 0))

        # Convert to Tensor (1, H, W) normalized to [0, 1]
        img_np = np.array(padded_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Tokens
        tokens = self.tokenizer.encode(label_str)
        return img_tensor, torch.tensor(tokens, dtype=torch.long), label_str


class SyntheticBanglaDataset(Dataset):
    """Dataset generating synthetic Bangla text samples on-the-fly."""

    def __init__(self, vocabulary: List[str], tokenizer: BanglaTokenizer,
                 num_samples: int = 2000, img_height: int = 64, max_width: int = 256,
                 is_training: bool = True):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.img_height = img_height
        self.max_width = max_width
        self.is_training = is_training
        self.generator = BanglaSyntheticTextGenerator(img_height=img_height)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        text = random.choice(self.vocabulary)
        img = self.generator.render_text(text)
        img = augment_handwriting(img, is_training=self.is_training)

        w, h = img.size
        if w > self.max_width:
            img = img.resize((self.max_width, self.img_height), Image.BICUBIC)
            target_w = self.max_width
        else:
            target_w = w

        padded_img = Image.new("L", (self.max_width, self.img_height), 255)
        padded_img.paste(img, (0, 0))

        img_np = np.array(padded_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        tokens = self.tokenizer.encode(text)
        return img_tensor, torch.tensor(tokens, dtype=torch.long), text


def HTRCollateFn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Collate function for padding images and label sequences in batches."""
    images, targets, texts = zip(*batch)
    batch_size = len(images)

    # Pad images to max width in batch
    max_w = max(img.shape[2] for img in images)
    c, h = images[0].shape[0], images[0].shape[1]

    padded_images = torch.ones((batch_size, c, h, max_w), dtype=torch.float32)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img
        input_lengths[i] = w // 4  # Assuming 4x spatial reduction in CNN backbone
        target_lengths[i] = len(targets[i])

    # Pad label targets with pad_token=1
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=1)

    return padded_images, padded_targets, input_lengths, target_lengths, list(texts)


if __name__ == "__main__":
    tokenizer = BanglaTokenizer("Bangla dataset/metaData_img.csv")
    sample_word = "বাংলাদেশ"
    tokens = tokenizer.encode(sample_word)
    decoded = tokenizer.decode(tokens)
    print(f"Word: {sample_word} | Tokens: {tokens} | Decoded: {decoded}")
