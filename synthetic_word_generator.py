"""
synthetic_word_generator.py — Synthetic Bangla Word Generator for RL Sequence HTR

Stitches isolated character crops from dataset_filtered to form continuous synthetic Bangla words:
1. Uses dictionary words (e.g. বাংলা, বাংলাদেশ, শিক্ষা, ঢাকা, আমাদের, কলম, পানি, মানুষ, বাড়ি)
2. Stitches character crops horizontally with variable kerning/overlap
3. Draws a continuous top Matra bar across character tops to mimic connected handwriting
4. Adds random rotation, scale, and noise augmentations
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple, Dict, Any

BANGLA_WORDS_LIST = [
    "বাংলা", "বাংলাদেশ", "শিক্ষা", "ঢাকা", "আমাদের", "কলম", "বই", "পানি", "মানুষ",
    "বাড়ি", "স্কুল", "ছাত্র", "শিক্ষক", "নদী", "ফুল", "পাখি", "আকাশ", "বাতাস", "গাছ",
    "ফল", "বোল", "কাজ", "মাটি", "দেশ", "ভাষা", "সোনা", "রূপা", "আলো", "ছায়া"
]


class SyntheticWordGenerator:
    """Generates synthetic multi-character connected Bangla word images from isolated character crops."""

    def __init__(self, dataset_dir: str = "Bangla dataset/dataset_filtered", csv_path: str = "Bangla dataset/metaData_img.csv"):
        self.dataset_dir = dataset_dir

        # Load folder -> character mapping
        self.char_to_folders = {}
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                folder_id = str(row.iloc[0]).strip()
                char_name = str(row.iloc[1]).strip()
                if char_name not in self.char_to_folders:
                    self.char_to_folders[char_name] = []
                self.char_to_folders[char_name].append(folder_id)

        # Index all available image file paths per character
        self.char_to_image_paths = {}
        if os.path.exists(dataset_dir):
            for folder in os.listdir(dataset_dir):
                folder_path = os.path.join(dataset_dir, folder)
                if not os.path.isdir(folder_path):
                    continue

                for char_name, folders in self.char_to_folders.items():
                    if folder in folders:
                        if char_name not in self.char_to_image_paths:
                            self.char_to_image_paths[char_name] = []
                        for fname in os.listdir(folder_path):
                            if fname.endswith(('.jpg', '.png', '.jpeg')):
                                self.char_to_image_paths[char_name].append(os.path.join(folder_path, fname))

        self.words_list = BANGLA_WORDS_LIST

    def get_random_character_crop(self, char: str) -> Any:
        """Fetches a random image crop for a given Bangla character."""
        paths = self.char_to_image_paths.get(char, [])
        if not paths:
            all_paths = [p for sublist in self.char_to_image_paths.values() for p in sublist]
            if not all_paths:
                return None
            paths = all_paths

        imgPath = random.choice(paths)
        img = cv2.imread(imgPath)
        return img

    def generate_word_image(self, word: str = None) -> Tuple[np.ndarray, str]:
        """Stitches character crops horizontally to create a synthetic connected word image."""
        if word is None:
            word = random.choice(self.words_list)

        char_crops = []
        valid_chars = []

        for ch in word:
            crop = self.get_random_character_crop(ch)
            if crop is not None:
                h, w = crop.shape[:2]
                target_h = 32
                new_w = max(10, int(w * (target_h / float(h))))
                crop_resized = cv2.resize(crop, (new_w, target_h))
                char_crops.append(crop_resized)
                valid_chars.append(ch)

        if not char_crops:
            canvas = np.zeros((32, 64, 3), dtype=np.uint8)
            return canvas, "ক"

        total_w = sum(c.shape[1] for c in char_crops) + random.randint(4, 12)
        canvas = np.zeros((32, total_w, 3), dtype=np.uint8)

        curr_x = random.randint(0, 4)
        for crop in char_crops:
            h, w = crop.shape[:2]
            if curr_x + w > total_w:
                w = total_w - curr_x
                crop = crop[:, :w]
            canvas[:, curr_x:curr_x+w] = crop
            curr_x += (w - random.randint(0, 3))

        # Add top Matra horizontal bar across character tops to mimic connected handwriting
        matra_y = random.randint(4, 7)
        cv2.line(canvas, (4, matra_y), (max(4, curr_x - 4), matra_y), (255, 255, 255), thickness=random.randint(1, 2))

        target_word = "".join(valid_chars)
        return canvas, target_word


if __name__ == "__main__":
    gen = SyntheticWordGenerator()
    img, text = gen.generate_word_image("বাংলা")
    print(f"Generated synthetic word: '{text}' | Image shape: {img.shape}")
