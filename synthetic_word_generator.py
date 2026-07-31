"""
synthetic_word_generator.py — Synthetic Bangla Word & Anthem Generator for RL Sequence HTR

Includes full vocabulary from 'Amar Shonar Bangla' (National Anthem of Bangladesh):
Stitches isolated character crops from dataset_filtered to form continuous synthetic Bangla words/phrases
with realistic Kerning, Matra top bars, and overlapping modifiers.
"""

import os
import random
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple, Dict, Any

AMAR_SHONAR_BANGLA_TEXT = """
আমার সোনার বাংলা আমি তোমায় ভালোবাসি
চিরদিন তোমার আকাশ তোমার বাতাস আমার প্রাণে বাজায় বাঁশি
ও মা ফাগুনে তোর আমের বনে ঘ্রাণে পাগল করে
মরি হায় হায় রে
ও মা অঘ্রানে তোর ভরা ক্ষেতে আমি কী দেখেছি মধুর হাসি
কী শোভা কী ছায়া গো কী স্নেহ কী মায়া গো
কী আঁচল বিছায়েছ বটের মূলে নদীর কূলে কূলে
মা তোর মুখের বাণী আমার কানে লাগে সুধার মতো
মরি হায় হায় রে
মা তোর বদনখানি মলিন হলে ও মা আমি নয়নজলে ভাসি
তোমার এই খেলাঘরে শিশুকাল কাটিলে রে
তোমারি ধুলামাটি অঙ্গে মাখি ধন্য জীবন মানি
তুই দিন ফুরালে সন্ধ্যাকালে কী দীপ জ্বালিস ঘরে
মরি হায় হায় রে
তখন খেলাধুলা সকল ফেলে ও মা তোমার কোলে ছুটে আসি
ধেনু চরা তোমার মাঠে পারে যাবার খেয়াঘাটে
সারা দিন পাখি ডাকা ছায়ায় ঢাকা তোমার পল্লীবাটে
তোমার ধানে ভরা আঙিনাতে জীবনের দিন কাটে
মরি হায় হায় রে
ও মা আমার যে ভাই তারা সবাই ও মা তোমার রাখাল তোমার চাষি
ও মা তোর চরণেতে দিলেম এই মাথা পেতে
দে গো তোর পায়ের ধুলা সে যে আমার মাথার মানিক হবে
ও মা গরিবের ধন যা আছে তাই দিব চরণতলে
মরি হায় হায় রে
আমি পরের ঘরে কিনব না আর মা তোর ভূষণ ব'লে গলার ফাঁসি
"""

# Extract unique words from Amar Shonar Bangla text
def extract_words(text: str) -> List[str]:
    cleaned = re.sub(r'[^\u0980-\u09FF\s]', ' ', text)
    tokens = cleaned.strip().split()
    unique_tokens = list(dict.fromkeys(tokens))
    return [t for t in unique_tokens if len(t) >= 2]

SHONAR_BANGLA_WORDS = extract_words(AMAR_SHONAR_BANGLA_TEXT)


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

        self.words_list = SHONAR_BANGLA_WORDS

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
                target_h = 64
                new_w = max(20, int(w * (target_h / float(h))))
                crop_resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                char_crops.append(crop_resized)
                valid_chars.append(ch)

        if not char_crops:
            canvas = np.zeros((64, 128, 3), dtype=np.uint8)
            return canvas, "ক"

        total_w = sum(c.shape[1] for c in char_crops) + random.randint(8, 20)
        canvas = np.zeros((64, total_w, 3), dtype=np.uint8)

        curr_x = random.randint(0, 8)
        for crop in char_crops:
            h, w = crop.shape[:2]
            if curr_x + w > total_w:
                w = total_w - curr_x
                crop = crop[:, :w]
            canvas[:, curr_x:curr_x+w] = crop
            curr_x += (w - random.randint(0, 5))

        # Character crops already contain their natural Matras, no extra line needed
        target_word = "".join(valid_chars)
        return canvas, target_word


if __name__ == "__main__":
    gen = SyntheticWordGenerator()
    print(f"Loaded {len(gen.words_list)} unique words from Amar Shonar Bangla!")
    sample_word = "ভালোবাসি"
    img, text = gen.generate_word_image(sample_word)
    print(f"Generated synthetic word: '{text}' | Image shape: {img.shape}")
