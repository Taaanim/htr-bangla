"""
predict.py - Offline Bangla HTR Inference & Text Prediction Pipeline

Loads trained HTR model checkpoint, preprocesses input handwriting images,
runs CTC/RL model sequence prediction with confidence scoring,
and applies Trie-based dictionary & Levenshtein post-processing refinement.
"""

import os
import argparse
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dataset import BanglaTokenizer, augment_handwriting
from model import HTRHybridModel
from dictionary import BanglaPostProcessor


class BanglaHTRPredictor:
    """End-to-End Inference Engine for Offline Bangla Handwritten Text Recognition."""

    def __init__(self, model_path: str, metadata_csv: str = "Bangla dataset/metaData_img.csv",
                 device: Optional[torch.device] = None, img_height: int = 64, img_width: int = 256):
        self.img_height = img_height
        self.img_width = img_width

        # Detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Initialize Tokenizer & Post-Processor
        self.tokenizer = BanglaTokenizer(metadata_csv)
        self.post_processor = BanglaPostProcessor()
        num_classes = len(self.tokenizer)

        # Build Model & Load Checkpoint
        self.model = HTRHybridModel(num_classes=num_classes, hidden_dim=256).to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded HTR model checkpoint from: {model_path}")
        else:
            print(f"Warning: Checkpoint path '{model_path}' not found. Using randomly initialized weights.")

        self.model.eval()

    def preprocess_image(self, image_input) -> torch.Tensor:
        """Preprocesses PIL Image or file path into normalized PyTorch Tensor (1, 1, H, W)."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("L")
        else:
            raise ValueError("Unsupported image input type.")

        # Resize keeping height constant & pad width
        w, h = img.size
        ratio = self.img_height / float(h)
        new_w = min(int(w * ratio), self.img_width)
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        padded_img = Image.new("L", (self.img_width, self.img_height), 255)
        padded_img.paste(img, (0, 0))

        img_np = np.array(padded_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_input) -> Dict[str, Any]:
        """
        Runs model inference on an input handwriting image.
        Returns dictionary containing raw text, corrected text, confidence scores, and dictionary flags.
        """
        tensor_img = self.preprocess_image(image_input)

        # Forward pass
        logits = self.model(tensor_img)  # (1, T, num_classes)
        probs = F.softmax(logits, dim=-1)  # (1, T, num_classes)

        max_probs, preds = torch.max(probs, dim=-1)  # (1, T)

        token_indices = preds[0].cpu().tolist()
        conf_scores = max_probs[0].cpu().tolist()

        # CTC Greedy Collapse & Confidence Calculation
        collapsed_tokens = []
        collapsed_confs = []
        prev = None
        for t, conf in zip(token_indices, conf_scores):
            if t != prev:
                if t != 0:  # 0 is CTC blank
                    collapsed_tokens.append(t)
                    collapsed_confs.append(conf)
                prev = t

        raw_text = self.tokenizer.decode(collapsed_tokens)

        # Post-Processing Correction
        corrected_text = self.post_processor.correct_sentence(raw_text)
        is_valid_dict = self.post_processor.is_valid_word(corrected_text)

        avg_confidence = float(np.mean(collapsed_confs)) if collapsed_confs else 0.0

        return {
            "raw_prediction": raw_text,
            "corrected_prediction": corrected_text,
            "confidence_score": round(avg_confidence * 100, 2),
            "is_in_dictionary": is_valid_dict,
            "raw_tokens": collapsed_tokens
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Bangla Text from Handwriting Image")
    parser.add_argument("--image", type=str, help="Path to input handwriting image")
    parser.add_argument("--model-path", type=str, default="checkpoints/final_htr_model.pth", help="Model checkpoint path")
    parser.add_argument("--metadata-csv", type=str, default="Bangla dataset/metaData_img.csv", help="Metadata CSV path")
    args = parser.parse_args()

    predictor = BanglaHTRPredictor(model_path=args.model_path, metadata_csv=args.metadata_csv)

    if args.image and os.path.exists(args.image):
        result = predictor.predict(args.image)
        print("\n" + "=" * 50)
        print("HTR PREDICTION RESULTS")
        print("=" * 50)
        print(f"Image Path          : {args.image}")
        print(f"Raw Model Prediction : '{result['raw_prediction']}'")
        print(f"Corrected Bangla Text: '{result['corrected_prediction']}'")
        print(f"Confidence Score     : {result['confidence_score']}%")
        print(f"Dictionary Validated : {result['is_in_dictionary']}")
        print("=" * 50)
    else:
        # Run demonstration on a sample from local dataset or synthetic text
        print("\nNo input image provided. Running demonstration on a synthetic sample...")
        from dataset import BanglaSyntheticTextGenerator
        gen = BanglaSyntheticTextGenerator()
        sample_img = gen.render_text("বাংলাদেশ")

        result = predictor.predict(sample_img)
        print("\n" + "=" * 50)
        print("HTR DEMO PREDICTION RESULTS")
        print("=" * 50)
        print(f"Ground Truth Text   : 'বাংলাদেশ'")
        print(f"Raw Model Prediction : '{result['raw_prediction']}'")
        print(f"Corrected Bangla Text: '{result['corrected_prediction']}'")
        print(f"Confidence Score     : {result['confidence_score']}%")
        print(f"Dictionary Validated : {result['is_in_dictionary']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
