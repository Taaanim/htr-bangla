"""
predict.py — Inference for Bangla Handwritten Character Recognition

Supports:
1. Single character image → class prediction
2. Document/page image → segment characters → classify each → combine text
3. JSON output for web app integration
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import base64

from model import BestCNN
from dataset import process_character, IMG_SIZE


class BanglaPredictor:
    """Inference engine for Bangla character recognition."""

    def __init__(self, weights_path: str = "checkpoints/best_cnn_model_weights.pth",
                 le_path: str = "checkpoints/label_encoder.pkl"):
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load label encoder
        if os.path.exists(le_path):
            self.le = joblib.load(le_path)
        elif os.path.exists("label_encoder.pkl"):
            self.le = joblib.load("label_encoder.pkl")
        else:
            raise FileNotFoundError(f"Label encoder not found at {le_path}")

        self.num_classes = len(self.le.classes_)

        # Load model
        self.model = BestCNN(self.num_classes).to(self.device)

        if os.path.exists(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device, weights_only=True)
            )
        elif os.path.exists("best_cnn_model_weights.pth"):
            self.model.load_state_dict(
                torch.load("best_cnn_model_weights.pth", map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        self.model.eval()

    def predict_image(self, img: np.ndarray) -> dict:
        """Predict a single preprocessed character image.

        Args:
            img: (H, W, 3) RGB uint8 image, already preprocessed

        Returns:
            dict with prediction, confidence, top5 predictions
        """
        # Normalize and convert to tensor
        img_float = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))  # (3, H, W)
        tensor = torch.tensor(img_chw).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, 1)

        pred_label = self.le.inverse_transform([pred_idx.item()])[0]
        conf = confidence.item() * 100

        # Top 5
        top5_probs, top5_idx = torch.topk(probs, min(5, self.num_classes), dim=-1)
        top5 = []
        for i in range(top5_idx.shape[1]):
            label = self.le.inverse_transform([top5_idx[0, i].item()])[0]
            prob = top5_probs[0, i].item() * 100
            top5.append({"label": label, "confidence": round(prob, 2)})

        return {
            "prediction": pred_label,
            "confidence": round(conf, 2),
            "top5": top5
        }

    def predict_file(self, image_path: str) -> dict:
        """Predict from an image file path."""
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not read image: {image_path}"}

        processed = process_character(img, size=IMG_SIZE)
        if processed is None:
            return {"error": "No character found in image"}

        result = self.predict_image(processed)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        result["img_b64"] = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        return result

    def predict_document(self, image_path: str) -> dict:
        """Segment a paragraph/document image into lines, words, and characters.

        1. Binarize & detect contours for all characters.
        2. Group bounding boxes into horizontal lines (by y position).
        3. Sort each line left-to-right (by x position).
        4. Detect space gaps between words.
        5. Classify each character with BestCNN.
        6. Reconstruct paragraph with spaces and newlines.
        """
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Could not read image"}

        # Handle RGBA
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            bg = np.ones_like(rgb) * 255
            alpha_f = alpha[:, :, np.newaxis] / 255.0
            img = (rgb * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Determine thresholding based on background polarity
        if gray.mean() > 127:  # White background, dark text
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:  # Dark background, light text
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # ── 1. Matra (মাত্রা) Line Removal for Connected Bangla Words ──────────
        # Detect top horizontal Matra line that connects Bangla letters in a word
        horiz_len = max(8, thresh.shape[1] // 30)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        matra_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

        # Subtract Matra line so joined characters separate into individual letters
        thresh_no_matra = cv2.subtract(thresh, matra_mask)

        # Dilate slightly to connect broken stroke parts of individual letters
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh_no_matra, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fall back to standard threshold if Matra removal produced no contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"prediction": "", "characters": [], "lines": []}

        # Filter out tiny noise contours
        raw_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 4 and h >= 4 and w * h >= 20:
                raw_boxes.append((x, y, w, h))

        if not raw_boxes:
            return {"prediction": "", "characters": [], "lines": []}

        # ── 2. Line Segmentation (Top-to-Bottom) ──────────────────────────────
        raw_boxes = sorted(raw_boxes, key=lambda b: b[1])
        lines = []
        for box in raw_boxes:
            x, y, w, h = box
            y_center = y + h / 2.0
            matched_line = False
            for line in lines:
                line_y_min = min(b[1] for b in line)
                line_y_max = max(b[1] + b[3] for b in line)
                line_h = max(20, line_y_max - line_y_min)
                if abs(y_center - (line_y_min + line_y_max) / 2.0) < line_h * 0.65:
                    line.append(box)
                    matched_line = True
                    break
            if not matched_line:
                lines.append([box])

        lines = sorted(lines, key=lambda l: np.mean([b[1] for b in l]))

        full_paragraph_text = []
        all_characters = []

        # ── 3. Character Classification & Word Reconstruction ─────────────────
        for line_idx, line_boxes in enumerate(lines):
            # Merge vertically aligned modifiers (e.g. vowel signs ি, ু, ৌ above/below base chars)
            line_boxes = self._merge_vertical_modifiers(line_boxes)
            # Merge only strictly overlapping boxes
            line_boxes = self._merge_boxes(line_boxes, threshold=0)
            # Split wide boxes that contain 2+ merged characters (e.g. w/h > 1.5)
            line_boxes = self._split_wide_boxes(line_boxes, max_aspect_ratio=1.5)
            line_boxes = sorted(line_boxes, key=lambda b: b[0])  # Sort left-to-right

            if not line_boxes:
                continue

            widths = [b[2] for b in line_boxes]
            median_w = np.median(widths) if widths else 20
            space_threshold = max(14, median_w * 0.75)

            line_text = []
            prev_right = None

            for x, y, w, h in line_boxes:
                # Detect word spaces between character bounding boxes
                if prev_right is not None and (x - prev_right) > space_threshold:
                    line_text.append(" ")
                prev_right = x + w

                pad = 4
                char_crop = img[max(0, y-pad):min(img.shape[0], y+h+pad),
                                max(0, x-pad):min(img.shape[1], x+w+pad)]

                processed = process_character(char_crop, size=IMG_SIZE)
                if processed is None:
                    continue

                result = self.predict_image(processed)
                char_pred = result["prediction"]

                # Encode processed 32x32 image to base64 PNG for UI display
                _, buffer = cv2.imencode('.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                img_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

                line_text.append(char_pred)
                all_characters.append({
                    "char": char_pred,
                    "confidence": result["confidence"],
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "line": int(line_idx),
                    "img_b64": img_b64,
                    "top5": result.get("top5", [])
                })

            full_paragraph_text.append("".join(line_text))

        final_result_str = "\n".join(full_paragraph_text)

        return {
            "prediction": final_result_str,
            "characters": all_characters,
            "num_chars": len(all_characters),
            "num_lines": len(lines)
        }

    @staticmethod
    def _merge_vertical_modifiers(boxes, x_overlap_threshold=0.5):
        """Merge vowel signs (কার/ফলা) positioned above or below a base character."""
        if not boxes:
            return []

        merged = []
        used = set()
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)  # Process larger base chars first

        for i in range(len(boxes)):
            if i in used:
                continue
            x1, y1, w1, h1 = boxes[i]
            curr_box = [x1, y1, w1, h1]

            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                x2, y2, w2, h2 = boxes[j]
                # Check horizontal overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                min_w = min(w1, w2)
                if min_w > 0 and (overlap_x / min_w) >= x_overlap_threshold:
                    # Merge into bounding box
                    nx1 = min(curr_box[0], x2)
                    ny1 = min(curr_box[1], y2)
                    nx2 = max(curr_box[0] + curr_box[2], x2 + w2)
                    ny2 = max(curr_box[1] + curr_box[3], y2 + h2)
                    curr_box = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                    used.add(j)

            merged.append(tuple(curr_box))

        return merged

    @staticmethod
    def _split_wide_boxes(boxes, max_aspect_ratio=1.5):
        """Split wide bounding boxes that merged 2 or more characters horizontally."""
        if not boxes:
            return []
        split_boxes = []
        for x, y, w, h in boxes:
            if h <= 0:
                continue
            aspect = w / float(h)
            if aspect > max_aspect_ratio:
                num_sub = max(2, int(np.round(aspect)))
                sub_w = w // num_sub
                for i in range(num_sub):
                    split_boxes.append((x + i * sub_w, y, sub_w, h))
            else:
                split_boxes.append((x, y, w, h))
        return split_boxes

    @staticmethod
    def _merge_boxes(boxes, threshold=5):
        """Merge overlapping/close bounding boxes."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []
        curr = list(boxes[0])
        for b in boxes[1:]:
            if b[0] <= curr[0] + curr[2] + threshold:
                x1 = min(curr[0], b[0])
                y1 = min(curr[1], b[1])
                x2 = max(curr[0] + curr[2], b[0] + b[2])
                y2 = max(curr[1] + curr[3], b[1] + b[3])
                curr = [x1, y1, x2 - x1, y2 - y1]
            else:
                merged.append(tuple(curr))
                curr = list(b)
        merged.append(tuple(curr))
        return merged


def main():
    parser = argparse.ArgumentParser(description="Bangla HTR Prediction")
    parser.add_argument("--image", type=str, help="Path to character/document image")
    parser.add_argument("--mode", type=str, choices=["char", "document"], default="char",
                        help="char: single character, document: full page OCR")
    parser.add_argument("--weights", type=str, default="checkpoints/best_cnn_model_weights.pth")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    try:
        predictor = BanglaPredictor(weights_path=args.weights)
    except FileNotFoundError as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
        sys.exit(1)

    if args.image and os.path.exists(args.image):
        if args.mode == "document":
            result = predictor.predict_document(args.image)
        else:
            result = predictor.predict_file(args.image)

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"\n{'='*50}")
            print(f"Bangla HTR Prediction")
            print(f"{'='*50}")
            print(f"Image: {args.image}")
            print(f"Mode:  {args.mode}")
            if "prediction" in result:
                print(f"Result: {result['prediction']}")
            if "confidence" in result:
                print(f"Confidence: {result['confidence']}%")
            if "top5" in result:
                print(f"Top 5:")
                for item in result["top5"]:
                    print(f"  {item['label']}: {item['confidence']}%")
            if "error" in result:
                print(f"Error: {result['error']}")
            print(f"{'='*50}")
    else:
        msg = "Usage: python3 predict.py --image <path> [--mode char|document] [--json]"
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(msg)


if __name__ == "__main__":
    main()
