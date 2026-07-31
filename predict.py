"""
predict.py — Inference for Bangla Handwritten Character & Document Recognition

Supports:
1. Single character image → class prediction
2. Document/page image → layout analysis → predict text → structured hierarchy
3. JSON output for web app integration
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import joblib
import base64
from typing import List, Dict, Any, Tuple

from model import BestCNN
from dataset import process_character, IMG_SIZE
from bengali_segmenter import BengaliSegmenter


class BanglaPredictor:
    """Inference engine for Bangla character recognition."""

    def __init__(self, weights_path: str = "checkpoints/best_cnn_model_weights.pth",
                 le_path: str = "checkpoints/label_encoder.pkl"):
        # Layout Segmenter
        self.segmenter = BengaliSegmenter()

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
        img_float = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        tensor = torch.tensor(img_chw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf_val, pred_class = torch.max(probs, dim=1)

            top5_probs, top5_idx = torch.topk(probs, min(5, self.num_classes), dim=1)

        pred_label = self.le.inverse_transform([pred_class.item()])[0]
        conf = conf_val.item() * 100.0

        top5 = []
        for i in range(top5_probs.size(1)):
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

    def predict_word_sliding_window(self, word_crop: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Recognizes a connected word image using sliding-window feature extraction & CTC-style sequence collapsing."""
        if word_crop is None or word_crop.size == 0:
            return "", 0.0, []

        h, w = word_crop.shape[:2]
        if h <= 0 or w <= 0:
            return "", 0.0, []

        win_w = max(16, int(h * 0.70))
        step = max(6, int(win_w * 0.35))

        windows = []
        for x in range(0, max(1, w - win_w + 1), step):
            windows.append((x, 0, min(win_w, w - x), h))

        if not windows:
            windows = [(0, 0, w, h)]

        raw_preds = []
        for wx, wy, ww, wh in windows:
            patch = word_crop[wy:wy+wh, wx:wx+ww]
            proc = process_character(patch, size=IMG_SIZE)
            if proc is None:
                continue
            res = self.predict_image(proc)
            raw_preds.append((res["prediction"], res["confidence"], wx, wy, ww, wh, proc, res))

        if not raw_preds:
            return "", 0.0, []

        collapsed_chars = []
        collapsed_confs = []
        symbol_objs = []

        last_char = None
        for char_pred, char_conf, wx, wy, ww, wh, proc, res in raw_preds:
            if char_pred != last_char and char_conf >= 20.0:
                collapsed_chars.append(char_pred)
                collapsed_confs.append(char_conf)
                last_char = char_pred

                _, buffer = cv2.imencode('.png', cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                img_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

                sym_bbox = [int(wx), int(wy), int(ww), int(wh)]
                sym_corners = self._get_corners(sym_bbox)

                symbol_objs.append({
                    "symbol": char_pred,
                    "confidence": char_conf,
                    "bbox": sym_bbox,
                    "corner_points": sym_corners,
                    "img_b64": img_b64,
                    "top5": res.get("top5", [])
                })

        word_text = "".join(collapsed_chars)
        avg_conf = round(float(np.mean(collapsed_confs)), 2) if collapsed_confs else 0.0
        return word_text, avg_conf, symbol_objs

    def predict_document(self, image_path: str) -> dict:
        """Hierarchical Document Layout Analysis & OCR Engine."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Could not read image"}

        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3]
            bg = np.ones_like(rgb) * 255
            alpha_f = alpha[:, :, np.newaxis] / 255.0
            img = (rgb * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)

        h_img, w_img = img.shape[:2]

        seg_res = self.segmenter.segment_image(img)
        rotation_angle = seg_res.get("rotation", 0.0)

        chars_meta = seg_res.get("characters", [])
        if not chars_meta:
            return {"prediction": "", "blocks": [], "lines": [], "words": [], "characters": []}

        line_indices = sorted(list(set(c["line"] for c in chars_meta)))

        structured_blocks = []
        structured_lines = []
        structured_words = []
        all_symbols = []
        full_paragraph_text = []

        for line_idx in line_indices:
            line_items = [c for c in chars_meta if c["line"] == line_idx]
            line_items = sorted(line_items, key=lambda c: c["primary_bbox"][0])

            if not line_items:
                continue

            word_groups = []
            curr_word = []
            widths = [c["primary_bbox"][2] for c in line_items]
            median_w = np.median(widths) if widths else 20
            space_thresh = max(14, median_w * 0.85)

            for c in line_items:
                bx, by, bw, bh = c["primary_bbox"]
                if curr_word and (bx - (curr_word[-1]["primary_bbox"][0] + curr_word[-1]["primary_bbox"][2])) > space_thresh:
                    word_groups.append(curr_word)
                    curr_word = []
                curr_word.append(c)
            if curr_word:
                word_groups.append(curr_word)

            line_word_strs = []
            line_confs = []

            for w_idx, w_items in enumerate(word_groups):
                wx1 = min(c["primary_bbox"][0] for c in w_items)
                wy1 = min(c["primary_bbox"][1] for c in w_items)
                wx2 = max(c["primary_bbox"][0] + c["primary_bbox"][2] for c in w_items)
                wy2 = max(c["primary_bbox"][1] + c["primary_bbox"][3] for c in w_items)
                word_bbox = [int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1)]
                word_corners = self._get_corners(word_bbox)

                word_chars = []
                word_confs = []

                for item in w_items:
                    px, py, pw, ph = item["primary_bbox"]
                    pad = 3
                    crop_y1 = max(0, py - pad)
                    crop_y2 = min(h_img, py + ph + pad)
                    crop_x1 = max(0, px - pad)
                    crop_x2 = min(w_img, px + pw + pad)
                    primary_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

                    p_processed = process_character(primary_crop, size=IMG_SIZE)
                    if p_processed is None:
                        continue

                    p_result = self.predict_image(p_processed)
                    primary_pred = p_result["prediction"]
                    primary_conf = p_result["confidence"]

                    aspect = pw / float(ph) if ph > 0 else 1.0
                    if primary_conf >= 50.0 or aspect < 1.35 or len(item["secondary_bboxes"]) <= 1:
                        final_boxes_to_use = [(px, py, pw, ph, primary_pred, primary_conf, p_result)]
                    else:
                        sec_evals = []
                        for sx, sy, sw, sh in item["secondary_bboxes"]:
                            scrop_y1 = max(0, sy - pad)
                            scrop_y2 = min(h_img, sy + sh + pad)
                            scrop_x1 = max(0, sx - pad)
                            scrop_x2 = min(w_img, sx + sw + pad)
                            sec_crop = img[scrop_y1:scrop_y2, scrop_x1:scrop_x2]

                            s_proc = process_character(sec_crop, size=IMG_SIZE)
                            if s_proc is None:
                                continue
                            s_res = self.predict_image(s_proc)
                            sec_evals.append((sx, sy, sw, sh, s_res["prediction"], s_res["confidence"], s_res))

                        if sec_evals:
                            final_boxes_to_use = sec_evals
                        else:
                            final_boxes_to_use = [(px, py, pw, ph, primary_pred, primary_conf, p_result)]

                    for cx, cy, cw, ch, char_pred, char_conf, result in final_boxes_to_use:
                        crop_y1 = max(0, cy - pad)
                        crop_y2 = min(h_img, cy + ch + pad)
                        crop_x1 = max(0, cx - pad)
                        crop_x2 = min(w_img, cx + cw + pad)
                        c_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                        proc_c = process_character(c_crop, size=IMG_SIZE)

                        if proc_c is None:
                            continue

                        _, buffer = cv2.imencode('.png', cv2.cvtColor(proc_c, cv2.COLOR_RGB2BGR))
                        img_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

                        sym_bbox = [int(cx), int(cy), int(cw), int(ch)]
                        sym_corners = self._get_corners(sym_bbox)

                        symbol_obj = {
                            "symbol": char_pred,
                            "confidence": char_conf,
                            "bbox": sym_bbox,
                            "corner_points": sym_corners,
                            "line": int(line_idx),
                            "img_b64": img_b64,
                            "top5": result.get("top5", [])
                        }
                        all_symbols.append(symbol_obj)
                        word_chars.append(char_pred)
                        word_confs.append(char_conf)

                word_text_str = "".join(word_chars)
                word_avg_conf = round(float(np.mean(word_confs)), 2) if word_confs else 0.0

                structured_words.append({
                    "text": word_text_str,
                    "confidence": word_avg_conf,
                    "bbox": word_bbox,
                    "corner_points": word_corners,
                    "line": int(line_idx)
                })

                line_word_strs.append(word_text_str)
                line_confs.extend(word_confs)

            lx1 = min(c["primary_bbox"][0] for c in line_items)
            ly1 = min(c["primary_bbox"][1] for c in line_items)
            lx2 = max(c["primary_bbox"][0] + c["primary_bbox"][2] for c in line_items)
            ly2 = max(c["primary_bbox"][1] + c["primary_bbox"][3] for c in line_items)
            line_bbox = [int(lx1), int(ly1), int(lx2 - lx1), int(ly2 - ly1)]
            line_corners = self._get_corners(line_bbox)

            line_text_str = " ".join(line_word_strs)
            line_avg_conf = round(float(np.mean(line_confs)), 2) if line_confs else 0.0

            structured_lines.append({
                "text": line_text_str,
                "confidence": line_avg_conf,
                "bbox": line_bbox,
                "corner_points": line_corners,
                "line": int(line_idx)
            })

            full_paragraph_text.append(line_text_str)

        final_paragraph_str = "\n".join(full_paragraph_text)
        block_bbox = [0, 0, img.shape[1], img.shape[0]]
        if structured_lines:
            bx1 = min(l["bbox"][0] for l in structured_lines)
            by1 = min(l["bbox"][1] for l in structured_lines)
            bx2 = max(l["bbox"][0] + l["bbox"][2] for l in structured_lines)
            by2 = max(l["bbox"][1] + l["bbox"][3] for l in structured_lines)
            block_bbox = [int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)]

        block_corners = self._get_corners(block_bbox)
        block_avg_conf = round(float(np.mean([l["confidence"] for l in structured_lines])), 2) if structured_lines else 0.0

        structured_blocks.append({
            "block_type": "Text",
            "text": final_paragraph_str,
            "confidence": block_avg_conf,
            "bbox": block_bbox,
            "corner_points": block_corners,
            "rotation": rotation_angle
        })

        return {
            "prediction": final_paragraph_str,
            "rotation": rotation_angle,
            "blocks": structured_blocks,
            "lines": structured_lines,
            "words": structured_words,
            "characters": all_symbols,
            "num_chars": len(all_symbols),
            "num_lines": len(structured_lines),
            "num_words": len(structured_words)
        }

    @staticmethod
    def _get_corners(bbox: List[int]) -> List[List[int]]:
        """Returns 4 corner points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] for a bbox."""
        x, y, w, h = bbox
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


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
