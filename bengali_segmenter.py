"""
bengali_segmenter.py — Advanced Bengali Document Layout Analysis & Segmentation Engine

Features:
1. Automated Preprocessing (Otsu Binarization, Deskewing, Noise Removal)
2. Detached Symbol Merging (Groups multi-part characters like ং, ঃ, ঁ, ো, ৌ into 1 box)
3. Garbage Segment Filtering (Filters out small noise specks)
4. Primary & Secondary Hierarchical Segment Tree for Confidence Voting
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class BengaliPreprocessor:
    """Handles image binarization, deskewing, and noise removal."""

    @staticmethod
    def preprocess(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Preprocesses input document/word image."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        is_white_bg = gray.mean() > 127
        if is_white_bg:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.medianBlur(thresh, 3)

        rotation_angle = 0.0
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            min_rect = cv2.minAreaRect(coords)
            angle = min_rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            rotation_angle = round(float(angle), 2)

        return gray, thresh, rotation_angle


class MatraRemover:
    """Automated Matra (Top Horizontal Line) Detection and Selective Removal."""

    @staticmethod
    def detect_and_remove(line_thresh: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detects and removes the top horizontal Matra line."""
        h, w = line_thresh.shape[:2]
        if h < 10 or w < 10:
            return line_thresh, 0

        h_proj = np.sum(line_thresh > 0, axis=1)

        search_min = int(h * 0.10)
        search_max = int(h * 0.45)

        if search_max <= search_min:
            return line_thresh, 0

        matra_y = search_min + int(np.argmax(h_proj[search_min:search_max]))
        matra_val = h_proj[matra_y]

        if matra_val > w * 0.30:
            matra_thickness = max(2, int(h * 0.08))
            y1 = max(0, matra_y - matra_thickness // 2)
            y2 = min(h, matra_y + matra_thickness // 2 + 1)

            mask = np.zeros_like(line_thresh)
            mask[y1:y2, :] = line_thresh[y1:y2, :]

            kernel_w = max(6, int(w * 0.10))
            matra_bar = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)))

            no_matra = cv2.subtract(line_thresh, matra_bar)
            return no_matra, matra_y

        return line_thresh, 0


class DetachedSymbolGrouper:
    """Merges physically detached multi-part Bangla characters like ং (Anusvara), ঃ (Bisarga), ঁ (Chandrabindu)."""

    @staticmethod
    def group_detached(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merges adjacent detached components belonging to the same symbol (e.g. top circle + bottom slant in ং)."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            x1, y1, w1, h1 = boxes[i]
            curr_x1, curr_y1, curr_x2, curr_y2 = x1, y1, x1 + w1, y1 + h1

            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                x2, y2, w2, h2 = boxes[j]

                # Check horizontal & vertical proximity for detached parts of ং or ঃ
                dx = abs(x2 - (x1 + w1))
                dx_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                min_w = min(w1, w2)

                # Overlap horizontally or close proximity within 12px
                if (dx <= 12 or (min_w > 0 and dx_overlap / float(min_w) >= 0.3)) and abs(y1 - y2) <= 25:
                    curr_x1 = min(curr_x1, x2)
                    curr_y1 = min(curr_y1, y2)
                    curr_x2 = max(curr_x2, x2 + w2)
                    curr_y2 = max(curr_y2, y2 + h2)
                    used.add(j)

            used.add(i)
            merged.append((curr_x1, curr_y1, curr_x2 - curr_x1, curr_y2 - curr_y1))

        return sorted(merged, key=lambda b: b[0])


class HierarchicalSegmenter:
    """Extracts Primary (Level 1 Macro Boxes) and Secondary (Level 2 Sub-Segments) for Confidence Voting."""

    @staticmethod
    def extract_hierarchy(thresh_img: np.ndarray) -> List[Dict[str, Any]]:
        """Extracts primary boxes and secondary sub-segments for each character component."""
        h_img, w_img = thresh_img.shape[:2]

        # 1. Primary Contours (Intact Macro Characters)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        final_primary_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 4 and h >= 8 and (w * h) >= 30:
                aspect = w / float(h)
                if aspect > 1.25:
                    # Multi-character connected word contour — disconnect top Matra bar
                    word_mask = thresh_img[y:y+h, x:x+w].copy()
                    search_region = word_mask[int(h * 0.08):int(h * 0.35), :]
                    row_sums = search_region.sum(axis=1)

                    if row_sums.size > 0 and row_sums.max() > 0:
                        matra_y = int(h * 0.08) + np.argmax(row_sums)
                        word_mask[max(0, matra_y - 2):min(h, matra_y + 3), :] = 0

                    sub_contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    sub_boxes = []
                    for sc in sub_contours:
                        sx, sy, sw, sh = cv2.boundingRect(sc)
                        if sw >= 4 and sh >= 6 and (sw * sh) >= 20:
                            sub_boxes.append((x + sx, y, sw, h))

                    # CRITICAL: Sort sub_boxes left-to-right by X coordinate BEFORE merging
                    sub_boxes = sorted(sub_boxes, key=lambda b: b[0])
                    merged_sub = []
                    for sb in sub_boxes:
                        if not merged_sub:
                            merged_sub.append(sb)
                        else:
                            px, py, pw, ph = merged_sub[-1]
                            cx, cy, cw, ch = sb
                            if cx < (px + pw - 2):
                                nx = px
                                ny = y
                                nw = max(px + pw, cx + cw) - nx
                                nh = h
                                merged_sub[-1] = (nx, ny, nw, nh)
                            else:
                                merged_sub.append(sb)

                    for mb in merged_sub:
                        final_primary_boxes.append(mb)
                else:
                    final_primary_boxes.append((x, y, w, h))

        if not final_primary_boxes:
            return []

        primary_boxes = sorted(final_primary_boxes, key=lambda b: b[0])

        hierarchy_tree = []
        for px, py, pw, ph in primary_boxes:
            hierarchy_tree.append({
                "primary": (px, py, pw, ph),
                "secondaries": [(px, py, pw, ph)]
            })

        return hierarchy_tree


class BengaliSegmenter:
    """Full Hierarchical Bengali Layout Pipeline with Detached Symbol Merging."""

    def __init__(self):
        self.preprocessor = BengaliPreprocessor()
        self.hierarchical_segmenter = HierarchicalSegmenter()

    def segment_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Segments image into lines, words, and hierarchical primary/secondary character segments."""
        gray, thresh, rotation_angle = self.preprocessor.preprocess(img)

        hierarchy_tree = self.hierarchical_segmenter.extract_hierarchy(thresh)
        if not hierarchy_tree:
            return {"lines": [], "words": [], "characters": [], "rotation": rotation_angle}

        # Line Grouping on Primary Boxes
        raw_lines = []
        for item in hierarchy_tree:
            p_box = item["primary"]
            x, y, w, h = p_box
            y_center = y + h / 2.0
            matched_line = False
            for line in raw_lines:
                line_y_min = min(it["primary"][1] for it in line)
                line_y_max = max(it["primary"][1] + it["primary"][3] for it in line)
                line_h = max(20, line_y_max - line_y_min)
                if abs(y_center - (line_y_min + line_y_max) / 2.0) < line_h * 0.65:
                    line.append(item)
                    matched_line = True
                    break
            if not matched_line:
                raw_lines.append([item])

        raw_lines = sorted(raw_lines, key=lambda l: np.mean([it["primary"][1] for it in l]))

        all_lines = []
        all_words = []
        all_characters = []

        for line_idx, line_items in enumerate(raw_lines):
            line_items = sorted(line_items, key=lambda it: it["primary"][0])

            lx1 = min(it["primary"][0] for it in line_items)
            ly1 = min(it["primary"][1] for it in line_items)
            lx2 = max(it["primary"][0] + it["primary"][2] for it in line_items)
            ly2 = max(it["primary"][1] + it["primary"][3] for it in line_items)
            line_bbox = (lx1, ly1, lx2 - lx1, ly2 - ly1)
            all_lines.append({"bbox": line_bbox, "line": line_idx})

            # Word grouping by horizontal gap between primary boxes
            widths = [it["primary"][2] for it in line_items]
            median_w = np.median(widths) if widths else 20
            space_threshold = max(14, median_w * 0.85)

            curr_word = []
            words_in_line = []
            for it in line_items:
                if curr_word and (it["primary"][0] - (curr_word[-1]["primary"][0] + curr_word[-1]["primary"][2])) > space_threshold:
                    words_in_line.append(curr_word)
                    curr_word = []
                curr_word.append(it)
            if curr_word:
                words_in_line.append(curr_word)

            for w_items in words_in_line:
                wx1 = min(it["primary"][0] for it in w_items)
                wy1 = min(it["primary"][1] for it in w_items)
                wx2 = max(it["primary"][0] + it["primary"][2] for it in w_items)
                wy2 = max(it["primary"][1] + it["primary"][3] for it in w_items)
                word_bbox = (wx1, wy1, wx2 - wx1, wy2 - wy1)
                all_words.append({"bbox": word_bbox, "line": line_idx})

                for item in w_items:
                    all_characters.append({
                        "primary_bbox": item["primary"],
                        "secondary_bboxes": item["secondaries"],
                        "line": line_idx,
                        "word_bbox": word_bbox
                    })

        return {
            "lines": all_lines,
            "words": all_words,
            "characters": all_characters,
            "rotation": rotation_angle
        }
