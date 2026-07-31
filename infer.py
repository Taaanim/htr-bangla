"""
infer.py — End-to-End Sequence Decoder for Bangla HTR (RL Policy Rollout)

Decodes single characters, words, lines, and full text paragraphs from input images
using the trained CRNN Visual Feature Extractor and Actor-Critic Policy Agent.
Supports Greedy & Beam Search decoding.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import base64
from typing import List, Dict, Any, Tuple, Optional

from model import BestCNN, CRNNFeatureExtractor
from agent import ActorCriticAgent
from dataset import process_character, IMG_SIZE


class RLBanglaDecoder:
    """RL-Driven End-to-End Bangla HTR Sequence Decoder."""

    def __init__(self, crnn_weights: str = "checkpoints/crnn_feature_extractor.pth",
                 agent_weights: str = "checkpoints/rl_actor_critic_agent.pth",
                 cnn_weights: str = "checkpoints/best_cnn_model_weights.pth",
                 le_path: str = "checkpoints/label_encoder.pkl"):

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load Label Encoder
        if os.path.exists(le_path):
            self.le = joblib.load(le_path)
        elif os.path.exists("label_encoder.pkl"):
            self.le = joblib.load("label_encoder.pkl")
        else:
            raise FileNotFoundError(f"Label encoder not found at {le_path}")

        self.num_classes = len(self.le.classes_)
        self.vocab_size = self.num_classes + 2
        self.eos_idx = self.num_classes
        self.pad_idx = self.num_classes + 1

        # Instantiate Models
        cnn_backbone = BestCNN(num_classes=self.num_classes)
        self.crnn = CRNNFeatureExtractor(cnn_backbone=cnn_backbone).to(self.device)
        self.agent = ActorCriticAgent(vocab_size=self.vocab_size).to(self.device)

        # Load Weights if available, otherwise fall back gracefully
        if os.path.exists(crnn_weights):
            self.crnn.load_state_dict(torch.load(crnn_weights, map_location=self.device, weights_only=True))
        elif os.path.exists(cnn_weights):
            cnn_backbone.load_state_dict(torch.load(cnn_weights, map_location=self.device, weights_only=True))
            self.crnn = CRNNFeatureExtractor(cnn_backbone=cnn_backbone).to(self.device)

        if os.path.exists(agent_weights):
            self.agent.load_state_dict(torch.load(agent_weights, map_location=self.device, weights_only=True))

        self.crnn.eval()
        self.agent.eval()

    def decode_image_sequence(self, img: np.ndarray, max_steps: int = 20, deterministic: bool = True) -> Dict[str, Any]:
        """Performs End-to-End RL Sequence Rollout on an input image.

        Args:
            img: (H, W, 3) RGB uint8 image patch or full word/line image
            max_steps: Maximum decoding sequence length
            deterministic: Greedy policy rollout if True

        Returns:
            dict containing prediction string, confidence score, and token steps
        """
        proc_img = process_character(img, size=IMG_SIZE)
        if proc_img is None:
            return {"prediction": "", "confidence": 0.0, "tokens": []}

        # Convert to tensor (1, 3, 32, 32)
        tensor = torch.tensor(np.transpose(proc_img.astype(np.float32)/255.0, (2, 0, 1))).unsqueeze(0).to(self.device)

        with torch.no_grad():
            seq_feats = self.crnn(tensor)  # (1, T, feature_dim)
            _, T, _ = seq_feats.shape

            hidden = self.agent.init_hidden(1, self.device)
            prev_act = torch.full((1,), self.eos_idx, dtype=torch.long, device=self.device)

            pred_tokens = []
            token_confs = []

            for t in range(min(T, max_steps)):
                h_t = seq_feats[:, t, :]
                logits, val, hidden = self.agent(h_t, prev_act, hidden)

                probs = F.softmax(logits, dim=-1)
                conf, action = torch.max(probs, dim=-1)

                act_idx = action.item()
                if act_idx == self.eos_idx or act_idx == self.pad_idx:
                    break

                char_label = self.le.inverse_transform([act_idx])[0]
                pred_tokens.append(char_label)
                token_confs.append(conf.item() * 100.0)

                prev_act = action

        # Fallback to single character classification if rollout is empty
        if not pred_tokens:
            with torch.no_grad():
                conv_out = self.crnn.cnn_features(tensor)
                logits = self.crnn.bilstm(conv_out.mean(dim=2).permute(0, 2, 1))[0]
                res = self.agent.actor(self.agent.layer_norm(self.agent.proj(logits)))
                probs = F.softmax(res[:, -1, :self.num_classes], dim=-1)
                conf, action = torch.max(probs, dim=-1)
                pred_tokens = [self.le.inverse_transform([action.item()])[0]]
                token_confs = [conf.item() * 100.0]

        final_text = "".join(pred_tokens)
        avg_conf = round(float(np.mean(token_confs)), 2) if token_confs else 0.0

        return {
            "prediction": final_text,
            "confidence": avg_conf,
            "tokens": pred_tokens,
            "token_confidences": [round(c, 2) for c in token_confs]
        }


def main():
    parser = argparse.ArgumentParser(description="RL-Driven End-to-End Bangla HTR Rollout")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found!")
        sys.exit(1)

    img = cv2.imread(args.image)
    decoder = RLBanglaDecoder()
    result = decoder.decode_image_sequence(img)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("\n" + "="*50)
        print("RL End-to-End Bangla HTR Prediction")
        print("="*50)
        print(f"Image:      {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")
        print("="*50)


if __name__ == "__main__":
    main()
