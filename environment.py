"""
environment.py — HTR Environment Wrapper for Reinforcement Learning (PPO/REINFORCE)

Manages sequence decoding environment transitions, step counters, and reward logic:
1. Terminal Reward: Normalized Levenshtein Edit Distance (1.0 - CER)
2. Lexicon Bonus: Reward boost (+0.15) if predicted word exists in valid Bangla dictionary
3. Step Penalty: Small step penalty (-0.01) to encourage sequence efficiency
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

# Levenshtein Edit Distance calculation
def levenshtein_distance(str1: str, str2: str) -> int:
    """Computes exact Levenshtein Edit Distance between two strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


class BanglaHTREnvironment:
    """RL Environment for Step-by-Step Bangla Text Sequence Generation."""

    def __init__(self, label_encoder, dictionary_words: Optional[set] = None):
        self.le = label_encoder
        self.num_classes = len(label_encoder.classes_)
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"

        # Load dictionary words if available
        self.dictionary_words = dictionary_words or set()
        if not self.dictionary_words:
            try:
                from dictionary import BANGLA_DICTIONARY
                self.dictionary_words = set(BANGLA_DICTIONARY)
            except Exception:
                self.dictionary_words = set()

    def decode_indices_to_string(self, token_indices: List[int]) -> str:
        """Converts sequence of token indices into a decoded Bangla text string."""
        chars = []
        for idx in token_indices:
            if idx >= self.num_classes:
                continue
            char = self.le.inverse_transform([idx])[0]
            if char in (self.eos_token, self.pad_token):
                break
            chars.append(char)
        return "".join(chars)

    def calculate_reward(self, pred_indices: List[int], target_indices: List[int]) -> Tuple[float, float, bool]:
        """Calculates normalized Levenshtein Edit Distance sequence reward + Lexicon Bonus.

        Returns:
            total_reward: Combined sequence reward
            edit_distance_reward: Normalized CER reward (1.0 - CER)
            is_valid_word: True if word exists in dictionary
        """
        pred_str = self.decode_indices_to_string(pred_indices)
        target_str = self.decode_indices_to_string(target_indices)

        max_len = max(len(target_str), 1)
        edit_dist = levenshtein_distance(pred_str, target_str)

        # 1. Terminal Normalized Edit Distance Reward (1.0 - CER)
        cer_reward = max(0.0, 1.0 - (edit_dist / float(max_len)))

        # 2. Lexicon Bonus (+0.15 if exact match in Bangla dictionary)
        is_valid_word = False
        lexicon_bonus = 0.0
        if pred_str in self.dictionary_words:
            is_valid_word = True
            lexicon_bonus = 0.15

        total_reward = cer_reward + lexicon_bonus
        return round(float(total_reward), 4), round(float(cer_reward), 4), is_valid_word
