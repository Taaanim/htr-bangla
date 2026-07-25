"""
rl_agent.py - Reinforcement Learning (RL) Policy Agent & Reward Engine

Implements Policy Gradient / Actor-Critic optimization for dynamic decoding path correction in HTR.
Reward function combines negative Character Error Rate (-CER), negative Word Error Rate (-WER),
and a dictionary bonus reward for lexicon-valid predictions.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dictionary import BanglaPostProcessor, levenshtein_distance
from dataset import BanglaTokenizer


def calculate_cer(pred_str: str, target_str: str) -> float:
    """Calculates Character Error Rate (CER) between predicted and target string."""
    if len(target_str) == 0:
        return 0.0 if len(pred_str) == 0 else 1.0
    dist = levenshtein_distance(pred_str, target_str)
    return float(dist) / float(len(target_str))


def calculate_wer(pred_str: str, target_str: str) -> float:
    """Calculates Word Error Rate (WER) between predicted and target sentence."""
    pred_words = pred_str.strip().split()
    target_words = target_str.strip().split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Word-level edit distance
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return float(dp[m][n]) / float(len(target_words))


class HTRRewardEngine:
    """Computes scalar reinforcement learning rewards for sequence decoding trajectories."""

    def __init__(self, post_processor: BanglaPostProcessor, cer_weight: float = 1.0,
                 wer_weight: float = 1.0, dict_bonus: float = 0.5):
        self.post_processor = post_processor
        self.cer_weight = cer_weight
        self.wer_weight = wer_weight
        self.dict_bonus = dict_bonus

    def compute_reward(self, pred_str: str, target_str: str) -> float:
        """
        Reward R = - (cer_weight * CER) - (wer_weight * WER) + (dict_bonus if pred in Dictionary)
        """
        cer = calculate_cer(pred_str, target_str)
        wer = calculate_wer(pred_str, target_str)

        bonus = 0.0
        words = pred_str.strip().split()
        if len(words) > 0:
            valid_count = sum(1 for w in words if self.post_processor.is_valid_word(w))
            bonus = self.dict_bonus * (valid_count / len(words))

        reward = - (self.cer_weight * cer) - (self.wer_weight * wer) + bonus
        return reward


class RLActorCriticAgent:
    """
    Actor-Critic Policy Gradient Optimization Agent for sequence decoding refinement.
    Uses policy advantage estimation and entropy regularization.
    """

    def __init__(self, model: nn.Module, tokenizer: BanglaTokenizer, reward_engine: HTRRewardEngine,
                 lr: float = 1e-4, gamma: float = 0.99, entropy_coef: float = 0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_engine = reward_engine
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, images: torch.Tensor, targets: torch.Tensor, target_texts: List[str]) -> dict:
        """
        Performs a single Actor-Critic policy gradient training step over a batch of images.
        """
        self.model.train()
        self.optimizer.zero_grad()

        encoded_seq = self.model.extract_features(images)  # (B, T, hidden_dim)
        action_logits, state_values = self.model.get_rl_predictions(encoded_seq)  # (B, T, num_classes), (B, T)

        batch_size, seq_len, num_classes = action_logits.shape
        dist = torch.distributions.Categorical(logits=action_logits)

        # Sample actions from policy
        actions = dist.sample()  # (B, T)
        log_probs = dist.log_prob(actions)  # (B, T)
        entropy = dist.entropy().mean()

        policy_loss = 0.0
        value_loss = 0.0
        total_rewards = []
        cers = []
        wers = []

        for b in range(batch_size):
            # Convert sampled sequence to decoded string
            sampled_tokens = actions[b].cpu().tolist()
            pred_text = self.tokenizer.decode_ctc(sampled_tokens)
            target_text = target_texts[b]

            # Calculate scalar trajectory reward
            r = self.reward_engine.compute_reward(pred_text, target_text)
            total_rewards.append(r)
            cers.append(calculate_cer(pred_text, target_text))
            wers.append(calculate_wer(pred_text, target_text))

            # Compute advantage A_t = R - V(s)
            returns = torch.full((seq_len,), r, device=images.device, dtype=torch.float32)
            values = state_values[b]
            advantages = returns - values.detach()

            # Actor & Critic losses
            b_policy_loss = -(log_probs[b] * advantages).mean()
            b_value_loss = F.mse_loss(values, returns)

            policy_loss += b_policy_loss
            value_loss += b_value_loss

        policy_loss = policy_loss / batch_size
        value_loss = value_loss / batch_size

        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        avg_reward = float(sum(total_rewards) / batch_size)
        avg_cer = float(sum(cers) / batch_size)
        avg_wer = float(sum(wers) / batch_size)

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward": avg_reward,
            "cer": avg_cer,
            "wer": avg_wer
        }


if __name__ == "__main__":
    from model import HTRHybridModel

    tokenizer = BanglaTokenizer("Bangla dataset/metaData_img.csv")
    post_processor = BanglaPostProcessor()
    reward_engine = HTRRewardEngine(post_processor)
    model = HTRHybridModel(num_classes=len(tokenizer))
    agent = RLActorCriticAgent(model, tokenizer, reward_engine)

    dummy_images = torch.randn(2, 1, 64, 256)
    dummy_targets = torch.tensor([[5, 62, 53], [5, 62, 53]], dtype=torch.long)
    target_texts = ["বাংলা", "বাংলাদেশ"]

    stats = agent.train_step(dummy_images, dummy_targets, target_texts)
    print("RL Train Step Stats:", stats)
