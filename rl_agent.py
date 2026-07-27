"""
rl_agent.py — REINFORCE Policy Gradient Agent for Bangla HTR Refinement

After the base SE-ResNet is trained with CrossEntropyLoss (Stage 1),
this RL agent fine-tunes the classifier head using policy gradient methods.

The agent treats classification as a decision problem:
- Action: selecting a class label
- Reward: +1 for correct, -1 for wrong, with confidence bonus/penalty
- Policy: softmax output of the classifier

This specifically targets hard/confused character pairs to push accuracy higher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class RLRefinementAgent:
    """REINFORCE agent that fine-tunes the classifier head of a trained CNN.

    Only updates the classifier parameters — feature extractor stays frozen.
    Uses baseline subtraction for variance reduction.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4,
                 gamma: float = 0.99, entropy_coeff: float = 0.01):
        self.model = model
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        # Only optimize the classifier head — freeze feature extractor
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(
            model.classifier.parameters(), lr=lr, weight_decay=1e-5
        )

        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = 0.99

    def compute_reward(self, pred_idx: int, true_idx: int,
                       confidence: float) -> float:
        """Computes scalar reward for a single prediction.

        Reward design:
        - Correct + high confidence: +1.0 + confidence_bonus
        - Correct + low confidence:  +0.5
        - Wrong:                     -1.0
        - Wrong + high confidence:   -1.5 (penalize confident mistakes)
        """
        if pred_idx == true_idx:
            if confidence > 0.9:
                return 1.0 + 0.5 * confidence
            elif confidence > 0.5:
                return 0.8
            else:
                return 0.5
        else:
            if confidence > 0.8:
                return -1.5  # Penalize confident wrong answers
            else:
                return -1.0

    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                   device: torch.device) -> Dict[str, float]:
        """One REINFORCE training step on a batch.

        Returns dict with metrics: reward, accuracy, loss, entropy.
        """
        self.model.train()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = self.model(images)
        probs = F.softmax(logits, dim=-1)

        # Sample actions from policy (or use argmax during later stages)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Compute per-sample rewards
        confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        rewards = []
        for i in range(len(actions)):
            r = self.compute_reward(
                actions[i].item(), labels[i].item(),
                confidences[i].item()
            )
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Baseline subtraction
        advantage = rewards_tensor - self.baseline
        self.baseline = (self.baseline_decay * self.baseline +
                         (1 - self.baseline_decay) * rewards_tensor.mean().item())

        # Policy gradient loss: -E[advantage * log_prob] - entropy_bonus
        policy_loss = -(advantage.detach() * log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropy
        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Metrics
        correct = (actions == labels).float().mean().item()
        return {
            "reward": rewards_tensor.mean().item(),
            "accuracy": correct * 100,
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
        }

    def unfreeze_all(self):
        """Unfreezes all model parameters (call after RL refinement if needed)."""
        for param in self.model.parameters():
            param.requires_grad = True


def run_rl_refinement(model: nn.Module, train_loader, val_loader,
                      device: torch.device, episodes: int = 10,
                      lr: float = 1e-4) -> nn.Module:
    """Runs RL refinement on a pre-trained model.

    Args:
        model: Pre-trained BestCNN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
        episodes: Number of RL training episodes
        lr: Learning rate for RL optimizer

    Returns:
        Refined model
    """
    agent = RLRefinementAgent(model, lr=lr)

    print(f"\n{'='*60}")
    print("STAGE 2: RL Policy Gradient Refinement")
    print(f"{'='*60}")
    print(f"Episodes: {episodes} | LR: {lr} | Entropy coeff: {agent.entropy_coeff}")
    print(f"Frozen: feature extractor | Trainable: classifier head\n")

    best_acc = 0.0
    best_state = None

    for episode in range(1, episodes + 1):
        # Train
        episode_rewards = []
        episode_accs = []
        for images, labels in train_loader:
            stats = agent.train_step(images, labels, device)
            episode_rewards.append(stats["reward"])
            episode_accs.append(stats["accuracy"])

        avg_reward = np.mean(episode_rewards)
        avg_train_acc = np.mean(episode_accs)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★ BEST"

        print(f"RL Episode {episode:02d}/{episodes} | "
              f"Reward: {avg_reward:+.3f} | "
              f"Train Acc: {avg_train_acc:.1f}% | "
              f"Val Acc: {val_acc:.2f}%{marker}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best RL model (Val Acc: {best_acc:.2f}%)")

    # Unfreeze everything for future fine-tuning if needed
    agent.unfreeze_all()

    return model
