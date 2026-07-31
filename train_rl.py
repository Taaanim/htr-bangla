"""
train_rl.py — 2-Phase Hybrid Training Engine (Supervised Warmup + PPO RL Fine-Tuning)

Phase 1 (Warmup): Pre-train CNN + BiLSTM + Actor-Critic using Cross-Entropy Loss
Phase 2 (RL Fine-tuning): PPO (Proximal Policy Optimization) Actor-Critic optimization with GAE
Supports Apple Silicon GPU (mps) acceleration.
"""

import os
import sys
import argparse
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import BestCNN, CRNNFeatureExtractor
from agent import ActorCriticAgent
from environment import BanglaHTREnvironment, levenshtein_distance
from dataset import BanglaDataset, process_character, IMG_SIZE


def train_hybrid_rl(epochs: int = 10, rl_episodes: int = 5, batch_size: int = 64, lr: float = 0.0003, ppo_epochs: int = 4, clip_eps: float = 0.2):
    """2-Phase Hybrid Training Engine for End-to-End Bangla HTR."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✓ Using Device: {device}")

    # Load dataset & label encoder
    dataset_dir = "Bangla dataset/dataset_filtered"
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found!")
        return

    full_ds = BanglaDataset(root_dir=dataset_dir)
    num_classes = full_ds.num_classes
    le = full_ds.label_encoder

    train_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    env = BanglaHTREnvironment(label_encoder=le)

    # Vocabulary size = character classes + <EOS> (122) + <PAD> (123)
    vocab_size = num_classes + 2
    eos_idx = num_classes
    pad_idx = num_classes + 1

    # Models
    cnn_backbone = BestCNN(num_classes=num_classes)
    feature_extractor = CRNNFeatureExtractor(cnn_backbone=cnn_backbone).to(device)
    agent = ActorCriticAgent(vocab_size=vocab_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(feature_extractor.parameters()) + list(agent.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # ── PHASE 1: Supervised Warmup ──────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Supervised Policy Warmup (CrossEntropy Pre-Training)")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    agent.train()

    for ep in range(1, epochs + 1):
        start_t = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            seq_feats = feature_extractor(imgs)  # (B, T, feature_dim)
            B, T, _ = seq_feats.shape

            hidden = agent.init_hidden(B, device)
            prev_act = torch.full((B,), eos_idx, dtype=torch.long, device=device)

            step_loss = 0.0
            for t in range(min(T, 4)):
                h_t = seq_feats[:, t, :]
                logits, val, hidden = agent(h_t, prev_act, hidden)

                loss = criterion(logits, labels)
                step_loss += loss

                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += B
                prev_act = labels

            step_loss.backward()
            optimizer.step()
            total_loss += step_loss.item()

        acc = (correct / float(total)) * 100.0 if total > 0 else 0.0
        elapsed = time.time() - start_t
        print(f"Warmup Epoch {ep:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}% | {elapsed:.1f}s")

    # ── PHASE 2: PPO RL Fine-Tuning ─────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: PPO Actor-Critic RL Fine-Tuning (Sequence Reward Optimization)")
    print("="*60)

    for rl_ep in range(1, rl_episodes + 1):
        start_t = time.time()
        ep_rewards = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            with torch.no_grad():
                seq_feats = feature_extractor(imgs)

            B, T, _ = seq_feats.shape
            hidden = agent.init_hidden(B, device)
            prev_act = torch.full((B,), eos_idx, dtype=torch.long, device=device)

            actions_batch, log_probs_batch, values_batch = [], [], []

            for t in range(min(T, 4)):
                h_t = seq_feats[:, t, :]
                logits, val, hidden = agent(h_t, prev_act, hidden)
                action, log_prob, _ = agent.sample_action(logits, deterministic=False)

                actions_batch.append(action.unsqueeze(1))
                log_probs_batch.append(log_prob.unsqueeze(1))
                values_batch.append(val)
                prev_act = action

            actions_tensor = torch.cat(actions_batch, dim=1)    # (B, T)
            old_log_probs = torch.cat(log_probs_batch, dim=1)   # (B, T)
            old_values = torch.cat(values_batch, dim=1).squeeze(-1) # (B, T)

            # Compute rewards
            batch_rewards = []
            for b in range(B):
                pred_list = actions_tensor[b].cpu().tolist()
                target_list = [labels[b].item()]
                rew, _, _ = env.calculate_reward(pred_list, target_list)
                batch_rewards.append(rew)

            ep_rewards.extend(batch_rewards)
            rewards_tensor = torch.tensor(batch_rewards, device=device).unsqueeze(1).repeat(1, actions_tensor.size(1))

            # PPO Update Step
            advantages = rewards_tensor - old_values.detach()

            for _ in range(ppo_epochs):
                new_log_probs, new_values, entropies = agent.evaluate_actions(seq_feats, actions_tensor)
                ratios = torch.exp(new_log_probs - old_log_probs.detach())

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(new_values, rewards_tensor)
                entropy_loss = -entropies.mean()

                ppo_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

                optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

        avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        elapsed = time.time() - start_t
        print(f"PPO Episode {rl_ep:02d}/{rl_episodes:02d} | Avg Reward: {avg_reward:.4f} | {elapsed:.1f}s")

    # Save Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(feature_extractor.state_dict(), "checkpoints/crnn_feature_extractor.pth")
    torch.save(agent.state_dict(), "checkpoints/rl_actor_critic_agent.pth")
    joblib.dump(le, "checkpoints/label_encoder.pkl")
    print("\n✓ Model checkpoints saved successfully to checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid RL-Based Bangla HTR")
    parser.add_argument("--epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--rl-episodes", type=int, default=3, help="PPO RL episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    args = parser.parse_args()

    train_hybrid_rl(epochs=args.epochs, rl_episodes=args.rl_episodes, batch_size=args.batch_size, lr=args.lr)
