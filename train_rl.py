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
from dataset import prepare_data, process_character, IMG_SIZE
from synthetic_word_generator import SyntheticWordGenerator, extract_words


def train_hybrid_rl(epochs: int = 8, rl_episodes: int = 5, batch_size: int = 64, lr: float = 0.0003, ppo_epochs: int = 4, clip_eps: float = 0.2):
    """2-Phase Hybrid Training Engine for End-to-End Bangla HTR."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✓ Using Device: {device}")

    # Load dataset & label encoder via prepare_data
    try:
        train_loader, val_loader, le, num_classes = prepare_data(batch_size=batch_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    env = BanglaHTREnvironment(label_encoder=le)

    # Vocabulary size = character classes + <EOS> (122) + <PAD> (123)
    vocab_size = num_classes + 2
    eos_idx = num_classes
    pad_idx = num_classes + 1

    # Models
    cnn_backbone = BestCNN(num_classes=num_classes)
    weights_path = "checkpoints/best_cnn_model_weights.pth"
    if os.path.exists(weights_path):
        try:
            cnn_backbone.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            print(f"✓ Loaded pre-trained BestCNN weights from {weights_path}")
        except Exception as e:
            print(f"Warning loading weights: {e}")
    elif os.path.exists("best_cnn_model_weights.pth"):
        try:
            cnn_backbone.load_state_dict(torch.load("best_cnn_model_weights.pth", map_location=device, weights_only=True))
            print("✓ Loaded pre-trained BestCNN weights from best_cnn_model_weights.pth")
        except Exception as e:
            print(f"Warning loading weights: {e}")

    feature_extractor = CRNNFeatureExtractor(cnn_backbone=cnn_backbone).to(device)
    agent = ActorCriticAgent(vocab_size=vocab_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(feature_extractor.parameters()) + list(agent.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # Synthetic Word Generator for multi-character sequence context
    synth_gen = SyntheticWordGenerator()

    # ── PHASE 1: Supervised Warmup (Single Chars + Synthetic Words) ────
    print("\n" + "="*60)
    print("PHASE 1: Supervised Policy Warmup (Single Chars + Synthetic Words)")
    print("="*60)

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    agent.train()

    for ep in range(1, epochs + 1):
        start_t = time.time()
        total_loss, correct, total = 0.0, 0, 0

        # 1. Single Character Batches
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

        # 2. Synthetic Word Sequence Batches with CTCLoss
        ctc_loss_fn = nn.CTCLoss(blank=eos_idx, zero_infinity=True)
        synth_batch_imgs = []
        synth_targets = []
        target_lengths = []

        for _ in range(batch_size):
            w_img, w_str = synth_gen.generate_word_image()
            proc_w = process_character(w_img, size=IMG_SIZE)
            if proc_w is not None and len(w_str) > 0:
                t_seq = [le.transform([ch])[0] for ch in w_str if ch in le.classes_]
                if t_seq:
                    synth_batch_imgs.append(np.transpose(proc_w.astype(np.float32)/255.0, (2, 0, 1)))
                    synth_targets.extend(t_seq)
                    target_lengths.append(len(t_seq))

        if synth_batch_imgs and target_lengths:
            s_imgs = torch.tensor(np.array(synth_batch_imgs), device=device)
            targets_tensor = torch.tensor(synth_targets, dtype=torch.long, device=device)
            target_lens_tensor = torch.tensor(target_lengths, dtype=torch.long, device=device)

            optimizer.zero_grad()
            s_feats = feature_extractor(s_imgs)  # (B, T, feature_dim)
            B_s, T_s, _ = s_feats.shape

            # Compute logits across full sequence length
            s_hidden = agent.init_hidden(B_s, device)
            s_prev = torch.full((B_s,), eos_idx, dtype=torch.long, device=device)

            logits_list = []
            for t in range(T_s):
                h_t = s_feats[:, t, :]
                logits, val, s_hidden = agent(h_t, s_prev, s_hidden)
                logits_list.append(logits.unsqueeze(1))
                s_prev = torch.argmax(logits, dim=-1)

            seq_logits = torch.cat(logits_list, dim=1)  # (B, T_s, vocab_size)
            log_probs_ctc = F.log_softmax(seq_logits, dim=-1).permute(1, 0, 2)  # (T_s, B, vocab_size)
            input_lens_tensor = torch.full((B_s,), T_s, dtype=torch.long, device=device)

            word_ctc_loss = ctc_loss_fn(
                log_probs_ctc.cpu(),
                targets_tensor.cpu(),
                input_lens_tensor.cpu(),
                target_lens_tensor.cpu()
            ).to(device)
            word_ctc_loss.backward()
            optimizer.step()

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
    parser.add_argument("--text-file", type=str, default="", help="Path to custom text file for synthetic word generator")
    args = parser.parse_args()

    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            custom_txt = f.read()
        custom_words = extract_words(custom_txt)
        if custom_words:
            print(f"✓ Loaded {len(custom_words)} custom synthetic training words from {args.text_file}")
            synth_gen = SyntheticWordGenerator()
            synth_gen.words_list = custom_words

    train_hybrid_rl(epochs=args.epochs, rl_episodes=args.rl_episodes, batch_size=args.batch_size, lr=args.lr)
