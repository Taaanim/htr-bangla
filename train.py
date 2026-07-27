"""
train.py — Two-Stage Training: Classification + RL Refinement

Stage 1: CrossEntropyLoss classification (50 epochs, early stopping)
Stage 2: REINFORCE RL refinement on classifier head (10 episodes)

Optimized for Apple Silicon MPS GPU.
"""

import os
import sys
import time
import argparse
import numpy as np

# Force unbuffered output so logs appear in real-time
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from model import BestCNN
from dataset import prepare_data
from rl_agent import run_rl_refinement


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✓ Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        print("⚠ MPS not available, using CPU")
        return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch, returns (avg_loss, accuracy%)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate, returns (avg_loss, accuracy%)."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return val_loss / len(loader), 100 * correct / total


def save_checkpoint(model, save_dir="checkpoints", filename="best_cnn_model_weights.pth"):
    """Saves model weights to checkpoints dir and project root."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), filename)  # Also save to project root
    return path


def main():
    parser = argparse.ArgumentParser(description="Bangla HTR Training")
    parser.add_argument("--epochs", type=int, default=50, help="Classification epochs")
    parser.add_argument("--rl-episodes", type=int, default=10, help="RL refinement episodes")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL refinement stage")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = get_device()

    # ══════════════════════════════════════════════════════════
    # DATA
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    train_loader, val_loader, le, num_classes = prepare_data(
        batch_size=args.batch_size
    )

    # ══════════════════════════════════════════════════════════
    # MODEL
    # ══════════════════════════════════════════════════════════
    model = BestCNN(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BestCNN (SE-ResNet) — {total_params:,} parameters")

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Classification Training
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 1: Classification Training (CrossEntropyLoss)")
    print("=" * 60)
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Patience: {args.patience}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Early stopping
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ★ BEST"
            save_checkpoint(model, args.save_dir)
        else:
            patience_counter += 1

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {lr_now:.6f} | {elapsed:.1f}s{marker}")

        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Restore best model from Stage 1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n{'='*60}")
    print(f"Stage 1 Complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    save_checkpoint(model, args.save_dir, "best_cnn_model_weights.pth")
    save_checkpoint(model, args.save_dir, "stage1_model.pth")

    # ══════════════════════════════════════════════════════════
    # STAGE 2: RL Refinement
    # ══════════════════════════════════════════════════════════
    if not args.skip_rl and args.rl_episodes > 0:
        model = run_rl_refinement(
            model, train_loader, val_loader, device,
            episodes=args.rl_episodes, lr=1e-4
        )
        save_checkpoint(model, args.save_dir, "best_cnn_model_weights.pth")
        save_checkpoint(model, args.save_dir, "rl_refined_model.pth")

    # ══════════════════════════════════════════════════════════
    # FINAL SAVE
    # ══════════════════════════════════════════════════════════
    final_path = save_checkpoint(model, args.save_dir, "best_cnn_model_weights.pth")
    print(f"\n✓ Final model saved to: {final_path}")
    print(f"✓ Label encoder saved to: checkpoints/label_encoder.pkl")
    print(f"✓ Model size: {os.path.getsize(final_path) / 1024 / 1024:.2f} MB")
    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
