"""
train.py - Three-Stage HTR Pre-Training, Fine-Tuning, & RL Policy Training Loop

Optimized for Apple Silicon GPU acceleration (device = torch.device("mps")) with AMP (Automatic Mixed Precision).
Stage 1: Pre-training on Synthetic Bangla Text using CTC Loss.
Stage 2: Domain Adaptation Fine-Tuning on 121-class local handwriting dataset.
Stage 3: Policy Gradient Reinforcement Learning (RL) fine-tuning for dynamic decoding optimization.
"""

import os
import sys

# Enable MPS fallback for un-implemented operators like _ctc_loss
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import time
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BanglaTokenizer, LocalBanglaDataset, SyntheticBanglaDataset, HTRCollateFn
from model import HTRHybridModel
from dictionary import BanglaPostProcessor
from rl_agent import HTRRewardEngine, RLActorCriticAgent, calculate_cer, calculate_wer


def get_device() -> torch.device:
    """Detects Apple Silicon MPS GPU or fallback CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU Acceleration (MPS)")
    else:
        device = torch.device("cpu")
        print("MPS GPU not available. Using CPU fallback.")
    return device


def train_ctc_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, ctc_loss_fn: nn.CTCLoss,
                    device: torch.device, epoch: int, total_epochs: int, stage_name: str = "Training",
                    use_amp: bool = True) -> float:
    """Runs one training epoch using CTC Loss with optional AMP and live tqdm progress bar."""
    model.train()
    total_loss = 0.0

    amp_device_type = "mps" if device.type == "mps" else "cpu"
    pbar = tqdm(dataloader, desc=f"{stage_name} Epoch [{epoch}/{total_epochs}]", unit="batch", leave=True)

    for step, (images, targets, input_lengths, target_lengths, texts) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward
        if use_amp and amp_device_type == "mps":
            with torch.amp.autocast(device_type="mps"):
                logits = model(images)  # (B, T, num_classes)
                log_probs = logits.permute(1, 0, 2).log_softmax(2)
                loss = ctc_loss_fn(log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu()).to(device)
        else:
            logits = model(images)
            log_probs = logits.permute(1, 0, 2).log_softmax(2)
            loss = ctc_loss_fn(log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu()).to(device)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        current_avg = total_loss / (step + 1)
        pbar.set_postfix({"Loss": f"{current_avg:.4f}"})

    scheduler.step()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, tokenizer: BanglaTokenizer,
                   post_processor: BanglaPostProcessor, device: torch.device) -> Tuple[float, float, float]:
    """Evaluates Average Character Error Rate (CER), Word Error Rate (WER), and Dictionary Match Rate."""
    model.eval()
    cers, wers, dict_matches = [], [], []

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch", leave=False)
    for images, targets, _, _, target_texts in pbar:
        images = images.to(device)
        logits = model(images)  # (B, T, num_classes)
        preds = logits.argmax(dim=-1)  # (B, T)

        for b in range(images.size(0)):
            tokens = preds[b].cpu().tolist()
            raw_text = tokenizer.decode_ctc(tokens)
            target_text = target_texts[b]

            cer = calculate_cer(raw_text, target_text)
            wer = calculate_wer(raw_text, target_text)
            cers.append(cer)
            wers.append(wer)

            words = raw_text.strip().split()
            if words:
                valid_ratio = sum(1 for w in words if post_processor.is_valid_word(w)) / len(words)
                dict_matches.append(valid_ratio)
            else:
                dict_matches.append(0.0)

    avg_cer = sum(cers) / max(len(cers), 1)
    avg_wer = sum(wers) / max(len(wers), 1)
    avg_dict_match = sum(dict_matches) / max(len(dict_matches), 1)

    return avg_cer, avg_wer, avg_dict_match


def main():
    parser = argparse.ArgumentParser(description="Offline Bangla HTR Three-Stage Training Script")
    parser.add_argument("--metadata-csv", type=str, default="Bangla dataset/metaData_img.csv", help="Metadata CSV path")
    parser.add_argument("--local-dataset-dir", type=str, default="Bangla dataset/dataset_filtered", help="Local dataset directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="Number of pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--rl-epochs", type=int, default=5, help="Number of RL policy training epochs")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip synthetic pre-training")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()

    # Initialize Tokenizer & Post-Processor
    tokenizer = BanglaTokenizer(args.metadata_csv)
    post_processor = BanglaPostProcessor()
    num_classes = len(tokenizer)

    print(f"Initialized Bangla Tokenizer with Vocabulary Size: {num_classes}")

    # Build Model
    model = HTRHybridModel(num_classes=num_classes, hidden_dim=256).to(device)

    # CTC Loss Function
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # ----------------------------------------------------
    # STAGE 1: Synthetic Pre-Training
    # ----------------------------------------------------
    if not args.skip_pretrain:
        print("\n" + "=" * 50)
        print("STAGE 1: Synthetic Pre-Training on Generated Bangla Sequences")
        print("=" * 50)

        vocab_list = list(post_processor.vocab_words)
        synthetic_dataset = SyntheticBanglaDataset(vocab_list, tokenizer, num_samples=1000)
        synth_loader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=HTRCollateFn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

        for epoch in range(1, args.pretrain_epochs + 1):
            start_time = time.time()
            loss = train_ctc_epoch(model, synth_loader, optimizer, scheduler, ctc_loss_fn, device, epoch, args.pretrain_epochs, stage_name="Pre-Train")
            elapsed = time.time() - start_time
            print(f"Pre-Train Epoch [{epoch}/{args.pretrain_epochs}] Completed | Loss: {loss:.4f} | Time: {elapsed:.2f}s")

        torch.save(model.state_dict(), os.path.join(args.save_dir, "pretrain_model.pth"))
        print(f"Saved pre-trained checkpoint to {args.save_dir}/pretrain_model.pth")

    # ----------------------------------------------------
    # STAGE 2: Local Dataset Fine-Tuning
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 2: Fine-Tuning on Local 121-Class Bangla Handwriting Dataset")
    print("=" * 50)

    if os.path.exists(args.local_dataset_dir):
        local_dataset = LocalBanglaDataset(args.local_dataset_dir, args.metadata_csv, tokenizer, is_training=True)
        print(f"Loaded {len(local_dataset)} local handwriting samples.")
        local_loader = DataLoader(local_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=HTRCollateFn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

        for epoch in range(1, args.finetune_epochs + 1):
            start_time = time.time()
            loss = train_ctc_epoch(model, local_loader, optimizer, scheduler, ctc_loss_fn, device, epoch, args.finetune_epochs, stage_name="Fine-Tune")
            cer, wer, dict_match = evaluate_model(model, local_loader, tokenizer, post_processor, device)
            elapsed = time.time() - start_time
            print(f"Fine-Tune Epoch [{epoch}/{args.finetune_epochs}] Completed | Loss: {loss:.4f} | CER: {cer:.4f} | WER: {wer:.4f} | Lexicon Match: {dict_match*100:.1f}% | Time: {elapsed:.2f}s")

        torch.save(model.state_dict(), os.path.join(args.save_dir, "finetune_model.pth"))
        print(f"Saved fine-tuned checkpoint to {args.save_dir}/finetune_model.pth")
    else:
        print(f"Local dataset path '{args.local_dataset_dir}' not found. Skipping Stage 2.")

    # ----------------------------------------------------
    # STAGE 3: RL Policy Agent Training
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 3: Reinforcement Learning (RL) Policy Agent Fine-Tuning")
    print("=" * 50)

    reward_engine = HTRRewardEngine(post_processor, cer_weight=1.0, wer_weight=1.0, dict_bonus=0.5)
    rl_agent = RLActorCriticAgent(model, tokenizer, reward_engine, lr=1e-4)

    vocab_list = list(post_processor.vocab_words)
    rl_dataset = SyntheticBanglaDataset(vocab_list, tokenizer, num_samples=500)
    rl_loader = DataLoader(rl_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=HTRCollateFn)

    for epoch in range(1, args.rl_epochs + 1):
        epoch_rewards, epoch_cers, epoch_wers = [], [], []
        start_time = time.time()
        pbar = tqdm(rl_loader, desc=f"RL-Agent Epoch [{epoch}/{args.rl_epochs}]", unit="batch", leave=True)

        for images, targets, _, _, target_texts in pbar:
            images = images.to(device)
            targets = targets.to(device)
            stats = rl_agent.train_step(images, targets, target_texts)
            epoch_rewards.append(stats['reward'])
            epoch_cers.append(stats['cer'])
            epoch_wers.append(stats['wer'])
            pbar.set_postfix({"Reward": f"{stats['reward']:.2f}", "CER": f"{stats['cer']:.2f}"})

        avg_r = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_c = sum(epoch_cers) / max(len(epoch_cers), 1)
        avg_w = sum(epoch_wers) / max(len(epoch_wers), 1)
        elapsed = time.time() - start_time
        print(f"RL Epoch [{epoch}/{args.rl_epochs}] Completed | Reward: {avg_r:.4f} | CER: {avg_c:.4f} | WER: {avg_w:.4f} | Time: {elapsed:.2f}s")

    final_model_path = os.path.join(args.save_dir, "final_htr_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSuccessfully completed all training stages! Final SOTA model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
