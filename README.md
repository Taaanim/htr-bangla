# Offline Bangla Handwritten Text Recognition (HTR)

A State-of-the-Art (SOTA) offline Bangla Handwritten Text Recognition (HTR) framework combining a hybrid ConvNeXt Visual Extractor, Bidirectional LSTM Sequence Encoder, Actor-Critic Reinforcement Learning (RL) Policy Agent, and Trie Lexicon dynamic programming post-processor.

## Architecture

```
Input Handwriting Image ──> Visual Feature Extractor (ConvNeXt) ──> Sequence Encoder (BiLSTM)
                                                                           │
                                ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
                                ▼                                                                                     ▼
                   Base CTC Logits Head                                                              RL Policy Agent (Actor-Critic)
                            │                                                                        State: Context + Prefix
                            │                                                                        Reward: -CER -WER + Lexicon Bonus
                            └──────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                                       ▼
                                                       Bangla Trie & Levenshtein Engine ──> Final Bangla Text
```

## Features

- **ConvNeXt Visual Feature Extractor**: Fine-grained stroke feature extraction for complex Bangla compound characters (*যুক্তাক্ষর*) and modifiers (*কার/ফলা*).
- **BiLSTM Sequence Encoder**: Multi-layer Bidirectional LSTM capturing temporal stroke context.
- **Actor-Critic RL Policy Agent**: Sequence decoding optimization using policy gradients with scalar CER, WER, and dictionary bonus rewards.
- **Bangla Trie & Levenshtein Engine**: Trie prefix lookup & dynamic programming edit-distance fallback post-processor.
- **Synthetic Data Expansion**: On-the-fly rendering with native Bangla TTF fonts (`KohinoorBangla`, `Bangla MN`), elastic deformation, noise, blur, and skew augmentations.
- **Apple Silicon (MPS) & AMP**: Optimized for GPU acceleration (`torch.device("mps")`) with Automatic Mixed Precision.

## Codebase Modules

- `dataset.py`: Local 121-class dataset loader, synthetic text line generator, and augmentation pipeline.
- `model.py`: Hybrid ConvNeXt + BiLSTM + CTC + RL model architecture (`HTRHybridModel`).
- `rl_agent.py`: Reward function definition (`HTRRewardEngine`) and Actor-Critic optimization (`RLActorCriticAgent`).
- `dictionary.py`: Trie dictionary structure (`BanglaTrie`) and Levenshtein post-processor (`BanglaPostProcessor`).
- `train.py`: 3-stage pre-training, fine-tuning, and RL policy training loop.
- `predict.py`: End-to-end inference script returning text predictions with confidence scores.

## Quick Start

### Installation
```bash
pip install torch torchvision numpy pandas pillow scipy
```

### Training
To run the full 3-stage training pipeline (Pre-training -> Fine-tuning -> RL Policy Training):
```bash
python3 train.py --pretrain-epochs 5 --finetune-epochs 10 --rl-epochs 5 --batch-size 16
```

### Inference
To run text recognition on a handwriting sample image:
```bash
python3 predict.py --image path/to/sample.png --model-path checkpoints/final_htr_model.pth
```
