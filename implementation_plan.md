# Implementation Plan - State-of-the-Art Offline Bangla Handwritten Text Recognition (HTR) System

Build a production-grade, modular, offline Bangla Handwritten Text Recognition (HTR) framework incorporating a hybrid Visual Feature Extractor (ResNet/ConvNeXt), Sequence Encoder (BiLSTM/Transformer), Reinforcement Learning (RL) Policy Agent (Actor-Critic / Policy Gradient with CER, WER, and Dictionary rewards), Trie-based Bangla Dictionary with Levenshtein distance fallback, Synthetic Expansion Pipeline, and Apple Silicon (MPS) GPU acceleration with AMP (Automatic Mixed Precision).

---

## Architecture Overview

```
 Input Handwriting Image
           │
           ▼
┌─────────────────────────┐
│ Synthetic / Local Data  │ (Data Augmentations: Elastic deformation, Skew, Blur, Noise)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Visual Feature         │ (ConvNeXt / ResNet / ViT Backbone)
│  Extractor              │ ──> Produces feature sequence X = (x_1, ..., x_T)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Sequence Encoder       │ (BiLSTM / Transformer Encoder)
│                         │ ──> Produces context vectors H = (h_1, ..., h_T)
└──────────┬──────────────┘
           │
 ┌─────────┴───────────────────────┐
 │                                 │
 ▼                                 ▼
┌─────────────────────────┐  ┌─────────────────────────────────────┐
│  Base CTC / Seq Head    │  │  RL Policy Agent (Actor-Critic)    │
│  (Base Logits & Tokens) │  │  State: Context + Decoded Prefix    │
└──────────┬──────────────┘  │  Action: Character Selection        │
           │                 │  Reward: -CER -WER + Lexicon Bonus  │
           │                 └──────────────────┬──────────────────┘
           │                                    │
           └──────────────────┬─────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Candidate String │
                     └────────┬─────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │ Trie Bangla Lexicon &        │
               │ Levenshtein Post-Processor   │
               └──────────────┬───────────────┘
                              │
                              ▼
                    Final Bangla Text Output
```

---

## User Review Required

> [!IMPORTANT]
> **Apple Silicon (MPS) & AMP Compatibility Note**:
> PyTorch on Apple Silicon supports MPS (`torch.device("mps")`) and Automatic Mixed Precision (`torch.amp.autocast("mps")`). The training scripts will automatically fall back gracefully to `cpu` if MPS is unavailable.

> [!NOTE]
> **121 Class Local Dataset Integration**:
> The local dataset inside `Bangla dataset/dataset_filtered` contains 122 subfolders (0 to 121 mapping to 121 character/stroke classes plus metadata). `dataset.py` will read `metaData_img.csv` to map directory indices to unicode Bangla characters, while synthetic word generation uses Bangla character combinations.

---

## Open Questions

None at this time. Standard Bangla unicode representation and font availability on macOS (`KohinoorBangla`, `Bangla MN`, `Bangla Sangam MN`) are verified and integrated into the synthetic generator.

---

## Proposed Changes

We will create 6 modular python files in the project root `/Users/muntasirabdullah/Desktop/HTR Bangla`:

### Module 1: `dataset.py`
#### [NEW] [dataset.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/dataset.py)
- **`BanglaCharacterDataset`**: PyTorch Dataset to load local 121-class dataset images from `Bangla dataset/dataset_filtered` using `metaData_img.csv`.
- **`SyntheticBanglaDataset`**: On-the-fly or offline synthetic text image generator using PIL/OpenCV and system/custom Bangla TTF/TTC fonts.
- **Data Augmentation Pipeline**: Elastic transformation (`scipy.ndimage.gaussian_filter`), random rotation/skewing, Gaussian noise, motion blur, line thickness modulation, contrast adjustments.
- **`HTRCollateFn`**: Handles padding of variable-length sequences and images into uniform tensor batches.

### Module 2: `model.py`
#### [NEW] [model.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/model.py)
- **`VisualFeatureExtractor`**: ResNet/ConvNeXt style CNN or ViT patch projection stem to extract high-dimensional feature maps from input handwriting images (e.g. 1x32xW or 1x64xW).
- **`SequenceEncoder`**: Multi-layer Bidirectional LSTM (BiLSTM) or Transformer Encoder for deep temporal/character context modelling across image width.
- **`HTRHybridModel`**: End-to-end model combining visual extractor, sequence encoder, CTC classification layer, and hidden state projections for the RL Agent.

### Module 3: `rl_agent.py`
#### [NEW] [rl_agent.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/rl_agent.py)
- **`HTREnvironment`**: Environment wrapper that tracks decoding state (current step $t$, context hidden vector $h_t$, previous predicted sequence prefix, candidate predictions).
- **`ActorCriticPolicy`**: Actor (Policy Network) producing character probability distributions over the 121-class vocabulary (+ blank/EOS tokens), and Critic (Value Network) estimating state values.
- **`RewardCalculator`**: Calculates scalar reward $R = - \alpha \cdot \text{CER} - \beta \cdot \text{WER} + \gamma \cdot \mathbb{I}(\text{word} \in \text{Lexicon})$.
- **`PPOAgent` / `PolicyGradientTrainer`**: Policy Gradient (REINFORCE / Actor-Critic) update step with advantage normalization and entropy regularization.

### Module 4: `dictionary.py`
#### [NEW] [dictionary.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/dictionary.py)
- **`TrieNode` & `BanglaTrie`**: Efficient prefix-tree data structure to insert and query Bangla vocabulary words.
- **`LevenshteinMatcher`**: Pure-Python / NumPy dynamic programming fuzzy search algorithm to find nearest valid Bangla dictionary words within edit distance $K$ when out-of-vocabulary predictions occur.
- **`BanglaPostProcessor`**: Integrates Trie prefix checks and Levenshtein candidates for final decoding refinement.

### Module 5: `train.py`
#### [NEW] [train.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/train.py)
- **Three-Stage Training Pipeline**:
  1. **Stage 1 (Pre-training)**: Train Visual Extractor + Sequence Encoder on synthetic Bangla text sequences using CTC Loss with AdamW and Cosine Annealing.
  2. **Stage 2 (Fine-tuning)**: Fine-tune on local handwriting dataset with domain adaptation.
  3. **Stage 3 (RL Optimization)**: Train RL Policy Agent using Actor-Critic / Policy Gradient on sequence decoding to maximize prediction accuracy under composite rewards.
- **Optimization & Acceleration**: Full support for Apple Silicon GPU (`mps`), `torch.amp.autocast`, checkpoint saving/loading, logging, evaluation metrics (CER, WER, Lexicon Match Rate).

### Module 6: `predict.py`
#### [NEW] [predict.py](file:///Users/muntasirabdullah/Desktop/HTR%20Bangla/predict.py)
- **`BanglaHTRPredictor`**: Unified inference engine. Loads trained model checkpoint and dictionary.
- Takes an input handwriting image, preprocesses it, runs hybrid visual-sequence feature extraction + RL policy decoding, and applies Trie post-processing.
- Returns predicted text, character-level confidence scores, and raw vs post-processed comparison.

---

## Verification Plan

### Automated Verification
1. **Module Unit Tests**: Execute each python module (`dataset.py`, `model.py`, `rl_agent.py`, `dictionary.py`, `predict.py`) directly with synthetic dummy tensors to verify input/output shapes and forward passes on `mps` and `cpu`.
2. **Pre-training & Fine-tuning Run**: Run `train.py --epochs 2 --batch-size 8 --pretrain` to verify synthetic training, fine-tuning, RL training, AMP compatibility, and checkpoint creation without runtime crashes.
3. **Inference Pipeline Test**: Run `predict.py --image <sample_path>` to ensure smooth end-to-end prediction with confidence scoring and Bangla text output.

### Manual Verification
- Inspect generated synthetic samples and accuracy logs to ensure zero NaNs, smooth loss convergence, and proper text decoding.
