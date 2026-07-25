"""
model.py - Hybrid Visual Feature Extractor (ResNet/ConvNeXt) + BiLSTM Sequence Encoder

Implements fine-grained visual stroke feature extraction, sequence context modeling via BiLSTM,
CTC character prediction output head, and policy/value projections for the RL Agent.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block for fine-grained visual stroke extraction."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input_x + x


class VisualFeatureExtractor(nn.Module):
    """Hybrid ConvNeXt/ResNet Visual Feature Extractor for Bangla HTR."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Stem: Downsample height by 4x, width by 2x
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # Stage 1: ConvNeXt blocks
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128)
        )

        # Downsample Stage 2: Reduce height further
        self.downsample = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Stage 3: Feature refinement
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(hidden_dim),
            ConvNeXtBlock(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: (B, T, D) where T = W_feat, D = hidden_dim
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)

        # Adaptive height pooling to collapse H dimension to 1
        x = F.adaptive_avg_pool2d(x, (1, None))  # (B, hidden_dim, 1, W_feat)
        x = x.squeeze(2)  # (B, hidden_dim, W_feat)
        x = x.permute(0, 2, 1)  # (B, W_feat, hidden_dim)
        return x


class SequenceEncoder(nn.Module):
    """Multi-layer Bidirectional LSTM Sequence Encoder."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, input_dim)
        Output: (B, T, hidden_dim)
        """
        lstm_out, _ = self.bilstm(x)
        out = self.proj(lstm_out)
        out = self.dropout(out)
        return out


class HTRHybridModel(nn.Module):
    """
    End-to-End Hybrid HTR Model:
    Visual Feature Extractor + Sequence Encoder + CTC Head + RL Projections
    """

    def __init__(self, num_classes: int = 127, hidden_dim: int = 256, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. Visual Feature Extractor
        self.visual_extractor = VisualFeatureExtractor(in_channels=in_channels, hidden_dim=hidden_dim)

        # 2. Sequence Encoder
        self.sequence_encoder = SequenceEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=2)

        # 3. Base CTC Classifier Head
        self.ctc_head = nn.Linear(hidden_dim, num_classes)

        # 4. Projections for RL Policy Agent
        self.rl_actor_head = nn.Linear(hidden_dim, num_classes)
        self.rl_critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for pre-training / CTC loss computation.
        Input: (B, C, H, W)
        Output: CTC Logits (B, T, num_classes)
        """
        features = self.visual_extractor(x)
        encoded_seq = self.sequence_encoder(features)
        ctc_logits = self.ctc_head(encoded_seq)
        return ctc_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts contextualized sequence embeddings (B, T, hidden_dim)."""
        features = self.visual_extractor(x)
        encoded_seq = self.sequence_encoder(features)
        return encoded_seq

    def get_rl_predictions(self, encoded_seq: torch.Tensor):
        """Returns RL Policy action logits (B, T, num_classes) and State values (B, T, 1)."""
        action_logits = self.rl_actor_head(encoded_seq)
        state_values = self.rl_critic_head(encoded_seq).squeeze(-1)
        return action_logits, state_values


if __name__ == "__main__":
    model = HTRHybridModel(num_classes=127, hidden_dim=256)
    dummy_input = torch.randn(4, 1, 64, 256)
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> CTC Logits shape: {logits.shape}")

    encoded = model.extract_features(dummy_input)
    act_logits, st_vals = model.get_rl_predictions(encoded)
    print(f"Encoded shape: {encoded.shape} -> Action Logits: {act_logits.shape}, State Values: {st_vals.shape}")
