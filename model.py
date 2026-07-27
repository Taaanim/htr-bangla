"""
model.py — SE-ResNet CNN for Bangla Character Classification

Proven architecture from the working AI Project II:
- Squeeze-and-Excitation (SE) attention blocks for channel re-weighting
- 4 residual stages: 64 → 128 → 256 → 512 channels
- Global Average Pooling → deep classifier head
- Kaiming initialization for stable training
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns to re-weight feature channels."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), -1, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block with SE attention and optional dropout."""
    def __init__(self, in_ch, out_ch, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BestCNN(nn.Module):
    """SE-ResNet classifier for Bangla handwritten character recognition.

    Input:  (B, 3, 32, 32) — RGB images resized to 32×32
    Output: (B, num_classes) — logits for each character class
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Stem: 3 -> 64, 32×32
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 1: 64->64, 32×32 -> 16×16
            ResidualBlock(64, 64, drop=0.05),
            ResidualBlock(64, 64, drop=0.05),
            nn.MaxPool2d(2, 2),

            # Block 2: 64->128, 16×16 -> 8×8
            ResidualBlock(64, 128, drop=0.10),
            ResidualBlock(128, 128, drop=0.10),
            nn.MaxPool2d(2, 2),

            # Block 3: 128->256, 8×8 -> 4×4
            ResidualBlock(128, 256, drop=0.15),
            ResidualBlock(256, 256, drop=0.15),
            nn.MaxPool2d(2, 2),

            # Block 4: 256->512, 4×4 -> global pool
            ResidualBlock(256, 512, drop=0.20),
            ResidualBlock(512, 512, drop=0.20),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        """Returns 512-dim feature vector (for RL agent)."""
        feat = self.features(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat


if __name__ == "__main__":
    model = BestCNN(num_classes=122)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BestCNN — {total_params:,} total params, {trainable:,} trainable")

    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
