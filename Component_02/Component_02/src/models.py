"""
models.py — SINGLE SOURCE OF TRUTH for every network architecture.

Fixes audit finding E-10: the _archive/ code duplicated the same ResNet in five
files (app.py, predict.py, evaluate_hybrid.py, train_ecg_only.py,
train_report_gen.py). Any edit had to be made five times or inference silently
diverged from training. Everything now imports from here.

Two architectures:
  ECGResNet      — bit-exact reimplementation of the shipped baseline so that
                   _archive/checkpoints_ecg_only/best_model.pt loads unchanged.
  ECGResNetSE    — the improved backbone (squeeze-excitation + multi-kernel stem
                   + attention pooling) used for the Component-02 retrain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
NUM_LEADS = 12
NUM_CLASSES = 5
SIGNAL_LENGTH = 5000
SAMPLING_RATE = 500

# Baseline hyper-parameters — must not change or the shipped checkpoint breaks.
BASE_CHANNELS = [64, 128, 192, 256]
BASE_KERNELS = [15, 7, 5, 3]


# ══════════════════════════════════════════════════════════════════════════
#  BASELINE (checkpoint-compatible)
# ══════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """1D residual block.

    NOTE: the archive used an in-place `out += residual`. That is replaced with
    an out-of-place add — in-place ops on a tensor that a full-backward hook is
    registered on can corrupt Grad-CAM gradients. Numerically identical.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ECGResNet(nn.Module):
    """Baseline 1D ResNet — loads _archive/checkpoints_ecg_only/best_model.pt."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        blocks, in_ch = [], NUM_LEADS
        for i, (out_ch, ks) in enumerate(zip(BASE_CHANNELS, BASE_KERNELS)):
            blocks.append(ResidualBlock(in_ch, out_ch, ks, stride=2,
                                        dropout=0.1 if i < 2 else 0.2))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(BASE_CHANNELS[-1], 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def features(self, x):
        """(B, 12, 5000) -> (B, C, T') feature map. Grad-CAM hooks attach here."""
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).squeeze(-1))

    @property
    def cam_layer(self) -> nn.Module:
        return self.backbone[-1]


# ══════════════════════════════════════════════════════════════════════════
#  IMPROVED BACKBONE (Component-02 retrain)
# ══════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-excitation over channels — lets the net re-weight leads/filters."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(ch // r, 4))
        self.fc2 = nn.Linear(max(ch // r, 4), ch)

    def forward(self, x):
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(-1)


class SEResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + residual)


class MultiKernelStem(nn.Module):
    """Parallel receptive fields: P/QRS (short) and T/ST (long) live at different scales."""

    def __init__(self, in_ch=NUM_LEADS, out_ch=64):
        super().__init__()
        per = out_ch // 3
        self.b1 = nn.Conv1d(in_ch, per, 7, stride=2, padding=3, bias=False)
        self.b2 = nn.Conv1d(in_ch, per, 15, stride=2, padding=7, bias=False)
        self.b3 = nn.Conv1d(in_ch, out_ch - 2 * per, 31, stride=2, padding=15, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)))


class AttentionPool(nn.Module):
    """Global average pooling throws away *when* something happened. This keeps it."""

    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                       # (B, C, T)
        w = torch.softmax(self.score(x), dim=2)  # (B, 1, T)
        return (x * w).sum(dim=2), w.squeeze(1)


class ECGResNetSE(nn.Module):
    """Improved backbone. ~2.4M params — still small enough for a 45-min L4 run."""

    def __init__(self, num_classes: int = NUM_CLASSES, channels=(64, 128, 256, 320),
                 dropout: float = 0.3):
        super().__init__()
        self.stem = MultiKernelStem(NUM_LEADS, channels[0])
        blocks, in_ch = [], channels[0]
        for i, out_ch in enumerate(channels):
            ks = [11, 7, 5, 3][min(i, 3)]
            blocks.append(SEResidualBlock(in_ch, out_ch, ks, stride=2,
                                          dropout=0.1 if i < 2 else 0.2))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.attn_pool = AttentionPool(channels[-1])
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def features(self, x):
        return self.backbone(self.stem(x))

    def forward(self, x):
        pooled, _ = self.attn_pool(self.features(x))
        return self.head(pooled)

    @property
    def cam_layer(self) -> nn.Module:
        return self.backbone[-1]


# ══════════════════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Multi-label focal loss.

    IMPORTANT (audit finding C-6): do NOT combine a non-trivial `alpha` with a
    WeightedRandomSampler. The archive did both, correcting class imbalance
    twice and destroying calibration (HYP predicted at 4.14x its prevalence).
    Use alpha=None when a balanced sampler is active.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            e = self.label_smoothing
            targets = targets * (1 - e) + 0.5 * e
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            w = w * (targets * self.alpha + (1 - targets))
        return (w * bce).mean()


def build_model(name: str = "resnet_se", **kw) -> nn.Module:
    if name in ("resnet", "baseline", "ecgresnet"):
        return ECGResNet(**kw)
    if name in ("resnet_se", "se", "improved"):
        return ECGResNetSE(**kw)
    raise ValueError(f"unknown model '{name}'")
