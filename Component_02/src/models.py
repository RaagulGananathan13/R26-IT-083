"""
models.py — SINGLE SOURCE OF TRUTH for every network architecture.

Fixes audit finding E-10: the _archive/ code duplicated the same ResNet in five
files (app.py, predict.py, evaluate_hybrid.py, train_ecg_only.py,
train_report_gen.py). Any edit had to be made five times or inference silently
diverged from training. Everything now imports from here.

Architectures are published through MODEL_REGISTRY rather than an if/elif chain,
so every consumer (training, calibration, audit, serving) names a model the same
way and can enumerate what exists:

  resnet            — bit-exact reimplementation of the shipped baseline so that
                      reference/checkpoints_ecg_only/best_model.pt loads unchanged.
  resnet_se         — the improved backbone (squeeze-excitation + multi-kernel
                      stem + attention pooling) used for the Component-02 retrain.
  resnet_se_no_se     resnet_se_no_stem  |  component-wise ablations of resnet_se. Each disables ONE
  resnet_se_no_attn  |  component and holds depth, width and training recipe
  resnet_se_plain   /   fixed, so the delta is attributable to that component.

The ablation variants exist because the shipped architecture carried the
headline result with no component-wise evidence that any of its three additions
helped (writeup gap S5). They are constructed from the same class via flags, so
there is still exactly one definition of the network.

Use `build_model(name)` to construct and `list_models()` / `describe_models()`
to enumerate. `resolve_model_name()` maps aliases to canonical names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Residual block with an optional squeeze-excitation branch.

    `use_se=False` swaps SE for an identity, which removes its parameters from
    the state_dict — that is the point: the ablation must not silently keep the
    parameter budget it is supposed to be testing.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1,
                 use_se: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
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


class SingleKernelStem(nn.Module):
    """Ablation counterpart of MultiKernelStem: one receptive field, same width.

    Kernel 15 is the middle of the 7/15/31 bank, so the comparison isolates
    *multi-scale* rather than *scale*.
    """

    def __init__(self, in_ch=NUM_LEADS, out_ch=64, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=2,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class AttentionPool(nn.Module):
    """Global average pooling throws away *when* something happened. This keeps it."""

    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                       # (B, C, T)
        w = torch.softmax(self.score(x), dim=2)  # (B, 1, T)
        return (x * w).sum(dim=2), w.squeeze(1)


class GlobalAvgPool(nn.Module):
    """Ablation counterpart of AttentionPool.

    Returns the same `(pooled, weights)` pair so callers — and the ablation
    variants — need no branching. The weights are uniform, which is exactly
    what average pooling assumes.
    """

    def forward(self, x):                       # (B, C, T)
        w = x.new_full((x.shape[0], x.shape[2]), 1.0 / x.shape[2])
        return x.mean(dim=2), w


class ECGResNetSE(nn.Module):
    """Improved backbone — 1.58M params, small enough for a 45-min L4 run.

    The three flags select the shipped model or one of its ablations. Defaults
    reproduce the shipped architecture exactly, so `checkpoints/best_model.pt`
    loads with `strict=True` and zero missing keys; do not change them.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, channels=(64, 128, 256, 320),
                 dropout: float = 0.3, use_se: bool = True,
                 use_multikernel_stem: bool = True,
                 use_attention_pool: bool = True):
        super().__init__()
        self.use_se = use_se
        self.use_multikernel_stem = use_multikernel_stem
        self.use_attention_pool = use_attention_pool

        self.stem = (MultiKernelStem(NUM_LEADS, channels[0]) if use_multikernel_stem
                     else SingleKernelStem(NUM_LEADS, channels[0]))
        blocks, in_ch = [], channels[0]
        for i, out_ch in enumerate(channels):
            ks = [11, 7, 5, 3][min(i, 3)]
            blocks.append(SEResidualBlock(in_ch, out_ch, ks, stride=2,
                                          dropout=0.1 if i < 2 else 0.2,
                                          use_se=use_se))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.attn_pool = (AttentionPool(channels[-1]) if use_attention_pool
                          else GlobalAvgPool())
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


# ══════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════
# One place that knows what architectures exist. Consumers (train_gpu.py,
# fit_calibration.py, the audits, the zoo, the API) resolve names through here
# instead of hard-coding a `choices=[...]` list that drifts out of date.

DEFAULT_MODEL = "resnet_se"


@dataclass(frozen=True)
class ModelSpec:
    """Everything the rest of the system needs to know about an architecture."""

    name: str
    factory: Callable[..., nn.Module]
    description: str
    family: str = "resnet"
    aliases: Tuple[str, ...] = ()
    #: name of the model this is a component-wise ablation of, if any
    ablation_of: Optional[str] = None
    #: which component the ablation removes — for ablation tables
    ablates: Optional[str] = None
    #: default constructor kwargs baked into this variant
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, **overrides) -> nn.Module:
        return self.factory(**{**self.kwargs, **overrides})

    @property
    def is_ablation(self) -> bool:
        return self.ablation_of is not None


MODEL_REGISTRY: Dict[str, ModelSpec] = {}
_ALIAS_TO_NAME: Dict[str, str] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Add an architecture to the registry.

    Raises on a duplicate name or a colliding alias — a silent overwrite here
    would mean a checkpoint loading into the wrong network.
    """
    if spec.name in MODEL_REGISTRY:
        raise ValueError(f"model '{spec.name}' is already registered")
    for key in (spec.name, *spec.aliases):
        owner = _ALIAS_TO_NAME.get(key)
        if owner is not None and owner != spec.name:
            raise ValueError(
                f"alias '{key}' for model '{spec.name}' already points at '{owner}'")
    MODEL_REGISTRY[spec.name] = spec
    for key in (spec.name, *spec.aliases):
        _ALIAS_TO_NAME[key] = spec.name
    return spec


def resolve_model_name(name: str) -> str:
    """Canonical registry name for `name`, accepting aliases. Raises if unknown."""
    key = (name or "").strip().lower()
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    raise ValueError(
        f"unknown model '{name}'. Available: "
        + ", ".join(sorted(MODEL_REGISTRY))
        + ".\nRun `python -m src.models` for a description of each.")


def get_spec(name: str) -> ModelSpec:
    return MODEL_REGISTRY[resolve_model_name(name)]


def build_model(name: str = DEFAULT_MODEL, **kw) -> nn.Module:
    """Construct an architecture by registry name or alias.

    `kw` overrides the variant's baked-in kwargs, so
    `build_model("resnet_se", num_classes=3)` works while
    `build_model("resnet_se_no_se")` keeps `use_se=False`.
    """
    return get_spec(name).build(**kw)


def list_models(family: Optional[str] = None,
                include_ablations: bool = True) -> List[ModelSpec]:
    """Registered specs, deployable models first, then ablations."""
    out = [s for s in MODEL_REGISTRY.values()
           if (family is None or s.family == family)
           and (include_ablations or not s.is_ablation)]
    return sorted(out, key=lambda s: (s.is_ablation, s.name))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def describe_models(with_params: bool = True) -> str:
    """Human-readable table — used by `--list-models` on the training CLIs."""
    rows = []
    for spec in list_models():
        params = ""
        if with_params:
            try:
                params = f"{count_parameters(spec.build()):>10,}"
            except Exception:                    # pragma: no cover - defensive
                params = "         ?"
        tag = f"ablation: -{spec.ablates}" if spec.is_ablation else "deployable"
        rows.append(f"  {spec.name:<20} {params}  {tag:<28} {spec.description}")
    header = f"  {'name':<20} {'params':>10}  {'role':<28} description"
    return "\n".join([header, "  " + "-" * 96, *rows])


register_model(ModelSpec(
    name="resnet",
    factory=ECGResNet,
    description="Progress-1 baseline 1D ResNet (checkpoint-compatible).",
    aliases=("baseline", "ecgresnet"),
))

register_model(ModelSpec(
    name="resnet_se",
    factory=ECGResNetSE,
    description="Shipped model: SE + multi-kernel stem + attention pooling.",
    aliases=("se", "improved", "ecgresnetse"),
))

for _name, _alias, _ablates, _kw, _desc in [
    ("resnet_se_no_se", "no_se", "squeeze-excitation",
     dict(use_se=False), "resnet_se without the SE channel re-weighting."),
    ("resnet_se_no_stem", "no_stem", "multi-kernel stem",
     dict(use_multikernel_stem=False), "resnet_se with a single 15-tap stem."),
    ("resnet_se_no_attn", "no_attn", "attention pooling",
     dict(use_attention_pool=False), "resnet_se with global average pooling."),
    ("resnet_se_plain", "plain", "all three additions",
     dict(use_se=False, use_multikernel_stem=False, use_attention_pool=False),
     "resnet_se depth/width only — the floor the three additions build on."),
]:
    register_model(ModelSpec(
        name=_name, factory=ECGResNetSE, description=_desc,
        aliases=(_alias,), ablation_of="resnet_se", ablates=_ablates, kwargs=_kw,
    ))
del _name, _alias, _ablates, _kw, _desc


if __name__ == "__main__":                       # python -m src.models
    print(describe_models())
