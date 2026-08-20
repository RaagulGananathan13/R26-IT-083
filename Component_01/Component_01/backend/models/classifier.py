"""
Stage 5 classifier — ConvNeXt-Base, 8 pathologies.

Architecture verified against checkpoints/stage5/best.pt:
    classifier.0 = LayerNorm(1024)
    classifier.2 = Linear(1024, 512)
    classifier.5 = Linear(512, 8)

DO NOT rebuild this from torchvision's ConvNeXt with a swapped head.
torchvision applies `classifier` to the 4-D (B,1024,1,1) avgpool output; Stage 5
flattens FIRST. nn.LayerNorm(1024) on a 4-D tensor normalises the last dimension
(size 1) instead of the channels -- it does not raise, it silently returns wrong
numbers that still look like probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CXRClassifier(nn.Module):
    """ConvNeXt-Base + multi-label head. `features` is the Grad-CAM hook point."""

    def __init__(self, num_labels: int = 8, p_drop: float = 0.3):
        super().__init__()
        base = models.convnext_base(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        d = base.classifier[2].in_features                      # 1024
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(p_drop), nn.Linear(d, 512),
            nn.GELU(), nn.Dropout(p_drop * 0.66), nn.Linear(512, num_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


def load_classifier(weights_path, device, num_labels: int = 8) -> CXRClassifier:
    """Load Stage 5 strictly, then apply EMA weights.

    EMA is not optional. Stage 5 selected its checkpoint on, and reported
    0.8554 with, the EMA weights. Loading only `model` silently serves a
    different, worse model than the one every number in the write-up describes.
    """
    m = CXRClassifier(num_labels)
    ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    got = ck.get("pathologies")
    if got is not None:
        from backend.config import LABEL_COLS
        if list(got) != list(LABEL_COLS):
            raise RuntimeError(
                "pathology ORDER differs from config -- every column would be "
                "silently permuted.\n  checkpoint: %s\n  config    : %s"
                % (list(got), list(LABEL_COLS)))

    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model"].items()}
    m.load_state_dict(sd, strict=True)

    n_ema = 0
    if ck.get("ema"):
        msd = m.state_dict()
        missing = [k for k in ck["ema"] if k not in msd]
        if missing:
            raise RuntimeError("EMA keys absent from model: %s" % missing[:5])
        for k, v in ck["ema"].items():
            msd[k].copy_(v.to(msd[k].dtype))
            n_ema += 1

    print("[classifier] Stage 5 epoch %s | val %.4f | EMA weights applied: %d"
          % (ck.get("epoch"), ck.get("best_metric", float("nan")), n_ema))
    return m.to(device).eval()
