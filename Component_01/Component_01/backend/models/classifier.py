"""
The cardiomegaly classifier (ConvNeXt-Base, 8 pathologies).

This has to match the saved checkpoint exactly:
    classifier.0 = LayerNorm(1024)
    classifier.2 = Linear(1024, 512)
    classifier.5 = Linear(512, 8)

IMPORTANT: don't rebuild this by taking torchvision's ConvNeXt and swapping the
head. torchvision keeps the pooled output as (B, 1024, 1, 1) and we flatten it
first. If you skip the flatten, LayerNorm(1024) ends up normalising a dimension
of size 1. It won't crash. You just get wrong numbers that still look like
probabilities, which is much worse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CXRClassifier(nn.Module):
    """ConvNeXt-Base with a small multi-label head on top.

    Grad-CAM hooks into `features`, so don't rename it.
    """

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
        # look at the image -> pool it down -> flatten -> predict 8 scores
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


def load_classifier(weights_path, device, num_labels: int = 8) -> CXRClassifier:
    """Load the trained weights, then overwrite them with the EMA copy.

    The EMA step is not optional. Training picked the best checkpoint using the
    EMA weights, and every result we report comes from those. If you load only
    the "model" weights you get a different (worse) model with no warning.
    """
    m = CXRClassifier(num_labels)
    ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    # The label order in the checkpoint must match config. If it doesn't, every
    # column would be shifted and the predictions would be quietly wrong.
    got = ck.get("pathologies")
    if got is not None:
        from backend.config import LABEL_COLS
        if list(got) != list(LABEL_COLS):
            raise RuntimeError(
                "pathology ORDER differs from config -- every column would be "
                "silently permuted.\n  checkpoint: %s\n  config    : %s"
                % (list(got), list(LABEL_COLS)))

    # Training used DataParallel, which prefixes every key with "module."
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model"].items()}
    m.load_state_dict(sd, strict=True)

    # Now swap in the EMA weights on top.
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
