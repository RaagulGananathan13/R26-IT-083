"""
UEF-Net :: Uncertainty-aware Ordinal Ejection-Fraction Network
==============================================================
Shared 18-layer spatio-temporal backbone (Kinetics-pretrained), with the stem
adapted to N-channel (grayscale + motion) input,
feeding two heads.  R(2+1)D-18 is the default and EchoNet's proven choice;
r3d_18 and mc3_18 are selectable via --backbone so the choice can be
tested rather than assumed.  Heads:

  * Regression head  -> standardized EF  (drives MAE)
  * CORAL ordinal head -> K-1 rank-consistent cumulative logits P(y>k)
                          (respects EF ordering; drives balanced classification)

Rank consistency (CORAL, Cao et al. 2020) comes from a single shared projection
plus K-1 independent biases, guaranteeing P(y>0) >= P(y>1) >= ... per sample.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """Legacy head retained so the two existing v1 checkpoints still load."""
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_classes - 1).float())

    def forward(self, x):                      # x: (B, feat)
        return self.proj(x) + self.bias        # (B, K-1)


class OrderedCoralHead(nn.Module):
    """Cumulative ordinal head with structurally ordered cut-points.

    A single severity score is compared with increasing learned cut-points.  This
    guarantees P(y>0) >= P(y>1) >= ... instead of repairing invalid probabilities
    after the forward pass.
    """

    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError("ordinal regression needs at least two classes")
        self.score = nn.Linear(feat_dim, 1)
        self.anchor = nn.Parameter(torch.tensor(-1.0))
        gap_init = math.log(math.expm1(1.0))
        self.raw_gaps = nn.Parameter(torch.full((n_classes - 2,), gap_init))

    def cutpoints(self) -> torch.Tensor:
        if self.raw_gaps.numel() == 0:
            return self.anchor.reshape(1)
        gaps = F.softplus(self.raw_gaps) + 1e-4
        return torch.cat([self.anchor.reshape(1), self.anchor + torch.cumsum(gaps, 0)])

    def forward(self, x):
        return self.score(x) - self.cutpoints().unsqueeze(0)


#: Backbones this network can be built on.  All three are 18-layer, Kinetics-400
#: pretrained, expose the same `stem[0]` conv and the same 512-d `fc` input, so
#: the stem adaptation and every head below work unchanged across them.
#:
#: r2plus1d_18  the shipped choice.  Factorises each 3-D convolution into a
#:              spatial (1,d,d) then temporal (t,1,1) step, which matches how
#:              ejection fraction is defined -- a temporal quantity computed
#:              from spatial structure.  Ouyang et al. found it best for EF on
#:              this dataset, and keeping it makes our MAE comparable to theirs.
#: r3d_18       full 3-D convolution: the un-factorised baseline R(2+1)D was
#:              designed to beat (Tran et al., CVPR 2018).  Nearly matched in
#:              capacity (33.4 M vs 31.5 M), so a comparison isolates the
#:              factorisation rather than confounding it with model size.
#: mc3_18       mixed 3-D early / 2-D late.  Included for completeness, but at
#:              11.7 M parameters a comparison against it conflates architecture
#:              with capacity and should be read with that in mind.
_BACKBONES = {
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights"),
    "r3d_18": ("r3d_18", "R3D_18_Weights"),
    "mc3_18": ("mc3_18", "MC3_18_Weights"),
}


def _build_backbone(cfg, legacy_stem: bool = False):
    import torchvision.models.video as tvv

    backbone = getattr(cfg, "backbone", "r2plus1d_18")
    if backbone not in _BACKBONES:
        raise ValueError(
            f"unsupported backbone={backbone!r}; expected one of "
            f"{sorted(_BACKBONES)}")
    factory_name, weights_name = _BACKBONES[backbone]
    factory = getattr(tvv, factory_name)
    weights_enum = getattr(tvv, weights_name)

    weights = weights_enum.KINETICS400_V1 if cfg.pretrained else None
    try:
        m = factory(weights=weights)
        loaded = weights is not None
    except Exception as e:                     # offline / download blocked
        if weights is not None and getattr(cfg, "require_pretrained", False):
            raise RuntimeError(
                "Kinetics weights are required by this run but could not be loaded. "
                "Download/cache them first or explicitly use --no-pretrained for a smoke test."
            ) from e
        print(f"[model][WARN] pretrained weights unavailable ({e}); random init.")
        m = factory(weights=None)
        loaded = False

    # adapt stem conv (3->in_channels) preserving pretrained response magnitude
    old = m.stem[0]
    new = nn.Conv3d(cfg.in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        w = old.weight                          # (out, 3, kt, kh, kw)
        mean_w = w.mean(dim=1, keepdim=True)     # (out, 1, ...)
        if legacy_stem:
            # Exact v1 behaviour for old checkpoints.
            new.weight.copy_(mean_w.repeat(1, cfg.in_channels, 1, 1, 1))
        else:
            new.weight.zero_()
            # Channel zero is grayscale. Summing RGB kernels makes a one-channel
            # grayscale input equivalent to repeating it over the original RGB stem.
            new.weight[:, 0].copy_(w.sum(dim=1))
            # Optional derived channels start as a small residual and are learned.
            if cfg.in_channels > 1:
                scale = float(getattr(cfg, "aux_channel_init_scale", 0.05))
                for c in range(1, cfg.in_channels):
                    new.weight[:, c].copy_(mean_w[:, 0] * scale)
        if new.bias is not None:
            if old.bias is None:
                new.bias.zero_()
            else:
                new.bias.copy_(old.bias)
    m.stem[0] = new

    feat_dim = m.fc.in_features                  # 512 for all three
    m.fc = nn.Identity()
    return m, feat_dim, loaded


class UEFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_version = getattr(cfg, "model_version", "uefnet_v1")
        self.is_v2 = self.model_version != "uefnet_v1"
        self.backbone, feat_dim, self.pretrained_loaded = _build_backbone(
            cfg, legacy_stem=not self.is_v2)
        self.dropout = nn.Dropout(cfg.dropout)
        self.reg_head = nn.Linear(feat_dim, 1)
        self.ord_head = (OrderedCoralHead(feat_dim, cfg.n_classes)
                         if self.is_v2 else CoralHead(feat_dim, cfg.n_classes))
        self.class_head = (nn.Linear(feat_dim, cfg.n_classes)
                           if self.is_v2 and getattr(cfg, "use_class_head", True) else None)
        self.log_var_head = (nn.Linear(feat_dim, 1)
                             if self.is_v2 and getattr(cfg, "predict_uncertainty", True) else None)

    def forward(self, x):
        f = self.backbone(x)                     # (B, feat)
        f = self.dropout(f)
        ef_z = self.reg_head(f).squeeze(1)       # (B,)
        ord_logits = self.ord_head(f)            # (B, K-1)
        if not self.is_v2:
            return ef_z, ord_logits, f
        aux = {
            "features": f,
            "class_logits": self.class_head(f) if self.class_head is not None else None,
            "log_var": (self.log_var_head(f).squeeze(1).clamp(-6.0, 4.0)
                        if self.log_var_head is not None else None),
        }
        return ef_z, ord_logits, aux

    def param_groups(self, lr_backbone, lr_head, weight_decay):
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        bb, hd = [], []
        for p in self.parameters():
            (bb if id(p) in backbone_ids else hd).append(p)
        return [
            {"params": bb, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": hd, "lr": lr_head, "weight_decay": weight_decay},
        ]


# ------------------------------- inference helpers ------------------------- #
def coral_probs(ord_logits: torch.Tensor) -> torch.Tensor:
    """P(y>k) for each threshold k -> (B, K-1)."""
    return torch.sigmoid(ord_logits)


def coral_predict(ord_logits: torch.Tensor) -> torch.Tensor:
    """Predicted class rank = number of thresholds passed (P(y>k) > 0.5)."""
    return (coral_probs(ord_logits) > 0.5).sum(dim=1)


def coral_class_distribution(ord_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert cumulative P(y>k) into a proper per-class distribution (B, K):
        P(y=0) = 1 - P(y>0)
        P(y=k) = P(y>k-1) - P(y>k)
        P(y=K-1) = P(y>K-2)
    """
    p = coral_probs(ord_logits)                  # (B, K-1)
    B, Km1 = p.shape
    ones = torch.ones(B, 1, device=p.device, dtype=p.dtype)
    zeros = torch.zeros(B, 1, device=p.device, dtype=p.dtype)
    p_gt = torch.cat([ones, p], dim=1)           # P(y> -1)=1 ... P(y>K-2)
    p_gt_next = torch.cat([p, zeros], dim=1)     # P(y>0) ... P(y>K-1)=0
    dist = (p_gt - p_gt_next).clamp(min=0.0)     # (B, K)
    return dist / dist.sum(dim=1, keepdim=True).clamp(min=1e-6)
