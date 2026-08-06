"""
COMPONENT_01 · STAGE 9B · LABEL-CONDITIONAL GRADIENT REVERSAL (reimplementation)
===============================================================================

Reimplements Pereira et al., MIDL 2023 (PMLR 227:1199-1210) on OUR data, OUR
backbone and OUR split, so the Stage 9A comparison stops being cross-dataset.

    L = L_disease(theta_fe, theta_d) + L_proj(theta_p) - lambda * L_proj(theta_fe)

The minus sign is implemented by a gradient-reversal layer between the pooled
feature vector and the projection head, NOT by negating a loss term. Negating
the loss would also flip the sign for theta_p, which must be trained NORMALLY
to stay a competent projection discriminator -- reversing both makes the
adversary useless and the whole method a no-op that still looks like it ran.

lambda = 0.1 matches Pereira's setup: they use lr 1e-4 for L_p(theta_p) and
1e-5 for the -L_p(theta_fe) component, a 0.1 ratio.

THE QUESTION THIS ANSWERS
-------------------------
Stage 9A showed TPR Disparity can be cut 73.3% by thresholding alone, with the
AUROC gap unchanged to 1e-12. Does the published ADVERSARIAL method move the
threshold-free gap, or does it too only move the manipulable metric?

Both answers are publishable:
  * gap unchanged  -> confirms Stage 9A; the prior method moved only the metric
  * gap reduced    -> the method genuinely works; 9A narrows to "metric overstates"

SAFETY
------
best.pt is opened READ-ONLY and never written. All outputs go to
checkpoints/stage9b/. The notebook asserts this and SHA-256 verifies the
checkpoint before and after.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
               "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]


# ====================================================================
# 1 · gradient reversal
# ====================================================================
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)          # identity forward

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward, negated-and-scaled gradient backward."""
    return _GradReverse.apply(x, lambd)


def lambda_at(step: int, total_steps: int, lambda_max: float = 0.1,
              warmup_frac: float = 0.15) -> float:
    """Ramp lambda 0 -> lambda_max over the first `warmup_frac` of training.

    Applying full adversarial pressure from step 0 is the standard way to make
    gradient reversal diverge: the projection head is still random, so its
    gradients are noise, and that noise is fed straight back into a pretrained
    feature extractor. Ganin & Lempitsky ramp for exactly this reason.
    """
    if total_steps <= 0:
        return lambda_max
    w = max(int(total_steps * warmup_frac), 1)
    return float(lambda_max * min(1.0, step / w))


# ====================================================================
# 2 · model
# ====================================================================
class CXRGradRev(nn.Module):
    """Stage 5's classifier, byte-compatible, plus a label-conditional
    projection adversary.

    `features` / `avgpool` / `classifier` replicate Stage 5 EXACTLY so
    best.pt loads with strict key matching on the disease path. Verified
    against the checkpoint: classifier.0=LayerNorm(1024),
    classifier.2=Linear(1024,512), classifier.5=Linear(512,8).

    DO NOT rebuild this from torchvision's ConvNeXt with a swapped head:
    torchvision applies `classifier` to the 4-D (B,1024,1,1) avgpool output,
    while Stage 5 flattens FIRST. nn.LayerNorm(1024) on a 4-D tensor
    normalises the last dim (size 1) instead of the channel dim -- it does not
    raise, it silently returns wrong numbers that still look like probabilities.
    """

    def __init__(self, n_path: int = 8, p_drop: float = 0.3,
                 emb_dim: int = 128, proj_hidden: int = 256):
        super().__init__()
        import torchvision
        base = torchvision.models.convnext_base(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        d = base.classifier[2].in_features                    # 1024
        self.feat_dim = d
        self.n_path = n_path

        # --- Stage 5 disease path (must stay bit-identical) ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(p_drop), nn.Linear(d, 512),
            nn.GELU(), nn.Dropout(p_drop * 0.66), nn.Linear(512, n_path))

        # --- NEW: label-conditional projection adversary (theta_p) ---
        # Pereira concatenate the disease-label embedding with the feature
        # vector so the adversary predicts P(AP | y_d) rather than P(AP),
        # which stops the alignment from erasing disease-correlated structure.
        self.label_emb = nn.Sequential(nn.Linear(n_path, emb_dim), nn.GELU())
        self.proj_head = nn.Sequential(
            nn.Linear(d + emb_dim, proj_hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(proj_hidden, 1))

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.features(x)).flatten(1)      # (B, 1024)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None,
                lambd: float = 0.0):
        """Returns (disease_logits, projection_logit).

        `y` is the ground-truth multi-hot label used ONLY during training.
        At inference it is None, and the label branch receives zeros so the
        adversary is measured label-free -- exactly Pereira's protocol, and
        the only way the reported projection AUC is comparable to a baseline
        that never had labels available.
        """
        feat = self.backbone(x)
        disease = self.classifier(feat)

        rev = grad_reverse(feat, lambd)                       # theta_fe only
        if y is None:
            y = torch.zeros(feat.size(0), self.n_path, device=feat.device,
                            dtype=feat.dtype)
        emb = self.label_emb(y.to(feat.dtype))
        proj = self.proj_head(torch.cat([rev, emb], dim=1)).squeeze(1)
        return disease, proj

    # ---------------- checkpoint plumbing ----------------
    def load_stage5(self, ckpt: dict, use_ema: bool = True) -> dict:
        """Load the Stage 5 disease path. The adversary stays freshly
        initialised. Raises on ANY unexpected key -- a silently partial load
        produces plausible-looking garbage."""
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in ckpt["model"].items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has keys this model lacks: {unexpected[:8]}")
        new = {m for m in missing}
        allowed = {n for n, _ in self.named_parameters()
                   if n.startswith(("label_emb", "proj_head"))}
        allowed |= {n for n, _ in self.named_buffers()
                    if n.startswith(("label_emb", "proj_head"))}
        stray = new - allowed
        if stray:
            raise RuntimeError(f"disease-path weights missing from checkpoint: {sorted(stray)[:8]}")
        if use_ema and ckpt.get("ema"):
            msd = self.state_dict()
            bad = [k for k in ckpt["ema"] if k not in msd]
            if bad:
                raise RuntimeError(f"EMA keys absent from model: {bad[:5]}")
            for k, v in ckpt["ema"].items():
                msd[k].copy_(v)
        return dict(loaded=len(sd), fresh=len(new))

    def param_groups(self, lr_backbone: float, lr_adv: float):
        """Pretrained weights move slowly; the randomly-initialised adversary
        needs a normal learning rate or it never becomes a real discriminator
        and the reversal has nothing meaningful to reverse."""
        pre, adv = [], []
        for n, p in self.named_parameters():
            (adv if n.startswith(("label_emb", "proj_head")) else pre).append(p)
        return [{"params": pre, "lr": lr_backbone},
                {"params": adv, "lr": lr_adv}]


# ====================================================================
# 3 · losses
# ====================================================================
class WeightedBCE(nn.Module):
    """Stage 5's disease loss: per-class pos_weight plus per-cell confidence
    weights (uncertain labels downweighted). Reused verbatim so the fine-tune
    optimises the same objective the checkpoint was trained on."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pw", pos_weight)

    def forward(self, logits, targets, cellw):
        l = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pw, reduction="none")
        return (l * cellw).sum() / cellw.sum().clamp(min=1.0)


def projection_loss(proj_logit: torch.Tensor, is_ap: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(proj_logit, is_ap.to(proj_logit.dtype))


# ====================================================================
# 4 · EMA (swap-based -- never deepcopy a model onto the GPU)
# ====================================================================
class EMA:
    """Shadow weights kept on CPU. `apply`/`restore` swap in place.

    Stage 5's first version called copy.deepcopy(model) every epoch, which
    allocates a second 89M-parameter model on the GPU right after the batch
    finder has already pushed VRAM to its limit. That is a guaranteed OOM
    hours into training.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().to("cpu", copy=True).float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, s in self.shadow.items():
            s.mul_(self.decay).add_(sd[k].detach().to("cpu", torch.float32),
                                    alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        sd = model.state_dict()
        self._backup = {k: sd[k].detach().to("cpu", copy=True) for k in self.shadow}
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None


def ema_decay_for(total_steps: int, window_frac: float = 0.05) -> float:
    """Decay whose averaging window is `window_frac` of the run.

    Stage 5 hard-coded 0.9998 for a run so short that 86% of the shadow was
    still initialisation at epoch 1, making the EMA checkpoint worse than the
    raw weights. Deriving it from run length removes that failure mode.
    """
    return float(np.clip(1.0 - 1.0 / max(1.0, window_frac * total_steps), 0.9, 0.9999))


# ====================================================================
# 5 · dataset
# ====================================================================
class CXRDataset(torch.utils.data.Dataset):
    """Returns (image, labels, cell_weights, is_AP)."""

    def __init__(self, df: pd.DataFrame, img_root, transform,
                 pathologies=PATHOLOGIES, uncertain_weight: float = 0.5):
        from pathlib import Path
        self.paths = [str(Path(img_root) / p) for p in df["image_path"]]
        self.y = df[pathologies].to_numpy(np.float32)
        w = np.ones_like(self.y)
        for j, k in enumerate(pathologies):
            c = k + "_uncertain"
            if c in df.columns:
                w[:, j] = np.where(df[c].to_numpy() > 0, uncertain_weight, 1.0)
        self.w = w.astype(np.float32)
        self.ap = (df["view"].astype(str).str.upper() == "AP").to_numpy(np.float32)
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return (self.tf(Image.open(self.paths[i])),
                torch.from_numpy(self.y[i]),
                torch.from_numpy(self.w[i]),
                torch.tensor(self.ap[i]))


# ====================================================================
# 6 · self-tests
# ====================================================================
def _selftest(verbose: bool = True) -> tuple[int, int]:
    P, F_ = [], []

    def g(name, ok, extra=""):
        (P if ok else F_).append(name)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{extra}")

    torch.manual_seed(0)

    # ---- gradient reversal -------------------------------------------------
    x = torch.ones(4, 6, requires_grad=True)
    grad_reverse(x, 0.1).sum().backward()
    g("GRL forward is identity", torch.equal(grad_reverse(x.detach(), 0.3), x.detach()))
    g("GRL negates and scales the gradient by lambda",
      torch.allclose(x.grad, torch.full_like(x, -0.1)), f"grad={x.grad[0,0].item():.3f}")
    x.grad = None
    grad_reverse(x, 0.0).sum().backward()
    g("lambda=0 blocks the adversarial gradient entirely",
      torch.allclose(x.grad, torch.zeros_like(x)))

    g("lambda ramps from 0", lambda_at(0, 1000, 0.1) == 0.0)
    g("lambda reaches its maximum after warmup",
      abs(lambda_at(1000, 1000, 0.1) - 0.1) < 1e-9)
    g("lambda is monotone during warmup",
      lambda_at(50, 1000, 0.1) < lambda_at(100, 1000, 0.1))

    # ---- model -------------------------------------------------------------
    m = CXRGradRev(8)
    xb = torch.randn(2, 3, 64, 64)
    yb = torch.randint(0, 2, (2, 8)).float()
    d, p = m(xb, yb, lambd=0.1)
    g("disease logits shape (B,8)", tuple(d.shape) == (2, 8), str(tuple(d.shape)))
    g("projection logit shape (B,)", tuple(p.shape) == (2,), str(tuple(p.shape)))
    d2, p2 = m(xb, None, lambd=0.0)
    g("label-free inference path runs", tuple(p2.shape) == (2,))
    # Must be checked in eval(): dropout is stochastic in train mode, so two
    # forward passes differ for reasons that have nothing to do with labels.
    m.eval()
    with torch.no_grad():
        dA, pA = m(xb, yb, lambd=0.1)
        dB, pB = m(xb, None, lambd=0.1)
    g("disease head ignores the label branch (no label leakage)",
      torch.allclose(dA, dB, atol=1e-6), "diagnosis identical with/without labels")
    g("adversary DOES use the label branch",
      not torch.allclose(pA, pB, atol=1e-6), "that is what makes it label-conditional")
    m.train()

    # head geometry must match the Stage 5 checkpoint exactly
    shapes = {n: tuple(t.shape) for n, t in m.classifier.state_dict().items()}
    g("classifier geometry matches best.pt",
      shapes.get("0.weight") == (1024,) and shapes.get("2.weight") == (512, 1024)
      and shapes.get("5.weight") == (8, 512), str(sorted(shapes)))

    # ---- checkpoint loading ------------------------------------------------
    ref = CXRGradRev(8)
    disease_only = {k: v for k, v in ref.state_dict().items()
                    if not k.startswith(("label_emb", "proj_head"))}
    info = m.load_stage5({"model": disease_only}, use_ema=False)
    g("loads a Stage-5-shaped checkpoint", info["loaded"] > 300, f"{info['loaded']} keys")
    g("adversary stays freshly initialised", info["fresh"] > 0, f"{info['fresh']} new keys")
    g("disease path actually matches after load",
      torch.allclose(m.classifier[2].weight, ref.classifier[2].weight))
    try:
        m.load_stage5({"model": {"nonexistent.key": torch.zeros(1)}}, use_ema=False)
        ok = False
    except RuntimeError:
        ok = True
    g("rejects an unexpected checkpoint key (fails loud)", ok)

    # ---- losses ------------------------------------------------------------
    crit = WeightedBCE(torch.ones(8))
    lg = torch.randn(4, 8, requires_grad=True)
    tg = torch.randint(0, 2, (4, 8)).float()
    cw = torch.ones(4, 8)
    l = crit(lg, tg, cw)
    g("disease loss finite and positive", torch.isfinite(l) and l.item() > 0,
      f"{l.item():.4f}")
    cw2 = cw.clone(); cw2[:, 0] = 0.0
    g("zero cell weight removes that label from the loss",
      not torch.allclose(crit(lg, tg, cw2), l))
    pw = WeightedBCE(torch.full((8,), 8.0))
    g("pos_weight changes the loss", not torch.allclose(pw(lg, tg, cw), l))

    lp = projection_loss(torch.randn(4), torch.tensor([1., 0., 1., 0.]))
    g("projection loss finite", torch.isfinite(lp) and lp.item() > 0, f"{lp.item():.4f}")

    # ---- the critical wiring test -----------------------------------------
    m.zero_grad()
    d, p = m(xb, yb, lambd=0.5)
    projection_loss(p, torch.tensor([1., 0.])).backward()
    gf = m.classifier[2].weight.grad
    gp = m.proj_head[0].weight.grad
    g("adversary receives a gradient", gp is not None and gp.abs().sum() > 0)
    g("projection loss does NOT touch the disease head",
      gf is None or gf.abs().sum() == 0,
      "theta_d must be updated by L_disease only")
    gb = m.features[0][0].weight.grad
    g("*** reversed gradient DOES reach the feature extractor ***",
      gb is not None and gb.abs().sum() > 0, f"|g|={gb.abs().sum():.3e}")

    m.zero_grad()
    d, p = m(xb, yb, lambd=0.0)
    projection_loss(p, torch.tensor([1., 0.])).backward()
    gb0 = m.features[0][0].weight.grad
    g("with lambda=0 the feature extractor is untouched by L_proj",
      gb0 is None or gb0.abs().sum() < 1e-12)

    # ---- EMA ---------------------------------------------------------------
    small = nn.Linear(4, 3)
    ema = EMA(small, 0.5)
    w0 = small.weight.detach().clone()
    with torch.no_grad():
        small.weight.add_(1.0)
    ema.update(small)
    ema.apply(small)
    g("EMA shadow sits between the old and new weights",
      bool(((small.weight > w0).all() & (small.weight < w0 + 1.0).all()).item()))
    ema.restore(small)
    g("EMA restore returns the live weights exactly",
      torch.allclose(small.weight, w0 + 1.0))
    g("EMA decay derived from run length (not hard-coded)",
      0.99 < ema_decay_for(20000) < 0.9999, f"{ema_decay_for(20000):.5f}")
    g("short runs get a faster decay than long runs",
      ema_decay_for(2000) < ema_decay_for(50000))

    if verbose:
        print(f"\n  {len(P)} passed, {len(F_)} failed")
        for f in F_:
            print(f"    - {f}")
    return len(P), len(F_)


if __name__ == "__main__":
    print("=" * 78)
    print(" STAGE 9B · stage9b_gradrev.py self-test")
    print("=" * 78)
    p, f = _selftest()
    raise SystemExit(1 if f else 0)
