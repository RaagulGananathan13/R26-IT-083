"""
xai.py — Thread-safe Grad-CAM, signed Integrated Gradients, and anatomical
localisation of lead attributions.

Fixes audit findings:
  C-5  the archive kept ONE module-level Grad-CAM object storing .activations
       and .gradients as instance state, under Flask's threaded server. Four
       concurrent calls were tested: 3 of 4 returned heatmaps that differed
       from the sequential result. Hooks are now created and removed inside a
       context manager per call, so no state is shared.
  7.4  compute_lead_saliency() applied .abs() before summing over time,
       discarding attribution sign — a lead arguing AGAINST the diagnosis was
       shown to the clinician as "important". Sign is now preserved and
       reported separately from magnitude.

Adds the machinery RESEARCH CONTRIBUTION 2 needs: mapping per-lead attributions
onto coronary territories so the explanation becomes report content rather than
a picture.
"""
from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .models import LEAD_NAMES

# Standard 12-lead territories. Lead order must match models.LEAD_NAMES.
TERRITORIES: Dict[str, List[str]] = {
    "anterior":      ["V1", "V2", "V3", "V4"],
    "lateral":       ["I", "aVL", "V5", "V6"],
    "inferior":      ["II", "III", "aVF"],
    "septal":        ["V1", "V2"],
    "anterolateral": ["V4", "V5", "V6", "I", "aVL"],
}
TERRITORY_ARTERY = {
    "anterior": "left anterior descending (LAD)",
    "septal": "proximal LAD / septal perforators",
    "lateral": "left circumflex (LCx)",
    "anterolateral": "LAD / LCx",
    "inferior": "right coronary artery (RCA)",
}


@dataclass
class Explanation:
    target_class: str
    cam: np.ndarray                                  # (T,) in [0,1], temporal
    cam_peaks_s: List[float] = field(default_factory=list)
    lead_signed: Dict[str, float] = field(default_factory=dict)   # % signed
    lead_magnitude: Dict[str, float] = field(default_factory=dict)  # % of |total|
    top_leads: List[str] = field(default_factory=list)
    territory: str | None = None
    territory_score: float = 0.0
    territory_artery: str | None = None


# ══════════════════════════════════════════════════════════════════════════
#  GRAD-CAM — no shared state
# ══════════════════════════════════════════════════════════════════════════
# One lock per model instance. Hooks are registered on a SHARED nn.Module, so
# per-call hook registration alone is not enough: thread A's hook still fires
# during thread B's forward pass and overwrites the captured activation. That is
# exactly the archive's bug (3 of 4 concurrent calls returned a corrupted map).
# Grad-CAM costs 60 ms, so serialising it is the right trade.
_MODEL_LOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _model_lock(model) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(model)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[model] = lock
        return lock


@contextmanager
def _hooks(layer):
    """Attach forward/backward hooks for the duration of ONE call, then remove."""
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("grad", go[0].detach()))
    try:
        yield store
    finally:
        h1.remove()
        h2.remove()


def grad_cam(model, x: torch.Tensor, class_idx: int, layer=None) -> np.ndarray:
    """1D Grad-CAM. Thread-safe: serialised per model instance.

    x: (1, 12, T) float tensor. Returns (T',) normalised to [0, 1].
    """
    layer = layer if layer is not None else model.cam_layer
    was_training = model.training
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    with _model_lock(model), _hooks(layer) as store:
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        act, grad = store["act"], store["grad"]
    w = grad.mean(dim=2, keepdim=True)
    cam = F.relu((w * act.detach()).sum(dim=1)).squeeze(0)
    denom = cam.max()
    cam = cam / denom if denom > 0 else torch.zeros_like(cam)
    if was_training:
        model.train()
    return cam.detach().cpu().numpy()


def integrated_gradients(model, x: torch.Tensor, class_idx: int,
                         steps: int = 64, baseline: torch.Tensor | None = None
                         ) -> np.ndarray:
    """Signed per-(lead, time) attribution. Returns (12, T).

    steps=64: the audit measured completeness error 1.3% at 30 steps and 0.2%
    at 100; 64 is the knee. Ranking is stable (Spearman 0.999 vs 200 steps).
    """
    was_training = model.training
    model.eval()
    base = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    # model.zero_grad() mutates shared parameter .grad buffers, so this loop is
    # serialised per model for the same reason grad_cam() is.
    with _model_lock(model):
        for i in range(1, steps + 1):
            s = (base + (float(i) / steps) * (x - base)).clone().detach().requires_grad_(True)
            out = model(s)
            model.zero_grad(set_to_none=True)
            out[0, class_idx].backward()
            total = total + s.grad.detach()
    attr = ((x - base) * (total / steps)).squeeze(0)     # (12, T), SIGNED
    if was_training:
        model.train()
    return attr.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════
#  LOCALISATION
# ══════════════════════════════════════════════════════════════════════════
def lead_attributions(attr: np.ndarray, lead_names: List[str] | None = None
                      ) -> tuple[Dict[str, float], Dict[str, float]]:
    """Collapse (12, T) attributions to per-lead signed % and magnitude %."""
    lead_names = lead_names or LEAD_NAMES
    signed = attr.sum(axis=1)                       # net evidence per lead
    magnitude = np.abs(attr).sum(axis=1)            # how much the lead mattered
    tot_mag = magnitude.sum()
    tot_signed = np.abs(signed).sum()
    signed_pct = {lead_names[i]: float(signed[i] / tot_signed * 100) if tot_signed > 0 else 0.0
                  for i in range(len(lead_names))}
    mag_pct = {lead_names[i]: float(magnitude[i] / tot_mag * 100) if tot_mag > 0 else 0.0
               for i in range(len(lead_names))}
    return signed_pct, mag_pct


def localise(signed_pct: Dict[str, float]) -> tuple[str | None, float]:
    """Which coronary territory do the SUPPORTING leads concentrate in?

    Only positive (supporting) attribution counts — a territory is not implicated
    by leads that argue against the finding.
    """
    pos = {k: max(v, 0.0) for k, v in signed_pct.items()}
    total = sum(pos.values())
    if total <= 0:
        return None, 0.0
    scores = {}
    for terr, leads in TERRITORIES.items():
        share = sum(pos.get(l, 0.0) for l in leads) / total
        # Normalise by territory size so 'anterolateral' (5 leads) is not favoured.
        scores[terr] = share / (len(leads) / 12.0)
    best = max(scores, key=scores.get)
    # Require meaningful concentration above a uniform baseline of 1.0.
    return (best, float(scores[best])) if scores[best] >= 1.35 else (None, float(scores[best]))


def cam_peaks(cam: np.ndarray, total_seconds: float = 10.0, top_k: int = 3,
              min_sep_s: float = 0.8) -> List[float]:
    """Times (seconds) of the strongest temporal activations."""
    if cam.size == 0 or cam.max() <= 0:
        return []
    t = np.linspace(0, total_seconds, len(cam))
    order = np.argsort(cam)[::-1]
    picked: List[float] = []
    for i in order:
        if cam[i] < 0.5 * cam.max():
            break
        ti = float(t[i])
        if all(abs(ti - p) >= min_sep_s for p in picked):
            picked.append(ti)
        if len(picked) >= top_k:
            break
    return sorted(picked)


def explain(model, x: torch.Tensor, class_idx: int, class_name: str,
            ig_steps: int = 64, seconds: float = 10.0) -> Explanation:
    """One call -> everything the report layer needs."""
    cam = grad_cam(model, x, class_idx)
    attr = integrated_gradients(model, x, class_idx, steps=ig_steps)
    signed, magnitude = lead_attributions(attr)
    terr, score = localise(signed)
    top = sorted(signed, key=lambda k: signed[k], reverse=True)[:3]
    return Explanation(
        target_class=class_name,
        cam=cam,
        cam_peaks_s=cam_peaks(cam, seconds),
        lead_signed=signed,
        lead_magnitude=magnitude,
        top_leads=[l for l in top if signed[l] > 0],
        territory=terr,
        territory_score=score,
        territory_artery=TERRITORY_ARTERY.get(terr) if terr else None,
    )


# ══════════════════════════════════════════════════════════════════════════
#  FAITHFULNESS (the evaluation contribution)
# ══════════════════════════════════════════════════════════════════════════
def deletion_insertion_auc(model, x: torch.Tensor, class_idx: int,
                           cam: np.ndarray,
                           fractions=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
                           rng: np.random.Generator | None = None) -> Dict[str, float]:
    """Deletion / insertion faithfulness for one record, vs a random baseline.

    Returns deletion & insertion AUCs and the IMD summary (insertion - deletion;
    higher is more faithful).
    """
    rng = rng or np.random.default_rng(0)
    T = x.shape[-1]
    cam_full = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(cam)), cam)
    order = np.argsort(cam_full)[::-1]
    rand_order = rng.permutation(T)
    src = x.squeeze(0).cpu().numpy()

    def p(arr):
        with torch.no_grad():
            return float(torch.sigmoid(model(torch.from_numpy(arr[None]).float()))[0, class_idx])

    d_c, d_r, i_c, i_r = [], [], [], []
    for f in fractions:
        n = int(f * T)
        a = src.copy()
        if n:
            a[:, order[:n]] = 0.0
        d_c.append(p(a))
        a = src.copy()
        if n:
            a[:, rand_order[:n]] = 0.0
        d_r.append(p(a))
        a = np.zeros_like(src)
        if n:
            a[:, order[:n]] = src[:, order[:n]]
        i_c.append(p(a))
        a = np.zeros_like(src)
        if n:
            a[:, rand_order[:n]] = src[:, rand_order[:n]]
        i_r.append(p(a))

    trapz = getattr(np, "trapezoid", np.trapz)
    out = dict(deletion_auc=float(trapz(d_c, fractions)),
               deletion_auc_random=float(trapz(d_r, fractions)),
               insertion_auc=float(trapz(i_c, fractions)),
               insertion_auc_random=float(trapz(i_r, fractions)))
    out["imd"] = out["insertion_auc"] - out["deletion_auc"]
    out["imd_random"] = out["insertion_auc_random"] - out["deletion_auc_random"]
    return out


def qrs_alignment(cam: np.ndarray, r_peaks: np.ndarray, n_samples: int,
                  fs: int = 500, window_ms: float = 120.0) -> float:
    """Fraction of Grad-CAM mass falling inside QRS complexes.

    The clinical sanity check nobody has run for PTB-XL: a conduction-disturbance
    explanation should concentrate on QRS; an ST/T explanation should not.
    """
    if cam.size == 0 or len(r_peaks) == 0:
        return float("nan")
    cam_full = np.interp(np.linspace(0, 1, n_samples), np.linspace(0, 1, len(cam)), cam)
    mask = np.zeros(n_samples, dtype=bool)
    half = int(window_ms / 1000.0 * fs / 2)
    for r in r_peaks:
        mask[max(0, r - half):min(n_samples, r + half)] = True
    total = cam_full.sum()
    if total <= 0:
        return float("nan")
    return float(cam_full[mask].sum() / total)
