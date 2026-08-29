"""
Robustness analysis: acquisition-subgroup breakdown and paired significance tests.
=================================================================================

Two things a reviewer asks that aggregate metrics cannot answer.

1. **Does performance hold across acquisition conditions?**
   EchoNet-Dynamic ships no patient demographics (FileList.csv carries only
   FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split),
   so demographic fairness cannot be assessed on this cohort and is not claimed.
   What *is* available are acquisition characteristics -- frame rate, spatial
   resolution, recording length -- and ventricular volumes.  Systematic failure
   on one acquisition setting is a real deployment risk, so it is measured here.

2. **Is a difference between two configurations real, or sampling noise?**
   Comparisons in this project share one test split, so the correct instrument
   is a *paired* test on per-study outcomes, not two independent intervals.  A
   paired bootstrap over studies and an exact McNemar test on the discordant
   pairs are both provided.

Both functions consume per-study prediction arrays saved by ``run_ensemble.py``
/ ``run_eval.py`` (``--save-predictions``), so no re-inference is needed once a
run has been evaluated.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from engine.metrics import classification_metrics, regression_metrics


# --------------------------------------------------------------------------- #
#  Subgroup analysis                                                          #
# --------------------------------------------------------------------------- #
def quantile_bins(values, n_bins: int = 3, labels: Optional[Sequence[str]] = None):
    """Split a continuous covariate into roughly equal-sized bins.

    Equal-frequency rather than equal-width, so every subgroup carries enough
    studies to support an interval.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    edges = np.quantile(v[finite], np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # np.unique guards against duplicate edges when a covariate is highly discrete.
    edges = np.unique(edges)
    idx = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    idx[~finite] = -1
    if labels is None:
        labels = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(len(edges) - 1)]
    return idx, list(labels)


def subgroup_report(y_true, y_pred, ef_true, ef_pred, covariates: Mapping[str, np.ndarray],
                    n_classes: int, n_bins: int = 3, min_n: int = 30) -> dict:
    """Per-subgroup metrics for each supplied covariate.

    `covariates` maps a name to a per-study value array (same order as y_true).
    Subgroups smaller than `min_n` are reported but flagged, since their
    intervals are too wide to support a conclusion.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    ef_true = np.asarray(ef_true, dtype=np.float64)
    ef_pred = np.asarray(ef_pred, dtype=np.float64)

    out = {}
    for name, values in covariates.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != y_true.shape:
            raise ValueError(f"covariate {name!r} has shape {values.shape}, expected {y_true.shape}")
        idx, labels = quantile_bins(values, n_bins=n_bins)
        groups = []
        for g, label in enumerate(labels):
            m = idx == g
            n = int(m.sum())
            if n == 0:
                continue
            cls = classification_metrics(y_true[m], y_pred[m], n_classes)
            reg = regression_metrics(ef_true[m], ef_pred[m])
            groups.append({
                "label": label,
                "n": n,
                "underpowered": bool(n < min_n),
                "mae": float(reg["mae"]),
                "overall_acc": float(cls["overall_acc"]),
                "balanced_acc": float(cls["balanced_acc"]),
                "min_class_recall": float(cls["min_class_recall"]),
                "per_class_recall": [None if r is None else float(r)
                                     for r in cls["per_class_recall"]],
                "class_support": np.bincount(y_true[m], minlength=n_classes).tolist(),
            })
        powered = [g for g in groups if not g["underpowered"]]
        spread = {}
        if len(powered) >= 2:
            for key in ("mae", "overall_acc", "balanced_acc"):
                vals = [g[key] for g in powered]
                spread[f"{key}_range"] = float(max(vals) - min(vals))
                spread[f"{key}_min"] = float(min(vals))
                spread[f"{key}_max"] = float(max(vals))
        out[name] = {"groups": groups, "spread_over_powered_groups": spread}
    return out


# --------------------------------------------------------------------------- #
#  Paired significance testing                                                #
# --------------------------------------------------------------------------- #
def paired_bootstrap(metric_fn, n: int, *, n_boot: int = 10000, seed: int = 1337,
                     alpha: float = 0.05) -> dict:
    """Paired bootstrap over study indices for an arbitrary metric difference.

    `metric_fn(idx)` must return the scalar difference (system A minus system B)
    computed on the resampled index array.  Resampling the *same* indices for
    both systems preserves the pairing, which is what makes the interval valid
    for a shared test split.
    """
    rng = np.random.default_rng(seed)
    observed = float(metric_fn(np.arange(n)))
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric_fn(idx)
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Two-sided bootstrap p-value: proportion of resamples on the far side of 0,
    # doubled, with the +1 correction so p is never exactly 0.
    if observed >= 0:
        tail = float((diffs <= 0).sum())
    else:
        tail = float((diffs >= 0).sum())
    p = min(1.0, 2.0 * (tail + 1.0) / (n_boot + 1.0))
    return {
        "observed_difference": observed,
        "ci_lower": float(lo),
        "ci_hi": float(hi),
        "ci_level": 1.0 - alpha,
        "p_value_two_sided": float(p),
        "n_bootstrap": int(n_boot),
        "significant_at_alpha": bool(p < alpha),
        "note": "paired bootstrap over studies; both systems resampled on identical indices",
    }


def compare_systems(y_true, pred_a, pred_b, ef_true, ef_a, ef_b, n_classes: int,
                    *, n_boot: int = 10000, seed: int = 1337, alpha: float = 0.05) -> dict:
    """Paired comparison of two systems on the same studies."""
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    ef_true = np.asarray(ef_true, dtype=np.float64)
    ef_a = np.asarray(ef_a, dtype=np.float64)
    ef_b = np.asarray(ef_b, dtype=np.float64)
    n = len(y_true)
    for arr, nm in ((pred_a, "pred_a"), (pred_b, "pred_b"), (ef_a, "ef_a"), (ef_b, "ef_b")):
        if len(arr) != n:
            raise ValueError(f"{nm} length {len(arr)} != {n}")

    err_a = np.abs(ef_a - ef_true)
    err_b = np.abs(ef_b - ef_true)
    ok_a = (pred_a == y_true).astype(np.float64)
    ok_b = (pred_b == y_true).astype(np.float64)

    results = {
        "n": int(n),
        "mae_a": float(err_a.mean()),
        "mae_b": float(err_b.mean()),
        "accuracy_a": float(ok_a.mean()),
        "accuracy_b": float(ok_b.mean()),
    }

    results["mae_difference"] = paired_bootstrap(
        lambda idx: float(err_a[idx].mean() - err_b[idx].mean()),
        n, n_boot=n_boot, seed=seed, alpha=alpha)
    results["accuracy_difference"] = paired_bootstrap(
        lambda idx: float(ok_a[idx].mean() - ok_b[idx].mean()),
        n, n_boot=n_boot, seed=seed, alpha=alpha)

    def _balanced(idx, pred):
        m = classification_metrics(y_true[idx], pred[idx], n_classes)
        return float(m["balanced_acc"])

    results["balanced_accuracy_difference"] = paired_bootstrap(
        lambda idx: _balanced(idx, pred_a) - _balanced(idx, pred_b),
        n, n_boot=max(1000, n_boot // 5), seed=seed, alpha=alpha)

    results["mcnemar"] = mcnemar_exact(ok_a.astype(bool), ok_b.astype(bool))
    return results


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on discordant classification outcomes.

    Exact rather than chi-squared, because the discordant count here is small
    enough that the asymptotic approximation is not reliable.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(a & ~b))          # A right, B wrong
    b10 = int(np.sum(~a & b))          # A wrong, B right
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p_value_two_sided": 1.0,
                "significant_at_0.05": False, "note": "systems never disagree"}
    k = min(b01, b10)
    # Two-sided exact binomial p under H0: p = 0.5.
    cdf = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2.0 ** n_disc)
    p = min(1.0, 2.0 * cdf)
    return {
        "b01_a_right_b_wrong": b01,
        "b10_a_wrong_b_right": b10,
        "n_discordant": n_disc,
        "p_value_two_sided": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "test": "exact binomial McNemar",
    }
