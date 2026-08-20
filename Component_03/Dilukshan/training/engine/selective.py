"""
Selective prediction (classification with a reject option) for EF severity grading.
==================================================================================

Motivation
----------
Residual error in this task is concentrated at the clinical decision boundaries.
Roughly 37% of studies lie within one mean-absolute-error of a boundary at 30, 40
or 55 EF points, and reported inter-observer variability is of the same order.
For those studies the label itself is not reliably determined, so forcing a
single grade is neither necessary nor clinically desirable.

A selective classifier is permitted to abstain.  Following Chow's rule, the model
returns a grade when its confidence exceeds a threshold and otherwise defers the
study for specialist review.  Recall is then reported *together with coverage*,
the fraction of studies the system chose to grade.  Reporting one without the
other would be meaningless, so every function here returns both.

Uncertainty signals
-------------------
All signals are derived from the model, not hand-designed:

  aleatoric      the learned predictive standard deviation from the log-variance
                 head, trained by the Gaussian negative log-likelihood term
  epistemic      disagreement between the deterministic test-time clips of the
                 same study, and between ensemble members
  total          sqrt(aleatoric^2 + epistemic^2), by the law of total variance
  boundary       |EF - nearest clinical boundary| / total   -- how many standard
                 deviations separate the prediction from the decision it changes
  margin         gap between the two largest class probabilities
  entropy        Shannon entropy of the predicted class distribution

The `boundary` signal is the clinically motivated one: a study is uncertain
precisely when its prediction interval straddles a threshold that would change
the grade.

Protocol
--------
The signal and its threshold are selected on validation data and then frozen.
The test split is scored once with the frozen rule.  No test data participates
in any fitting decision.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np

from engine.metrics import classification_metrics


# --------------------------------------------------------------------------- #
#  Uncertainty signals                                                        #
# --------------------------------------------------------------------------- #
def _safe(a, n) -> np.ndarray:
    return np.zeros(n, dtype=np.float64) if a is None else np.asarray(a, dtype=np.float64)


def total_uncertainty(predictions: Mapping) -> np.ndarray:
    """sqrt(aleatoric^2 + epistemic^2) in EF points."""
    n = len(np.asarray(predictions["ef_pred"]))
    aleatoric = _safe(predictions.get("ef_aleatoric_std"), n)
    epistemic = _safe(predictions.get("ef_pred_std"), n)
    return np.sqrt(np.square(aleatoric) + np.square(epistemic))


def uncertainty_signals(predictions: Mapping, ef_used, thresholds) -> dict:
    """Compute every candidate uncertainty score (higher = less confident)."""
    ef_used = np.asarray(ef_used, dtype=np.float64)
    n = len(ef_used)
    thr = np.asarray(list(thresholds), dtype=np.float64).reshape(1, -1)

    aleatoric = _safe(predictions.get("ef_aleatoric_std"), n)
    epistemic = _safe(predictions.get("ef_pred_std"), n)
    total = np.sqrt(np.square(aleatoric) + np.square(epistemic))
    # A floor keeps the ratio finite when the model reports near-zero spread.
    floor = max(float(np.median(total[total > 1e-6])) * 0.25, 0.25) if np.any(total > 1e-6) else 1.0
    total_floored = np.maximum(total, floor)

    distance = np.abs(ef_used.reshape(-1, 1) - thr).min(axis=1)
    # Small value = prediction sits close to a boundary relative to its own
    # uncertainty; negated so that "higher = less confident" holds throughout.
    boundary = -(distance / total_floored)

    signals = {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
        "boundary": boundary,
    }

    probability = predictions.get("class_dist")
    if probability is None:
        probability = predictions.get("ord_dist")
    if probability is not None:
        p = np.asarray(probability, dtype=np.float64)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        ordered = np.sort(p, axis=1)
        signals["margin"] = -(ordered[:, -1] - ordered[:, -2])
        signals["entropy"] = -(p * np.log(p)).sum(axis=1)
        # Combined clinical score: boundary proximity tempered by class ambiguity.
        b = signals["boundary"]
        m = signals["margin"]
        signals["boundary_margin"] = _standardise(b) + _standardise(m)

    return signals


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(x.std())
    return (x - float(x.mean())) / (s if s > 1e-12 else 1.0)


# --------------------------------------------------------------------------- #
#  Selective evaluation                                                       #
# --------------------------------------------------------------------------- #
def evaluate_at_coverage(score, y_true, y_pred, n_classes, coverage) -> dict:
    """Metrics over the most-confident `coverage` fraction of studies."""
    score = np.asarray(score, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(score)
    keep = max(1, int(round(float(coverage) * n)))
    # Lowest uncertainty first; ties broken deterministically by index.
    order = np.lexsort((np.arange(n), score))
    selected = np.zeros(n, dtype=bool)
    selected[order[:keep]] = True

    metrics = classification_metrics(y_true[selected], y_pred[selected], n_classes)
    recalls = [r for r in metrics["per_class_recall"] if r is not None]
    metrics["coverage"] = float(selected.sum()) / float(n)
    metrics["n_covered"] = int(selected.sum())
    metrics["n_deferred"] = int(n - selected.sum())
    metrics["min_class_recall"] = float(min(recalls)) if recalls else float("nan")
    metrics["selected"] = selected
    return metrics


def coverage_curve(score, y_true, y_pred, n_classes,
                   coverages=None) -> list:
    """Metrics across a sweep of coverage levels."""
    if coverages is None:
        coverages = np.round(np.arange(1.00, 0.499, -0.02), 4)
    out = []
    for c in coverages:
        m = evaluate_at_coverage(score, y_true, y_pred, n_classes, c)
        m.pop("selected", None)
        out.append(m)
    return out


def fit_selective_rule(predictions: Mapping, y_pred, ef_used, cfg,
                       target: float = 0.75,
                       min_coverage: float = 0.60) -> dict:
    """Choose the uncertainty signal and threshold on validation data.

    The rule selected is the one that reaches `target` recall on every class at
    the highest possible coverage.  If no signal reaches the target above
    `min_coverage`, the signal with the best worst-class recall is returned and
    the shortfall is reported rather than hidden.
    """
    y_true = np.asarray(predictions["y_true"], dtype=np.int64)
    n_classes = int(cfg.n_classes)
    signals = uncertainty_signals(predictions, ef_used, cfg.EF_THRESHOLDS)
    coverages = np.round(np.arange(1.00, float(min_coverage) - 1e-9, -0.01), 4)

    best_meeting = None      # (coverage, name, threshold, metrics)
    best_overall = None      # fallback: (min_recall, coverage, name, threshold, metrics)

    for name, score in signals.items():
        score = np.asarray(score, dtype=np.float64)
        if not np.isfinite(score).all():
            continue
        for c in coverages:
            m = evaluate_at_coverage(score, y_true, y_pred, n_classes, c)
            selected = m.pop("selected")
            # Threshold = the largest score still accepted at this coverage.
            cut = float(np.max(score[selected])) if selected.any() else float("inf")
            mr = float(m["min_class_recall"])
            if mr >= target:
                key = (m["coverage"],)
                if best_meeting is None or key > (best_meeting[0],):
                    best_meeting = (m["coverage"], name, cut, m)
            key2 = (mr, m["coverage"])
            if best_overall is None or key2 > (best_overall[0], best_overall[1]):
                best_overall = (mr, m["coverage"], name, cut, m)

    if best_meeting is not None:
        coverage, name, cut, metrics = best_meeting
        meets = True
    else:
        _, coverage, name, cut, metrics = best_overall
        meets = False

    return {
        "schema_version": 1,
        "signal": name,
        "threshold": float(cut),
        "target_recall": float(target),
        "meets_target_on_validation": bool(meets),
        "validation_coverage": float(coverage),
        "validation_min_class_recall": float(metrics["min_class_recall"]),
        "validation_per_class_recall": [
            None if r is None else float(r) for r in metrics["per_class_recall"]],
        "validation_balanced_acc": float(metrics["balanced_acc"]),
        "n_validation": int(len(y_true)),
    }


def apply_selective_rule(predictions: Mapping, y_pred, ef_used, cfg,
                         rule: Mapping) -> dict:
    """Apply a frozen selective rule to an untouched split."""
    y_true = np.asarray(predictions["y_true"], dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    signals = uncertainty_signals(predictions, ef_used, cfg.EF_THRESHOLDS)
    name = str(rule["signal"])
    if name not in signals:
        raise ValueError(f"frozen selective signal {name!r} is not available for this split")
    score = np.asarray(signals[name], dtype=np.float64)
    selected = score <= float(rule["threshold"])
    if not selected.any():
        raise RuntimeError("frozen selective threshold accepted no studies")

    covered = classification_metrics(y_true[selected], y_pred[selected], int(cfg.n_classes))
    recalls = [r for r in covered["per_class_recall"] if r is not None]
    n = len(y_true)
    out = {
        "signal": name,
        "threshold": float(rule["threshold"]),
        "coverage": float(selected.sum()) / float(n),
        "n_total": int(n),
        "n_covered": int(selected.sum()),
        "n_deferred": int(n - selected.sum()),
        "covered": covered,
        "min_class_recall": float(min(recalls)) if recalls else float("nan"),
        "selected": selected,
    }
    if (~selected).any():
        deferred = classification_metrics(
            y_true[~selected], y_pred[~selected], int(cfg.n_classes))
        out["deferred"] = deferred
        # Sanity signal: the deferred subset should be harder than the covered one.
        out["deferred_accuracy"] = float(deferred["overall_acc"])
    return out
