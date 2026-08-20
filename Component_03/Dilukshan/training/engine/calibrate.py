"""Post-training calibration for EF regression and ordinal classification.

Calibration is deliberately performed only after model weights are frozen.  The
primary clinical result always uses the published EF boundaries (30/40/55) on
the *raw* regression output.  Affine EF correction, learned decision thresholds,
and probability blends are useful operational rules, but are reported under
separate names so they cannot be mistaken for the clinical result.

The module also keeps support for the variance-expansion dictionaries written by
the original v1 pipeline, allowing historical reports/checkpoints to be opened.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import numpy as np

from engine.metrics import classify_ef, classification_metrics, per_class_recall


def apply_expansion(ef, params) -> np.ndarray:
    """Apply the legacy v1 variance expansion."""
    ef = np.asarray(ef, dtype=np.float64)
    if not params:
        return ef
    return np.clip(
        float(params["target_mean"])
        + float(params["k"]) * (ef - float(params["pred_mean"])),
        0.0,
        100.0,
    )


def apply_ef_calibration(ef, params) -> np.ndarray:
    """Apply a frozen affine EF calibration (or a legacy v1 expansion)."""
    ef = np.asarray(ef, dtype=np.float64)
    if not params:
        return np.clip(ef, 0.0, 100.0)
    if "slope" in params or "intercept" in params:
        slope = float(params.get("slope", 1.0))
        intercept = float(params.get("intercept", 0.0))
        return np.clip(intercept + slope * ef, 0.0, 100.0)
    if {"target_mean", "pred_mean", "k"}.issubset(params):
        return apply_expansion(ef, params)
    raise ValueError("unrecognised EF calibration parameters")


def fit_affine_ef(ef_pred, ef_true, slope_bounds=(0.5, 1.5)) -> dict:
    """Fit least-squares ``true = intercept + slope * prediction``.

    The conservative slope bounds prevent an unstable small calibration subset
    from producing a clinically implausible extrapolation.
    """
    x = np.asarray(ef_pred, dtype=np.float64)
    y = np.asarray(ef_true, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("affine EF calibration needs paired one-dimensional arrays")
    var = float(np.mean((x - x.mean()) ** 2))
    slope = 1.0 if var <= 1e-12 else float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    lo, hi = map(float, slope_bounds)
    if not (0.0 < lo <= hi and math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("invalid affine slope bounds")
    unclipped = slope
    slope = float(np.clip(slope, lo, hi))
    intercept = float(y.mean() - slope * x.mean())
    return {
        "method": "least_squares_affine",
        "slope": slope,
        "intercept": intercept,
        "slope_unclipped": unclipped,
        "slope_bounds": [lo, hi],
    }


def fit_variance_expansion(ef_pred, ef_true, k_bounds=(1.0, 1.7)) -> dict:
    """Fit a variance-expansion around the mean to decompress regression-to-mean.

    A regressor on an imbalanced target shrinks predictions toward the mean, so
    the extreme-low (Severe) tail is systematically over-predicted into Moderate.
    We rescale distances from the mean by ``k = std(true)/std(pred)`` (>= 1) so the
    predicted EF distribution recovers the true spread, pushing true-Severe cases
    back below the EF=30 boundary.  Uses the classic v1 expansion parameterization
    ``EF' = target_mean + k*(EF - pred_mean)`` so apply_ef_calibration handles it.
    """
    x = np.asarray(ef_pred, dtype=np.float64)
    y = np.asarray(ef_true, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("variance expansion needs paired one-dimensional arrays")
    sp = float(x.std())
    st = float(y.std())
    k = 1.0 if sp <= 1e-6 else st / sp
    lo, hi = map(float, k_bounds)
    k = float(np.clip(k, lo, hi))
    return {
        "method": "variance_expansion",
        "pred_mean": float(x.mean()),
        "target_mean": float(y.mean()),
        "k": k,
        "k_unclipped": (1.0 if sp <= 1e-6 else st / sp),
    }


def optimize_thresholds(
    ef_pred,
    y_true,
    n_classes=4,
    clinical_thresholds=(30.0, 40.0, 55.0),
    radius=(5.0, 8.0, 6.0),
    min_gap=2.0,
    step=0.5,
):
    """Find an explicitly operational threshold rule on calibration data.

    The objective is lexicographic: worst-class recall, balanced accuracy,
    macro-F1, then proximity to the clinical boundaries.  Search is constrained
    around those boundaries to reduce pathological calibration-set overfitting.
    """
    ef_pred = np.asarray(ef_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    base = tuple(float(v) for v in clinical_thresholds)
    if n_classes != 4 or len(base) != 3:
        raise ValueError("threshold optimization currently supports four ordered EF classes")
    radii = tuple(float(v) for v in radius)
    grids = [np.arange(t - r, t + r + 1e-9, step) for t, r in zip(base, radii)]
    best = None
    for t1 in grids[0]:
        for t2 in grids[1]:
            if t2 - t1 < min_gap:
                continue
            for t3 in grids[2]:
                if t3 - t2 < min_gap:
                    continue
                thresholds = (float(t1), float(t2), float(t3))
                metrics = classification_metrics(
                    y_true, classify_ef(ef_pred, thresholds), n_classes
                )
                recalls = np.nan_to_num(metrics["per_class_recall"], nan=0.0)
                distance = -float(np.sum(np.abs(np.asarray(thresholds) - np.asarray(base))))
                score = (
                    float(np.min(recalls)),
                    float(np.mean(recalls)),
                    float(metrics["macro_f1"]),
                    distance,
                )
                if best is None or score > best[0]:
                    best = (score, thresholds)
    if best is None:
        raise RuntimeError("no valid ordered threshold combination was generated")
    return best[0][0], best[0][1], best[1]


def _normalise_probabilities(probabilities) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, K), K >= 2")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or infinity")
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    """Temperature-scale probabilities via their normalized log-probabilities."""
    p = _normalise_probabilities(probabilities)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(p) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, y_true) -> dict:
    """Fit a scalar temperature by calibration-set multiclass NLL."""
    p = _normalise_probabilities(probabilities)
    y = np.asarray(y_true, dtype=np.int64)
    if len(y) != len(p):
        raise ValueError("probability and target lengths differ")
    # A dense log grid is deterministic and avoids adding a scipy dependency.
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    best = None
    for temperature in candidates:
        scaled = apply_temperature(p, float(temperature))
        nll = float(-np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, 1.0)).mean())
        key = (nll, abs(math.log(float(temperature))))
        if best is None or key < best[0]:
            best = (key, float(temperature))
    return {"method": "scalar_temperature", "temperature": best[1], "calibration_nll": best[0][0]}


def ef_class_distribution(ef, thresholds, sigma: float) -> np.ndarray:
    """Turn continuous EF into smooth probability mass over clinical intervals."""
    ef = np.asarray(ef, dtype=np.float64).reshape(-1, 1)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
    sigma = max(float(sigma), 1e-6)
    z = (thr - ef) / (sigma * math.sqrt(2.0))
    # numpy does not expose erf on every supported build; vectorize math.erf.
    cdf = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(z))
    pieces = [cdf[:, :1]]
    if cdf.shape[1] > 1:
        pieces.append(cdf[:, 1:] - cdf[:, :-1])
    pieces.append(1.0 - cdf[:, -1:])
    return _normalise_probabilities(np.concatenate(pieces, axis=1))


def _metrics(y_true, y_pred, n, ef_true=None, ef_used=None):
    metrics = classification_metrics(y_true, y_pred, n)
    if ef_true is not None and ef_used is not None:
        metrics["mae"] = float(
            np.mean(np.abs(np.asarray(ef_used, dtype=float) - np.asarray(ef_true, dtype=float)))
        )
    return metrics


def _strategy_score(metrics: Mapping) -> tuple:
    values = (
        float(metrics["min_class_recall"]),
        float(metrics["balanced_acc"]),
        float(metrics["macro_f1"]),
        float(metrics["overall_acc"]),
    )
    return tuple(v if math.isfinite(v) else -1.0 for v in values)


def fit_conformal(ef_true, ef_pred, pred_std=None, alpha: float = 0.10) -> dict:
    """Fit a finite-sample split-conformal absolute-residual interval."""
    y = np.asarray(ef_true, dtype=np.float64)
    pred = np.asarray(ef_pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("conformal calibration needs non-empty paired vectors")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("conformal alpha must be in (0, 1)")
    scale = None
    mode = "absolute_residual"
    if pred_std is not None:
        candidate = np.asarray(pred_std, dtype=np.float64)
        if candidate.shape == pred.shape and np.isfinite(candidate).all() and np.any(candidate > 1e-6):
            floor = max(float(np.quantile(candidate[candidate > 1e-6], 0.10)), 0.5)
            scale = np.maximum(candidate, floor)
            mode = "uncertainty_scaled_residual"
    scores = np.abs(y - pred) if scale is None else np.abs(y - pred) / scale
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1.0 - float(alpha))) / n)
    try:
        q_hat = float(np.quantile(scores, level, method="higher"))
    except TypeError:  # numpy < 1.22
        q_hat = float(np.quantile(scores, level, interpolation="higher"))
    return {
        "method": mode,
        "alpha": float(alpha),
        "q_hat": q_hat,
        "std_floor": (None if scale is None else floor),
        "n_calibration": int(n),
        "finite_sample_quantile_level": float(level),
    }


def apply_conformal(ef_pred, params: Mapping, pred_std=None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(ef_pred, dtype=np.float64)
    q_hat = float(params["q_hat"])
    if params.get("method") == "uncertainty_scaled_residual":
        if pred_std is None:
            raise ValueError("uncertainty-scaled interval requires prediction std")
        scale = np.maximum(np.asarray(pred_std, dtype=np.float64), float(params["std_floor"]))
        width = q_hat * scale
    else:
        width = np.full_like(pred, q_hat)
    return np.clip(pred - width, 0.0, 100.0), np.clip(pred + width, 0.0, 100.0)


def _best_probability_blend(
    y_true,
    n_classes: int,
    sources: Mapping[str, np.ndarray],
    step: float = 0.10,
) -> Optional[dict]:
    """Select a small convex probability blend on calibration data."""
    names = list(sources)
    if len(names) < 2:
        return None
    arrays = {name: _normalise_probabilities(sources[name]) for name in names}
    n_steps = int(round(1.0 / step))
    best = None

    def compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest

    for integer_weights in compositions(n_steps, len(names)):
        if sum(v > 0 for v in integer_weights) < 2:
            continue
        weights = np.asarray(integer_weights, dtype=np.float64) / n_steps
        probability = sum(weights[i] * arrays[name] for i, name in enumerate(names))
        pred = probability.argmax(axis=1).astype(np.int64)
        metrics = classification_metrics(y_true, pred, n_classes)
        # Prefer a simpler/equal blend only after all performance tie-breakers.
        complexity_tie = -float(np.square(weights - 1.0 / len(names)).sum())
        score = _strategy_score(metrics) + (complexity_tie,)
        if best is None or score > best[0]:
            best = (
                score,
                {
                    "weights": {name: float(weights[i]) for i, name in enumerate(names)},
                    "metrics": metrics,
                },
            )
    return None if best is None else best[1]


def calibrate(
    ef_pred,
    y_true,
    ord_pred,
    cfg,
    ef_true=None,
    ord_dist=None,
    class_dist=None,
    pred_std=None,
) -> dict:
    """Fit all post-hoc choices on a calibration subset and freeze the result."""
    n = int(cfg.n_classes)
    ef_pred = np.asarray(ef_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    ord_pred = np.asarray(ord_pred, dtype=np.int64)
    if not (len(ef_pred) == len(y_true) == len(ord_pred)):
        raise ValueError("calibration prediction/target lengths differ")

    strategies = {}
    clinical = tuple(float(v) for v in cfg.EF_THRESHOLDS)

    raw_clinical_pred = classify_ef(ef_pred, clinical)
    strategies["reg_clinical_raw"] = dict(
        kind="regression_thresholds",
        thresholds=list(clinical),
        ef_calibration=None,
        **_metrics(y_true, raw_clinical_pred, n, ef_true, ef_pred),
    )

    affine = None
    ef_affine = ef_pred
    if ef_true is not None:
        bounds = getattr(cfg, "affine_slope_bounds", (0.5, 1.5))
        affine = fit_affine_ef(ef_pred, ef_true, slope_bounds=bounds)
        ef_affine = apply_ef_calibration(ef_pred, affine)
        affine_pred = classify_ef(ef_affine, clinical)
        strategies["reg_clinical_affine"] = dict(
            kind="regression_thresholds",
            thresholds=list(clinical),
            ef_calibration=affine,
            **_metrics(y_true, affine_pred, n, ef_true, ef_affine),
        )

    _, _, learned_thresholds = optimize_thresholds(
        ef_affine,
        y_true,
        n_classes=n,
        clinical_thresholds=clinical,
        radius=getattr(cfg, "threshold_search_radius", (5.0, 8.0, 6.0)),
        min_gap=float(getattr(cfg, "threshold_min_gap", 2.0)),
        step=float(getattr(cfg, "threshold_search_step", 0.5)),
    )
    learned_pred = classify_ef(ef_affine, learned_thresholds)
    strategies["reg_operational_thresholds"] = dict(
        kind="regression_thresholds",
        thresholds=list(learned_thresholds),
        ef_calibration=affine,
        **_metrics(y_true, learned_pred, n, ef_true, ef_affine),
    )

    # Variance-expansion candidates: decompress the regressed EF so extreme-tail
    # (Severe) cases fall back across the EF=30 boundary instead of piling into
    # Moderate.  Selected only if it improves VAL min-recall.
    if ef_true is not None:
        expansion = fit_variance_expansion(ef_pred, ef_true)
        ef_expanded = apply_ef_calibration(ef_pred, expansion)
        strategies["reg_clinical_expanded"] = dict(
            kind="regression_thresholds",
            thresholds=list(clinical),
            ef_calibration=expansion,
            **_metrics(y_true, classify_ef(ef_expanded, clinical), n, ef_true, ef_expanded),
        )
        _, _, expanded_thresholds = optimize_thresholds(
            ef_expanded, y_true, n_classes=n, clinical_thresholds=clinical,
            radius=getattr(cfg, "threshold_search_radius", (5.0, 8.0, 6.0)),
            min_gap=float(getattr(cfg, "threshold_min_gap", 2.0)),
            step=float(getattr(cfg, "threshold_search_step", 0.5)),
        )
        strategies["reg_operational_expanded"] = dict(
            kind="regression_thresholds",
            thresholds=list(expanded_thresholds),
            ef_calibration=expansion,
            **_metrics(y_true, classify_ef(ef_expanded, expanded_thresholds), n, ef_true, ef_expanded),
        )

    strategies["ordinal_rank"] = dict(
        kind="ordinal_rank",
        thresholds=None,
        ef_calibration=None,
        **_metrics(y_true, ord_pred, n),
    )

    probability_calibration = {}
    probability_sources = {
        "regression": ef_class_distribution(
            ef_affine, clinical, float(getattr(cfg, "calibration_ef_sigma", 4.0))
        )
    }
    if ord_dist is not None:
        ord_temp = fit_temperature(ord_dist, y_true)
        probability_calibration["ordinal"] = ord_temp
        ord_probability = apply_temperature(ord_dist, ord_temp["temperature"])
        probability_sources["ordinal"] = ord_probability
        strategies["ordinal_argmax"] = dict(
            kind="probability_argmax",
            probability_source="ordinal",
            thresholds=None,
            ef_calibration=None,
            **_metrics(y_true, ord_probability.argmax(axis=1), n),
        )
    if class_dist is not None:
        class_temp = fit_temperature(class_dist, y_true)
        probability_calibration["class"] = class_temp
        class_probability = apply_temperature(class_dist, class_temp["temperature"])
        probability_sources["class"] = class_probability
        strategies["class_argmax"] = dict(
            kind="probability_argmax",
            probability_source="class",
            thresholds=None,
            ef_calibration=None,
            **_metrics(y_true, class_probability.argmax(axis=1), n),
        )

    blend = _best_probability_blend(y_true, n, probability_sources)
    if blend is not None:
        strategies["probability_blend"] = dict(
            kind="probability_blend",
            blend_weights=blend["weights"],
            thresholds=None,
            ef_calibration=affine,
            **blend["metrics"],
        )

    operational_names = [name for name in strategies if name != "reg_clinical_raw"]
    best_name = max(operational_names, key=lambda name: _strategy_score(strategies[name]))
    conformal = None
    if ef_true is not None:
        conformal = fit_conformal(
            ef_true,
            ef_affine,
            pred_std=pred_std,
            alpha=float(getattr(cfg, "conformal_alpha", 0.10)),
        )
    return {
        "schema_version": 2,
        "clinical_primary": "reg_clinical_raw",
        "best_strategy": best_name,
        "strategies": strategies,
        "best": strategies[best_name],
        "ef_calibration": affine,
        "probability_calibration": probability_calibration,
        "probability_blend_weights": (None if blend is None else blend["weights"]),
        "conformal": conformal,
        "n_calibration": int(len(y_true)),
    }


def apply_frozen_strategy(predictions: Mapping, calibration: Mapping, cfg) -> dict:
    """Apply a strategy selected on calibration data to an untouched split."""
    strategy_name = str(calibration["best_strategy"])
    strategy = calibration["strategies"][strategy_name]
    ef_raw = np.asarray(predictions["ef_pred"], dtype=np.float64)
    ef_calibrated = apply_ef_calibration(ef_raw, calibration.get("ef_calibration"))
    clinical_pred = classify_ef(ef_raw, cfg.EF_THRESHOLDS)

    ordinal_probability = None
    if predictions.get("ord_dist") is not None:
        temperature = calibration.get("probability_calibration", {}).get(
            "ordinal", {}
        ).get("temperature", 1.0)
        ordinal_probability = apply_temperature(predictions["ord_dist"], temperature)
    class_probability = None
    if predictions.get("class_dist") is not None:
        temperature = calibration.get("probability_calibration", {}).get(
            "class", {}
        ).get("temperature", 1.0)
        class_probability = apply_temperature(predictions["class_dist"], temperature)

    kind = strategy.get("kind")
    operational_probability = None
    if kind == "regression_thresholds":
        ef_for_rule = apply_ef_calibration(ef_raw, strategy.get("ef_calibration"))
        operational_pred = classify_ef(ef_for_rule, strategy["thresholds"])
    elif kind == "ordinal_rank":
        operational_pred = np.asarray(predictions["ord_pred"], dtype=np.int64)
    elif kind == "probability_argmax":
        source = strategy["probability_source"]
        operational_probability = ordinal_probability if source == "ordinal" else class_probability
        if operational_probability is None:
            raise ValueError(f"frozen strategy requires missing {source} probabilities")
        operational_pred = operational_probability.argmax(axis=1).astype(np.int64)
    elif kind == "probability_blend":
        sources = {
            "regression": ef_class_distribution(
                ef_calibrated,
                cfg.EF_THRESHOLDS,
                float(getattr(cfg, "calibration_ef_sigma", 4.0)),
            ),
            "ordinal": ordinal_probability,
            "class": class_probability,
        }
        weights = strategy["blend_weights"]
        missing = [name for name, weight in weights.items() if weight and sources.get(name) is None]
        if missing:
            raise ValueError(f"frozen blend requires missing probabilities: {missing}")
        operational_probability = sum(
            float(weight) * sources[name] for name, weight in weights.items()
        )
        operational_probability = _normalise_probabilities(operational_probability)
        operational_pred = operational_probability.argmax(axis=1).astype(np.int64)
    else:
        raise ValueError(f"unknown frozen strategy kind: {kind!r}")

    out = {
        "strategy": strategy_name,
        "ef_raw": ef_raw,
        "ef_calibrated": ef_calibrated,
        "clinical_pred": clinical_pred,
        "operational_pred": operational_pred,
        "operational_probability": operational_probability,
        "ordinal_probability": ordinal_probability,
        "class_probability": class_probability,
    }
    if calibration.get("conformal") is not None:
        out["interval_low"], out["interval_high"] = apply_conformal(
            ef_calibrated,
            calibration["conformal"],
            pred_std=predictions.get("ef_pred_std"),
        )
    return out
