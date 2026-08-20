"""Classification + regression metrics for EF grading."""
from __future__ import annotations
import numpy as np


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return [float("nan"), float("nan")]
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def classify_ef(ef: np.ndarray, thresholds) -> np.ndarray:
    """Bin continuous EF into class ids using (t1,t2,t3)."""
    ef = np.asarray(ef)
    t1, t2, t3 = thresholds
    out = np.full(ef.shape, 3, dtype=np.int64)
    out[ef < t3] = 2
    out[ef < t2] = 1
    out[ef < t1] = 0
    return out


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def per_class_recall(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rec = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y_true == c
        if m.sum() > 0:
            rec[c] = (y_pred[m] == c).mean()
    return rec


def per_class_f1(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1 = np.full(n_classes, np.nan)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    rec = per_class_recall(y_true, y_pred, n_classes)
    f1 = per_class_f1(y_true, y_pred, n_classes)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision = np.zeros(n_classes, dtype=np.float64)
    specificity = np.zeros(n_classes, dtype=np.float64)
    support = cm.sum(axis=1)
    recall_ci = []
    total = int(cm.sum())
    for c in range(n_classes):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp); tn = total - tp - fp - fn
        precision[c] = tp / (tp + fp) if tp + fp else 0.0
        specificity[c] = tn / (tn + fp) if tn + fp else 0.0
        recall_ci.append(_wilson_interval(tp, int(support[c])))
    return {
        "per_class_recall": rec.tolist(),
        "per_class_recall_ci95_wilson": recall_ci,
        "per_class_precision": precision.tolist(),
        "per_class_specificity": specificity.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "min_class_recall": float(np.nanmin(rec)),
        "balanced_acc": float(np.nanmean(rec)),
        "overall_acc": float((y_true == y_pred).mean()),
        "within_one_class_acc": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(np.nanmean(f1)),
        "macro_precision": float(np.nanmean(precision)),
        "macro_specificity": float(np.nanmean(specificity)),
        "confusion": cm.tolist(),
    }


def regression_metrics(ef_true, ef_pred) -> dict:
    ef_true = np.asarray(ef_true, dtype=np.float64)
    ef_pred = np.asarray(ef_pred, dtype=np.float64)
    err = ef_pred - ef_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((ef_true - ef_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bias = float(err.mean())
    error_sd = float(err.std(ddof=1)) if len(err) > 1 else 0.0
    corr = float(np.corrcoef(ef_true, ef_pred)[0, 1]) if len(err) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": corr,
            "bias": bias,
            "within_5_ef": float((np.abs(err) <= 5.0).mean()),
            "within_10_ef": float((np.abs(err) <= 10.0).mean()),
            "bland_altman_loa95": [bias - 1.96 * error_sd, bias + 1.96 * error_sd]}


def regression_metrics_by_class(ef_true, ef_pred, y_true, n_classes: int) -> list:
    ef_true = np.asarray(ef_true); ef_pred = np.asarray(ef_pred); y_true = np.asarray(y_true)
    out = []
    for c in range(n_classes):
        mask = y_true == c
        out.append(dict(n=int(mask.sum()), **regression_metrics(ef_true[mask], ef_pred[mask]))
                   if mask.any() else {"n": 0})
    return out


def probability_metrics(y_true, probabilities, n_bins: int = 15) -> dict:
    """Multiclass Brier score, NLL and equal-width expected calibration error."""
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    one_hot = np.eye(p.shape[1], dtype=np.float64)[y_true]
    nll = float(-np.log(p[np.arange(len(y_true)), y_true]).mean())
    brier = float(np.square(p - one_hot).sum(axis=1).mean())
    conf = p.max(axis=1); pred = p.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(float((pred[mask] == y_true[mask]).mean()) - float(conf[mask].mean()))
    return {"nll": nll, "brier": brier, "ece": float(ece), "n_bins": int(n_bins)}


def bootstrap_regression_ci(ef_true, ef_pred, n_bootstrap: int = 2000,
                            confidence: float = 0.95, seed: int = 2027) -> dict:
    """Patient-level percentile bootstrap intervals for core regression metrics."""
    y = np.asarray(ef_true, dtype=np.float64)
    pred = np.asarray(ef_pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1 or len(y) < 2:
        raise ValueError("bootstrap requires paired one-dimensional arrays with N >= 2")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    values = {name: np.empty(n_bootstrap, dtype=np.float64)
              for name in ("mae", "rmse", "r2", "bias")}
    n = len(y)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = regression_metrics(y[idx], pred[idx])
        for name in values:
            values[name][b] = metrics[name]
    tail = (1.0 - confidence) / 2.0
    intervals = {}
    for name, samples in values.items():
        finite = samples[np.isfinite(samples)]
        intervals[name] = ([float("nan"), float("nan")] if len(finite) == 0 else
                           [float(np.quantile(finite, tail)),
                            float(np.quantile(finite, 1.0 - tail))])
    return {"method": "patient_percentile_bootstrap", "confidence": float(confidence),
            "n_bootstrap": int(n_bootstrap), "seed": int(seed), "intervals": intervals}


def prediction_interval_metrics(ef_true, lower, upper) -> dict:
    """Empirical coverage and width of prediction intervals."""
    y = np.asarray(ef_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("interval and target shapes differ")
    if np.any(hi < lo):
        raise ValueError("prediction interval has upper < lower")
    return {
        "empirical_coverage": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width": float((hi - lo).mean()),
        "median_width": float(np.median(hi - lo)),
    }
