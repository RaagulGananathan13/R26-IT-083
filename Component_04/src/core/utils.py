"""
Component 04 — shared utilities: logging, metrics, bootstrap CIs, plotting.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import FIGURE_DIR

# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------
_W = 78


def banner(title: str, ch: str = "=") -> None:
    print("\n" + ch * _W)
    print(f"  {title}")
    print(ch * _W)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, _W - len(title) - 6))


def kv(label: str, value, width: int = 34) -> None:
    print(f"  {label:<{width}} {value}")


@contextmanager
def timer(label: str):
    t0 = time.time()
    print(f"  > {label} ...")
    yield
    print(f"  < {label} done in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# GPU detection
# --------------------------------------------------------------------------
class BinBudget:
    """
    Shared, self-tuning histogram width.

    max_bin is pushed high on purpose: bigger histograms mean more GPU memory
    resident and finer candidate split points.  If the card cannot hold it at
    the configured concurrency, the budget halves itself once and every
    subsequent fit uses the reduced value, so a search never dies on an OOM
    that only affects the largest setting.
    """

    def __init__(self, max_bin: int = 4096, floor: int = 256, enabled: bool = True):
        self.value = int(max_bin)
        self.floor = int(floor)
        self.enabled = enabled
        self._lock = __import__("threading").Lock()

    def _is_oom(self, err: Exception) -> bool:
        m = str(err).lower()
        return any(t in m for t in ("out of memory", "oom", "cudaerrormemoryallocation",
                                    "cuda error", "bad_alloc", "resource_exhausted"))

    def run(self, fn):
        """fn(max_bin) -> result, retried at a smaller width on GPU OOM."""
        while True:
            mb = self.value
            try:
                return fn(mb)
            except Exception as e:
                if not (self.enabled and self._is_oom(e) and mb > self.floor):
                    raise
                with self._lock:
                    if self.value == mb:
                        self.value = max(self.floor, mb // 2)
                        print(f"\n  [gpu] OOM at max_bin={mb} -> retrying at "
                              f"{self.value} (this becomes the new budget)")


def resolve_device(requested: str = "cuda") -> str:
    """Return 'cuda' only if XGBoost can actually reach a GPU."""
    if requested != "cuda":
        return "cpu"
    try:
        import xgboost as xgb

        probe = xgb.XGBClassifier(
            device="cuda", tree_method="hist", n_estimators=2, verbosity=0
        )
        probe.fit(np.random.rand(32, 4), np.random.randint(0, 2, 32))
        return "cuda"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Recall / precision / F1 / support / accuracy for every class."""
    labels = list(range(len(class_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        out[name] = {
            "recall": float(rec[i]),          # == per-class accuracy (sensitivity)
            "precision": float(prec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
            "correct": int(cm[i, i]),
            "specificity": _specificity(cm, i),
        }
    return out


def _specificity(cm: np.ndarray, i: int) -> float:
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def summarise(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    y_prob: np.ndarray | None = None,
) -> Dict:
    labels = list(range(len(class_names)))
    res = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels,
                                      zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class_report(y_true, y_pred, class_names),
    }
    res["min_recall"] = float(min(v["recall"] for v in res["per_class"].values()))
    res["min_f1"] = float(min(v["f1"] for v in res["per_class"].values()))
    if y_prob is not None:
        res.update(_prob_metrics(y_true, y_prob, class_names))
    return res


def _prob_metrics(y_true, y_prob, class_names) -> Dict:
    out: Dict = {}
    try:
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            p = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
            out["auroc"] = float(roc_auc_score(y_true, p))
            out["auprc"] = float(average_precision_score(y_true, p))
        else:
            out["auroc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            aps = []
            for i in range(len(class_names)):
                aps.append(average_precision_score((y_true == i).astype(int), y_prob[:, i]))
            out["auprc_macro"] = float(np.mean(aps))
            out["auprc_per_class"] = {
                class_names[i]: float(aps[i]) for i in range(len(class_names))
            }
    except Exception:
        pass
    return out


def print_report(name: str, res: Dict, class_names: Sequence[str],
                 floor: float = 0.75) -> None:
    section(name)
    kv("Balanced accuracy", f"{res['balanced_accuracy']*100:6.2f}%")
    kv("Macro F1", f"{res['macro_f1']:6.4f}")
    for k in ("auroc", "auprc", "auroc_ovr_macro", "auprc_macro"):
        if k in res:
            kv(k.upper(), f"{res[k]:6.4f}")
    print()
    print(f"  {'class':<10}{'recall':>9}{'prec':>9}{'F1':>9}{'spec':>9}{'n':>8}   status")
    print("  " + "-" * 62)
    for c in class_names:
        m = res["per_class"][c]
        ok = "PASS" if (m["recall"] >= floor and m["f1"] >= floor) else \
             ("recall-ok" if m["recall"] >= floor else "BELOW")
        print(f"  {c:<10}{m['recall']*100:8.2f}%{m['precision']*100:8.2f}%"
              f"{m['f1']*100:8.2f}%{m['specificity']*100:8.2f}%{m['support']:8d}   {ok}")
    print("  " + "-" * 62)
    cm = np.array(res["confusion_matrix"])
    print("  Confusion matrix (rows = truth):")
    print("            " + "".join(f"{c:>10}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"    {c:<8}" + "".join(f"{v:>10,}" for v in cm[i]))


# --------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> Dict:
    """Stratified (optionally cluster/patient-level) bootstrap CIs."""
    rng = np.random.RandomState(seed)
    N = len(y_true)
    acc: Dict[str, List[float]] = {f"{c}_{m}": [] for c in class_names
                                   for m in ("recall", "f1", "precision")}
    acc["macro_f1"] = []
    acc["balanced_accuracy"] = []

    if groups is not None:
        uniq = np.unique(groups)
        index_of = {g: np.where(groups == g)[0] for g in uniq}

    for _ in range(n):
        if groups is not None:                     # cluster bootstrap on patients
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([index_of[g] for g in picked])
        else:
            idx = rng.randint(0, N, N)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        pc = per_class_report(yt, yp, class_names)
        for c in class_names:
            for m in ("recall", "f1", "precision"):
                acc[f"{c}_{m}"].append(pc[c][m])
        acc["macro_f1"].append(
            f1_score(yt, yp, average="macro", labels=list(range(len(class_names))),
                     zero_division=0)
        )
        acc["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        k: {
            "mean": float(np.mean(v)),
            "lo": float(np.percentile(v, lo)),
            "hi": float(np.percentile(v, hi)),
        }
        for k, v in acc.items() if len(v) > 0
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], title: str,
                   fname: str, normalise: bool = True) -> str:
    plt = _mpl()
    cm = np.asarray(cm, dtype=float)
    disp = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalise else cm
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(class_names),
                                    1.4 + 0.95 * len(class_names)))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.size else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    ax.grid(False)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{disp[i,j]*100:.1f}%\n({int(cm[i,j]):,})" if normalise else f"{int(cm[i,j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if disp[i, j] > disp.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, fname)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_per_class_bars(res: Dict, class_names: Sequence[str], title: str,
                        fname: str, floor: float = 0.75) -> str:
    plt = _mpl()
    metrics = ("recall", "precision", "f1")
    x = np.arange(len(class_names)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.9 + 1.4 * len(class_names), 3.8))
    colors = ["#2E5EAA", "#57A0D3", "#9BC4E2"]
    for k, m in enumerate(metrics):
        vals = [res["per_class"][c][m] for c in class_names]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=m.capitalize(), color=colors[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v*100:.0f}",
                    ha="center", fontsize=7)
    ax.axhline(floor, ls="--", c="#C0392B", lw=1.2,
               label=f"target {floor*100:.0f}%")
    ax.set_xticks(x, class_names); ax.set_ylim(0, 1.08)
    ax.set_ylabel("score"); ax.set_title(title)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, fname)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating))
                         else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
