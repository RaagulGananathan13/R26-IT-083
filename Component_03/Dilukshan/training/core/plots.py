"""
Report-quality plots (training curves + confusion matrix).
Uses a non-interactive Agg backend so it runs headless. matplotlib is optional:
if unavailable the functions no-op with a warning instead of crashing training.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:                                  # pragma: no cover
    _HAS_MPL = False


def plot_training_curves(csv_path, out_path):
    if not _HAS_MPL:
        print("[plots] matplotlib unavailable; skipping training curves.")
        return
    import pandas as pd
    csv_path, out_path = Path(csv_path), Path(out_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = df.drop_duplicates("epoch", keep="last").sort_values("epoch")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(df["epoch"], df["train_loss"], label="train loss", color="#4C78A8")
    ax[0].plot(df["epoch"], df["val_mae"], label="val MAE", color="#E45756")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss / MAE")
    ax[0].set_title("Loss & MAE"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(df["epoch"], df["val_min_recall"], label="val min-recall", color="#54A24B")
    ax[1].plot(df["epoch"], df["val_balanced_acc"], label="val balanced-acc", color="#B279A2")
    ax[1].axhline(0.75, ls="--", color="gray", label="0.75 target")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
    ax[1].set_ylim(0, 1); ax[1].set_title("Per-class objective"); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plots] training curves -> {out_path}")


def plot_confusion(cm, class_names, out_path, recalls=None):
    if not _HAS_MPL:
        print("[plots] matplotlib unavailable; skipping confusion matrix.")
        return
    cm = np.asarray(cm, dtype=np.int64)
    out_path = Path(out_path)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axx = plt.subplots(figsize=(6.2, 5.4))
    im = axx.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    axx.set_xticks(range(len(class_names))); axx.set_yticks(range(len(class_names)))
    short = [n.split("(")[0] for n in class_names]
    axx.set_xticklabels(short, rotation=30, ha="right"); axx.set_yticklabels(short)
    axx.set_xlabel("Predicted"); axx.set_ylabel("True")
    title = "Confusion (row-normalised)"
    if recalls is not None:
        title += f"\nmin per-class recall = {np.nanmin(recalls):.3f}"
    axx.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axx.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center",
                     color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plots] confusion matrix -> {out_path}")
