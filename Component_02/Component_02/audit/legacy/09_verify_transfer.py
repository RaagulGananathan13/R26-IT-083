"""
09_verify_transfer.py — Prove the Colab artefacts arrived intact and consistent.

Downloading a model from one machine and its logits from another is exactly where
silent corruption hides. This checks:

  1. best_model.pt loads as the architecture it claims, with sane metadata
  2. the saved logits actually come FROM that model (re-runs inference locally on
     the packed data and compares) — catches a mismatched download
  3. the saved labels match the CSV ground truth, in the same row order
  4. the test metrics reproduce what Colab printed
  5. the stale archive-model calibrator/conformal files are detected

Usage: python -X utf8 Component_02/audit/09_verify_transfer.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import preprocess as pp                    # noqa: E402
from src.models import CLASS_NAMES, build_model     # noqa: E402

DATA = os.path.join(ROOT, "_archive", "data")
PACK = os.path.join(COMP, "data")
CKPT = os.path.join(COMP, "checkpoints")

npass = nfail = 0
def check(name, ok, detail=""):
    global npass, nfail
    npass, nfail = npass + ok, nfail + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
def hdr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


# ── 1. checkpoint ────────────────────────────────────────────────────────
hdr("1. CHECKPOINT INTEGRITY")
p = os.path.join(CKPT, "best_model.pt")
st = torch.load(p, map_location="cpu", weights_only=False)
name = st.get("model_name", "?")
print(f"  keys        : {sorted(st.keys())}")
print(f"  model_name  : {name}")
print(f"  best epoch  : {st.get('epoch')}")
print(f"  best val AUROC (Colab): {st.get('best_auroc', float('nan')):.4f}")
print(f"  EMA weights : {st.get('ema')}")
a = st.get("args", {})
print(f"  trained with: seed={a.get('seed')} epochs={a.get('epochs')} "
      f"batch={a.get('batch')} lr={a.get('lr')} patience={a.get('patience')}")

check("model_name is resnet_se", name == "resnet_se", name)
model = build_model(name)
missing, unexpected = model.load_state_dict(st["model_state"], strict=False)
check("state_dict loads with no missing/unexpected keys",
      not missing and not unexpected,
      f"missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
model.eval()
n_par = sum(x.numel() for x in model.parameters())
check("parameter count matches training log (1,584,326)", n_par == 1584326, f"{n_par:,}")
finite = all(torch.isfinite(v).all() for v in st["model_state"].values()
             if v.dtype.is_floating_point)
check("all weights finite (no NaN/Inf from a truncated download)", finite)

# ── 2. labels ────────────────────────────────────────────────────────────
hdr("2. LABEL CONSISTENCY")
Yv = np.load(os.path.join(CKPT, "val_labels.npy"))
Yt = np.load(os.path.join(CKPT, "test_labels.npy"))
val = pd.read_csv(os.path.join(DATA, "val.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
Yv_csv = val[[f"label_{c}" for c in CLASS_NAMES]].values.astype(float)
Yt_csv = test[[f"label_{c}" for c in CLASS_NAMES]].values.astype(float)
check("val labels match val.csv row-for-row", np.array_equal(Yv, Yv_csv),
      f"{int((Yv != Yv_csv).sum())} mismatched cells")
check("test labels match test.csv row-for-row", np.array_equal(Yt, Yt_csv),
      f"{int((Yt != Yt_csv).sum())} mismatched cells")
print(f"  val positives per class : {Yv.sum(0).astype(int).tolist()}")
print(f"  test positives per class: {Yt.sum(0).astype(int).tolist()}")

# ── 3. do the logits come from THIS model? ───────────────────────────────
hdr("3. LOGIT PROVENANCE (re-run inference locally and compare)")
Lv = np.load(os.path.join(CKPT, "val_logits_seed0.npy"))
Lt = np.load(os.path.join(CKPT, "test_logits_seed0.npy"))
check("logit shapes are (n, 5)", Lv.shape == (len(Yv), 5) and Lt.shape == (len(Yt), 5),
      f"{Lv.shape} / {Lt.shape}")
check("logits are finite", np.isfinite(Lv).all() and np.isfinite(Lt).all())

if not os.path.exists(os.path.join(PACK, "test_X.npy")):
    print("  packed data missing locally -> skipping the recomputation check")
else:
    X = np.asarray(np.load(os.path.join(PACK, "test_X.npy"), mmap_mode="r"),
                   dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out.append(model(torch.from_numpy(X[i:i + 64])).numpy())
    Lt_local = np.concatenate(out)
    d = np.abs(Lt_local - Lt)
    # Colab ran under bfloat16 autocast; local is fp32. Differences of order 1e-2
    # in logit space are expected and harmless; a mismatched model would be O(1).
    check("recomputed test logits match the downloaded ones",
          float(d.max()) < 0.5,
          f"max|diff|={d.max():.4f}  mean={d.mean():.5f}  "
          f"(bf16-vs-fp32 noise expected, a wrong model would be O(1)+)")
    agree = ((Lt_local >= 0) == (Lt >= 0)).mean()
    check("sign agreement > 99.5%", agree > 0.995, f"{agree*100:.2f}%")

# ── 4. metrics reproduce Colab ───────────────────────────────────────────
hdr("4. DO THE TEST METRICS REPRODUCE THE COLAB LOG?")
P = 1 / (1 + np.exp(-Lt))
au = [roc_auc_score(Yt[:, k], P[:, k]) for k in range(5)]
ap = [average_precision_score(Yt[:, k], P[:, k]) for k in range(5)]
f1 = [f1_score(Yt[:, k], (P[:, k] >= 0.5).astype(int), zero_division=0) for k in range(5)]
COLAB = {"NORM": (0.9574, 0.9253, 0.8647), "MI": (0.9487, 0.7833, 0.6915),
         "STTC": (0.9315, 0.8280, 0.7420), "CD": (0.9141, 0.8646, 0.7557),
         "HYP": (0.9085, 0.5842, 0.5348)}
print(f"  {'class':<6} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}   {'colab AUROC':>12} {'delta':>8}")
ok_all = True
for k, c in enumerate(CLASS_NAMES):
    d = au[k] - COLAB[c][0]
    ok_all &= abs(d) < 5e-3
    print(f"  {c:<6} {au[k]:>8.4f} {ap[k]:>8.4f} {f1[k]:>8.4f}   "
          f"{COLAB[c][0]:>12.4f} {d:>+8.4f}")
print(f"  {'MACRO':<6} {np.mean(au):>8.4f} {np.mean(ap):>8.4f} {np.mean(f1):>8.4f}"
      f"   {0.9320:>12.4f} {np.mean(au)-0.9320:>+8.4f}")
check("per-class AUROC reproduces Colab within 0.005", ok_all)
print(f"\n  baseline (archive): AUROC 0.9297  AUPRC 0.7864  F1 0.7172")
print(f"  this model        : AUROC {np.mean(au):.4f}  AUPRC {np.mean(ap):.4f}  "
      f"F1 {np.mean(f1):.4f}")

# ── 5. stale artefacts ───────────────────────────────────────────────────
hdr("5. ARE THE CALIBRATOR / CONFORMAL FILES STALE?")
for f in ("calibrator.json", "conformal_triage.json"):
    fp = os.path.join(CKPT, f)
    if not os.path.exists(fp):
        check(f"{f} present", False, "missing")
        continue
    age = os.path.getmtime(fp) - os.path.getmtime(os.path.join(CKPT, "best_model.pt"))
    stale = age < 0
    check(f"{f} is newer than best_model.pt", not stale,
          "STALE — fitted for a different model, must be refitted" if stale else "ok")

hdr(f"SUMMARY: {npass} passed, {nfail} failed")
sys.exit(0 if nfail == 0 else 1)
