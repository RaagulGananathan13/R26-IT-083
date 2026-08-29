"""
Post-hoc experiment: can we lift min-per-class recall WITHOUT retraining, by
correcting the regression-to-the-mean COMPRESSION in the predicted EF?

The confusion matrix shows extreme classes leaking inward (Severe->Moderate,
Normal->Mild) because predicted EF is squeezed toward the mean. We test a
variance-matching expansion  EF' = mean_t + (std_t/std_p)*(EF_pred - mean_p),
fitted on VAL, then re-optimise thresholds on VAL and report on TEST.
All fitting is on VAL only (no test leakage).

Run:  python experiments/posthoc_calibration.py   (from the training/ folder)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from config import CFG
CFG.run_name = "uefnet_r2p1d"
CFG.pretrained = False

import torch
from torch.utils.data import DataLoader
from core.common import get_device, load_checkpoint
from data.dataset import EchoClipDataset
from models.uef_net import UEFNet
from engine.evaluate import run_inference
from engine.calibrate import optimize_thresholds
from engine.metrics import classify_ef, per_class_recall


def infer(model, dev, split):
    ds = EchoClipDataset(split, CFG, train=False, n_views=CFG.n_tta_clips)
    ld = DataLoader(ds, batch_size=max(2, CFG.batch_size // 2), shuffle=False,
                    num_workers=CFG.num_workers, pin_memory=True)
    p = run_inference(model, ld, CFG, dev, ds.ef_mean, ds.ef_std, desc=split)
    return p["ef_pred"], p["y_true"], p["ef_true"]


def report(name, ef_val, vy, ef_test, ty, tt, n):
    _, _, thr = optimize_thresholds(ef_val, vy, n)
    yp = classify_ef(ef_test, thr)
    rec = per_class_recall(ty, yp, n)
    mae = float(np.mean(np.abs(ef_test - tt)))
    print(f"{name:26s} thr={[round(x,1) for x in thr]}  MAE={mae:.2f}  "
          f"per-class={np.round(rec,3).tolist()}  MIN={np.nanmin(rec):.3f}")
    return np.nanmin(rec)


def main():
    dev = get_device("cuda")
    model = UEFNet(CFG).to(dev)
    load_checkpoint(CFG.CKPT_BEST, model, map_location=dev)
    model.eval()

    vp, vy, vt = infer(model, dev, "VAL")
    tp, ty, tt = infer(model, dev, "TEST")
    n = CFG.n_classes

    print("\n========== POST-HOC STRATEGIES (fit on VAL, report on TEST) ==========")
    report("raw (baseline)", vp, vy, tp, ty, tt, n)

    # variance-matching expansion, params fitted on VAL only
    mp, sp = float(vp.mean()), float(vp.std())
    mt_, st = float(vt.mean()), float(vt.std())
    k = st / (sp + 1e-6)
    print(f"[expansion] std_pred={sp:.2f} std_true={st:.2f} -> full gain k={k:.3f}")

    ve = np.clip(mt_ + k * (vp - mp), 0, 100)
    te = np.clip(mt_ + k * (tp - mp), 0, 100)
    report("variance-expanded", ve, vy, te, ty, tt, n)

    for kk in (1.15, 1.3, 1.5):
        ve = np.clip(mt_ + kk * (vp - mp), 0, 100)
        te = np.clip(mt_ + kk * (tp - mp), 0, 100)
        report(f"expand k={kk}", ve, vy, te, ty, tt, n)
    print("======================================================================")


if __name__ == "__main__":
    main()
