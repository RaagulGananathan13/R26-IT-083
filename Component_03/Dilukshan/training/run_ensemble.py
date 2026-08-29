"""
UEF-Net ensemble evaluation.
============================

Averages the per-video outputs of several independently-trained runs (different
seeds, optionally CAMUS co-training / logit adjustment), then applies ONE
decision strategy that is selected on the VAL calibration split and frozen for
TEST -- so there is no test-set tuning, exactly like run_eval.py.

    python run_ensemble.py --runs uefnet_v2 uefnet_v2b uefnet_v2c --n-tta 10

Each run is loaded with its OWN saved config + frozen normalization statistics.
All runs must share the same VAL/TEST split (the EchoNet split), which they do
because CAMUS/extra manifests only ever enter TRAIN.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import CFG


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names to ensemble")
    ap.add_argument("--split", default="TEST", choices=["TEST", "VAL"])
    ap.add_argument("--n-tta", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--out", default=None, help="output json path (default: outputs/ensemble_report.json)")
    ap.add_argument("--save-predictions", action="store_true",
                    help="also write per-study predictions (.npz) so subgroup and "
                         "paired-significance analysis can run without re-inference")
    return ap.parse_args()


def print_confusion(cm, names):
    cm = np.asarray(cm)
    w = max(len(n) for n in names) + 2
    print(" " * (w + 2) + "".join(f"{n[:8]:>10}" for n in names) + "   (pred)")
    for i, n in enumerate(names):
        print(f"{n:>{w}}  " + "".join(f"{cm[i, j]:>10}" for j in range(len(names))))
    print("  (true, rows)")


def run_predictions(run_name, split, n_tta, device_str, num_workers):
    """Restore one run and return its per-video predictions on `split`."""
    import torch
    from torch.utils.data import DataLoader
    from core.common import (get_device, load_checkpoint, make_torch_generator,
                             seed_everything, seed_worker)
    from data.dataset import EchoClipDataset
    from models.uef_net import UEFNet
    from engine.evaluate import run_inference

    # fresh restore into the global CFG for this run
    CFG.run_name = run_name
    snap = CFG.OUT_DIR / "config.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            CFG.restore_for_evaluation(json.load(f))
    CFG.run_name = run_name
    CFG.pretrained = False
    CFG.n_tta_clips = n_tta
    CFG.num_workers = num_workers
    if device_str is not None:
        CFG.device = device_str
    if not CFG.CKPT_BEST.exists():
        sys.exit(f"[ensemble] checkpoint not found for run {run_name!r}: {CFG.CKPT_BEST}")
    # frozen normalization statistics for THIS run
    if CFG.FROZEN_NORM_JSON.exists():
        CFG.freeze_norm_stats(overwrite=False)

    seed_everything(CFG.seed)
    device = get_device(CFG.device)
    ds = EchoClipDataset(split, CFG, train=False, n_views=n_tta)
    loader = DataLoader(ds, batch_size=max(2, CFG.batch_size // 2), shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker, generator=make_torch_generator(CFG.seed))
    model = UEFNet(CFG).to(device)
    load_checkpoint(CFG.CKPT_BEST, model, map_location=device)
    model.eval()
    preds = run_inference(model, loader, CFG, device, ds.ef_mean, ds.ef_std,
                          desc=f"{run_name} {split} TTA-{n_tta}")
    return preds


def average_predictions(pred_list):
    """Mean per-video EF and class distributions across runs (aligned order)."""
    ref = pred_list[0]
    n = len(ref["y_true"])
    for p in pred_list:
        if len(p["y_true"]) != n or not np.array_equal(p["y_true"], ref["y_true"]):
            raise RuntimeError("runs disagree on the evaluation set / ordering; "
                               "ensembling requires the same split for every run")
    out = {
        "y_true": ref["y_true"].astype(np.int64),
        "ef_true": ref["ef_true"].astype(np.float64),
        "ef_pred": np.mean([p["ef_pred"] for p in pred_list], axis=0),
        "ef_pred_std": np.mean([p["ef_pred_std"] for p in pred_list], axis=0),
        "ord_dist": np.mean([p["ord_dist"] for p in pred_list], axis=0),
    }
    out["ord_pred"] = out["ord_dist"].argmax(axis=1).astype(np.int64)
    if all("class_dist" in p for p in pred_list):
        out["class_dist"] = np.mean([p["class_dist"] for p in pred_list], axis=0)
        out["class_pred"] = out["class_dist"].argmax(axis=1).astype(np.int64)
    return out


def main():
    a = parse_args()
    from engine.calibrate import calibrate, apply_frozen_strategy
    from data.dataset import EchoClipDataset
    from engine.metrics import classify_ef, classification_metrics, regression_metrics
    from core.common import write_json_atomic

    # 1) VAL: gather per-run predictions and select ONE strategy on the ensemble.
    val_runs = [run_predictions(r, "VAL", a.n_tta, a.device, a.num_workers) for r in a.runs]
    val_ens = average_predictions(val_runs)
    calibration = calibrate(
        val_ens["ef_pred"], val_ens["y_true"], val_ens["ord_pred"], CFG,
        ef_true=val_ens["ef_true"], ord_dist=val_ens.get("ord_dist"),
        class_dist=val_ens.get("class_dist"), pred_std=val_ens.get("ef_pred_std"))
    print(f"\n[ensemble] VAL-selected strategy: {calibration['best_strategy']}")

    # 2) Target split: same averaging, then apply the frozen strategy.
    if a.split == "VAL":
        test_ens = val_ens
    else:
        test_runs = [run_predictions(r, a.split, a.n_tta, a.device, a.num_workers) for r in a.runs]
        test_ens = average_predictions(test_runs)

    frozen = apply_frozen_strategy(test_ens, calibration, CFG)
    y_pred = frozen["operational_pred"]
    ef_used = np.asarray(frozen["ef_calibrated"], dtype=np.float64)
    clinical_pred = frozen["clinical_pred"]

    cls = classification_metrics(test_ens["y_true"], y_pred, CFG.n_classes)
    reg = regression_metrics(test_ens["ef_true"], ef_used)
    clin_cls = classification_metrics(test_ens["y_true"], clinical_pred, CFG.n_classes)
    clin_reg = regression_metrics(test_ens["ef_true"], np.asarray(frozen["ef_raw"], dtype=np.float64))

    print(f"\n============ ENSEMBLE {a.split} RESULTS ({len(a.runs)} runs) ============")
    print(f"  runs               : {', '.join(a.runs)}")
    print(f"  strategy           : {calibration['best_strategy']} (frozen on VAL)")
    print(f"  n videos           : {len(test_ens['y_true'])}")
    print(f"  MAE                : {reg['mae']:.3f} EF pts")
    print(f"  RMSE / R2          : {reg['rmse']:.3f} / {reg['r2']:.3f}")
    print(f"  overall accuracy   : {cls['overall_acc']:.3f}")
    print(f"  balanced accuracy  : {cls['balanced_acc']:.3f}")
    print(f"  macro-F1           : {cls['macro_f1']:.3f}")
    print(f"  MIN class recall   : {cls['min_class_recall']:.3f}  "
          f"({'>=75% ALL CLASSES' if cls['min_class_recall'] >= 0.75 else 'below 0.75'})")
    print("\n  Per-class recall:")
    for c, name in enumerate(CFG.CLASS_NAMES):
        r = cls["per_class_recall"][c]
        flag = "OK " if (r is not None and r >= 0.75) else "<75"
        print(f"    [{flag}] {name:<16} {'n/a' if r is None else f'{r:.3f}'}")
    print("\n  Confusion matrix:")
    print_confusion(cls["confusion"], CFG.CLASS_NAMES)
    print(f"\n  clinical reference (raw avg-EF @30/40/55): MAE {clin_reg['mae']:.3f} | "
          f"overall {clin_cls['overall_acc']:.3f} | min-recall {clin_cls['min_class_recall']:.3f}")

    out_path = a.out or str(CFG.TRAIN_DIR / "outputs" / "ensemble_report.json")
    write_json_atomic(out_path, dict(
        runs=a.runs, split=a.split, n_tta=a.n_tta,
        strategy=calibration["best_strategy"], n=len(test_ens["y_true"]),
        regression=reg, classification=cls,
        clinical_reference=dict(regression=clin_reg, classification=clin_cls)))
    print(f"\n  report -> {out_path}")

    if a.save_predictions:
        # Per-study arrays, in dataset order, so subgroup and paired-significance
        # analysis can be run later without repeating inference.
        ds_order = EchoClipDataset(a.split, CFG, train=False, n_views=1).files
        pred_path = str(CFG.TRAIN_DIR / "outputs" /
                        f"predictions_{a.split.lower()}_{'_'.join(a.runs)}.npz")
        np.savez_compressed(
            pred_path,
            file_name=np.asarray(ds_order, dtype=object),
            y_true=test_ens["y_true"], ef_true=test_ens["ef_true"],
            ef_pred_raw=np.asarray(frozen["ef_raw"], dtype=np.float64),
            ef_pred_calibrated=ef_used,
            y_pred_operational=y_pred, y_pred_clinical=clinical_pred,
            ef_pred_std=test_ens.get("ef_pred_std", np.zeros(len(y_pred))),
            ord_dist=test_ens.get("ord_dist", np.zeros((len(y_pred), CFG.n_classes))),
            class_dist=test_ens.get("class_dist", np.zeros((len(y_pred), CFG.n_classes))),
            strategy=np.asarray(calibration["best_strategy"]),
            runs=np.asarray(a.runs, dtype=object))
        print(f"  predictions -> {pred_path}")
    print("=====================================================")


if __name__ == "__main__":
    main()
