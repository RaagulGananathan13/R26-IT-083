"""
UEF-Net final evaluation on the TEST split, with multi-clip TTA and the
val-frozen decision strategy/thresholds.

    python run_eval.py --run-name uefnet_r2p1d

Writes outputs/<run>/test_report.json and prints the confusion matrix + the
per-class accuracy table that answers the 75%-per-class question.
Run from the training/ folder.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import CFG


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=CFG.run_name)
    ap.add_argument("--split", default="TEST", choices=["TEST", "VAL"])
    ap.add_argument("--n-tta", type=int, default=None,
                    help="number of deterministic full-video views (default: saved run config)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    return ap.parse_args()


def print_confusion(cm, names):
    cm = np.asarray(cm)
    w = max(len(n) for n in names) + 2
    print(" " * (w + 2) + "".join(f"{n[:8]:>10}" for n in names) + "   (pred)")
    for i, n in enumerate(names):
        print(f"{n:>{w}}  " + "".join(f"{cm[i, j]:>10}" for j in range(len(names))))
    print("  (true, rows)")


def main():
    a = parse_args()
    CFG.run_name = a.run_name
    try:
        CFG.validate(for_training=False)  # validate identity before constructing paths
    except ValueError as e:
        sys.exit(f"[run_eval] invalid configuration: {e}")

    # Restore architecture, clip semantics, class definitions, and the exact
    # preprocessing statistics from the run being evaluated.  Absolute paths
    # in historical snapshots are intentionally ignored for portability.
    snapshot_path = CFG.OUT_DIR / "config.json"
    snapshot = {}
    if snapshot_path.exists():
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        CFG.restore_for_evaluation(snapshot)
    else:
        print(f"[run_eval][WARN] saved config not found: {snapshot_path}; "
              "using current code defaults for this legacy checkpoint")
    CFG.run_name = a.run_name                  # never adopt snapshot path identity
    CFG.pretrained = False                     # weights come from checkpoint
    if a.n_tta is not None:
        CFG.n_tta_clips = a.n_tta
    if a.num_workers is not None:
        CFG.num_workers = a.num_workers
    if a.device is not None:
        CFG.device = a.device
    try:
        CFG.validate(for_training=False)
    except ValueError as e:
        sys.exit(f"[run_eval] invalid configuration: {e}")

    if not CFG.CKPT_BEST.exists():
        sys.exit(f"[run_eval] checkpoint not found: {CFG.CKPT_BEST}. Train first.")
    if not CFG.THRESH_JSON.exists():
        sys.exit(f"[run_eval] thresholds not found: {CFG.THRESH_JSON}. Train first.")

    with open(CFG.THRESH_JSON, "r", encoding="utf-8") as f:
        strat = json.load(f)

    norm_source = "saved config"
    if CFG.FROZEN_NORM_JSON.exists():
        # Prefer the immutable run-local artifact.  This also validates fields.
        try:
            CFG.freeze_norm_stats(overwrite=False)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            sys.exit(f"[run_eval] invalid frozen normalization file: {e}")
        norm_source = str(CFG.FROZEN_NORM_JSON)
    elif CFG.norm_stats:
        norm_source = "embedded saved config"
    else:
        # Old checkpoints did not snapshot pixel statistics.  Preserve their
        # saved EF scaling from thresholds.json, and loudly disclose that the
        # current preprocessing pixel statistics are the only recoverable copy.
        live_norm = CFG.PREP_DIR / "artifacts" / "norm_stats.json"
        if not live_norm.exists():
            sys.exit("[run_eval] this legacy run has no frozen normalization statistics "
                     f"and the preprocessing copy is missing: {live_norm}")
        with open(live_norm, "r", encoding="utf-8") as f:
            CFG.norm_stats = json.load(f)
        for key in ("ef_mean", "ef_std"):
            if key in strat:
                CFG.norm_stats[key] = strat[key]
        try:
            CFG._validate_norm_stats(CFG.norm_stats, live_norm)
        except ValueError as e:
            sys.exit(f"[run_eval] invalid normalization statistics: {e}")
        norm_source = f"legacy live fallback: {live_norm}"
        print("[run_eval][WARN] legacy run has no frozen pixel normalization; "
              "using the current preprocessing copy. Reproduce old results only if "
              "that file has not changed.")

    import torch
    from torch.utils.data import DataLoader
    from core.common import (get_device, load_checkpoint, make_torch_generator,
                             seed_everything, seed_worker, write_json_atomic)
    from data.dataset import EchoClipDataset
    from models.uef_net import UEFNet
    from engine.evaluate import run_inference
    from engine.metrics import classify_ef, classification_metrics, regression_metrics
    from engine.calibrate import apply_expansion, apply_frozen_strategy

    seed_everything(CFG.seed)
    device = get_device(CFG.device)
    schema_version = int(strat.get("schema_version", 1))
    calibration = strat.get("calibration")
    use_frozen_strategy = schema_version >= 2 and isinstance(calibration, dict)
    val_min = strat.get("val_min_recall")
    val_min_text = "unknown" if val_min is None else f"{float(val_min):.3f}"
    if use_frozen_strategy:
        print(f"[run_eval] frozen VAL-calibrated strategy={strat.get('strategy')} "
              f"(schema v{schema_version}) | clinical_primary={calibration.get('clinical_primary')} "
              f"| ef_calibration={strat.get('ef_calibration')} "
              f"| n_calibration={calibration.get('n_calibration')} (val min-recall {val_min_text})")
    else:
        print(f"[run_eval] frozen strategy={strat['strategy']} thresholds={strat.get('thresholds')} "
              f"expansion={strat.get('expansion')} (val min-recall {val_min_text})")
    protocol = str(strat.get("protocol", ""))
    # The frozen calibration must use the SAME clip count as this evaluation.
    # Accept both the legacy ("TTA_Nclip") and current ("label_free_TTA_Nclip")
    # protocol strings; only warn when the view count actually differs.
    if protocol and f"{CFG.n_tta_clips}clip" not in protocol:
        print(f"[run_eval][WARN] calibration protocol is {protocol}, but evaluation uses "
              f"{CFG.n_tta_clips} TTA views; recalibrate VAL with the same view count "
              f"for a final report")
    if CFG.eval_use_keyframes:
        print("[run_eval][WARN] eval_use_keyframes=True uses privileged tracing annotations "
              "and is not a deployable or label-free benchmark")

    ds = EchoClipDataset(a.split, CFG, train=False, n_views=CFG.n_tta_clips)
    loader = DataLoader(ds, batch_size=max(2, CFG.batch_size // 2), shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker,
                        generator=make_torch_generator(CFG.seed))

    model = UEFNet(CFG).to(device)
    load_checkpoint(CFG.CKPT_BEST, model, map_location=device)
    model.eval()

    preds = run_inference(model, loader, CFG, device, ds.ef_mean, ds.ef_std,
                          desc=f"{a.split} TTA-{CFG.n_tta_clips}")

    # Apply the decision rule that was frozen on VAL.  No test-set tuning ever
    # happens here: for schema v2 the whole strategy (EF affine, thresholds,
    # temperature, blend weights) was selected on a held-out VAL calibration
    # split; for legacy runs we replay the stored expansion/thresholds.
    clinical_cls = None
    clinical_reg = None
    if use_frozen_strategy:
        frozen = apply_frozen_strategy(preds, calibration, CFG)
        # Deployable rule the model actually selected on VAL calibration.
        y_pred = frozen["operational_pred"]
        ef_used = np.asarray(frozen["ef_calibrated"], dtype=np.float64)
        # Primary clinical result: RAW regression on published EF boundaries,
        # with no post-hoc decision rule — reported alongside for honesty.
        clinical_pred = frozen["clinical_pred"]
        clinical_ef = np.asarray(frozen["ef_raw"], dtype=np.float64)
        clinical_cls = classification_metrics(preds["y_true"], clinical_pred, CFG.n_classes)
        clinical_reg = regression_metrics(preds["ef_true"], clinical_ef)
    elif strat["strategy"] == "ordinal_argmax":
        y_pred = preds["ord_pred"]
        ef_used = preds["ef_pred"]
    else:
        ef_used = apply_expansion(preds["ef_pred"], strat.get("expansion"))
        thr = strat["thresholds"] if strat["thresholds"] else CFG.EF_THRESHOLDS
        y_pred = classify_ef(ef_used, thr)

    cls = classification_metrics(preds["y_true"], y_pred, CFG.n_classes)
    reg = regression_metrics(preds["ef_true"], ef_used)   # MAE on the corrected EF

    print("\n================ {} RESULTS ================".format(a.split))
    print(f"  n videos           : {len(preds['y_true'])}")
    print(f"  strategy           : {strat.get('strategy')}"
          f"{'  (frozen on VAL calibration split)' if use_frozen_strategy else ''}")
    print(f"  MAE                : {reg['mae']:.3f}  EF pts")
    print(f"  RMSE / R2          : {reg['rmse']:.3f} / {reg['r2']:.3f}")
    print(f"  overall accuracy   : {cls['overall_acc']:.3f}")
    print(f"  balanced accuracy  : {cls['balanced_acc']:.3f}")
    print(f"  macro-F1           : {cls['macro_f1']:.3f}")
    print(f"  MIN class recall   : {cls['min_class_recall']:.3f}  "
          f"({'>=75% ALL CLASSES ACHIEVED' if cls['min_class_recall'] >= 0.75 else 'below 0.75 target'})")
    print("\n  Per-class accuracy (recall):")
    for c, name in enumerate(CFG.CLASS_NAMES):
        r = cls["per_class_recall"][c]
        flag = "OK " if (r is not None and r >= 0.75) else "<75"
        value = "n/a" if r is None else f"{r:.3f}"
        print(f"    [{flag}] {name:<16} {value}")
    print("\n  Confusion matrix:")
    print_confusion(cls["confusion"], CFG.CLASS_NAMES)

    if clinical_cls is not None:
        print("\n  --- clinical reference (RAW regression @ 30/40/55, no decision rule) ---")
        print(f"  MAE {clinical_reg['mae']:.3f} | overall {clinical_cls['overall_acc']:.3f} | "
              f"balanced {clinical_cls['balanced_acc']:.3f} | macro-F1 {clinical_cls['macro_f1']:.3f} | "
              f"min-recall {clinical_cls['min_class_recall']:.3f}")
        parts = []
        for c, name in enumerate(CFG.CLASS_NAMES):
            r = clinical_cls["per_class_recall"][c]
            parts.append(f"{name.split('(')[0]} " + ("n/a" if r is None else f"{r:.3f}"))
        print("  per-class recall: " + "; ".join(parts))

    report = dict(split=a.split, strategy=strat.get("strategy"),
                  schema_version=schema_version,
                  thresholds=strat.get("thresholds"), n=len(preds["y_true"]),
                  regression=reg, classification=cls, n_tta=CFG.n_tta_clips,
                  calibration_protocol=protocol or None,
                  label_free=not CFG.eval_use_keyframes,
                  normalization_source=norm_source,
                  mean_tta_ef_std=float(np.mean(preds["ef_pred_std"])))
    if clinical_cls is not None:
        report["clinical_reference"] = dict(
            rule="raw_regression_published_boundaries",
            regression=clinical_reg, classification=clinical_cls)
    report_path = CFG.report_path(a.split)
    write_json_atomic(report_path, report)

    # report-quality confusion matrix plot
    from core.plots import plot_confusion
    cm_png = CFG.OUT_DIR / f"confusion_{a.split.lower()}.png"
    plot_confusion(cls["confusion"], CFG.CLASS_NAMES, cm_png, recalls=cls["per_class_recall"])

    print(f"\n  report -> {report_path}")
    print(f"  confusion plot -> {cm_png}")
    print("=====================================================")


if __name__ == "__main__":
    main()
