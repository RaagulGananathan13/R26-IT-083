"""
UEF-Net training entry point.

    python run_train.py                       # full training (default config)
    python run_train.py --smoke               # 1-epoch, few-iter wiring test
    python run_train.py --epochs 60 --batch-size 8 --grad-accum 4
    python run_train.py --no-pretrained       # skip Kinetics download (offline)

Run from the training/ folder.  Reads preprocessing artifacts automatically.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=CFG.run_name)
    # None means "use config default" for a new run and "use snapshot" for a
    # resume.  This prevents current source defaults from silently changing an
    # existing experiment.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--clip-len", type=int, default=None)
    ap.add_argument("--sampling-period", type=int, default=None)
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--drw-epoch", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop patience on val min-recall")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--n-tta", type=int, default=None,
                    help="number of deterministic label-free VAL/TEST views")
    ap.add_argument("--tta-forward-batch", type=int, default=None)
    ap.add_argument("--cycle-aware-probability", type=float, default=None,
                    help="fraction of train clips using annotated ED/ES guidance")
    ap.add_argument("--no-ema", action="store_true", help="disable weight EMA")
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--model-version", choices=["uefnet_v1", "uefnet_v2"], default=None,
                    help="uefnet_v2 activates the ordered-cutpoint ordinal head, softmax "
                         "class head, uncertainty + rank losses, DRW re-weighting and the "
                         "honest tune/calibration split")
    ap.add_argument("--v2", action="store_true", help="shortcut for --model-version uefnet_v2")
    ap.add_argument("--extra-manifest", action="append", default=None,
                    help="additional TRAIN manifest to co-train on (e.g. CAMUS, "
                         "rich in low/mid-EF cases). Path relative to preprocessing/ "
                         "or absolute. Repeatable. VAL/TEST are never affected.")
    ap.add_argument("--logit-adjustment-tau", type=float, default=None,
                    help="tau for logit-adjusted training (Menon et al. 2021); "
                         "shifts logits by tau*log(class_prior) to lift rare/middle "
                         "classes. 0 disables. Typical 0.5-1.5.")
    ap.add_argument("--backbone", choices=["r2plus1d_18", "r3d_18", "mc3_18"],
                    default=None,
                    help="spatio-temporal backbone. r2plus1d_18 (default) is the "
                         "shipped choice and EchoNet's; r3d_18 is the un-factorised "
                         "baseline it was designed to beat, at matched capacity.")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true",
                    help="continue from outputs/<run>/last.pt (restores optimizer, "
                         "epoch, best score, patience and RNG state)")
    mode.add_argument("--calibrate-only", action="store_true",
                    help="skip training; just re-run the TTA bias-correction + threshold "
                         "calibration on an existing best.pt and rewrite thresholds.json")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    CFG.run_name = a.run_name

    # Resume/calibration must instantiate exactly the architecture and data
    # semantics that produced the checkpoint. Explicit CLI flags below may
    # still extend the epoch count or adjust runtime settings intentionally.
    if a.resume or a.calibrate_only:
        snapshot_path = CFG.OUT_DIR / "config.json"
        if not snapshot_path.exists():
            sys.exit(f"[run_train] saved config required for resume/calibration: {snapshot_path}")
        with open(snapshot_path, "r", encoding="utf-8") as f:
            CFG.restore_for_evaluation(json.load(f))
        CFG.run_name = a.run_name

    overrides = {
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "grad_accum": a.grad_accum,
        "clip_len": a.clip_len,
        "sampling_period": a.sampling_period,
        "lr_backbone": a.lr_backbone,
        "lr_head": a.lr_head,
        "num_workers": a.num_workers,
        "drw_epoch": a.drw_epoch,
        "early_stop_patience": a.patience,
        "seed": a.seed,
        "device": a.device,
        "n_tta_clips": a.n_tta,
        "tta_forward_batch": a.tta_forward_batch,
        "cycle_aware_probability": a.cycle_aware_probability,
        "ema_decay": a.ema_decay,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(CFG, key, value)
    if a.no_ema:
        CFG.use_ema = False
    if a.no_pretrained:
        CFG.pretrained = False
    if a.no_amp:
        CFG.amp = False

    # Model architecture version. --v2 is a shortcut for --model-version uefnet_v2.
    requested_version = "uefnet_v2" if a.v2 else a.model_version
    if requested_version is not None:
        # Changing architecture on an existing checkpoint would make its weights
        # unloadable, so only a fresh run may switch versions from the CLI.
        if (a.resume or a.calibrate_only) and requested_version != CFG.model_version:
            sys.exit(f"[run_train] cannot change model_version from {CFG.model_version!r} "
                     f"to {requested_version!r} on --resume/--calibrate-only; the saved "
                     f"checkpoint is a {CFG.model_version!r} network.")
        CFG.model_version = requested_version

    # Backbone. Guarded exactly like model_version: swapping the architecture
    # under an existing checkpoint would make its weights unloadable, so only a
    # fresh run may choose one from the CLI.
    if a.backbone is not None:
        if (a.resume or a.calibrate_only) and a.backbone != CFG.backbone:
            sys.exit(f"[run_train] cannot change backbone from {CFG.backbone!r} to "
                     f"{a.backbone!r} on --resume/--calibrate-only; the saved "
                     f"checkpoint is a {CFG.backbone!r} network. Start a fresh "
                     f"--run-name instead.")
        CFG.backbone = a.backbone

    # Extra co-training manifests (e.g. CAMUS). Only override when explicitly
    # given, so a resume keeps whatever the snapshot recorded.
    if a.extra_manifest is not None:
        CFG.extra_manifests = tuple(a.extra_manifest)
    if a.logit_adjustment_tau is not None:
        CFG.logit_adjustment_tau = a.logit_adjustment_tau

    if a.num_workers is None and a.smoke:
        CFG.num_workers = 0            # simplest/safest for the wiring test

    try:
        CFG.validate(for_training=not a.calibrate_only)
    except ValueError as e:
        sys.exit(f"[run_train] invalid configuration: {e}")

    if not CFG.MANIFEST.exists():
        sys.exit(f"[run_train] manifest not found: {CFG.MANIFEST}\n"
                 f"Run the preprocessing pipeline first.")

    # Never let a routine new training invocation silently replace a completed
    # experiment. Resume/calibration are the only modes allowed to reuse it.
    if not (a.resume or a.calibrate_only) and (CFG.CKPT_BEST.exists() or CFG.CKPT_LAST.exists()):
        sys.exit(f"[run_train] run {CFG.run_name!r} already has checkpoints in {CFG.OUT_DIR}. "
                 "Choose a new --run-name or use --resume.")

    try:
        frozen_norm = CFG.freeze_norm_stats(overwrite=not (a.resume or a.calibrate_only))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        sys.exit(f"[run_train] cannot freeze normalization statistics: {e}")

    from engine.trainer import Trainer      # imported after path/CFG set
    print(f"[run_train] run={CFG.run_name} | model={CFG.model_version} "
          f"| clip={CFG.clip_len}x{CFG.sampling_period} "
          f"| batch={CFG.batch_size}x{CFG.grad_accum} | epochs={CFG.epochs} "
          f"| workers={CFG.num_workers} | amp={CFG.amp} | pretrained={CFG.pretrained} "
          f"| TTA={CFG.n_tta_clips} label-free views | resume={a.resume}\n"
          f"[run_train] frozen normalization -> {frozen_norm}")
    trainer = Trainer(CFG, smoke=a.smoke, resume=a.resume)

    if a.calibrate_only:
        if not CFG.CKPT_BEST.exists():
            sys.exit(f"[run_train] --calibrate-only needs an existing {CFG.CKPT_BEST}")
        print("[run_train] --calibrate-only: recalibrating existing best.pt "
              "(TTA + bias-correction) ...")
        trainer.finalize_calibration()
        print(f"\n[run_train] done. Evaluate with:\n"
              f"    python run_eval.py --run-name {CFG.run_name}")
        return

    trainer.fit()

    if not a.smoke:
        print("\n[run_train] training complete. Evaluate on TEST with:")
        print(f"    python run_eval.py --run-name {CFG.run_name}")


if __name__ == "__main__":
    main()
