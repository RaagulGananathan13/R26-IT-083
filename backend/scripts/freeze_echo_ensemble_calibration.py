"""
Fit and persist Component 03's ENSEMBLE-level decision rule.

WHY
---
Component 03's published headline (MAE 3.979, min-class recall 0.723) comes
from a decision rule fitted on the *ensemble's* validation predictions.
`run_ensemble.py` refits that rule on every invocation and never writes it out;
the only persisted rules are each member's own `outputs/<run>/thresholds.json`.

So a served study could only be graded by a *member's* rule applied to the
ensemble average -- close, but not the rule the reported numbers describe. This
script closes that gap: it reproduces `run_ensemble.py`'s calibration step
exactly, using the component's own `run_predictions` and `calibrate`, and
writes the resulting calibration to the backend cache. The echo adapter prefers
it when present and says so in `model.decision_rule`.

Nothing inside Component 03 is written to.

USAGE
-----
    python scripts/freeze_echo_ensemble_calibration.py
    python scripts/freeze_echo_ensemble_calibration.py --n-tta 10 --device cuda

Cost: one full pass over the 1,288-study validation split for every ensemble
member. Expect several minutes on a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tta", type=int, default=None,
                        help="clips per study (default: the service's setting)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from cvxai.core.logging import configure_logging, get_logger
    from cvxai.core.registry import get_registry
    from cvxai.settings import get_settings

    configure_logging("INFO")
    log = get_logger("freeze_echo")
    settings = get_settings()
    adapter = get_registry(settings).get("echo")

    if adapter.root is None:
        log.error("Component 03 root not found")
        return 2

    n_tta = args.n_tta or settings.echo_tta_clips
    runs = adapter._serving_runs()                       # noqa: SLF001
    log.info("members: %s | n_tta=%d", ", ".join(runs), n_tta)

    started = time.perf_counter()
    with adapter.sandbox.active():
        # The component's own routines, imported not reimplemented.
        sys.path.insert(0, str(adapter.root / "training"))
        import run_ensemble as RE                        # type: ignore
        from config import CFG                           # type: ignore
        from engine.calibrate import calibrate           # type: ignore

        device = args.device or ("cuda" if settings.device != "cpu" else "cpu")
        predictions = []
        for run in runs:
            log.info("predicting VAL with %s ...", run)
            predictions.append(RE.run_predictions(
                run, "VAL", n_tta, device, args.num_workers))

        ensemble = RE.average_predictions(predictions)
        log.info("ensemble VAL predictions: %d studies", len(ensemble["y_true"]))

        # Called exactly as run_ensemble.py calls it, so the fitted rule is the
        # one the published figures describe.
        calibration = calibrate(
            ensemble["ef_pred"], ensemble["y_true"], ensemble["ord_pred"], CFG,
            ef_true=ensemble["ef_true"], ord_dist=ensemble.get("ord_dist"),
            class_dist=ensemble.get("class_dist"),
            pred_std=ensemble.get("ef_pred_std"))

    strategy = calibration.get("best_strategy")
    best = calibration["strategies"][strategy]
    log.info("VAL-selected strategy : %s", strategy)
    log.info("  min-class recall    : %.4f", best["min_class_recall"])
    log.info("  balanced accuracy   : %.4f", best["balanced_acc"])
    log.info("  macro F1            : %.4f", best["macro_f1"])

    payload = {
        "schema_version": 2,
        "source": "ensemble VAL calibration, fitted by "
                  "backend/scripts/freeze_echo_ensemble_calibration.py",
        "runs": runs,
        "n_tta": n_tta,
        "split": "VAL",
        "n_calibration": int(len(ensemble["y_true"])),
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": strategy,
        "calibration": calibration,
    }
    out_path = Path(args.out) if args.out else (
        settings.cache_dir / "echo" / "ensemble_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")

    log.info("written: %s", out_path)
    log.info("elapsed: %.1f min", (time.perf_counter() - started) / 60.0)
    log.info("Restart the service; the echo adapter will pick it up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
