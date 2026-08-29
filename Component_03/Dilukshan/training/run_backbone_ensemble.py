"""
Backbone ablation at ensemble scale: three R3D seeds against three R(2+1)D seeds.

WHY THIS EXISTS
---------------
`run_backbone_ablation.py` compares ONE R3D model against ONE R(2+1)D model.
That answers "did this seed do better?", and a panel can fairly reply: "you
trained one of each -- maybe you got lucky." Averaging three seeds per
architecture removes seed variance from the comparison, so a difference that
survives is an architecture difference rather than a draw from the seed
distribution.

It also matches like with like. The shipped headline (MAE 3.979, min-recall
0.723) is a THREE-SEED ensemble. Comparing it against a single R3D model reads
variance reduction as an architecture effect. This script builds the R3D
counterpart so ensemble is compared against ensemble.

THE PARITY PROBLEM THIS SCRIPT EXISTS TO PREVENT
------------------------------------------------
The first backbone ablation was not single-variable. `run_backbone_ablation.py`
forwarded `--extra-manifest` from the baseline snapshot but nothing else, so the
challenger silently fell back to `config.py` defaults for everything the
baseline had set from the command line:

    logit_adjustment_tau    baseline 0.5   challenger 0.0     loss differs
    n_tta_clips             baseline 5     challenger 10      model selection
                                                              and calibration differ

Two variables moved besides the backbone. The result was still a null, but a
null on "backbone and loss and TTA, jointly", which is not what was claimed.

So this script does two things differently:

  1. It inherits EVERY hyperparameter from the baseline snapshot, by mapping
     config keys back to their CLI flags, rather than listing a few by hand.
  2. Before it reports any comparison it AUDITS the resulting configs and
     REFUSES to print numbers if anything except the backbone, the seed and
     their consequences differs. The audit is fail-closed: a config field added
     to the project later is audited automatically, because the exclusion list
     names what may differ rather than what must match.

An ablation you cannot trust is worse than no ablation, because you will quote
it.

COST
----
Three R3D runs at roughly 4.0 h each on the documented RTX 4060 Laptop, so
about 12 h, plus ~30 min for ensembling and the paired tests. Safe to interrupt
at any point: `--resume` picks up from wherever it stopped, including part-way
through an epoch of a part-finished run.

PROGRESS REPORTING
------------------
Training progress is the per-epoch bar that `run_train.py` already prints; this
script deliberately does not draw a second live bar over it, because two bars
writing to the same terminal corrupt each other. Instead each run is preceded by
a banner giving position, elapsed time and a projected finish based on the runs
already measured.

USAGE
-----
    python run_backbone_ensemble.py                  # train, ensemble, compare
    python run_backbone_ensemble.py --resume         # continue after a stop
    python run_backbone_ensemble.py --compare-only   # skip training
    python run_backbone_ensemble.py --audit-only     # just check config parity
    python run_backbone_ensemble.py --smoke          # wiring test, minutes
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
OUT_DIR = TRAIN_DIR / "outputs"
sys.path.insert(0, str(TRAIN_DIR))

#: Report produced by the shipped three-seed R(2+1)D ensemble. Its `runs` list
#: is the authority on which runs form the baseline, so the two ensembles are
#: guaranteed to be built from the same seeds.
BASELINE_ENSEMBLE = OUT_DIR / "ensemble_report.json"

#: Written by this script after `run_train.py` returns 0. `last.pt` alone cannot
#: distinguish "stopped early by patience" from "interrupted", and reading the
#: epoch back out of a 500 MB checkpoint to find out is wasteful.
TRAIN_DONE = ".train_complete"

#: Config keys allowed to differ between baseline and challenger.
#:
#: Everything NOT named here must match, so a hyperparameter added to the
#: project in future is audited by default instead of being silently ignored.
#: This is the fail-closed direction: a new field causes a loud mismatch rather
#: than a quiet hole in the comparison.
PARITY_EXEMPT = {
    "backbone",           # the variable under test
    "seed",               # varies per ensemble member by design
    "run_name",
    "params_M",           # a consequence of the backbone
    "feat_dim",           # ditto
    "pretrained_loaded",  # ditto (both are 512-d, but recorded per run)
    "device",             # runtime, not science
    "num_workers",        # runtime -- but see the warning in audit_parity()
    "norm_stats",         # frozen stats blob, compared via NORM_JSON instead
}

#: Config keys that change how augmentation RNG is consumed. They do not change
#: the science, but they do stop two runs being bit-identical, so a mismatch is
#: reported as a warning rather than an error.
PARITY_SOFT = {"num_workers"}

#: config key -> run_train.py CLI flag. Anything present in the baseline
#: snapshot and listed here is forwarded, so the challenger cannot fall back to
#: a config.py default the baseline had overridden.
CONFIG_TO_FLAG = {
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "grad_accum": "--grad-accum",
    "clip_len": "--clip-len",
    "sampling_period": "--sampling-period",
    "lr_backbone": "--lr-backbone",
    "lr_head": "--lr-head",
    "drw_epoch": "--drw-epoch",
    "early_stop_patience": "--patience",
    "n_tta_clips": "--n-tta",
    "tta_forward_batch": "--tta-forward-batch",
    "cycle_aware_probability": "--cycle-aware-probability",
    "ema_decay": "--ema-decay",
    "logit_adjustment_tau": "--logit-adjustment-tau",
    "model_version": "--model-version",
}


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def config_of(run: str) -> dict | None:
    return read_json(OUT_DIR / run / "config.json")


def hours(seconds: float) -> str:
    return "%.2f h" % (seconds / 3600.0)


def rule(char: str = "=") -> str:
    return char * 78


def banner(title: str) -> None:
    print("\n" + rule())
    print("  " + title)
    print(rule(), flush=True)


def spawn(cmd: list[str]) -> int:
    """Run a child process, inheriting stdout so its tqdm bar stays live."""
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(TRAIN_DIR))


# --------------------------------------------------------------------------- #
#  planning
# --------------------------------------------------------------------------- #
def discover_baseline() -> tuple[list[str], dict]:
    """Baseline member runs and the snapshot every challenger inherits from."""
    report = read_json(BASELINE_ENSEMBLE)
    if not report or not report.get("runs"):
        sys.exit("[ensemble-ablation] %s is missing or has no 'runs' list.\n"
                 "                    Build the baseline ensemble first:\n"
                 "                        python run_ensemble.py --runs uefnet_v3 "
                 "uefnet_v3b uefnet_v3c --n-tta 10" % BASELINE_ENSEMBLE)
    members = list(report["runs"])

    snapshot = config_of(members[0])
    if snapshot is None:
        sys.exit("[ensemble-ablation] cannot read config for baseline member %r" % members[0])
    return members, snapshot


def member_seeds(members: list[str]) -> list[int]:
    """Seeds of the baseline members, so challengers are matched seed-for-seed."""
    seeds = []
    for run in members:
        cfg = config_of(run)
        if cfg is None or cfg.get("seed") is None:
            sys.exit("[ensemble-ablation] baseline member %r has no recorded seed" % run)
        seeds.append(int(cfg["seed"]))
    return seeds


def train_command(run: str, backbone: str, seed: int, snapshot: dict,
                  num_workers: int | None, smoke: bool) -> list[str]:
    """Full training command with every hyperparameter inherited explicitly."""
    cmd = [sys.executable, "run_train.py",
           "--run-name", run,
           "--backbone", backbone,
           "--seed", str(seed)]

    for key, flag in CONFIG_TO_FLAG.items():
        if key in snapshot and snapshot[key] is not None:
            cmd += [flag, str(snapshot[key])]

    # Repeatable flag, so it cannot go through CONFIG_TO_FLAG.
    for manifest in snapshot.get("extra_manifests") or []:
        cmd += ["--extra-manifest", str(manifest)]

    # Boolean switches are expressed as negations on the CLI.
    if snapshot.get("use_ema") is False:
        cmd.append("--no-ema")
    if snapshot.get("pretrained") is False:
        cmd.append("--no-pretrained")
    if snapshot.get("amp") is False:
        cmd.append("--no-amp")

    # num_workers changes the augmentation RNG stream, so it is inherited by
    # default rather than left to the machine's convenience.
    workers = num_workers if num_workers is not None else snapshot.get("num_workers")
    if workers is not None:
        cmd += ["--num-workers", str(workers)]

    if smoke:
        cmd.append("--smoke")
    return cmd


def run_state(run: str) -> str:
    """One of: done, needs-eval, resumable, pending."""
    directory = OUT_DIR / run
    trained = (directory / TRAIN_DONE).exists()
    evaluated = (directory / "test_report.json").exists()
    if trained and evaluated:
        return "done"
    if trained:
        return "needs-eval"
    if (directory / "last.pt").exists() or (directory / "best.pt").exists():
        return "resumable"
    return "pending"


# --------------------------------------------------------------------------- #
#  the parity audit -- the reason this script exists
# --------------------------------------------------------------------------- #
def audit_parity(baseline: str, challengers: list[str]) -> tuple[bool, list[str]]:
    """Confirm only the backbone, the seed and their consequences differ.

    Returns (ok, lines). `ok` is False if any hard mismatch was found, in which
    case the caller must not report a comparison.
    """
    reference = config_of(baseline)
    if reference is None:
        return False, ["cannot read baseline config for %r" % baseline]

    lines: list[str] = []
    hard_failures = 0

    for run in challengers:
        candidate = config_of(run)
        if candidate is None:
            lines.append("  %-22s CONFIG MISSING" % run)
            hard_failures += 1
            continue

        # Path keys are recorded in SHOUTING_CASE and legitimately differ
        # between machines and after a project move.
        keys = {k for k in set(reference) | set(candidate) if not k.isupper()}
        keys -= PARITY_EXEMPT

        mismatches, softs, drifts = [], [], []
        for key in sorted(keys):
            here, there = reference.get(key, "<absent>"), candidate.get(key, "<absent>")
            if here == there:
                continue
            if "<absent>" in (here, there):
                # One snapshot predates the field. Schema drift, not a
                # different experiment -- report it but do not fail on it.
                drifts.append((key, here, there))
            elif key in PARITY_SOFT:
                softs.append((key, here, there))
            else:
                mismatches.append((key, here, there))

        if mismatches:
            hard_failures += 1
            lines.append("  %-22s %d MISMATCH%s" % (run, len(mismatches),
                                                    "" if len(mismatches) == 1 else "ES"))
            for key, here, there in mismatches:
                lines.append("      %-26s baseline=%-18s challenger=%s"
                             % (key, str(here)[:18], str(there)[:18]))
        else:
            lines.append("  %-22s matched" % run)

        for key, here, there in softs:
            lines.append("      warn  %-20s baseline=%-18s challenger=%s"
                         % (key, str(here)[:18], str(there)[:18]))
            lines.append("            (does not change the science; stops the runs "
                         "being bit-identical)")
        for key, here, there in drifts:
            lines.append("      drift %-20s baseline=%-18s challenger=%s"
                         % (key, str(here)[:18], str(there)[:18]))
            lines.append("            (field added after the baseline was trained)")

    return hard_failures == 0, lines


# --------------------------------------------------------------------------- #
#  reporting
# --------------------------------------------------------------------------- #
def summarise(report: dict | None) -> dict:
    if not report:
        return {}
    regression = report.get("regression", {}) or {}
    classification = report.get("classification", {}) or {}
    return {
        "MAE": regression.get("mae"),
        "R2": regression.get("r2"),
        "min-class recall": classification.get("min_class_recall"),
        "balanced accuracy": classification.get("balanced_acc"),
        "overall accuracy": classification.get("overall_acc"),
        "macro F1": classification.get("macro_f1"),
        "within-one-class": classification.get("within_one_class_acc"),
    }


def print_comparison(baseline_report: Path, challenger_report: Path,
                     baseline_name: str, challenger_name: str) -> None:
    left = summarise(read_json(baseline_report))
    right = summarise(read_json(challenger_report))
    if not left or not right:
        print("\n[ensemble-ablation] a report is missing; cannot compare.")
        return

    banner("ENSEMBLE vs ENSEMBLE -- three seeds each, matched seed-for-seed")
    print("  %-22s %14s %14s %12s" % ("metric", baseline_name, challenger_name, "delta"))
    print("  " + "-" * 66)
    lower_is_better = {"MAE"}
    for key in left:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            continue
        delta = b - a
        improved = (delta < 0) if key in lower_is_better else (delta > 0)
        if abs(delta) < 1e-9:
            mark = "  same"
        else:
            mark = "  better" if improved else "  worse"
        print("  %-22s %14.4f %14.4f %+12.4f%s" % (key, a, b, delta, mark))
    print(rule())
    print("  Both sides are three-seed ensembles over the same seeds, so seed")
    print("  variance is removed from the comparison rather than absorbed into it.")
    print("  Read the paired p-values below, not these point differences.")
    print(rule())


# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="r3d_18", choices=["r3d_18", "mc3_18"],
                        help="challenger architecture (default: r3d_18)")
    parser.add_argument("--prefix", default=None,
                        help="run-name prefix (default: uefnet_<backbone-short>)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="default: the baseline members' seeds, so they match")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="default: the baseline's, which keeps the RNG stream matched")
    parser.add_argument("--ensemble-n-tta", type=int, default=None,
                        help="default: whatever the baseline ensemble used")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted job (safe to pass always)")
    parser.add_argument("--compare-only", action="store_true",
                        help="skip training and ensembling; report what exists")
    parser.add_argument("--rebuild-ensemble", action="store_true",
                        help="regenerate the challenger ensemble even if its report exists")
    parser.add_argument("--audit-only", action="store_true",
                        help="only check config parity, then exit")
    parser.add_argument("--skip-audit", action="store_true",
                        help="report even if parity fails (you must justify this)")
    parser.add_argument("--smoke", action="store_true",
                        help="few-iteration wiring test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    members, snapshot = discover_baseline()
    seeds = args.seeds if args.seeds is not None else member_seeds(members)
    short = {"r3d_18": "r3d", "mc3_18": "mc3"}[args.backbone]
    # A smoke run writes a checkpoint after a handful of iterations. Under the
    # real run name the resume logic would later find it and continue a proper
    # 45-epoch run from a few junk steps, silently poisoning the experiment.
    # Give smoke its own namespace so that cannot happen.
    default_prefix = ("smoketest_%s" % short) if args.smoke else ("uefnet_%s" % short)
    prefix = args.prefix or default_prefix
    challengers = ["%s_s%d" % (prefix, seed) for seed in seeds]

    baseline_report = BASELINE_ENSEMBLE
    challenger_report = OUT_DIR / ("ensemble_report_%s.json" % short)
    baseline_n_tta = (read_json(baseline_report) or {}).get("n_tta", 10)
    ensemble_n_tta = args.ensemble_n_tta if args.ensemble_n_tta is not None else baseline_n_tta

    banner("BACKBONE ABLATION AT ENSEMBLE SCALE")
    print("  baseline    : %s" % ", ".join(members))
    print("                (%s, seeds %s)"
          % (snapshot.get("backbone"), ", ".join(str(s) for s in member_seeds(members))))
    print("  challenger  : %s" % ", ".join(challengers))
    print("                (%s, seeds %s)"
          % (args.backbone, ", ".join(str(s) for s in seeds)))
    print("  inherited   : %d hyperparameters forwarded from %s"
          % (sum(1 for k in CONFIG_TO_FLAG if k in snapshot), members[0]))
    print("  varied      : backbone only (audited before any number is reported)")
    print("  ensemble TTA: %d views, matching the baseline" % ensemble_n_tta)

    if args.audit_only:
        banner("CONFIG PARITY AUDIT")
        on_disk = [r for r in challengers if (OUT_DIR / r / "config.json").exists()]
        if not on_disk:
            # An audit of nothing is not a pass. Say so, rather than printing a
            # green verdict that only means "no runs exist yet".
            print("  No challenger runs on disk yet -- nothing to audit.")
            print("  Train them first, then re-run this to verify parity.")
            print(rule())
            return 0
        ok, lines = audit_parity(members[0], on_disk)
        print("\n".join(lines))
        print(rule())
        print("  %s" % ("PASS -- only the backbone and seed differ." if ok
                        else "FAIL -- see mismatches above."))
        return 0 if ok else 1

    # ---------------- phase 1+2: train and evaluate each seed ---------------- #
    if not args.compare_only:
        states = {run: run_state(run) for run in challengers}
        todo = [r for r in challengers if states[r] != "done"]

        print("\n  plan:")
        for index, run in enumerate(challengers, 1):
            print("    [%d/%d] %-22s %s" % (index, len(challengers), run, states[run]))
        if not todo:
            print("\n  All challenger runs are already trained and evaluated.")
        else:
            estimate = 4.0 if args.backbone == "r3d_18" else 4.0
            print("\n  %d run(s) to go, roughly %.0f h total. Safe to interrupt; "
                  "re-run with --resume." % (len(todo), estimate * len(todo)))

        durations: list[float] = []
        for index, run in enumerate(challengers, 1):
            state = run_state(run)
            if state == "done":
                print("\n[%d/%d] %s -- already complete, skipping." % (index, len(challengers), run))
                continue

            done_count = len(durations)
            if durations:
                mean = sum(durations) / len(durations)
                remaining = mean * (len(todo) - done_count)
                eta = " | est. %s remaining after this one" % hours(remaining - mean) \
                    if len(todo) - done_count > 1 else ""
            else:
                eta = ""
            # ASCII only: the Windows console this runs on renders U+00B7 as
            # mojibake under the default code page.
            banner("[%d/%d] %s | %s%s" % (index, len(challengers), run, state, eta))

            started = time.perf_counter()
            if state in ("pending", "resumable"):
                if state == "resumable":
                    # The snapshot supplies backbone/seed/epochs on resume, and
                    # passing them again trips run_train's change guards.
                    cmd = [sys.executable, "run_train.py", "--run-name", run, "--resume"]
                    workers = args.num_workers if args.num_workers is not None \
                        else snapshot.get("num_workers")
                    if workers is not None:
                        cmd += ["--num-workers", str(workers)]
                else:
                    cmd = train_command(run, args.backbone, seeds[index - 1], snapshot,
                                        args.num_workers, args.smoke)

                code = spawn(cmd)
                if code != 0:
                    print("\n[ensemble-ablation] %s exited with code %d.\n"
                          "                    Re-run with --resume to continue from last.pt."
                          % (run, code))
                    return code
                (OUT_DIR / run / TRAIN_DONE).write_text("ok\n", encoding="utf-8")
                elapsed = time.perf_counter() - started
                durations.append(elapsed)
                print("\n[ensemble-ablation] %s trained in %s" % (run, hours(elapsed)))

            if args.smoke:
                print("[ensemble-ablation] --smoke: stopping before evaluation.")
                return 0

            if spawn([sys.executable, "run_eval.py", "--run-name", run]) != 0:
                print("[ensemble-ablation] evaluation failed for %s." % run)
                return 1

    # ---------------- phase 3: parity audit, fail-closed ---------------- #
    banner("CONFIG PARITY AUDIT")
    print("  Confirming the only thing that changed is the backbone.\n")
    ok, lines = audit_parity(members[0], challengers)
    print("\n".join(lines))
    print(rule())
    if ok:
        print("  PASS -- only the backbone, the seed and their consequences differ.")
    else:
        print("  FAIL -- the runs differ in more than the backbone.")
        print("  A comparison built on these would not be a backbone ablation.")
        if not args.skip_audit:
            print("\n  Refusing to report. Retrain the mismatched runs, or pass")
            print("  --skip-audit and state the confound wherever you quote the result.")
            return 2
        print("\n  --skip-audit given: reporting anyway. State the confound.")
    print(rule())

    # ---------------- phase 4: build the challenger ensemble ---------------- #
    # Resume-consistent with the training phase: existing work is not redone.
    # Ensembling is ~25 min of TTA inference, so silently repeating it on every
    # invocation would make the script painful to re-run for the report alone.
    if challenger_report.exists() and not args.rebuild_ensemble:
        banner("CHALLENGER ENSEMBLE ALREADY BUILT")
        print("  %s" % challenger_report.name)
        print("  Pass --rebuild-ensemble to regenerate it.")
    else:
        banner("ENSEMBLING %d %s MODELS" % (len(challengers), args.backbone))
        code = spawn([sys.executable, "run_ensemble.py",
                      "--runs", *challengers,
                      "--n-tta", str(ensemble_n_tta),
                      "--save-predictions",
                      "--out", str(challenger_report)])
        if code != 0:
            print("[ensemble-ablation] ensembling failed.")
            return 1

    # ---------------- phase 5: report ---------------- #
    print_comparison(baseline_report, challenger_report,
                     "%s x3" % snapshot.get("backbone", "baseline"),
                     "%s x3" % args.backbone)

    # run_ensemble.py names its prediction dump after every run that went into
    # it: predictions_<split>_<run1>_<run2>_<run3>.npz. Naming a single member
    # here would silently compare one model against an ensemble, which is the
    # exact error this script exists to avoid.
    def predictions_path(runs: list[str]) -> Path:
        return OUT_DIR / ("predictions_test_%s.npz" % "_".join(runs))

    challenger_npz = predictions_path(challengers)
    baseline_npz = predictions_path(members)
    robustness_out = OUT_DIR / ("robustness_ensemble_%s.json" % short)

    banner("PAIRED SIGNIFICANCE TESTS")
    if not baseline_npz.exists():
        # The shipped baseline ensemble was built without --save-predictions,
        # so there is nothing to pair against yet. Rebuilding is deterministic:
        # the calibration is frozen, so the report reproduces and only the
        # missing .npz is added.
        print("  The baseline ensemble has no saved per-study predictions, so the")
        print("  paired test cannot run yet. Rebuild it (deterministic -- the")
        print("  frozen calibration reproduces the published report):\n")
        print("    python run_ensemble.py --runs %s \\" % " ".join(members))
        print("        --n-tta %d --save-predictions \\" % ensemble_n_tta)
        print("        --out outputs/ensemble_report_baseline_verify.json")
        print("\n  Writing to a verify path rather than over ensemble_report.json")
        print("  keeps the published headline intact and lets you diff the two.")
        print("\n  Then:\n")
        print("    python run_robustness.py \\")
        print("        --predictions outputs/%s \\" % challenger_npz.name)
        print("        --compare-with outputs/%s \\" % baseline_npz.name)
        print("        --out outputs/%s" % robustness_out.name)
        print(rule())
        return 0

    print("  Pairing %d studies, ensemble against ensemble.\n" % (read_json(
        challenger_report) or {}).get("n", 0))
    code = spawn([sys.executable, "run_robustness.py",
                  "--predictions", str(challenger_npz),
                  "--compare-with", str(baseline_npz),
                  "--out", str(robustness_out)])
    if code != 0:
        print("\n[ensemble-ablation] the paired test failed; run it manually.")
        return 1
    print("\n[ensemble-ablation] paired report -> %s" % robustness_out)
    print(rule())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
