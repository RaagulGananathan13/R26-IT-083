"""
Paired comparison of two trained architectures on the untouched test fold.

WHY A PAIRED TEST AND NOT TWO MEANS
-----------------------------------
Three seeds each gives two means and two spreads, and eyeballing whether the
error bars overlap is not a test. Both architectures are scored on the SAME
1,711 records, so the comparison can be paired: resample record indices, score
both systems on the identical resample, and take the difference. That removes
the variance contributed by which records happen to be in the test fold, which
is shared and therefore not evidence either way.

Two questions are answered separately, because they are different questions:

  per-seed        does a single model of architecture A beat a single model of
                  architecture B? Averaged over seeds, with the seed spread
                  reported so a lucky draw is visible.
  ensemble        does the 3-seed ensemble of A beat the 3-seed ensemble of B?
                  This is the deployable comparison, and it removes seed
                  variance rather than reporting it.

USAGE
-----
    python train/compare_architectures.py \\
        --a checkpoints/resnet --b checkpoints/resnet_se_rerun
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)

from src.models import CLASS_NAMES  # noqa: E402


def load_seeds(directory: str):
    """Every seed's test logits from one run directory, plus the shared labels."""
    labels = os.path.join(directory, "test_labels.npy")
    if not os.path.exists(labels):
        raise SystemExit("no test_labels.npy in %s" % directory)
    Y = np.load(labels)

    logits = []
    for seed in range(16):
        path = os.path.join(directory, f"test_logits_seed{seed}.npy")
        if os.path.exists(path):
            logits.append(np.load(path))
    if not logits:
        raise SystemExit("no test_logits_seed*.npy in %s" % directory)
    return logits, Y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def macro_auroc(P, Y, idx=None):
    if idx is not None:
        P, Y = P[idx], Y[idx]
    scores = []
    for c in range(Y.shape[1]):
        # A resample can omit a rare class entirely; AUROC is undefined there.
        if 0 < Y[:, c].sum() < len(Y):
            scores.append(roc_auc_score(Y[:, c], P[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def macro_auprc(P, Y, idx=None):
    if idx is not None:
        P, Y = P[idx], Y[idx]
    scores = [average_precision_score(Y[:, c], P[:, c])
              for c in range(Y.shape[1]) if Y[:, c].sum() > 0]
    return float(np.mean(scores)) if scores else float("nan")


def paired_bootstrap(Pa, Pb, Y, n=10000, seed=1337):
    """Difference in macro-AUROC, resampling records identically for both.

    Returns (observed, lo, hi, p). The p-value is two-sided and counts how often
    the resampled difference crosses zero, which is the fraction of resamples
    that would have pointed the other way.
    """
    rng = np.random.default_rng(seed)
    observed = macro_auroc(Pa, Y) - macro_auroc(Pb, Y)
    diffs = np.empty(n)
    size = len(Y)
    for i in range(n):
        idx = rng.integers(0, size, size)
        diffs[i] = macro_auroc(Pa, Y, idx) - macro_auroc(Pb, Y, idx)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    crossings = float(np.mean(diffs <= 0) if observed > 0 else np.mean(diffs >= 0))
    return observed, lo, hi, min(1.0, 2 * crossings)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run directory for architecture A")
    ap.add_argument("--b", required=True, help="run directory for architecture B")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    name_a = args.name_a or os.path.basename(args.a.rstrip("/\\"))
    name_b = args.name_b or os.path.basename(args.b.rstrip("/\\"))

    La, Ya = load_seeds(args.a)
    Lb, Yb = load_seeds(args.b)
    if Ya.shape != Yb.shape or not np.array_equal(Ya, Yb):
        raise SystemExit(
            "The two runs were scored on different labels. A paired test needs "
            "the same records in the same order.")
    Y = Ya

    print("=" * 74)
    print("  %s  vs  %s" % (name_a, name_b))
    print("  test fold: %d records, %d classes" % (len(Y), Y.shape[1]))
    print("=" * 74)

    # -- per seed ----------------------------------------------------------
    print("\nPER SEED")
    print("  %-14s %8s %8s" % ("", "AUROC", "AUPRC"))
    rows = {}
    for name, L in ((name_a, La), (name_b, Lb)):
        au = [macro_auroc(sigmoid(x), Y) for x in L]
        pr = [macro_auprc(sigmoid(x), Y) for x in L]
        rows[name] = (au, pr)
        for i, (a, p) in enumerate(zip(au, pr)):
            print("  %-14s %8.4f %8.4f" % ("%s s%d" % (name, i), a, p))
        print("  %-14s %8.4f %8.4f   (mean +/- %.4f)"
              % ("%s MEAN" % name, np.mean(au), np.mean(pr),
                 np.std(au, ddof=1) if len(au) > 1 else 0.0))
    gap = np.mean(rows[name_a][0]) - np.mean(rows[name_b][0])
    print("\n  mean AUROC difference: %+.4f  (%s %s)"
          % (gap, name_a if gap > 0 else name_b, "ahead"))

    # -- ensembles ---------------------------------------------------------
    Pa = sigmoid(np.mean(La, axis=0))
    Pb = sigmoid(np.mean(Lb, axis=0))
    print("\nENSEMBLE OF %d SEEDS EACH  (the deployable comparison)" % len(La))
    print("  %-14s AUROC %.4f   AUPRC %.4f" % (name_a, macro_auroc(Pa, Y), macro_auprc(Pa, Y)))
    print("  %-14s AUROC %.4f   AUPRC %.4f" % (name_b, macro_auroc(Pb, Y), macro_auprc(Pb, Y)))

    observed, lo, hi, p = paired_bootstrap(Pa, Pb, Y, n=args.n_bootstrap)
    print("\nPAIRED BOOTSTRAP  (%d resamples, identical record indices)" % args.n_bootstrap)
    print("  macro-AUROC difference %+.4f   95%% CI [%+.4f, %+.4f]   p = %.4f"
          % (observed, lo, hi, p))
    print("  %s" % ("SIGNIFICANT at alpha=0.05" if p < 0.05
                    else "not significant at alpha=0.05"))

    # -- per class ---------------------------------------------------------
    print("\nPER CLASS (ensembles)")
    print("  %-8s %9s %9s %9s" % ("class", name_a[:9], name_b[:9], "delta"))
    for c, cls in enumerate(CLASS_NAMES):
        a = roc_auc_score(Y[:, c], Pa[:, c])
        b = roc_auc_score(Y[:, c], Pb[:, c])
        print("  %-8s %9.4f %9.4f %+9.4f" % (cls, a, b, a - b))

    print("\n" + "=" * 74)
    print("  Both architectures trained with the same script, data, schedule and")
    print("  seeds. The only difference is the network.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
