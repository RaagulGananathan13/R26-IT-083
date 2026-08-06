"""
COMPONENT_01 · STAGE 9A · OPERATING-POINT ANALYSIS OF PROJECTION FAIRNESS
========================================================================

THE CLAIM THIS FILE TESTS
-------------------------
Pereira et al. (MIDL 2023, PMLR 227:1199-1210) mitigate chest-radiograph
projection bias with label-conditional gradient reversal, and report:

    macro AUC   84.57 -> 83.66   (-0.91, the price paid)
    TPR Disp    18.19 ->  9.69   (-8.50, the headline gain)
    delta       66.39 -> 73.97
    proj AUC    99.28 -> 61.18

Their headline fairness metric, TPR Disparity, is defined at a single
operating point. Our hypothesis:

    TPR Disparity is a property of the THRESHOLD, not of the model.
    It can be driven toward zero by choosing per-projection thresholds --
    no retraining, no architecture change, no accuracy cost -- while the
    underlying discrimination gap is mathematically untouched.

WHY THE UNDERLYING GAP CANNOT MOVE
----------------------------------
AUROC is computed over the full ranking of scores. A threshold is a single
cut through that ranking. Applying a different cut per group changes which
cases are called positive; it cannot reorder any case. Therefore:

    AUROC_AP, AUROC_PA, and the PA-AP gap are INVARIANT to thresholding.

That invariance is asserted numerically in the notebook, not assumed. If a
strategy ever appears to change AUROC, the harness is broken.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
IT DOES show that the fairness metric the prior work optimises is
manipulable at zero cost, so a reduction in it is weak evidence that a
model became fairer in any clinically meaningful sense.

IT DOES NOT show our system is better than theirs at diagnosis. Different
dataset (MIMIC-CXR vs ChestX-Ray14), different label set (8 vs 14),
different backbone. Absolute numbers are NOT head-to-head and must never
be tabulated as if they were.

IT DOES NOT show thresholding is free in every sense. Equalising TPR moves
cost onto the false-positive rate. That cost is measured and reported here
rather than hidden -- reporting only the improved metric would repeat the
exact error being criticised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# Pereira et al., MIDL 2023 -- Table 2, macro-average row. Percentages.
# Recorded for CONTEXT ONLY. Different dataset/labels/backbone: never place
# these in the same column as our numbers and call it a comparison.
PEREIRA_2023 = {
    "baseline": dict(auc=84.57, tpr_ap=74.77, tpr_pa=71.66, tpr_disp=18.19,
                     delta=66.39, proj_auc=99.28),
    "gradient_reversal": dict(auc=83.66, tpr_ap=77.50, tpr_pa=72.63, tpr_disp=9.69,
                              delta=73.97, proj_auc=61.18),
    "dataset": "ChestX-Ray14 (112,120 images, 14 findings)",
    "backbone": "DenseNet-121 @ 224x224, 5-fold CV",
    "citation": "Pereira et al., MIDL 2023, PMLR 227:1199-1210",
}


# ====================================================================
# 1 · operating-point primitives
# ====================================================================
def tpr_fpr_at(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    """True- and false-positive rate at a threshold. `>=` is the convention
    throughout; mixing `>` and `>=` between fitting and evaluation shifts
    results by one sample and is a classic silent off-by-one."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p, dtype=np.float64) >= thr
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, dtype=np.float64) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return float(2 * pr * rc / (pr + rc + EPS))


def best_f1_threshold(y: np.ndarray, p: np.ndarray, n_grid: int = 512) -> float:
    """F1-optimal threshold. Candidates are score quantiles, so the grid
    adapts to the score distribution instead of assuming it spans [0,1] --
    a well-calibrated rare-disease model may put every score below 0.2."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_grid)))
    scores = [f1_at(y, p, t) for t in qs]
    return float(qs[int(np.argmax(scores))])


def threshold_for_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    """Lowest threshold achieving at least `target_tpr`.

    TPR(t) = fraction of positives scoring >= t, so the threshold delivering
    a given TPR is the (1 - TPR) quantile of the POSITIVE scores. Negatives
    are irrelevant here by construction -- that asymmetry is the whole reason
    TPR disparity is cheap to manipulate.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    pos = p[y == 1]
    if pos.size == 0:
        return float("nan")
    t = float(np.quantile(pos, np.clip(1.0 - target_tpr, 0.0, 1.0)))
    return float(np.nextafter(t, -np.inf))    # make the boundary case inclusive


# ====================================================================
# 2 · the three operating-point strategies
# ====================================================================
def fit_thresholds(y_val: np.ndarray, p_val: np.ndarray, ap_val: np.ndarray,
                   strategy: str) -> dict:
    """Fit on VALIDATION ONLY and return {'AP': thr, 'PA': thr}.

    Fitting on test would guarantee a flattering result that does not
    generalise -- the entire point is to show the effect SURVIVES transfer
    to unseen data, so the test-set disparity will not be exactly zero.
    """
    y, p, ap = (np.asarray(y_val).astype(int), np.asarray(p_val, dtype=np.float64),
                np.asarray(ap_val).astype(bool))

    if strategy == "global":
        t = best_f1_threshold(y, p)
        return {"AP": t, "PA": t}

    if strategy == "per_group_f1":
        return {"AP": best_f1_threshold(y[ap], p[ap]),
                "PA": best_f1_threshold(y[~ap], p[~ap])}

    if strategy == "equal_tpr":
        # Target = the TPR the pooled F1-optimal threshold already achieves.
        # Matching an existing operating point rather than inventing one keeps
        # the comparison honest: overall sensitivity is held roughly fixed.
        t0 = best_f1_threshold(y, p)
        target, _ = tpr_fpr_at(y, p, t0)
        return {"AP": threshold_for_tpr(y[ap], p[ap], target),
                "PA": threshold_for_tpr(y[~ap], p[~ap], target)}

    raise ValueError("unknown strategy " + repr(strategy))


def evaluate_operating_point(y: np.ndarray, p: np.ndarray, ap: np.ndarray,
                             thr: dict) -> dict:
    """All operating-point metrics for ONE pathology on the TEST set."""
    y, p, ap = (np.asarray(y).astype(int), np.asarray(p, dtype=np.float64),
                np.asarray(ap).astype(bool))
    tpr_ap, fpr_ap = tpr_fpr_at(y[ap], p[ap], thr["AP"])
    tpr_pa, fpr_pa = tpr_fpr_at(y[~ap], p[~ap], thr["PA"])

    # Reconstruct predictions under the per-group thresholds for pooled F1.
    pred = np.where(ap, p >= thr["AP"], p >= thr["PA"]).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc + EPS) if tp else 0.0

    return dict(thr_ap=thr["AP"], thr_pa=thr["PA"],
                tpr_ap=tpr_ap, tpr_pa=tpr_pa, tpr_disp=abs(tpr_ap - tpr_pa),
                fpr_ap=fpr_ap, fpr_pa=fpr_pa, fpr_disp=abs(fpr_ap - fpr_pa),
                precision=pr, recall=rc, f1=f1)


def run_strategies(labels_val, probs_val, ap_val, labels_test, probs_test, ap_test,
                   pathologies, auroc_fn, strategies=("global", "per_group_f1", "equal_tpr")):
    """Full grid: every strategy x every pathology, fit on val, scored on test.

    `auroc_fn` is injected so this module stays independent of stage6_acr and
    the AUROC used here is provably the same function Stage 6 reported with.
    """
    out = {}
    for s in strategies:
        per = {}
        for k in pathologies:
            thr = fit_thresholds(labels_val[k].to_numpy(), probs_val[k].to_numpy(),
                                 ap_val, s)
            m = evaluate_operating_point(labels_test[k].to_numpy(),
                                         probs_test[k].to_numpy(), ap_test, thr)
            # Threshold-FREE quantities. Identical across strategies by
            # construction; carried along so the notebook can assert it.
            yk, pk = labels_test[k].to_numpy(), probs_test[k].to_numpy()
            m["auroc"] = auroc_fn(yk, pk)
            m["auroc_ap"] = auroc_fn(yk[ap_test], pk[ap_test])
            m["auroc_pa"] = auroc_fn(yk[~ap_test], pk[~ap_test])
            m["auroc_gap"] = m["auroc_pa"] - m["auroc_ap"]
            per[k] = m
        agg = {f"mean_{f}": float(np.nanmean([per[k][f] for k in pathologies]))
               for f in ("auroc", "auroc_gap", "tpr_ap", "tpr_pa", "tpr_disp",
                         "fpr_ap", "fpr_pa", "fpr_disp", "f1", "precision", "recall")}
        # Pereira's trade-off metric, on their percentage scale.
        agg["delta"] = 100.0 * agg["mean_auroc"] - 100.0 * agg["mean_tpr_disp"]
        out[s] = dict(per_pathology=per, **agg)
    return out


def bootstrap_disparity(y, p, ap, thr, n: int = 1000, seed: int = 0) -> tuple:
    """Stratified bootstrap CI for TPR disparity at a FIXED operating point.

    Thresholds are held fixed rather than refit inside each replicate: we are
    quantifying uncertainty in the disparity of a chosen operating point, not
    in the threshold-selection procedure.
    """
    y, p, ap = (np.asarray(y).astype(int), np.asarray(p, dtype=np.float64),
                np.asarray(ap).astype(bool))
    a_i, p_i = np.where(ap)[0], np.where(~ap)[0]
    rng = np.random.default_rng(seed)
    pt = evaluate_operating_point(y, p, ap, thr)["tpr_disp"]
    vals = np.empty(n)
    for b in range(n):
        ia = rng.choice(a_i, a_i.size, replace=True)
        ip = rng.choice(p_i, p_i.size, replace=True)
        ta, _ = tpr_fpr_at(y[ia], p[ia], thr["AP"])
        tp_, _ = tpr_fpr_at(y[ip], p[ip], thr["PA"])
        vals[b] = abs(ta - tp_)
    vals = vals[np.isfinite(vals)]
    return float(pt), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


# ====================================================================
# 3 · self-tests
# ====================================================================
def _selftest(verbose: bool = True) -> tuple[int, int]:
    P, F = [], []

    def g(name, ok, extra=""):
        (P if ok else F).append(name)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{extra}")

    rng = np.random.default_rng(0)

    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    g("tpr_fpr_at: perfect split", tpr_fpr_at(y, s, 0.5) == (1.0, 0.0))
    g("tpr_fpr_at: threshold above all -> (0,0)", tpr_fpr_at(y, s, 1.1) == (0.0, 0.0))
    g("tpr_fpr_at uses >= (boundary is inclusive)", tpr_fpr_at(y, s, 0.6)[0] == 1.0)
    g("f1_at perfect = 1.0", abs(f1_at(y, s, 0.5) - 1.0) < 1e-9)
    g("f1_at with no positives predicted = 0", f1_at(y, s, 1.1) == 0.0)

    yb = (rng.random(2000) < 0.3).astype(int)
    pb = np.clip(yb * 0.35 + rng.normal(0.3, 0.18, 2000), 0, 1)
    t = best_f1_threshold(yb, pb)
    g("best_f1_threshold beats a fixed 0.5", f1_at(yb, pb, t) >= f1_at(yb, pb, 0.5),
      f"F1 {f1_at(yb,pb,t):.4f} vs {f1_at(yb,pb,0.5):.4f} @0.5")
    g("best_f1_threshold lies inside the score range",
      pb.min() <= t <= pb.max(), f"thr={t:.4f}")

    for tgt in (0.5, 0.8, 0.95):
        th = threshold_for_tpr(yb, pb, tgt)
        got, _ = tpr_fpr_at(yb, pb, th)
        g(f"threshold_for_tpr hits target {tgt}", abs(got - tgt) < 0.02, f"got {got:.4f}")

    # --- the core claim, on data with a KNOWN injected projection gap -------
    n = 6000
    ap = rng.random(n) < 0.6
    y2 = (rng.random(n) < 0.35).astype(int)
    # AP scores are noisier => genuinely worse discrimination on AP
    sc = np.where(ap, y2 * 0.30 + rng.random(n) * 0.70,
                  y2 * 0.80 + rng.random(n) * 0.40)
    half = n // 2
    Lv = pd.DataFrame({"D": y2[:half]}); Pv = pd.DataFrame({"D": sc[:half]})
    Lt = pd.DataFrame({"D": y2[half:]}); Pt = pd.DataFrame({"D": sc[half:]})
    av, at = ap[:half], ap[half:]

    def _auroc(yy, ss):
        yy = np.asarray(yy).astype(int); ss = np.asarray(ss, dtype=np.float64)
        npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
        if npos == 0 or nneg == 0:
            return float("nan")
        r = pd.Series(ss).rank(method="average").to_numpy()
        return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    R = run_strategies(Lv, Pv, av, Lt, Pt, at, ["D"], _auroc)

    g("equal_tpr reduces TPR disparity vs global",
      R["equal_tpr"]["mean_tpr_disp"] < R["global"]["mean_tpr_disp"],
      f"{R['global']['mean_tpr_disp']:.4f} -> {R['equal_tpr']['mean_tpr_disp']:.4f}")

    aurocs = [round(R[s]["mean_auroc"], 12) for s in R]
    g("*** AUROC is IDENTICAL across all strategies ***",
      len(set(aurocs)) == 1, f"{aurocs[0]:.6f}")
    gaps = [round(R[s]["mean_auroc_gap"], 12) for s in R]
    g("*** AUROC GAP is IDENTICAL across all strategies ***",
      len(set(gaps)) == 1, f"{gaps[0]:+.6f}")
    g("the injected discrimination gap is detected and survives",
      gaps[0] > 0.05, f"gap={gaps[0]:+.4f}")

    g("equalising TPR shifts cost onto FPR (reported, not hidden)",
      R["equal_tpr"]["mean_fpr_disp"] >= R["global"]["mean_fpr_disp"] - 1e-9,
      f"FPR disp {R['global']['mean_fpr_disp']:.4f} -> {R['equal_tpr']['mean_fpr_disp']:.4f}")
    g("delta is on Pereira's percentage scale",
      0 < R["global"]["delta"] < 100, f"{R['global']['delta']:.2f}")

    pt, lo, hi = bootstrap_disparity(Lt["D"].to_numpy(), Pt["D"].to_numpy(), at,
                                     {"AP": .5, "PA": .5}, n=200)
    g("bootstrap CI brackets the disparity", lo <= pt <= hi,
      f"[{lo:.4f}, {pt:.4f}, {hi:.4f}]")

    g("Pereira reference numbers are intact",
      PEREIRA_2023["gradient_reversal"]["tpr_disp"] == 9.69
      and PEREIRA_2023["baseline"]["tpr_disp"] == 18.19)

    if verbose:
        print(f"\n  {len(P)} passed, {len(F)} failed")
        for f in F:
            print(f"    - {f}")
    return len(P), len(F)


if __name__ == "__main__":
    print("=" * 78)
    print(" STAGE 9A · stage9_fairness.py self-test")
    print("=" * 78)
    p, f = _selftest()
    raise SystemExit(1 if f else 0)
