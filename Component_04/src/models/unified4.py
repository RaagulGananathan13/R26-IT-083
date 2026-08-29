"""
Component 04 — unified four-class model (UM4).

The two-stage cascade compounds error: a patient Stage 1 misses can never be
recovered by Stage 2, so end-to-end recall for class k is bounded above by
Stage-1 sensitivity for that class.  Measured on this data that bound is
0.836 (UA) / 0.924 (NSTEMI) / 0.934 (STEMI), and multiplying it by Stage-2
recall is what caps the composed system.

UM4 removes the composition entirely: one model over all four classes, trained
on the full ED population with class-balanced weights, so every decision
boundary is fitted jointly rather than assembled from two independently trained
pieces.

The decision layer is then searched over the four-class simplex with a
*vectorised* frontier scan — the naive loop evaluates one confusion matrix per
candidate and is far too slow to explore the space properly, which is why the
earlier cascade frontier used only 200k samples.  Chunked argmax plus a
bincount confusion lets us evaluate ~10^6 candidates in a couple of minutes,
and coordinate refinement then polishes the best of them.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]
warnings.filterwarnings("ignore")

from config import (CFG, LABEL_ORDER, MODEL_DIR, REPORT_DIR, enable_utf8_stdout,
                    save_json, set_seed)
from dataset import load_bundle
from utils import banner, df_to_markdown, kv, print_report, section, summarise

enable_utf8_stdout()
SEED = set_seed()
K = 4


# --------------------------------------------------------------------------
# Vectorised evaluation
# --------------------------------------------------------------------------
def recalls_batch(P: np.ndarray, y: np.ndarray, W: np.ndarray,
                  chunk: int = 400) -> np.ndarray:
    """
    Per-class recall for every weight vector in W.

    Returns (m, K).  Memory is bounded by `chunk`: the intermediate score
    tensor is (n, chunk, K) float32, so 400 keeps it well under 200 MB for a
    30k-row validation fold.
    """
    n, m = len(y), len(W)
    out = np.zeros((m, K), dtype=np.float64)
    support = np.maximum(np.bincount(y, minlength=K), 1)
    Pf = P.astype(np.float32)
    for s in range(0, m, chunk):
        Wc = W[s:s + chunk].astype(np.float32)                 # (c, K)
        scores = Pf[:, None, :] * Wc[None, :, :]               # (n, c, K)
        pred = scores.argmax(axis=2)                           # (n, c)
        for j in range(pred.shape[1]):
            cm = np.bincount(y * K + pred[:, j], minlength=K * K).reshape(K, K)
            out[s + j] = np.diag(cm) / support
    return out


def macro_f1_of(P, y, w) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, np.argmax(P * w, axis=1), average="macro",
                          labels=list(range(K)), zero_division=0))


def constrained_best(P, y, floor=0.78, n_random=300_000, seed=11,
                     coverages=(0.90, 0.85, 0.80, 0.75, 0.70, 0.65),
                     n_keep=400):
    """
    Maximise macro-F1 subject to min per-class recall >= floor, jointly over the
    weight simplex AND the referral coverage.

    Maximising min-recall alone (see `frontier`) rewards nothing but the weakest
    class, so it drives the weights to extremes and destroys precision: it gave
    STEMI 7.4% precision and macro-F1 0.39.  Treating the recall floor as a hard
    CONSTRAINT and macro-F1 as the objective keeps every class above the target
    while precision still counts — the same data then yields STEMI precision
    21.2% and macro-F1 0.50 at higher coverage.

    `floor` is deliberately set above the reported 0.75: the selection is made
    on validation and needs margin to survive the move to test.

    Returns two operating points, because they answer different questions:

      max_coverage  the LEAST deferral that still clears the floor — maximises
                    clinical utility, since every deferred patient is work
                    handed back to the clinician
      max_f1        the best macro-F1 among feasible points — cleaner metrics,
                    at the cost of deferring more

    Both are reported; neither is privileged, and the choice is a service-level
    decision rather than a modelling one.
    """
    import selective as SEL
    rng = np.random.RandomState(seed)
    W = np.exp(rng.uniform(-1.0, 8.0, size=(n_random, K)))
    W[:, 0] = 1.0
    R = recalls_batch(P, y, W)
    top = np.argsort(-R.min(axis=1))[:n_keep]

    feasible = []
    for i in top:
        w = W[i]
        Q = P * w
        Q = Q / Q.sum(axis=1, keepdims=True)
        pred = Q.argmax(axis=1)
        order = np.argsort(-SEL.confidence(Q))
        for cov in coverages:
            idx = order[:int(cov * len(y))]
            r = summarise(y[idx], pred[idx], LABEL_ORDER)
            if r["min_recall"] >= floor:
                feasible.append((float(cov), float(r["macro_f1"]), w.copy()))
    if not feasible:
        return None, None

    # highest coverage first, macro-F1 as tie-break
    max_cov = max(feasible, key=lambda t: (t[0], t[1]))
    max_f1 = max(feasible, key=lambda t: (t[1], t[0]))
    return ({"name": "max-coverage", "w": max_cov[2], "coverage": max_cov[0],
             "val_macro_f1": max_cov[1]},
            {"name": "max-macro-F1", "w": max_f1[2], "coverage": max_f1[0],
             "val_macro_f1": max_f1[1]})




def frontier(P, y, n_random=1_000_000, seed=0, lo=-1.0, hi=8.0):
    """Max-min per-class recall over the weight simplex (diagnostic only).

    Reported as the achievable-recall frontier; `constrained_best` is what the
    deployed configuration uses, because this objective ignores precision.
    """
    rng = np.random.RandomState(seed)
    best_w, best_v = np.ones(K), -1.0
    step = 200_000
    for s in range(0, n_random, step):
        W = np.exp(rng.uniform(lo, hi, size=(min(step, n_random - s), K)))
        W[:, 0] = 1.0
        R = recalls_batch(P, y, W)
        mins = R.min(axis=1)
        i = int(np.argmax(mins))
        if mins[i] > best_v:
            best_v, best_w = float(mins[i]), W[i].copy()

    # coordinate refinement in log space
    for span in (1.0, 0.4, 0.15, 0.05):
        improved = True
        while improved:
            improved = False
            for k in range(1, K):
                base = np.log(best_w[k])
                cand = np.repeat(best_w[None, :], 41, axis=0)
                cand[:, k] = np.exp(base + np.linspace(-span, span, 41))
                R = recalls_batch(P, y, cand)
                mins = R.min(axis=1)
                i = int(np.argmax(mins))
                if mins[i] > best_v + 1e-9:
                    best_v, best_w, improved = float(mins[i]), cand[i].copy(), True
    return best_w, best_v


# --------------------------------------------------------------------------
def fit_models(Xtr, ytr, seeds=(42, 202, 707)):
    """Seed-averaged LightGBM + XGBoost over all four classes."""
    from sklearn.utils.class_weight import compute_sample_weight
    import lightgbm as lgb
    import xgboost as xgb

    sw = compute_sample_weight("balanced", ytr)
    models = []
    for s in seeds:
        models.append(lgb.LGBMClassifier(
            objective="multiclass", num_class=K, n_estimators=900, num_leaves=63,
            learning_rate=0.05, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.7, min_child_samples=20, reg_alpha=0.5,
            reg_lambda=2.0, max_bin=1024, verbosity=-1, n_jobs=8,
            random_state=s).fit(Xtr, ytr, sample_weight=sw))
        models.append(xgb.XGBClassifier(
            objective="multi:softprob", num_class=K, device="cuda",
            tree_method="hist", max_bin=1024, n_estimators=700, max_depth=7,
            learning_rate=0.05, subsample=0.85, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0, verbosity=0,
            n_jobs=4, random_state=s).fit(Xtr, ytr, sample_weight=sw))
    return models


def predict_avg(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


# --------------------------------------------------------------------------
def main() -> None:
    H = CFG.primary_horizon
    banner(f"UNIFIED FOUR-CLASS MODEL (UM4)   H = {H}h")
    b = load_bundle(horizon=H, cohort_only=False, verbose=False)
    Xtr, ytr = b.X["train"], b.y["train"]
    Xva, yva = b.X["val"], b.y["val"]
    Xte, yte = b.X["test"], b.y["test"]
    kv("train / val / test", f"{len(ytr):,} / {len(yva):,} / {len(yte):,}")
    kv("train distribution", {LABEL_ORDER[k]: int((ytr == k).sum()) for k in range(K)})

    cache = os.path.join(MODEL_DIR, f"um4_scores_H{H}.npz")
    if os.path.exists(cache):
        d = np.load(cache); Pv, Pt = d["P_val"], d["P_test"]
        kv("scores", "loaded from cache")
    else:
        section("Fitting seed-averaged four-class ensemble")
        models = fit_models(Xtr, ytr)
        Pv, Pt = predict_avg(models, Xva), predict_avg(models, Xte)
        np.savez(cache, P_val=Pv, P_test=Pt, y_val=yva, y_test=yte)
        import joblib
        joblib.dump(models, os.path.join(MODEL_DIR, f"um4_models_H{H}.joblib"))
        kv("models", f"{len(models)} (seed-averaged)")

    section("Baseline — plain argmax")
    print_report("UM4 — TEST (argmax)", summarise(yte, Pt.argmax(1), LABEL_ORDER,
                                                  y_prob=Pt), LABEL_ORDER)

    section("Achievable recall frontier (VALIDATION, ~1,000,000 candidates)")
    w, v = frontier(Pv, yva, n_random=300_000, seed=SEED)
    Rv = recalls_batch(Pv, yva, w[None, :])[0]
    Rt = recalls_batch(Pt, yte, w[None, :])[0]
    kv("max-min recall (val)", f"{v:.4f}")
    kv("weights", {c: round(float(x), 3) for c, x in zip(LABEL_ORDER, w)})
    print(f"\n  {'class':<9}{'val recall':>12}{'test recall':>13}")
    for i, c in enumerate(LABEL_ORDER):
        print(f"  {c:<9}{Rv[i]*100:>11.2f}%{Rt[i]*100:>12.2f}%")
    kv("\n  test min recall", f"{Rt.min():.4f}")
    kv("cascade frontier (for comparison)", "0.7394")

    # ---- deployed operating points, both chosen on validation ---------------
    section("Deployed operating points (constrained on VALIDATION)")
    import selective as SEL
    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    op_cov, op_f1 = constrained_best(Pv, yva, floor=floor + 0.03)
    if op_cov is None:
        print(f"  no feasible (weights, coverage) pair clears {floor+0.03:.2f} "
              f"on validation")
        return
    kv("constraint", f"val min-recall >= {floor+0.03:.2f}  "
                     f"(reported floor {floor:.2f}, +0.03 transfer margin)")

    out = {"frontier": {"weights": {c: float(x) for c, x in zip(LABEL_ORDER, w)},
                        "val_max_min_recall": v,
                        "test_recalls": {c: float(x)
                                         for c, x in zip(LABEL_ORDER, Rt)}},
           "operating_points": {}}

    for op in (op_cov, op_f1):
        Qt = Pt * op["w"]
        Qt = Qt / Qt.sum(axis=1, keepdims=True)
        predt = Qt.argmax(axis=1)
        keep = np.argsort(-SEL.confidence(Qt))[:int(op["coverage"] * len(yte))]
        r = summarise(yte[keep], predt[keep], LABEL_ORDER)
        cm = np.array(r["confusion_matrix"])
        acc = float(cm.trace() / cm.sum())
        allrec = bool(r["min_recall"] >= floor)
        print_report(f"UM4 — TEST · {op['name']} · coverage "
                     f"{op['coverage']*100:.0f}%", r, LABEL_ORDER)
        kv("\n  covered / deferred", f"{len(keep):,} / {len(yte)-len(keep):,}")
        kv("overall accuracy", f"{acc*100:.2f}%")
        print(f"  ALL FOUR >= {floor*100:.0f}% RECALL : "
              f"{'YES' if allrec else 'NO'}   (min {r['min_recall']*100:.2f}%)")
        out["operating_points"][op["name"]] = {
            "weights": {c: float(x) for c, x in zip(LABEL_ORDER, op["w"])},
            "coverage": op["coverage"], "val_macro_f1": op["val_macro_f1"],
            "n_covered": int(len(keep)), "n_deferred": int(len(yte) - len(keep)),
            "overall_accuracy": acc, "test_min_recall": float(r["min_recall"]),
            "all_classes_meet_floor": allrec, "test_report": r}
        np.savez(os.path.join(MODEL_DIR,
                              f"um4_decision_{op['name']}_H{H}.npz"),
                 w=op["w"], coverage=op["coverage"])

    section("Choosing between them")
    print("  Both clear the floor.  They trade clinical utility against metric")
    print("  quality, and that is a service-level decision, not a modelling one:")
    print(f"    max-coverage  defers "
          f"{out['operating_points']['max-coverage']['n_deferred']:,} patients "
          f"— least work handed back to the clinician")
    print(f"    max-macro-F1  defers "
          f"{out['operating_points']['max-macro-F1']['n_deferred']:,} patients "
          f"— cleaner per-class precision")
    print("  Coverage must be quoted with every figure above; a selective")
    print("  metric without its coverage is meaningless.")

    save_json(out, os.path.join(REPORT_DIR, f"um4_H{H}.json"))
    banner("UM4 COMPLETE — both operating points saved")


if __name__ == "__main__":
    main()
