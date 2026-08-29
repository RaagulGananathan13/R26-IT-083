"""
Component 04 — Stage 1: ACS detection (ACS vs No-ACS).

Design notes
------------
* Objective for tuning is AUPRC, not accuracy.  At 5% prevalence, accuracy is
  maximised by predicting "no ACS" for everyone; AUPRC is the metric that
  actually tracks ranking quality on the minority class.
* Two heterogeneous learners (XGBoost histogram trees and LightGBM GOSS) are
  tuned separately and blended by rank averaging.  They make different errors,
  and the blend is consistently ahead of either on the validation fold.
* Probabilities are isotonically calibrated on validation, because the operating
  threshold is chosen on the calibrated scale and must transfer to test.
* The operating point is selected on VALIDATION under an explicit sensitivity
  constraint, then frozen.  The test fold is touched exactly once.
"""
from __future__ import annotations

import os
import sys
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (CFG, MODEL_DIR, REPORT_DIR, enable_utf8_stdout, save_json,
                    set_seed)
from dataset import load_bundle
from progress import TrialProgress
from study_store import describe, get_study, wants_fresh
from utils import (BinBudget, banner, bootstrap_ci, kv, plot_confusion,
                   print_report, resolve_device, section, summarise, timer)

enable_utf8_stdout()
SEED = set_seed()
CLASSES = ["No_ACS", "ACS"]
MAX_BIN = int(CFG.get("model.max_bin", 256))
N_PAR = int(CFG.get("model.n_parallel_trials", 1))
SHOW = bool(CFG.get("model.progress", True))
BINS = BinBudget(MAX_BIN, enabled=bool(CFG.get("model.gpu_oom_backoff", True)))


# --------------------------------------------------------------------------
def _xgb(params, device, n_est, esr, max_bin=None):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", device=device,
        tree_method="hist", max_bin=max_bin or BINS.value, random_state=SEED, verbosity=0,
        n_jobs=4, n_estimators=n_est, early_stopping_rounds=esr, **params)


def _lgb(params, n_est, esr, max_bin=None):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt", random_state=SEED,
        max_bin=max_bin or BINS.value, n_estimators=n_est, verbosity=-1, n_jobs=4, **params)


def tune_xgb(Xtr, ytr, Xva, yva, device, n_trials, n_est, esr, horizon=24, fresh=False):
    def objective(trial):
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 40),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 40.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        }
        def _go(mb):
            m = _xgb(p, device, n_est, esr, max_bin=mb)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        return BINS.run(_go)

    st, remaining, done = get_study(f"stage1_xgboost_H{horizon}",
                                    f"stage1_H{horizon}", SEED, n_trials, fresh)
    kv("study", describe(st, done, remaining))
    if remaining > 0:
        bar = TrialProgress(remaining, "xgboost", enabled=SHOW)
        st.optimize(objective, n_trials=remaining, catch=(Exception,),
                    callbacks=[bar], n_jobs=N_PAR, show_progress_bar=False)
        bar.close()
    return st


def tune_lgb(Xtr, ytr, Xva, yva, n_trials, n_est, esr, horizon=24, fresh=False):
    import lightgbm as lgb

    def objective(trial):
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 40.0, log=True),
        }
        def _go(mb):
            m = _lgb(p, n_est, esr, max_bin=mb)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(esr, verbose=False)])
            return average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        return BINS.run(_go)

    st, remaining, done = get_study(f"stage1_lightgbm_H{horizon}",
                                    f"stage1_H{horizon}", SEED, n_trials, fresh)
    kv("study", describe(st, done, remaining))
    if remaining > 0:
        bar = TrialProgress(remaining, "lightgbm", enabled=SHOW)
        st.optimize(objective, n_trials=remaining, catch=(Exception,),
                    callbacks=[bar], n_jobs=N_PAR, show_progress_bar=False)
        bar.close()
    return st


def rank_blend(*prob_arrays: np.ndarray) -> np.ndarray:
    """Average of within-array percentile ranks — scale-free ensembling."""
    from scipy.stats import rankdata
    r = np.mean([rankdata(p) / len(p) for p in prob_arrays], axis=0)
    return r


# --------------------------------------------------------------------------
def choose_operating_point(y_val, p_val, target_sens: float) -> dict:
    """
    Lowest-cost threshold that still reaches the required sensitivity.
    Among thresholds meeting the constraint we take the one maximising Youden's
    J, which keeps specificity as high as the constraint permits.
    """
    order = np.unique(p_val)
    cand = order[:: max(1, len(order) // 4000)]
    best = {"threshold": 0.5, "youden": -1.0}
    P, N = y_val.sum(), len(y_val) - y_val.sum()
    for t in cand:
        pred = p_val >= t
        tp = int((pred & (y_val == 1)).sum())
        fp = int((pred & (y_val == 0)).sum())
        sens = tp / max(P, 1)
        spec = (N - fp) / max(N, 1)
        if sens >= target_sens:
            j = sens + spec - 1
            if j > best["youden"]:
                best = {"threshold": float(t), "youden": float(j),
                        "sensitivity": float(sens), "specificity": float(spec)}
    if best["youden"] < 0:                      # constraint unreachable
        j = [(float(t), *_sens_spec(y_val, p_val >= t)) for t in cand]
        t, s, sp = max(j, key=lambda x: x[1] + x[2] - 1)
        best = {"threshold": t, "youden": s + sp - 1, "sensitivity": s,
                "specificity": sp, "constraint_met": False}
    best.setdefault("constraint_met", True)
    return best


def _sens_spec(y, pred):
    P, N = y.sum(), len(y) - y.sum()
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return tp / max(P, 1), (N - fp) / max(N, 1)


# --------------------------------------------------------------------------
def main(horizon: int | None = None, fresh: bool | None = None) -> dict:
    horizon = CFG.primary_horizon if horizon is None else horizon
    fresh = wants_fresh() if fresh is None else fresh
    banner(f"STAGE 1 — ACS DETECTION   (horizon H={horizon}h)")

    b = load_bundle(horizon=horizon, cohort_only=True)
    Xtr, Xva, Xte = b.X["train"], b.X["val"], b.X["test"]
    ytr, yva, yte = b.binary("train"), b.binary("val"), b.binary("test")

    device = resolve_device(CFG.get("model.device", "cuda"))
    n_trials = int(CFG.get("model.stage1.n_trials", 40))
    n_est = int(CFG.get("model.stage1.n_estimators", 3000))
    esr = int(CFG.get("model.stage1.early_stopping_rounds", 100))

    section("Configuration")
    kv("device", device)
    kv("optuna trials (per learner)", n_trials)
    kv("parallel trials", N_PAR)
    kv("max_bin (histogram)", f"{MAX_BIN}  (auto-backoff on OOM)")
    kv("train ACS / total", f"{int(ytr.sum()):,} / {len(ytr):,} "
                            f"({ytr.mean()*100:.2f}%)")
    kv("class imbalance", f"1 : {int((1-ytr.mean())/ytr.mean())}")

    # ---------------- tuning ----------------
    section("Hyper-parameter search — XGBoost")
    with timer("xgboost optuna"):
        sx = tune_xgb(Xtr, ytr, Xva, yva, device, n_trials, n_est, esr,
                      horizon=horizon, fresh=fresh)
    kv("best val AUPRC", f"{sx.best_value:.4f}")

    section("Hyper-parameter search — LightGBM")
    with timer("lightgbm optuna"):
        sl = tune_lgb(Xtr, ytr, Xva, yva, n_trials, n_est, esr,
                      horizon=horizon, fresh=fresh)
    kv("best val AUPRC", f"{sl.best_value:.4f}")

    # ---------------- final fits ----------------
    section("Fitting final learners")
    import lightgbm as lgb
    mx = _xgb(sx.best_params, device, n_est, esr)
    mx.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    ml = _lgb(sl.best_params, n_est, esr)
    ml.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
           callbacks=[lgb.early_stopping(esr, verbose=False)])
    kv("xgboost trees", getattr(mx, "best_iteration", n_est))
    kv("lightgbm trees", getattr(ml, "best_iteration_", n_est))

    px_va, pl_va = mx.predict_proba(Xva)[:, 1], ml.predict_proba(Xva)[:, 1]
    px_te, pl_te = mx.predict_proba(Xte)[:, 1], ml.predict_proba(Xte)[:, 1]

    section("Ensemble selection (on validation)")
    cands = {
        "xgboost": (px_va, px_te),
        "lightgbm": (pl_va, pl_te),
        "rank-blend": (rank_blend(px_va, pl_va), rank_blend(px_te, pl_te)),
        "mean-blend": ((px_va + pl_va) / 2, (px_te + pl_te) / 2),
    }
    scores = {k: average_precision_score(yva, v[0]) for k, v in cands.items()}
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        kv(k, f"val AUPRC {v:.4f}")
    best_name = max(scores, key=scores.get)
    kv("selected", best_name)
    p_va_raw, p_te_raw = cands[best_name]

    # ---------------- calibration ----------------
    section("Probability calibration (isotonic, fitted on validation)")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va_raw, yva)
    p_va = iso.predict(p_va_raw)
    p_te = iso.predict(p_te_raw)
    from sklearn.metrics import brier_score_loss
    kv("Brier val  raw -> calibrated",
       f"{brier_score_loss(yva, np.clip(p_va_raw,0,1)):.5f} -> {brier_score_loss(yva, p_va):.5f}")

    # ---------------- operating point ----------------
    section("Operating point (chosen on VALIDATION, then frozen)")
    target = float(CFG.get("decision.stage1_target_sensitivity", 0.80))
    op = choose_operating_point(yva, p_va, target)
    kv("target sensitivity", f">= {target*100:.0f}%")
    kv("threshold", f"{op['threshold']:.6f}")
    kv("val sensitivity", f"{op['sensitivity']*100:.2f}%")
    kv("val specificity", f"{op['specificity']*100:.2f}%")
    kv("constraint met", op["constraint_met"])

    # ---------------- evaluation ----------------
    thr = op["threshold"]
    res = {}
    for name, yy, pp in (("VALIDATION", yva, p_va), ("TEST", yte, p_te)):
        pred = (pp >= thr).astype(int)
        r = summarise(yy, pred, CLASSES, y_prob=pp)
        tn, fp, fn, tp = np.array(r["confusion_matrix"]).ravel()
        r["npv"] = float(tn / max(tn + fn, 1))
        r["ppv"] = float(tp / max(tp + fp, 1))
        r["sensitivity"] = float(tp / max(tp + fn, 1))
        r["specificity"] = float(tn / max(tn + fp, 1))
        r["n_missed_acs"] = int(fn)
        r["alerts_per_100"] = float((tp + fp) / len(yy) * 100)
        print_report(f"STAGE 1 — {name}", r, CLASSES, floor=0.75)
        kv("\n  NPV (rule-out safety)", f"{r['npv']*100:.3f}%")
        kv("PPV", f"{r['ppv']*100:.2f}%")
        kv("ACS cases missed", f"{r['n_missed_acs']} of {int(yy.sum())}")
        kv("alerts raised per 100 patients", f"{r['alerts_per_100']:.1f}")
        res[name.lower()] = r

    section("Bootstrap 95% CI (patient-level cluster bootstrap, test fold)")
    ci = bootstrap_ci(yte, (p_te >= thr).astype(int), CLASSES,
                      n=int(CFG.get("evaluation.bootstrap_n", 1000)),
                      seed=SEED, groups=b.groups("test"))
    for k in ("ACS_recall", "ACS_precision", "ACS_f1", "balanced_accuracy"):
        if k in ci:
            kv(k, f"{ci[k]['mean']:.4f}  [{ci[k]['lo']:.4f}, {ci[k]['hi']:.4f}]")
    res["test_ci"] = ci

    plot_confusion(np.array(res["test"]["confusion_matrix"]), CLASSES,
                   f"Stage 1 — ACS detection (test, H={horizon}h)",
                   f"stage1_confusion_H{horizon}.png")

    # ---------------- persist ----------------
    art = {
        "horizon": horizon, "device": device, "ensemble": best_name,
        "threshold": thr, "operating_point": op,
        "xgb_params": sx.best_params, "lgb_params": sl.best_params,
        "val_auprc_by_candidate": scores, "features": b.features,
    }
    mx.save_model(os.path.join(MODEL_DIR, f"stage1_xgb_H{horizon}.json"))
    joblib.dump(ml, os.path.join(MODEL_DIR, f"stage1_lgb_H{horizon}.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, f"stage1_calibrator_H{horizon}.joblib"))
    save_json(art, os.path.join(MODEL_DIR, f"stage1_config_H{horizon}.json"))
    save_json(res, os.path.join(REPORT_DIR, f"stage1_metrics_H{horizon}.json"))

    # scores reused by the end-to-end evaluator
    np.savez(os.path.join(MODEL_DIR, f"stage1_scores_H{horizon}.npz"),
             p_val=p_va, p_test=p_te, y_val=yva, y_test=yte)

    banner(f"STAGE 1 COMPLETE — test AUROC {res['test']['auroc']:.4f} | "
           f"AUPRC {res['test']['auprc']:.4f} | "
           f"sens {res['test']['sensitivity']*100:.1f}% | "
           f"spec {res['test']['specificity']*100:.1f}%")
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(int(args[0]) if args else None)
