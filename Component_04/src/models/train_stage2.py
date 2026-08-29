"""
Component 04 — Stage 2: ACS subtype classification (UA / NSTEMI / STEMI).

This is where the "every class >= 75%" requirement is met, and it is the right
place for it: subtyping is a 3-way decision among patients already flagged as
ACS, so precision is not crushed by a 1:19 prevalence the way it is in Stage 1.

Pipeline
  1. Grouped 5-fold CV on the training fold for hyper-parameter search.  With
     ~3.5k ACS cases a single validation fold is far too noisy to tune on; CV
     over patient-disjoint folds gives a much steadier signal and leaves the
     real validation fold untouched for the decision layer.
  2. XGBoost + LightGBM, class-balanced sample weights, soft-probability blend.
  3. Constrained Cost-Sensitive Decision Layer fitted on validation, frozen.
  4. Single pass over the test fold.
"""
from __future__ import annotations

import os
import sys
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (CFG, MODEL_DIR, REPORT_DIR, SUBTYPE_ORDER, enable_utf8_stdout,
                    save_json, set_seed)
from dataset import load_bundle
from decision_layer import ConstrainedDecisionLayer
from progress import TrialProgress
from study_store import describe, get_study, wants_fresh
from utils import (BinBudget, banner, bootstrap_ci, kv, plot_confusion,
                   plot_per_class_bars, print_report, resolve_device, section,
                   summarise, timer)

enable_utf8_stdout()
SEED = set_seed()
K = 3
MAX_BIN = int(CFG.get("model.max_bin", 256))
N_PAR = int(CFG.get("model.n_parallel_trials", 1))
SHOW = bool(CFG.get("model.progress", True))
CV_FOLDS = int(CFG.get("model.stage2.cv_folds", 5))
BINS = BinBudget(MAX_BIN, enabled=bool(CFG.get("model.gpu_oom_backoff", True)))


# --------------------------------------------------------------------------
def _xgb(p, device, n_est, esr=None, max_bin=None):
    import xgboost as xgb
    kw = dict(objective="multi:softprob", num_class=K, eval_metric="mlogloss",
              device=device, tree_method="hist", max_bin=max_bin or BINS.value,
              random_state=SEED, verbosity=0, n_jobs=4,
              n_estimators=n_est, **p)
    if esr:
        kw["early_stopping_rounds"] = esr
    return xgb.XGBClassifier(**kw)


def _lgb(p, n_est, max_bin=None):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=K,
                              random_state=SEED, n_estimators=n_est,
                              max_bin=max_bin or BINS.value, verbosity=-1, n_jobs=4, **p)


def cv_score(make_model, X, y, groups, n_splits=None, kind="xgb", esr=60):
    """Mean macro-F1 over patient-disjoint stratified folds."""
    from sklearn.metrics import f1_score
    import lightgbm as lgb

    n_splits = n_splits or CV_FOLDS
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        sw = compute_sample_weight("balanced", ytr)
        m = make_model()
        if kind == "xgb":
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        else:
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(esr, verbose=False)])
        scores.append(f1_score(yva, m.predict(Xva), average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def tune(X, y, groups, device, n_trials, n_est, esr, horizon=24, fresh=False):
    def obj_x(trial):
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 6.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 6),
        }
        m, _ = BINS.run(lambda mb: cv_score(
            lambda: _xgb(p, device, n_est, esr, max_bin=mb), X, y, groups, kind="xgb"))
        return m

    def obj_l(trial):
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 120, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
        m, _ = BINS.run(lambda mb: cv_score(
            lambda: _lgb(p, n_est, max_bin=mb), X, y, groups, kind="lgb", esr=esr))
        return m

    out = {}
    for name, fn in (("xgboost", obj_x), ("lightgbm", obj_l)):
        section(f"Hyper-parameter search — {name}  (grouped {CV_FOLDS}-fold CV macro-F1)")
        st, remaining, done = get_study(f"stage2_{name}_H{horizon}",
                                        f"stage2_H{horizon}", SEED, n_trials, fresh)
        kv("study", describe(st, done, remaining))
        if remaining > 0:
            bar = TrialProgress(remaining, name, enabled=SHOW)
            with timer(f"{name} optuna ({remaining} trials x {CV_FOLDS}-fold CV)"):
                st.optimize(fn, n_trials=remaining, catch=(Exception,),
                            callbacks=[bar], n_jobs=N_PAR, show_progress_bar=False)
            bar.close()
        kv("best CV macro-F1", f"{st.best_value:.4f}")
        out[name] = st
    return out


# --------------------------------------------------------------------------
def main(horizon: int | None = None, fresh: bool | None = None) -> dict:
    horizon = CFG.primary_horizon if horizon is None else horizon
    fresh = wants_fresh() if fresh is None else fresh
    banner(f"STAGE 2 — ACS SUBTYPE CLASSIFICATION   (horizon H={horizon}h)")

    # Evaluation uses the Intended Use Population; training may additionally
    # use ACS cases that fall outside it (still patient-disjoint), because more
    # positives help and the test fold is unaffected.
    b_all = load_bundle(horizon=horizon, cohort_only=False, verbose=False)
    b = load_bundle(horizon=horizon, cohort_only=True)

    Xtr, ytr, mtr = b_all.acs_only("train")
    Xva, yva, mva = b.acs_only("val")
    Xte, yte, mte = b.acs_only("test")

    device = resolve_device(CFG.get("model.device", "cuda"))
    n_trials = int(CFG.get("model.stage2.n_trials", 60))
    n_est = int(CFG.get("model.stage2.n_estimators", 2000))
    esr = int(CFG.get("model.stage2.early_stopping_rounds", 80))

    section("Configuration")
    kv("device", device)
    kv("optuna trials (per learner)", n_trials)
    kv("parallel trials", N_PAR)
    kv("CV folds per trial", CV_FOLDS)
    kv("max_bin (histogram)", f"{MAX_BIN}  (auto-backoff on OOM)")
    kv("model fits in search", f"{n_trials*CV_FOLDS*2:,}")
    for nm, yy in (("train", ytr), ("val", yva), ("test", yte)):
        kv(nm, f"{len(yy):>5,}  " +
           "  ".join(f"{c}={int((yy==i).sum()):,}" for i, c in enumerate(SUBTYPE_ORDER)))

    kv("resume", "disabled (--fresh)" if fresh else "enabled (SQLite-backed studies)")
    studies = tune(Xtr, ytr, mtr.subject_id.to_numpy(), device, n_trials,
                   n_est, esr, horizon=horizon, fresh=fresh)

    # ---------------- final fits ----------------
    section("Fitting final learners on the full training fold")
    import lightgbm as lgb
    sw = compute_sample_weight("balanced", ytr)
    mx = _xgb(studies["xgboost"].best_params, device, n_est, esr)
    mx.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    ml = _lgb(studies["lightgbm"].best_params, n_est)
    ml.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
           eval_metric="multi_logloss",
           callbacks=[lgb.early_stopping(esr, verbose=False)])
    kv("xgboost trees", getattr(mx, "best_iteration", n_est))
    kv("lightgbm trees", getattr(ml, "best_iteration_", n_est))

    Px_va, Pl_va = mx.predict_proba(Xva), ml.predict_proba(Xva)
    Px_te, Pl_te = mx.predict_proba(Xte), ml.predict_proba(Xte)

    section("Ensemble weight selection (validation macro-F1, argmax decision)")
    from sklearn.metrics import f1_score
    best_a, best_s = 0.5, -1.0
    for a in np.linspace(0, 1, 21):
        s = f1_score(yva, np.argmax(a * Px_va + (1 - a) * Pl_va, axis=1),
                     average="macro", zero_division=0)
        if s > best_s:
            best_a, best_s = float(a), float(s)
    kv("alpha (xgboost share)", f"{best_a:.2f}")
    kv("val macro-F1 before CDL", f"{best_s:.4f}")
    P_va = best_a * Px_va + (1 - best_a) * Pl_va
    P_te = best_a * Px_te + (1 - best_a) * Pl_te

    # ---------------- constrained decision layer ----------------
    section("Constrained Cost-Sensitive Decision Layer (fitted on VALIDATION)")
    cdl = ConstrainedDecisionLayer(
        SUBTYPE_ORDER,
        floor_ladder=[float(x) for x in CFG.get("decision.floor_relaxation",
                                                [0.75, 0.74, 0.73, 0.72, 0.70])],
        n_bootstrap=40, seed=SEED,
    )
    with timer("CDL search"):
        cdl.fit(P_va, yva, groups=mva.subject_id.to_numpy())
    print(cdl.report())

    # ---------------- evaluation ----------------
    res = {}
    section("Baseline: plain argmax (no decision layer)")
    base = summarise(yte, np.argmax(P_te, axis=1), SUBTYPE_ORDER, y_prob=P_te)
    print_report("STAGE 2 — TEST (argmax baseline)", base, SUBTYPE_ORDER)
    res["test_argmax"] = base

    for name, yy, PP, mm in (("VALIDATION", yva, P_va, mva),
                             ("TEST", yte, P_te, mte)):
        pred = cdl.predict(PP)
        r = summarise(yy, pred, SUBTYPE_ORDER, y_prob=cdl.transform_proba(PP))
        print_report(f"STAGE 2 — {name} (with CDL)", r, SUBTYPE_ORDER)
        res[name.lower()] = r

    section("Bootstrap 95% CI (patient-level cluster bootstrap, test fold)")
    ci = bootstrap_ci(yte, cdl.predict(P_te), SUBTYPE_ORDER,
                      n=int(CFG.get("evaluation.bootstrap_n", 1000)),
                      seed=SEED, groups=mte.subject_id.to_numpy())
    for c in SUBTYPE_ORDER:
        kv(f"{c} recall", f"{ci[c+'_recall']['mean']:.4f}  "
                          f"[{ci[c+'_recall']['lo']:.4f}, {ci[c+'_recall']['hi']:.4f}]")
        kv(f"{c} F1", f"{ci[c+'_f1']['mean']:.4f}  "
                      f"[{ci[c+'_f1']['lo']:.4f}, {ci[c+'_f1']['hi']:.4f}]")
    kv("macro-F1", f"{ci['macro_f1']['mean']:.4f}  "
                   f"[{ci['macro_f1']['lo']:.4f}, {ci['macro_f1']['hi']:.4f}]")
    res["test_ci"] = ci

    # ---------------- target verdict ----------------
    section("Requirement check — every class >= 75% recall AND >= 75% F1")
    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    ok = True
    for c in SUBTYPE_ORDER:
        m = res["test"]["per_class"][c]
        good = m["recall"] >= floor and m["f1"] >= floor
        ok &= good
        print(f"  {c:<8} recall={m['recall']*100:6.2f}%  F1={m['f1']*100:6.2f}%  "
              f"precision={m['precision']*100:6.2f}%   "
              f"{'PASS' if good else 'BELOW TARGET'}")
    print(f"\n  OVERALL: {'ALL CLASSES MEET TARGET' if ok else 'TARGET NOT FULLY MET'}")
    res["meets_target"] = bool(ok)

    plot_confusion(np.array(res["test"]["confusion_matrix"]), SUBTYPE_ORDER,
                   f"Stage 2 — subtype (test, H={horizon}h)",
                   f"stage2_confusion_H{horizon}.png")
    plot_per_class_bars(res["test"], SUBTYPE_ORDER,
                        f"Stage 2 per-class performance (test, H={horizon}h)",
                        f"stage2_per_class_H{horizon}.png", floor=floor)

    # ---------------- persist ----------------
    mx.save_model(os.path.join(MODEL_DIR, f"stage2_xgb_H{horizon}.json"))
    joblib.dump(ml, os.path.join(MODEL_DIR, f"stage2_lgb_H{horizon}.joblib"))
    joblib.dump(cdl, os.path.join(MODEL_DIR, f"stage2_cdl_H{horizon}.joblib"))
    save_json({"horizon": horizon, "device": device, "alpha_xgb": best_a,
               "xgb_params": studies["xgboost"].best_params,
               "lgb_params": studies["lightgbm"].best_params,
               "cv_macro_f1": {k: v.best_value for k, v in studies.items()},
               "cdl": cdl.info, "features": b.features},
              os.path.join(MODEL_DIR, f"stage2_config_H{horizon}.json"))
    save_json(res, os.path.join(REPORT_DIR, f"stage2_metrics_H{horizon}.json"))
    np.savez(os.path.join(MODEL_DIR, f"stage2_scores_H{horizon}.npz"),
             P_val=P_va, P_test=P_te, y_val=yva, y_test=yte)

    banner(f"STAGE 2 COMPLETE — test macro-F1 {res['test']['macro_f1']:.4f} | "
           f"min recall {res['test']['min_recall']*100:.1f}% | "
           f"min F1 {res['test']['min_f1']*100:.1f}%")
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(int(args[0]) if args else None)
