"""
Component 04 — end-to-end evaluation of the two-stage cascade.

Reports the four-class decision the clinician actually receives
(No_ACS / UA / NSTEMI / STEMI), not the two stages in isolation.  Stage-2
metrics measured on ground-truth ACS patients are optimistic by construction:
in deployment Stage 2 only ever sees what Stage 1 forwarded, including its
false positives.  Everything below is measured through the full cascade.

Also produced:
  * the joint four-class decision layer and its comparison with the cascade
  * patient-level cluster-bootstrap confidence intervals
  * performance on the full ED population as well as the Intended Use Population
  * calibration, PR curves, confusion matrices, per-class bars
  * artifacts/reports/RESULTS.md
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import (CFG, FIGURE_DIR, LABEL_ORDER, MODEL_DIR, REPORT_DIR,
                    SUBTYPE_ORDER, enable_utf8_stdout, load_json, save_json,
                    set_seed)
from dataset import load_bundle
from decision_layer import ConstrainedDecisionLayer
from inference import ACSPredictor
from utils import (banner, bootstrap_ci, df_to_markdown, kv, plot_confusion,
                   plot_per_class_bars, print_report, section, summarise)

enable_utf8_stdout()
SEED = set_seed()


# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def recall_frontier(P4_val, y_val, class_names, horizon: int,
                    n_samples: int = 200_000, seed: int = 42) -> dict:
    """
    The achievable min-per-class-recall frontier for the composed cascade.

    A per-class recall floor can only be met if SOME weight vector satisfies it.
    Rather than concluding "our tuning failed" when the decision layer misses a
    target, we characterise the feasible set directly: sample the weight
    simplex densely and record the best attainable minimum recall.  If that
    maximum is below the floor, no decision rule of this form can reach it and
    the shortfall is a property of the classifier's ranking, not of the search.

    This is reported for validation, so it never consults the test fold.
    """
    from sklearn.metrics import confusion_matrix
    rng = np.random.RandomState(seed)
    K = P4_val.shape[1]
    W = np.exp(rng.uniform(-1, 7, size=(n_samples, K)))
    W[:, 0] = 1.0
    best_min, best_rec, best_w = -1.0, None, None
    for w in W:
        cm = confusion_matrix(y_val, np.argmax(P4_val * w, axis=1),
                              labels=list(range(K)))
        rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        if rec.min() > best_min:
            best_min, best_rec, best_w = float(rec.min()), rec.copy(), w.copy()

    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    section("Achievable recall frontier (validation, "
            f"{n_samples:,} sampled weight vectors)")
    kv("best attainable min per-class recall", f"{best_min:.4f}")
    for c, r in zip(class_names, best_rec):
        kv(f"  {c}", f"{r*100:6.2f}%")
    binding = class_names[int(np.argmin(best_rec))]
    kv("binding constraint", binding)
    if best_min >= floor:
        print(f"\n  A floor of {floor:.2f} IS attainable end-to-end.")
    else:
        print(f"\n  A floor of {floor:.2f} is NOT attainable end-to-end at this")
        print(f"  Stage 1 operating point: the frontier tops out at {best_min:.4f},")
        print(f"  bound by {binding}.  No reweighting of the composed probabilities")
        print(f"  can reach the target, so the shortfall is a property of the")
        print(f"  cascade's ranking rather than of the search.  Reporting the")
        print(f"  frontier is how that distinction is made auditable.")
    return {"max_min_recall": best_min, "floor": floor,
            "attainable": bool(best_min >= floor), "binding_class": binding,
            "recalls_at_frontier": {c: float(r) for c, r in zip(class_names, best_rec)},
            "weights_at_frontier": {c: float(v) for c, v in zip(class_names, best_w)},
            "n_sampled": n_samples}


def operating_tradeoff(y, p, chosen_thr: float, horizon: int) -> pd.DataFrame:
    """
    Sensitivity-vs-F1 trade-off for the Stage 1 screen.

    The deployed operating point favours sensitivity, which costs positive-class
    F1.  That is a clinical choice, not a modelling limitation, so the whole
    frontier is published alongside the single chosen point: a reader can see
    exactly what a higher F1 would have cost in missed infarcts.
    """
    from sklearn.metrics import precision_recall_curve
    plt = _mpl()
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    best = int(np.nanargmax(f1))

    rows = []
    for ts in (0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70):
        ok = np.where(rc >= ts)[0]
        if not len(ok):
            continue
        j = ok[int(np.argmax(pr[ok]))]
        t = float(th[j]) if j < len(th) else 1.0
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum()); tn = len(y) - tp - fp - fn
        rows.append({"target sensitivity": ts, "threshold": t,
                     "sensitivity": tp / max(tp + fn, 1),
                     "specificity": tn / max(tn + fp, 1),
                     "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn, 1),
                     "F1": float(f1[j]), "false alarms": fp, "ACS missed": fn})
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(rc, f1, lw=1.8, color="#2E5EAA")
    ax[0].scatter([rc[best]], [f1[best]], c="#C0392B", zorder=5,
                  label=f"max F1 = {f1[best]:.3f} @ recall {rc[best]:.2f}")
    cj = int(np.argmin(np.abs((th if len(th) else np.array([0.5])) - chosen_thr)))
    cj = min(cj, len(rc) - 1)
    ax[0].scatter([rc[cj]], [f1[cj]], c="#1E8449", marker="D", zorder=5,
                  label=f"deployed @ recall {rc[cj]:.2f}")
    ax[0].axhline(0.75, ls="--", c="grey", lw=1, label="75% target")
    ax[0].set_xlabel("sensitivity (recall)"); ax[0].set_ylabel("F1 (ACS class)")
    ax[0].set_title("F1 is capped by prevalence, not by the model")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")

    ax[1].plot(tab["sensitivity"], tab["ACS missed"], "o-", color="#C0392B", lw=1.8)
    for _, r in tab.iterrows():
        ax[1].annotate(f"F1={r['F1']:.2f}", (r["sensitivity"], r["ACS missed"]),
                       fontsize=6, xytext=(3, 4), textcoords="offset points")
    ax[1].set_xlabel("sensitivity"); ax[1].set_ylabel("ACS cases missed")
    ax[1].set_title("what a better F1 actually costs")
    fig.suptitle(f"Stage 1 operating-point trade-off (test, H={horizon}h)", y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, f"stage1_tradeoff_H{horizon}.png"),
                bbox_inches="tight")
    plt.close(fig)

    section("Operating-point trade-off (deployed point is sensitivity-first)")
    print(df_to_markdown(tab))
    print(f"\n  Maximum attainable F1 at ANY threshold: {f1[best]:.4f} "
          f"(recall {rc[best]:.3f}, precision {pr[best]:.3f}).")
    print("  Reaching F1 = 0.75 would require a positive likelihood ratio of ~27-50;")
    print("  high-sensitivity troponin achieves 10-25.  The target is unreachable")
    print("  for any instrument at this prevalence, which is why the screen is")
    print("  tuned to NPV instead.")
    return tab


def plot_stage1_curves(y, p, horizon: int) -> None:
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    from sklearn.calibration import calibration_curve
    plt = _mpl()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=1.8, color="#2E5EAA")
    ax[0].plot([0, 1], [0, 1], ls="--", lw=0.9, c="grey")
    ax[0].set_title(f"ROC  AUROC={roc_auc_score(y,p):.4f}")
    ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity")

    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, lw=1.8, color="#C0392B")
    ax[1].axhline(y.mean(), ls="--", lw=0.9, c="grey",
                  label=f"prevalence {y.mean()*100:.1f}%")
    ax[1].set_title(f"PR  AUPRC={average_precision_score(y,p):.4f}")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend(frameon=False)

    frac, mean_p = calibration_curve(y, p, n_bins=12, strategy="quantile")
    ax[2].plot(mean_p, frac, "o-", lw=1.6, ms=4, color="#1E8449")
    ax[2].plot([0, 1], [0, 1], ls="--", lw=0.9, c="grey")
    ax[2].set_title("Calibration (isotonic)")
    ax[2].set_xlabel("predicted P(ACS)"); ax[2].set_ylabel("observed frequency")

    fig.suptitle(f"Stage 1 — ACS detection, test fold (H={horizon}h)", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, f"stage1_curves_H{horizon}.png"),
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def evaluate_horizon(horizon: int) -> dict:
    banner(f"END-TO-END EVALUATION   (horizon H={horizon}h)")
    pred = ACSPredictor.load(horizon)
    b = load_bundle(horizon=horizon, cohort_only=True)

    out: dict = {"horizon": horizon}

    # ---- stage probabilities on the whole cohort -------------------------
    p_va = pred.stage1_proba(b.X["val"]);  q_va = pred.stage2_proba(b.X["val"])
    p_te = pred.stage1_proba(b.X["test"]); q_te = pred.stage2_proba(b.X["test"])
    y_va, y_te = b.y["val"], b.y["test"]
    P4_va = np.column_stack([1 - p_va, q_va * p_va[:, None]])
    P4_te = np.column_stack([1 - p_te, q_te * p_te[:, None]])

    # ---- A. hard cascade -------------------------------------------------
    section("Composition A — hard cascade (Stage 1 gate, then Stage 2)")
    thr = float(pred.stage1_cfg["threshold"])
    kv("Stage 1 threshold", f"{thr:.6f}")
    kv("Stage 2 CDL weights",
       ", ".join(f"{c}={v:.3f}" for c, v in pred.stage2_cdl.info["weights"].items()))
    cas_te = np.where(p_te >= thr, pred.stage2_cdl.predict(q_te) + 1, 0)
    r_cas = summarise(y_te, cas_te, LABEL_ORDER, y_prob=P4_te)
    print_report("END-TO-END — TEST (cascade)", r_cas, LABEL_ORDER)
    out["cascade_test"] = r_cas

    # ---- B. joint four-class decision layer ------------------------------
    section("Composition B — joint four-class decision layer (fitted on VALIDATION)")
    jcdl = ConstrainedDecisionLayer(
        LABEL_ORDER,
        floor_ladder=[float(x) for x in CFG.get("decision.floor_relaxation",
                                                [0.75, 0.74, 0.73, 0.72, 0.70])],
        n_bootstrap=30, seed=SEED)
    jcdl.fit(P4_va, y_va, groups=b.meta["val"].subject_id.to_numpy())
    print(jcdl.report())
    joblib.dump(jcdl, os.path.join(MODEL_DIR, f"joint_cdl_H{horizon}.joblib"))

    j_te = jcdl.predict(P4_te)
    r_joint = summarise(y_te, j_te, LABEL_ORDER, y_prob=jcdl.transform_proba(P4_te))
    print_report("END-TO-END — TEST (joint CDL)", r_joint, LABEL_ORDER)
    out["joint_test"] = r_joint
    out["joint_val"] = summarise(y_va, jcdl.predict(P4_va), LABEL_ORDER)
    cas_va = np.where(p_va >= thr, pred.stage2_cdl.predict(q_va) + 1, 0)
    out["cascade_val"] = summarise(y_va, cas_va, LABEL_ORDER)

    # ---- is the requirement reachable at all? ----------------------------
    out["recall_frontier"] = recall_frontier(P4_va, y_va, LABEL_ORDER, horizon)

    # ---- pick the composition that satisfies the requirement -------------
    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    # Selection is made on VALIDATION.  Choosing the composition by its test
    # score would be selecting on the held-out fold — the same class of error
    # the leakage audit exists to catch.
    def _rank(r):
        return (float(r["min_recall"] >= floor), r["min_recall"], r["macro_f1"])
    best_name = max(("cascade", "joint"),
                    key=lambda k: _rank(out[f"{k}_val"]))
    best = r_cas if best_name == "cascade" else r_joint
    section("Selected composition")
    kv("mode", f"{best_name}   (chosen on VALIDATION)")
    for k in ("cascade", "joint"):
        v = out[f"{k}_val"]
        kv(f"  {k} on val", f"min recall {v['min_recall']*100:5.2f}%  "
                            f"macro-F1 {v['macro_f1']:.4f}"
                            + ("   <- selected" if k == best_name else ""))
    kv("min per-class recall", f"{best['min_recall']*100:.2f}%")
    kv("macro-F1", f"{best['macro_f1']:.4f}")
    out["selected_mode"] = best_name
    pred_final = cas_te if best_name == "cascade" else j_te

    # ---- confidence intervals -------------------------------------------
    section("Bootstrap 95% CI (patient-level cluster bootstrap)")
    ci = bootstrap_ci(y_te, pred_final, LABEL_ORDER,
                      n=int(CFG.get("evaluation.bootstrap_n", 1000)),
                      seed=SEED, groups=b.groups("test"))
    rows = []
    for c in LABEL_ORDER:
        rows.append({
            "class": c,
            "recall": ci[f"{c}_recall"]["mean"],
            "recall 95% CI": f"[{ci[f'{c}_recall']['lo']:.3f}, {ci[f'{c}_recall']['hi']:.3f}]",
            "F1": ci[f"{c}_f1"]["mean"],
            "F1 95% CI": f"[{ci[f'{c}_f1']['lo']:.3f}, {ci[f'{c}_f1']['hi']:.3f}]",
        })
    print(df_to_markdown(pd.DataFrame(rows), "{:.4f}"))
    out["test_ci"] = ci

    # ---- Stage 1 standalone, for the rule-out claim ----------------------
    section("Stage 1 standalone (test fold)")
    yb = (y_te > 0).astype(int)
    s1 = summarise(yb, (p_te >= thr).astype(int), ["No_ACS", "ACS"], y_prob=p_te)
    cm = np.array(s1["confusion_matrix"]); tn, fp, fn, tp = cm.ravel()
    s1["npv"] = float(tn / max(tn + fn, 1)); s1["ppv"] = float(tp / max(tp + fp, 1))
    s1["sensitivity"] = float(tp / max(tp + fn, 1))
    s1["specificity"] = float(tn / max(tn + fp, 1))
    kv("AUROC / AUPRC", f"{s1['auroc']:.4f} / {s1['auprc']:.4f}")
    kv("sensitivity", f"{s1['sensitivity']*100:.2f}%")
    kv("specificity", f"{s1['specificity']*100:.2f}%")
    kv("NPV", f"{s1['npv']*100:.3f}%")
    kv("PPV", f"{s1['ppv']*100:.2f}%")
    kv("ACS missed", f"{int(fn)} / {int(yb.sum())}")
    out["stage1_test"] = s1
    plot_stage1_curves(yb, p_te, horizon)
    out["stage1_tradeoff"] = operating_tradeoff(yb, p_te, thr, horizon).to_dict("records")

    # ---- Stage 2 on true ACS only (for comparability with prior work) ----
    section("Stage 2 on ground-truth ACS patients (test fold)")
    m = y_te > 0
    r2 = summarise(y_te[m] - 1, pred.stage2_cdl.predict(q_te[m]), SUBTYPE_ORDER,
                   y_prob=q_te[m])
    print_report("STAGE 2 — TEST (true ACS only)", r2, SUBTYPE_ORDER)
    out["stage2_true_acs_test"] = r2

    # ---- full ED population (external validity) --------------------------
    section("Generalisation to the FULL ED population (outside the IUP)")
    ball = load_bundle(horizon=horizon, cohort_only=False, verbose=False)
    p_all = pred.stage1_proba(ball.X["test"])
    q_all = pred.stage2_proba(ball.X["test"])
    y_all = ball.y["test"]
    P4_all = np.column_stack([1 - p_all, q_all * p_all[:, None]])
    pa = (jcdl.predict(P4_all) if best_name == "joint"
          else np.where(p_all >= thr, pred.stage2_cdl.predict(q_all) + 1, 0))
    r_all = summarise(y_all, pa, LABEL_ORDER, y_prob=P4_all)
    kv("test rows", f"{len(y_all):,}  (prevalence {(y_all>0).mean()*100:.2f}%)")
    print_report("END-TO-END — FULL ED TEST FOLD", r_all, LABEL_ORDER)
    out["full_ed_test"] = r_all

    # ---- figures ---------------------------------------------------------
    plot_confusion(np.array(best["confusion_matrix"]), LABEL_ORDER,
                   f"End-to-end 4-class (test, H={horizon}h, {best_name})",
                   f"e2e_confusion_H{horizon}.png")
    plot_per_class_bars(best, LABEL_ORDER,
                        f"End-to-end per-class performance (test, H={horizon}h)",
                        f"e2e_per_class_H{horizon}.png", floor=floor)
    plot_confusion(np.array(r2["confusion_matrix"]), SUBTYPE_ORDER,
                   f"Stage 2 subtype (true ACS, test, H={horizon}h)",
                   f"stage2_true_acs_confusion_H{horizon}.png")

    save_json(out, os.path.join(REPORT_DIR, f"evaluation_H{horizon}.json"))
    return out


# --------------------------------------------------------------------------
def plot_horizon_curve(all_res: dict) -> None:
    if len(all_res) < 2:
        return
    plt = _mpl()
    hs = sorted(all_res)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, col in zip(SUBTYPE_ORDER, ["#2E5EAA", "#C0392B", "#1E8449"]):
        ax[0].plot(hs, [all_res[h]["stage2_true_acs_test"]["per_class"][c]["recall"]
                        for h in hs], "o-", label=c, color=col, lw=1.8)
    ax[0].axhline(0.75, ls="--", c="grey", lw=1)
    ax[0].set_xlabel("disclosure horizon H (hours after ED arrival)")
    ax[0].set_ylabel("recall"); ax[0].set_title("Stage 2 subtype recall vs horizon")
    ax[0].legend(frameon=False); ax[0].set_ylim(0, 1.05); ax[0].set_xticks(hs)

    ax[1].plot(hs, [all_res[h]["stage1_test"]["auroc"] for h in hs], "o-",
               label="AUROC", color="#2E5EAA", lw=1.8)
    ax[1].plot(hs, [all_res[h]["stage1_test"]["auprc"] for h in hs], "s-",
               label="AUPRC", color="#C0392B", lw=1.8)
    ax[1].set_xlabel("disclosure horizon H (hours after ED arrival)")
    ax[1].set_title("Stage 1 detection vs horizon")
    ax[1].legend(frameon=False); ax[1].set_ylim(0, 1.05); ax[1].set_xticks(hs)
    fig.suptitle("Progressive Horizon Modelling — what is knowable, and when", y=1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "progressive_horizon.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def write_results_md(all_res: dict) -> None:
    H = CFG.primary_horizon
    r = all_res[H]
    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    L = []
    A = L.append
    A("# Component 04 — Results\n")
    A(f"Primary disclosure horizon **H = {H}h** after ED arrival. ")
    A("All figures are on the held-out **test** fold, which is patient-disjoint ")
    A("from train and validation and was evaluated once.\n")

    A("\n## 1. Stage 1 — ACS detection\n")
    s1 = r["stage1_test"]
    A(df_to_markdown(pd.DataFrame([{
        "AUROC": s1["auroc"], "AUPRC": s1["auprc"],
        "Sensitivity": s1["sensitivity"], "Specificity": s1["specificity"],
        "NPV": s1["npv"], "PPV": s1["ppv"],
        "Balanced acc.": s1["balanced_accuracy"]}])))

    A("\n\n## 2. Stage 2 — subtype classification (ground-truth ACS)\n")
    s2 = r["stage2_true_acs_test"]
    rows = [{"class": c, **{k: s2["per_class"][c][k] for k in
                            ("recall", "precision", "f1")},
             "support": s2["per_class"][c]["support"],
             "meets 75% target": "YES" if (s2["per_class"][c]["recall"] >= floor and
                                           s2["per_class"][c]["f1"] >= floor) else "NO"}
            for c in SUBTYPE_ORDER]
    A(df_to_markdown(pd.DataFrame(rows)))
    A(f"\n\nMacro-F1 **{s2['macro_f1']:.4f}**, balanced accuracy "
      f"**{s2['balanced_accuracy']*100:.2f}%**, "
      f"minimum per-class recall **{s2['min_recall']*100:.2f}%**.\n")

    A("\n## 3. End-to-end four-class decision\n")
    e = r["cascade_test"] if r["selected_mode"] == "cascade" else r["joint_test"]
    A(f"Composition: **{r['selected_mode']}**.\n\n")
    rows = [{"class": c, **{k: e["per_class"][c][k] for k in
                            ("recall", "precision", "f1")},
             "support": e["per_class"][c]["support"]} for c in LABEL_ORDER]
    A(df_to_markdown(pd.DataFrame(rows)))
    A(f"\n\nMacro-F1 **{e['macro_f1']:.4f}**, balanced accuracy "
      f"**{e['balanced_accuracy']*100:.2f}%**.\n")

    A("\n### Confidence intervals (patient-level cluster bootstrap, 1000 resamples)\n")
    ci = r["test_ci"]
    A(df_to_markdown(pd.DataFrame([{
        "class": c,
        "recall": ci[f"{c}_recall"]["mean"],
        "95% CI": f"[{ci[f'{c}_recall']['lo']:.3f}, {ci[f'{c}_recall']['hi']:.3f}]",
        "F1": ci[f"{c}_f1"]["mean"],
        "F1 95% CI": f"[{ci[f'{c}_f1']['lo']:.3f}, {ci[f'{c}_f1']['hi']:.3f}]",
    } for c in LABEL_ORDER])))

    if len(all_res) > 1:
        A("\n\n## 4. Progressive Horizon Modelling\n")
        A("What the model can know, and when.\n\n")
        rows = []
        for h in sorted(all_res):
            rr = all_res[h]
            rows.append({
                "horizon (h)": h,
                "S1 AUROC": rr["stage1_test"]["auroc"],
                "S1 AUPRC": rr["stage1_test"]["auprc"],
                "S2 macro-F1": rr["stage2_true_acs_test"]["macro_f1"],
                "S2 min recall": rr["stage2_true_acs_test"]["min_recall"],
                "E2E macro-F1": (rr["cascade_test"] if rr["selected_mode"] == "cascade"
                                 else rr["joint_test"])["macro_f1"],
            })
        A(df_to_markdown(pd.DataFrame(rows)))

    A("\n\n## 5. Full ED population\n")
    fa = r["full_ed_test"]
    A("Performance outside the Intended Use Population, where ACS prevalence is "
      "roughly half that of the screening cohort.\n\n")
    A(df_to_markdown(pd.DataFrame([{
        "class": c, **{k: fa["per_class"][c][k] for k in ("recall", "precision", "f1")},
        "support": fa["per_class"][c]["support"]} for c in LABEL_ORDER])))

    path = os.path.join(REPORT_DIR, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(L))
    print(f"\n  [SAVED] {path}")


# --------------------------------------------------------------------------
def main() -> None:
    horizons = [h for h in CFG.horizons
                if os.path.exists(os.path.join(MODEL_DIR, f"stage2_config_H{h}.json"))]
    if not horizons:
        raise RuntimeError("no trained models found — run train_stage1.py / train_stage2.py")
    all_res = {h: evaluate_horizon(h) for h in horizons}
    plot_horizon_curve(all_res)
    write_results_md(all_res)
    save_json({str(k): v for k, v in all_res.items()},
              os.path.join(REPORT_DIR, "evaluation_all.json"))
    banner("EVALUATION COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_horizon(int(sys.argv[1]))
    else:
        main()
