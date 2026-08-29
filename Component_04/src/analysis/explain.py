"""
Component 04 — tri-level explainability.

  Level 1  FEATURE   TreeSHAP attributions per input feature, per stage,
                     per class.
  Level 2  MODALITY  SHAP mass aggregated over the eight modality groups.
                     A clinician asks "was this a lab call or an ECG call?"
                     long before asking which of 221 columns mattered, and a
                     modality breakdown is also what makes the missingness-aware
                     design auditable: if a channel is absent for a patient its
                     contribution should visibly collapse.
  Level 3  TOKEN     span-level highlighting of the chief complaint, driven by
                     the clinical lexicon with negation scoping.

Level 2 is computed per patient as well as globally, so the same decomposition
that appears in the paper appears in the bedside output.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import (CFG, FIGURE_DIR, LABEL_ORDER, REPORT_DIR, SUBTYPE_ORDER,
                    enable_utf8_stdout, save_json, set_seed)
from dataset import load_bundle, modality_groups
from inference import ACSPredictor
import text_features as TF
from utils import banner, df_to_markdown, kv, section, timer

enable_utf8_stdout()
SEED = set_seed()

MODALITY_COLORS = {
    "vitals": "#2E5EAA", "labs": "#C0392B", "ecg": "#1E8449",
    "text": "#8E44AD", "demographics": "#D68910", "medications": "#16A085",
    "history": "#7F8C8D", "interaction": "#34495E", "other": "#BDC3C7",
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


# --------------------------------------------------------------------------
def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """(n, f) for binary, (n, f, k) for multiclass."""
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)


def modality_attribution(sv: np.ndarray, features: List[str],
                         modality: Dict[str, str]) -> pd.Series:
    """Share of total |SHAP| mass carried by each modality."""
    mass = np.abs(sv)
    if mass.ndim == 3:
        mass = mass.mean(axis=2)
    per_feature = mass.mean(axis=0)
    s = pd.Series(per_feature, index=features).groupby(
        pd.Series({f: modality.get(f, "other") for f in features})).sum()
    return (s / s.sum()).sort_values(ascending=False)


def per_patient_modality(sv_row: np.ndarray, features: List[str],
                         modality: Dict[str, str]) -> Dict[str, float]:
    mass = np.abs(sv_row)
    if mass.ndim == 2:
        mass = mass.mean(axis=1)
    s = pd.Series(mass, index=features).groupby(
        pd.Series({f: modality.get(f, "other") for f in features})).sum()
    tot = s.sum()
    return {k: float(v / tot) if tot else 0.0 for k, v in s.sort_values(ascending=False).items()}


# --------------------------------------------------------------------------
def plot_modality_bars(shares: Dict[str, pd.Series], fname: str, title: str) -> None:
    plt = _mpl()
    keys = list(shares)
    mods = sorted({m for s in shares.values() for m in s.index},
                  key=lambda m: -max(shares[k].get(m, 0) for k in keys))
    x = np.arange(len(mods)); w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(mods), 3.8))
    for i, k in enumerate(keys):
        vals = [shares[k].get(m, 0.0) for m in mods]
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w, label=k)
        for b, v in zip(bars, vals):
            if v > 0.015:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                        f"{v*100:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x, mods, rotation=25, ha="right")
    ax.set_ylabel("share of |SHAP| mass"); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, fname), bbox_inches="tight")
    plt.close(fig)


def plot_top_features(sv: np.ndarray, features: List[str], modality: Dict[str, str],
                      title: str, fname: str, k: int = 22) -> pd.DataFrame:
    plt = _mpl()
    mass = np.abs(sv)
    if mass.ndim == 3:
        mass = mass.mean(axis=2)
    imp = pd.Series(mass.mean(axis=0), index=features).sort_values(ascending=False)
    top = imp.head(k)[::-1]
    colors = [MODALITY_COLORS.get(modality.get(f, "other"), "#BDC3C7") for f in top.index]
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(top) + 1.4))
    ax.barh(np.arange(len(top)), top.to_numpy(), color=colors)
    ax.set_yticks(np.arange(len(top)), top.index, fontsize=8)
    ax.set_xlabel("mean |SHAP|"); ax.set_title(title)
    seen = {}
    for f in top.index:
        seen[modality.get(f, "other")] = MODALITY_COLORS.get(modality.get(f, "other"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
    ax.legend(handles, list(seen), frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    return imp.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})


def plot_beeswarm(sv, X, title, fname, max_display=18):
    try:
        import shap
        plt = _mpl()
        v = sv.mean(axis=2) if sv.ndim == 3 else sv
        plt.figure(figsize=(8, 0.32 * max_display + 1.6))
        shap.summary_plot(v, X, show=False, max_display=max_display, plot_size=None)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, fname), bbox_inches="tight")
        plt.close("all")
    except Exception as e:                     # beeswarm is a nicety, not a gate
        print(f"    [skip beeswarm {fname}: {e}]")


# --------------------------------------------------------------------------
def main(horizon: int | None = None) -> dict:
    horizon = CFG.primary_horizon if horizon is None else horizon
    banner(f"EXPLAINABILITY   (horizon H={horizon}h)")

    pred = ACSPredictor.load(horizon)
    b = load_bundle(horizon=horizon, cohort_only=True, verbose=False)
    groups = modality_groups(b)
    kv("modalities", ", ".join(f"{k}({len(v)})" for k, v in
                               sorted(groups.items(), key=lambda x: -len(x[1]))))

    rng = np.random.RandomState(SEED)
    Xte = b.X["test"]; yte = b.y["test"]
    n = min(4000, len(Xte))
    idx = rng.choice(len(Xte), n, replace=False)
    Xs = Xte.iloc[idx]
    out: dict = {"horizon": horizon}

    # ---------------- Stage 1 ----------------
    section("Stage 1 — feature and modality attribution")
    with timer(f"TreeSHAP on {n:,} test rows"):
        sv1 = shap_values(pred.stage1_xgb, Xs)
    imp1 = plot_top_features(sv1, b.features, b.modality,
                             f"Stage 1 — top features (H={horizon}h)",
                             f"shap_stage1_top_H{horizon}.png")
    plot_beeswarm(sv1, Xs, f"Stage 1 — SHAP distribution (H={horizon}h)",
                  f"shap_stage1_beeswarm_H{horizon}.png")
    m1 = modality_attribution(sv1, b.features, b.modality)
    print("\n  Modality attribution — Stage 1:")
    for k, v in m1.items():
        print(f"    {k:<14} {v*100:5.1f}%  " + "#" * int(v * 60))
    out["stage1_modality"] = m1.to_dict()
    out["stage1_top20"] = imp1.head(20).to_dict("records")

    # ---------------- Stage 2 ----------------
    section("Stage 2 — per-subtype attribution")
    m = yte > 0
    Xa = Xte[m]
    with timer(f"TreeSHAP on {len(Xa):,} ACS test rows"):
        sv2 = shap_values(pred.stage2_xgb, Xa)
    imp2 = plot_top_features(sv2, b.features, b.modality,
                             f"Stage 2 — top features (H={horizon}h)",
                             f"shap_stage2_top_H{horizon}.png")
    m2 = modality_attribution(sv2, b.features, b.modality)
    print("\n  Modality attribution — Stage 2:")
    for k, v in m2.items():
        print(f"    {k:<14} {v*100:5.1f}%  " + "#" * int(v * 60))
    out["stage2_modality"] = m2.to_dict()
    out["stage2_top20"] = imp2.head(20).to_dict("records")

    per_class = {}
    if sv2.ndim == 3:
        for i, c in enumerate(SUBTYPE_ORDER):
            s = pd.Series(np.abs(sv2[:, :, i]).mean(axis=0), index=b.features)
            per_class[c] = s.sort_values(ascending=False).head(15).to_dict()
            plot_top_features(sv2[:, :, i], b.features, b.modality,
                              f"Stage 2 — drivers of {c} (H={horizon}h)",
                              f"shap_stage2_{c}_H{horizon}.png", k=16)
            print(f"\n  Top drivers of {c}:")
            for f, v in list(per_class[c].items())[:8]:
                print(f"    {f:<28} {v:.4f}   [{b.modality.get(f,'other')}]")
    out["stage2_per_class_top"] = per_class

    plot_modality_bars({"Stage 1 (detection)": m1, "Stage 2 (subtyping)": m2},
                       f"modality_attribution_H{horizon}.png",
                       f"Modality attribution from SHAP mass (H={horizon}h)")

    # ---------------- availability sanity check ----------------
    section("Missingness-aware check — does attribution follow availability?")
    rows = []
    if "trop_available" in b.features:
        j = b.features.index("trop_available")
        has = Xa["trop_available"].to_numpy() == 1
        lab_idx = [i for i, f in enumerate(b.features)
                   if b.modality.get(f) == "labs"]
        mass = np.abs(sv2).mean(axis=2) if sv2.ndim == 3 else np.abs(sv2)
        for label, sel in (("troponin present", has), ("troponin absent", ~has)):
            if sel.sum() == 0:
                continue
            tot = mass[sel].sum(axis=1).mean()
            lab = mass[sel][:, lab_idx].sum(axis=1).mean()
            rows.append({"subgroup": label, "n": int(sel.sum()),
                         "lab share of |SHAP|": float(lab / tot) if tot else 0.0})
        print(df_to_markdown(pd.DataFrame(rows)))
        print("\n  The laboratory channel should carry substantially less mass when")
        print("  no biomarker exists — that is the missingness-aware encoding working,")
        print("  and it is what lets the model degrade gracefully instead of imputing.")
    out["availability_check"] = rows

    # ---------------- worked examples ----------------
    section("Worked patient-level explanations")
    examples = []
    meta = b.meta["test"]
    for cls in range(4):
        cand = np.where(yte == cls)[0]
        if len(cand) == 0:
            continue
        i = int(cand[0])
        row = Xte.iloc[[i]]
        ex = pred.explain_row(row, 0, meta.iloc[i].get("chiefcomplaint_raw", ""))
        sv_row = shap_values(pred.stage1_xgb, row)[0]
        ex["true_label"] = LABEL_ORDER[cls]
        ex["chief_complaint"] = str(meta.iloc[i].get("chiefcomplaint_raw", ""))
        ex["modality_contribution"] = per_patient_modality(sv_row, b.features, b.modality)
        contrib = pd.Series(sv_row if sv_row.ndim == 1 else sv_row.mean(axis=1),
                            index=b.features)
        ex["top_features"] = [
            {"feature": f, "shap": float(contrib[f]), "value": float(row.iloc[0][f]),
             "modality": b.modality.get(f, "other")}
            for f in contrib.abs().sort_values(ascending=False).head(8).index]
        examples.append(ex)

        print(f"\n  --- true={ex['true_label']}  predicted={ex['prediction']} "
              f"P(ACS)={ex['p_acs']:.3f} ---")
        print(f"      complaint: {ex['chief_complaint'][:70]}")
        print("      modality: " + "  ".join(
            f"{k}={v*100:.0f}%" for k, v in
            list(ex["modality_contribution"].items())[:5]))
        for t in ex["top_features"][:5]:
            print(f"        {t['feature']:<26} value={t['value']:>9.3f} "
                  f"shap={t['shap']:+.4f}  [{t['modality']}]")
        for tok in ex.get("text_attribution", [])[:4]:
            print(f"        text: '{tok['term']}' -> {tok['category']} "
                  f"(w={tok['weight']:+.1f}{', NEGATED' if tok['negated'] else ''})")
    out["examples"] = examples

    # ---------------- text component report ----------------
    if b.embedder is not None:
        section("Text representation — highest-loading terms per SVD component")
        tt = b.embedder.top_terms(k=8)
        for comp in list(tt)[:6]:
            print(f"    {comp}: {', '.join(tt[comp])}")
        out["svd_top_terms"] = tt

    save_json(out, os.path.join(REPORT_DIR, f"explainability_H{horizon}.json"))
    banner("EXPLAINABILITY COMPLETE")
    return out


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(h)
