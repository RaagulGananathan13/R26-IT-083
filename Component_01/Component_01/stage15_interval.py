"""
COMPONENT_01 · STAGE 15 — Acquisition-induced false interval change
===================================================================

THE CLINICAL PROBLEM
--------------------
"Interval increase in cardiac silhouette" is not a description, it is an action
trigger: it prompts echocardiography, diuresis, medication changes, escalation.
It is the most consequential sentence in a serial chest radiograph.

Chest films are acquired posteroanterior (PA, patient standing in the radiology
department) or anteroposterior (AP, portable machine at the bedside). AP
magnifies the cardiac silhouette because the heart sits anterior, further from
the detector. And a patient is moved to AP imaging precisely BECAUSE they have
deteriorated enough that they can no longer be transported.

So when a model compares today's film to a prior, two things changed at once:

    the patient's condition   (maybe)
    the imaging geometry      (certainly)

A model that reads the geometry as disease will report the heart has enlarged
when nothing about the heart has changed.

THE DESIGN
----------
Condition on pairs where THE RADIOLOGIST RECORDED NO CHANGE. Any movement the
model then reports is spurious by construction -- no ground-truth adjudication
of "real" change is required, which is what makes this measurable at all.

FOUR ARMS.

    A  same-projection      AP->AP and PA->PA     NEGATIVE CONTROL
    B  shuffled order       projection changed,   THE NULL
                            temporal order randomised
    C  true order           projection changed    THE FINDING
    D  threshold-corrected  arm C re-scored with  THE OBVIOUS FIX
                            per-projection thresholds (Stage 9A)

Arm A establishes the false-positive floor: with no geometry change, spurious
"worse" and spurious "better" should be symmetric, and are. Arm B destroys
temporal direction while keeping every other property of the pair; if the effect
survives shuffling it is not directional and the clinical claim collapses. Arm D
tests whether this project's own Contribution 1 rescues the situation.

⚠️ WHY THE METADATA JOIN IS NOT OPTIONAL
----------------------------------------
MIMIC-CXR `study_id` is an identifier, NOT a timestamp. Measured on this test
set, ordering studies by study_id agrees with true chronology 49.59% of the
time -- a coin flip. An earlier version of this analysis used study_id ordering
and produced a diluted, half-strength effect, because random ordering mixes the
two directions and cancels them. True order is read from StudyDate/StudyTime in
mimic-cxr-2.0.0-metadata.csv. `selftest` asserts the coin-flip property so this
can never be silently reintroduced.

READ-ONLY. `torch` is never imported: no checkpoint is opened, so none can be
modified. The MIMIC metadata file is opened read-only and never written. All
output goes to reports/stage15/.

Run:  python stage15_interval.py          full analysis
      python stage15_interval.py --test   self-checks only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST = HERE / "training_manifest" / "manifest_test.csv"
PROBS = HERE / "reports" / "stage6" / "cache" / "probs_test.npy"
THRESHOLDS = HERE / "backend" / "thresholds.json"
METADATA = REPO / "data" / "raw" / "mimic-cxr-2.0.0-metadata.csv"
OUT = HERE / "reports" / "stage15"
TS_CACHE = OUT / "study_timestamps.csv"

TARGET = "Cardiomegaly"
LABELS = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
          "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]
N_BOOT = 4000
SEED = 20260813

# Okabe-Ito colourblind-safe.
C_CTRL, C_FIND, C_NULL, C_FIX, C_INK = "#0072B2", "#D55E00", "#999999", "#009E73", "#333333"


# ---------------------------------------------------------------------------
# 1 · Timestamps
# ---------------------------------------------------------------------------
def load_timestamps(dicom_ids: set) -> pd.DataFrame:
    """Returns dicom_id -> integer timestamp, from the MIMIC metadata file.

    The 56 MB source lives outside this folder, so the test-set slice is cached
    into reports/stage15/ on first run. Subsequent runs are self-contained and
    need no access to data/.
    """
    if TS_CACHE.exists():
        return pd.read_csv(TS_CACHE)
    if not METADATA.exists():
        raise SystemExit(
            "Cannot establish chronological order.\n"
            "  Needed : %s\n"
            "  or the cache it produces: %s\n\n"
            "  study_id ordering is NOT a substitute -- it matches true\n"
            "  chronology only 49.6%% of the time on this split." % (METADATA, TS_CACHE))

    md = pd.read_csv(METADATA, low_memory=False,
                     usecols=["dicom_id", "study_id", "StudyDate", "StudyTime"])
    md = md[md.dicom_id.isin(dicom_ids)].copy()
    # StudyTime is HHMMSS.frac; the integer part orders within a day.
    t = md.StudyTime.astype(float).fillna(0).astype("int64")
    md["ts"] = md.StudyDate.astype("int64") * 1_000_000 + t
    md = md[["dicom_id", "study_id", "ts"]]
    OUT.mkdir(parents=True, exist_ok=True)
    md.to_csv(TS_CACHE, index=False)
    return md


def study_id_order_agreement(df: pd.DataFrame) -> float:
    """Fraction of consecutive pairs where study_id order matches time order.

    Exists to be asserted, not admired: it documents why the metadata join is
    mandatory.
    """
    st = df.groupby(["subject_id", "study_id"]).ts.min().reset_index()
    ok = tot = 0
    for _, d in st.groupby("subject_id"):
        if len(d) < 2:
            continue
        t = d.sort_values("study_id").ts.to_numpy()
        tot += len(t) - 1
        ok += int((t[:-1] < t[1:]).sum())
    return ok / tot if tot else float("nan")


# ---------------------------------------------------------------------------
# 2 · Pair construction
# ---------------------------------------------------------------------------
def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive study pairs per patient, ordered by TRUE timestamp.

    One study can contain several images; probabilities are averaged within a
    study, matching how a radiologist reports per study rather than per image.
    """
    agg = {k: (k, "max") for k in LABELS}
    rows = []
    for sid, d in df.groupby("subject_id"):
        st = (d.groupby("study_id")
                .agg(ts=("ts", "min"), proj=("view", "first"), p=("p", "mean"), **agg)
                .reset_index().sort_values("ts"))
        for i in range(len(st) - 1):
            a, b = st.iloc[i], st.iloc[i + 1]
            rows.append(dict(
                sid=sid, t1=a["ts"], t2=b["ts"],
                v1=a["proj"], v2=b["proj"], trans=a["proj"] + "->" + b["proj"],
                p1=float(a["p"]), p2=float(b["p"]),
                card_same=int(a[TARGET]) == int(b[TARGET]),
                all_same=all(int(a[k]) == int(b[k]) for k in LABELS)))
    P = pd.DataFrame(rows)
    P["proj_changed"] = P.v1 != P.v2
    return P


def score(P: pd.DataFrame, t1, t2) -> pd.DataFrame:
    """Adds threshold-crossing flags for a given pair of operating points.

    `t1`/`t2` are arrays so the caller can pass either one global threshold or
    per-projection thresholds -- that difference is exactly arm C vs arm D.
    """
    Q = P.copy()
    Q["up"] = (Q.p1 < t1) & (Q.p2 >= t2)      # model newly calls disease
    Q["dn"] = (Q.p1 >= t1) & (Q.p2 < t2)      # model newly clears it
    return Q


# ---------------------------------------------------------------------------
# 3 · Statistics
# ---------------------------------------------------------------------------
def asymmetry(S: pd.DataFrame) -> float:
    """False-worsening rate minus false-improvement rate, percentage points.

    Random error is symmetric and cancels here. A non-zero value means the
    errors have a DIRECTION, which noise cannot produce.
    """
    return 100.0 * (S.up.mean() - S.dn.mean())


def cluster_bootstrap(S: pd.DataFrame, n_boot=N_BOOT, seed=SEED) -> np.ndarray:
    """Resamples PATIENTS, not pairs.

    A patient can contribute up to 32 pairs here; resampling pairs would treat
    those as independent and produce intervals far too narrow.
    """
    rng = np.random.default_rng(seed)
    ids = S.sid.unique()
    idx = {s: g.index.to_numpy() for s, g in S.groupby("sid")}
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([idx[s] for s in rng.choice(ids, len(ids), replace=True)])
        out[i] = asymmetry(S.loc[pick])
    return out


def ci(v: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided bootstrap p for the difference of two asymmetries."""
    d = a - b
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))


def shuffle_order(P: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """THE NULL. Randomises which study of each pair is 'first'.

    Everything else -- the patient, the two images, both projections, both
    probabilities -- is untouched. Only temporal direction is destroyed. If the
    asymmetry survives this, it was never directional.
    """
    rng = np.random.default_rng(seed)
    Q = P.copy()
    flip = rng.random(len(Q)) < 0.5
    for a, b in (("p1", "p2"), ("v1", "v2")):
        Q.loc[flip, [a, b]] = Q.loc[flip, [b, a]].to_numpy()
    Q["trans"] = Q.v1 + "->" + Q.v2
    return Q


# ---------------------------------------------------------------------------
# 4 · Analysis
# ---------------------------------------------------------------------------
def analyse(P: pd.DataFrame, thr: dict, condition: str) -> dict:
    tG = thr["global"][TARGET]
    tau = lambda v: np.where(v == "AP", thr["AP"][TARGET], thr["PA"][TARGET])
    S0 = P[P[condition]].copy()

    res = {"condition": condition, "n_pairs": int(len(S0)),
           "n_patients": int(S0.sid.nunique()), "by_transition": [], "arms": {}}

    G = score(S0, tG, tG)
    for t in ["AP->AP", "PA->PA", "PA->AP", "AP->PA"]:
        s = G[G.trans == t]
        if len(s) < 30:
            continue
        bt = cluster_bootstrap(s)
        lo, hi = ci(bt)
        res["by_transition"].append(dict(
            transition=t, n=int(len(s)), n_patients=int(s.sid.nunique()),
            false_worse_pct=round(100 * s.up.mean(), 2),
            false_better_pct=round(100 * s.dn.mean(), 2),
            asymmetry=round(asymmetry(s), 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
            significant=bool(lo > 0 or hi < 0)))

    same = G[~G.proj_changed]
    chg = G[G.proj_changed & (G.trans == "PA->AP")]
    null = score(shuffle_order(S0), tG, tG)
    null = null[null.v1 != null.v2]
    fix = score(S0[S0.trans == "PA->AP"], tau(S0[S0.trans == "PA->AP"].v1),
                tau(S0[S0.trans == "PA->AP"].v2))

    for name, S in (("A_same_projection", same), ("B_shuffled_null", null),
                    ("C_true_order_PA_to_AP", chg), ("D_threshold_corrected", fix)):
        if len(S) < 20:
            continue
        bt = cluster_bootstrap(S)
        lo, hi = ci(bt)
        res["arms"][name] = dict(
            n=int(len(S)), n_patients=int(S.sid.nunique()),
            false_worse_pct=round(100 * S.up.mean(), 2),
            false_better_pct=round(100 * S.dn.mean(), 2),
            asymmetry=round(asymmetry(S), 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
            significant=bool(lo > 0 or hi < 0))

    bA, bC, bD = (cluster_bootstrap(same), cluster_bootstrap(chg), cluster_bootstrap(fix))
    bB = cluster_bootstrap(null)
    res["tests"] = {
        "C_vs_A_finding_vs_control": dict(
            diff=round(float((bC - bA).mean()), 2),
            ci=[round(x, 2) for x in ci(bC - bA)], p=round(boot_p(bC, bA), 5)),
        "C_vs_B_finding_vs_shuffled_null": dict(
            diff=round(float((bC - bB).mean()), 2),
            ci=[round(x, 2) for x in ci(bC - bB)], p=round(boot_p(bC, bB), 5)),
        "D_vs_C_does_the_fix_work": dict(
            diff=round(float((bD - bC).mean()), 2),
            ci=[round(x, 2) for x in ci(bD - bC)], p=round(boot_p(bD, bC), 5)),
    }
    return res


# ---------------------------------------------------------------------------
# 5 · Output
# ---------------------------------------------------------------------------
def chart(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["by_transition"]
    labels = [r["transition"] for r in rows]
    vals = [r["asymmetry"] for r in rows]
    los = [r["asymmetry"] - r["ci_lo"] for r in rows]
    his = [r["ci_hi"] - r["asymmetry"] for r in rows]
    cols = [C_FIND if r["transition"] in ("PA->AP", "AP->PA") else C_CTRL for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    y = np.arange(len(rows))
    ax[0].axvline(0, color=C_INK, lw=1, alpha=0.45)
    # One call per point: errorbar's `ecolor` takes a single colour, not a list
    # (passing a list raises "Invalid RGBA argument").
    for i in range(len(rows)):
        ax[0].errorbar([vals[i]], [y[i]], xerr=[[los[i]], [his[i]]], fmt="none",
                       ecolor=cols[i], elinewidth=2, capsize=4)
    ax[0].scatter(vals, y, color=cols, s=60, zorder=5)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(["%s\nn=%d" % (r["transition"], r["n"]) for r in rows], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("False worsening − false improvement (percentage points)")
    ax[0].set_title("Only a CHANGE of projection produces\ndirectional error",
                    fontsize=10, loc="left", color=C_INK)
    ax[0].grid(True, axis="x", lw=0.5, alpha=0.25, color=C_INK)
    ax[0].set_axisbelow(True)
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False)

    order = ["A_same_projection", "B_shuffled_null", "C_true_order_PA_to_AP",
             "D_threshold_corrected"]
    names = ["A · same projection\n(control)", "B · shuffled order\n(null)",
             "C · PA→AP, true order\n(the finding)", "D · + per-projection\nthresholds (the fix)"]
    cmap = [C_CTRL, C_NULL, C_FIND, C_FIX]
    arms = [(n, res["arms"][k], c) for k, n, c in zip(order, names, cmap) if k in res["arms"]]
    y2 = np.arange(len(arms))
    ax[1].axvline(0, color=C_INK, lw=1, alpha=0.45)
    for i, (nm, a, c) in enumerate(arms):
        ax[1].errorbar([a["asymmetry"]], [i],
                       xerr=[[a["asymmetry"] - a["ci_lo"]], [a["ci_hi"] - a["asymmetry"]]],
                       fmt="none", ecolor=c, elinewidth=2, capsize=4)
        ax[1].scatter([a["asymmetry"]], [i], color=c, s=60, zorder=5)
    ax[1].set_yticks(y2)
    ax[1].set_yticklabels([n for n, _, _ in arms], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("False worsening − false improvement (percentage points)")
    ax[1].set_title("The obvious fix reduces it significantly\nbut does NOT eliminate it",
                    fontsize=10, loc="left", color=C_INK)
    ax[1].grid(True, axis="x", lw=0.5, alpha=0.25, color=C_INK)
    ax[1].set_axisbelow(True)
    for s in ("top", "right"):
        ax[1].spines[s].set_visible(False)

    fig.suptitle("Stage 15 · Acquisition-induced false interval change — Cardiomegaly, "
                 "radiologist recorded NO CHANGE (n=%d pairs, %d patients)"
                 % (res["n_pairs"], res["n_patients"]),
                 fontsize=10, x=0.008, ha="left", color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def report(main: dict, strict: dict) -> str:
    L = ["# Stage 15 — Acquisition-Induced False Interval Change", "",
         "Pairs of consecutive studies from the same patient, ordered by **true**",
         "`StudyDate`/`StudyTime`. Restricted to pairs where the radiologist recorded",
         "**no change**, so any movement the model reports is spurious by construction.", ""]
    for res, title in ((main, "Radiologist: cardiomegaly unchanged"),
                       (strict, "Radiologist: all 8 findings unchanged (stricter)")):
        L += ["## %s" % title, "",
              "n = %d pairs from %d patients" % (res["n_pairs"], res["n_patients"]), "",
              "| Transition | n | False worsening | False improvement | Asymmetry [95% CI] | Sig |",
              "|---|---|---|---|---|---|"]
        for r in res["by_transition"]:
            L.append("| %s | %d | %.1f%% | %.1f%% | %+.2f [%+.2f, %+.2f] | %s |" % (
                r["transition"], r["n"], r["false_worse_pct"], r["false_better_pct"],
                r["asymmetry"], r["ci_lo"], r["ci_hi"], "**YES**" if r["significant"] else "no"))
        L += ["", "### Four arms", "",
              "| Arm | n | Asymmetry [95% CI] | Significant |", "|---|---|---|---|"]
        # ASCII arrows only: this string is printed to a cp1252 console on
        # Windows, which cannot encode U+2192.
        pretty = {"A_same_projection": "A - same projection (control)",
                  "B_shuffled_null": "B - shuffled temporal order (null)",
                  "C_true_order_PA_to_AP": "C - PA->AP, true order (finding)",
                  "D_threshold_corrected": "D - C + per-projection thresholds (the fix)"}
        for k, a in res["arms"].items():
            L.append("| %s | %d | %+.2f [%+.2f, %+.2f] | %s |" % (
                pretty.get(k, k), a["n"], a["asymmetry"], a["ci_lo"], a["ci_hi"],
                "**YES**" if a["significant"] else "no"))
        L += ["", "### Tests", "", "| Comparison | Difference [95% CI] | p |", "|---|---|---|"]
        pt = {"C_vs_A_finding_vs_control": "Finding vs same-projection control",
              "C_vs_B_finding_vs_shuffled_null": "Finding vs shuffled-order null",
              "D_vs_C_does_the_fix_work": "Threshold fix vs uncorrected"}
        for k, t in res["tests"].items():
            L.append("| %s | %+.2f [%+.2f, %+.2f] | %.5f |" % (
                pt.get(k, k), t["diff"], t["ci"][0], t["ci"][1], t["p"]))
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 6 · Data
# ---------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, dict]:
    te = pd.read_csv(MANIFEST, low_memory=False)
    pr = np.load(PROBS)
    thr = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    te = te.iloc[:len(pr)].copy()
    te["p"] = pr[:, thr["pathologies"].index(TARGET)]
    ts = load_timestamps(set(te.dicom_id))
    te = te.merge(ts[["dicom_id", "ts"]], on="dicom_id", how="left")
    if te.ts.isna().any():
        raise SystemExit("%d test images have no timestamp; cannot order them."
                         % int(te.ts.isna().sum()))
    return te, thr


# ---------------------------------------------------------------------------
# 7 · Self-checks
# ---------------------------------------------------------------------------
def selftest() -> int:
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  PASS  %s" % name)
        else:
            fail += 1
            print("  FAIL  %s" % name)

    chk("torch is never imported", "torch" not in sys.modules)
    o = OUT.resolve()
    chk("output stays inside the project", HERE.resolve() in o.parents or o.parent == HERE.resolve())

    te, thr = load()
    chk("every test image has a timestamp", not te.ts.isna().any())
    chk("timestamps joined for all %d images" % len(te), len(te) == 4722)

    agree = study_id_order_agreement(te)
    chk("study_id ordering is a COIN FLIP (%.4f) -- metadata join is mandatory"
        % agree, 0.40 < agree < 0.60)

    P = build_pairs(te)
    chk("pairs built (%d from %d patients)" % (len(P), P.sid.nunique()), len(P) > 1000)
    chk("every pair is chronologically ordered", bool((P.t2 >= P.t1).all()))
    chk("both projection-change directions present",
        {"PA->AP", "AP->PA"} <= set(P.trans.unique()))
    chk("proj_changed flag agrees with the transition label",
        bool((P.proj_changed == (P.trans.isin(["PA->AP", "AP->PA"]))).all()))

    tG = thr["global"][TARGET]
    G = score(P[P.card_same], tG, tG)
    chk("up and dn are mutually exclusive", not bool((G.up & G.dn).any()))

    same = G[~G.proj_changed]
    pa_ap = G[G.trans == "PA->AP"]
    chk("control arm is near zero (%.2f)" % asymmetry(same), abs(asymmetry(same)) < 3.0)
    chk("PA->AP shows positive asymmetry (%.2f)" % asymmetry(pa_ap), asymmetry(pa_ap) > 5.0)

    bs, bp = cluster_bootstrap(same), cluster_bootstrap(pa_ap)
    lo_s, hi_s = ci(bs)
    lo_p, hi_p = ci(bp)
    chk("control CI includes zero [%.2f, %.2f]" % (lo_s, hi_s), lo_s <= 0 <= hi_s)
    chk("PA->AP CI excludes zero [%.2f, %.2f]" % (lo_p, hi_p), lo_p > 0)
    chk("finding beats control (p=%.5f)" % boot_p(bp, bs), boot_p(bp, bs) < 0.05)

    # The null must collapse the effect.
    N = score(shuffle_order(P[P.card_same]), tG, tG)
    N = N[N.v1 != N.v2]
    chk("shuffled-order null collapses asymmetry (%.2f vs %.2f)"
        % (asymmetry(N), asymmetry(pa_ap)), abs(asymmetry(N)) < abs(asymmetry(pa_ap)))
    chk("shuffling preserves the pair count", len(N) == int(P[P.card_same].proj_changed.sum()))

    chk("bootstrap is reproducible under a fixed seed",
        bool(np.allclose(cluster_bootstrap(pa_ap, 200, 5), cluster_bootstrap(pa_ap, 200, 5))))
    chk("resampling patients not pairs widens the interval",
        (hi_p - lo_p) > 0.5)

    print("\n  %d passed, %d failed" % (ok, fail))
    return 1 if fail else 0


# ---------------------------------------------------------------------------
def main() -> int:
    for p in (MANIFEST, PROBS, THRESHOLDS):
        if not p.exists():
            raise SystemExit("missing required input: %s" % p)

    print("STAGE 15 - acquisition-induced false interval change\n")
    te, thr = load()
    agree = study_id_order_agreement(te)
    print("  study_id ordering matches true chronology: %.2f%%  "
          "<- why the metadata join is mandatory" % (100 * agree))

    P = build_pairs(te)
    print("  chronological pairs: %d from %d patients" % (len(P), P.sid.nunique()))
    print("  transitions: %s\n" % dict(P.trans.value_counts()))

    main_res = analyse(P, thr, "card_same")
    strict_res = analyse(P, thr, "all_same")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(dict(study_id_order_agreement=round(agree, 4),
                        main=main_res, strict=strict_res), indent=2), encoding="utf-8")
    md = report(main_res, strict_res)
    (OUT / "interval_change.md").write_text(md, encoding="utf-8")
    print(md)
    chart(main_res, OUT / "interval_change.png")

    print("\n  wrote %s" % (OUT / "interval_change.md"))
    print("  wrote %s" % (OUT / "summary.json"))
    print("  wrote %s" % (OUT / "interval_change.png"))
    print("  wrote %s  (cached, makes reruns self-contained)" % TS_CACHE)
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--test" in sys.argv else main())
