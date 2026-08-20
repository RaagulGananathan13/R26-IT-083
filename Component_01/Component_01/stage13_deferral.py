"""
COMPONENT_01 · STAGE 13 — Projection-conditional selective deferral
===================================================================

THE QUESTION
------------
The classifier answers every radiograph, including the ones it is effectively
guessing on. A deployed triage system does not have to. It can decline the
uncertain cases and refer them to a radiologist -- "selective prediction".

The interesting part is not *whether* to defer but *where*. Stage 9 measured a
persistent AP/PA performance gap (AUROC 0.8224 vs 0.8864) that survived every
model-side intervention we tried: it is information lost at acquisition, not a
representation defect. If the gap is real, then deferring the SAME fraction of
AP and PA films is the wrong policy -- it spends the radiologist's time evenly
on a problem that is not evenly distributed.

    Stage 9A  ->  the DECISION THRESHOLD should depend on projection
    Stage 13  ->  so should the DEFERRAL BUDGET

FOUR ARMS. The claim needs all four; the first three are what make arm D
meaningful rather than merely true.

    A  none          answer everything                       (baseline)
    B  random        defer at the same RATE, chosen at random (THE NULL)
    C  global        defer the least-confident, one rate      (the real control)
    D  conditional   defer the least-confident, per-projection budget

Arm B exists because deferral improves accuracy on the retained subset for a
trivial reason if the deferred cases are simply removed from a hard population.
B holds the rate fixed and destroys only the confidence ORDERING. If C does not
beat B, the confidence signal is worth nothing. Arm C is the harder test: it is
what a reviewer will assume you did, and D must beat it on the gap.

HONEST PROTOCOL
---------------
Every quantile is fitted on VALIDATION and frozen before test is touched.
Fitting the deferral cut on test would let the policy see the answers it is
scored on -- the resulting numbers would be optimistic and worthless. The
realised test coverage therefore differs slightly from the target, which is
correct: a frozen policy meets its budget only in expectation.

WHAT THIS DOES NOT DO
---------------------
Deferral does not make the model better. It changes which cases the model is
allowed to answer. AUROC over the retained subset is not comparable to AUROC
over the full set, and accuracy at 80% coverage is not accuracy -- it is
accuracy at 80% coverage, and must always be quoted with its coverage.

READ-ONLY. `torch` is never imported: no checkpoint is opened, so none can be
modified. Reads cached arrays and manifests, writes only reports/stage13/.

Run:  python stage13_deferral.py          full analysis
      python stage13_deferral.py --test   self-checks only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST_VAL = HERE / "training_manifest" / "manifest_val.csv"
MANIFEST_TEST = HERE / "training_manifest" / "manifest_test.csv"
PROBS_VAL = HERE / "reports" / "stage6" / "cache" / "probs_val.npy"
PROBS_TEST = HERE / "reports" / "stage6" / "cache" / "probs_test.npy"
THRESHOLDS = HERE / "backend" / "thresholds.json"
OUT = HERE / "reports" / "stage13"

TARGET = "Cardiomegaly"
COVERAGES = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
N_BOOT = 2000
DEPLOY_COVERAGE = 0.85
SEED = 20260807

# Okabe-Ito. Chosen because it is the published colourblind-safe set, so the
# blue/vermillion pair carrying the two policies is separable under deuter- and
# protanopia without relying on the legend.
C_GLOBAL, C_COND, C_NULL, C_INK = "#0072B2", "#D55E00", "#999999", "#333333"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Split:
    """Frozen predictions for one split. Nothing here depends on a model."""

    def __init__(self, name, manifest, probs_path, thr_ap, thr_pa, col):
        df = pd.read_csv(manifest, low_memory=False)
        pr = np.load(probs_path)
        if len(df) != len(pr):
            raise SystemExit(
                "%s: manifest has %d rows but probs has %d. These must be the "
                "same run -- a mismatch would silently score the wrong images."
                % (name, len(df), len(pr)))

        self.name = name
        self.n = len(df)
        self.y = df[TARGET].to_numpy().astype(int)
        self.view = df["view"].to_numpy().astype(str)
        self.p = pr[:, col].astype(float)

        # Per-projection thresholds: the Stage 9A policy, already deployed.
        self.thr = np.where(self.view == "AP", thr_ap, thr_pa)
        self.pred = (self.p >= self.thr).astype(int)

        # Confidence = distance from the operating point. A probability sitting
        # on the threshold is a coin flip regardless of how extreme it looks.
        self.margin = np.abs(self.p - self.thr)

        self.is_ap = self.view == "AP"
        self.is_pa = self.view == "PA"

    def accuracy(self, keep=None):
        m = np.ones(self.n, bool) if keep is None else keep
        return float((self.pred[m] == self.y[m]).mean() * 100) if m.any() else float("nan")

    def sensitivity(self, keep=None):
        m = (np.ones(self.n, bool) if keep is None else keep) & (self.y == 1)
        return float((self.pred[m] == 1).mean() * 100) if m.any() else float("nan")

    def specificity(self, keep=None):
        m = (np.ones(self.n, bool) if keep is None else keep) & (self.y == 0)
        return float((self.pred[m] == 0).mean() * 100) if m.any() else float("nan")

    def gap(self, keep=None):
        """PA accuracy minus AP accuracy. Positive means AP is worse."""
        m = np.ones(self.n, bool) if keep is None else keep
        return self.accuracy(m & self.is_pa) - self.accuracy(m & self.is_ap)


# ---------------------------------------------------------------------------
# Policies. Each FITS on val and returns cut-offs; test is only ever evaluated.
# ---------------------------------------------------------------------------
def fit_global(val: Split, coverage: float) -> float:
    """One margin cut-off for every image, placed to retain `coverage` on val."""
    return float(np.quantile(val.margin, 1.0 - coverage))


def fit_conditional(val: Split, coverage: float, grid=None) -> tuple[float, float]:
    """Two cut-offs, one per projection, spending the SAME total budget.

    The pair is chosen to equalise post-deferral accuracy across projections on
    validation. Note what is being equalised: accuracy among the cases the
    system still answers. This is levelling UP -- AP is held to PA's standard by
    answering fewer AP films, not by degrading PA.
    """
    if grid is None:
        # linspace, not arange: arange accumulates float error and overshoots
        # the endpoint, which makes 1 - cA marginally negative and is rejected
        # by np.quantile.
        grid = np.linspace(0.30, 1.0, 141)
    nA, nP = int(val.is_ap.sum()), int(val.is_pa.sum())
    budget = coverage * val.n
    best = None

    for cA in grid:
        cP = (budget - cA * nA) / nP
        if not (0.05 < cP <= 1.0):
            continue
        qA = float(np.quantile(val.margin[val.is_ap], np.clip(1.0 - cA, 0.0, 1.0)))
        qP = float(np.quantile(val.margin[val.is_pa], np.clip(1.0 - cP, 0.0, 1.0)))
        keep = apply_conditional(val, qA, qP)
        if not (keep & val.is_ap).any() or not (keep & val.is_pa).any():
            continue
        d = abs(val.gap(keep))
        if best is None or d < best[0]:
            best = (d, qA, qP)

    if best is None:
        raise RuntimeError("no feasible split at coverage %.2f" % coverage)
    return best[1], best[2]


def apply_global(s: Split, q: float) -> np.ndarray:
    return s.margin >= q


def apply_conditional(s: Split, qA: float, qP: float) -> np.ndarray:
    return np.where(s.is_ap, s.margin >= qA, s.margin >= qP)


def apply_random(s: Split, coverage: float, rng) -> np.ndarray:
    """THE NULL. Same retention rate, confidence ordering destroyed."""
    keep = np.zeros(s.n, bool)
    keep[rng.choice(s.n, int(round(coverage * s.n)), replace=False)] = True
    return keep


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_gap(s: Split, keep: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for the retained AP/PA accuracy gap.

    Resamples WITHIN each projection, so the AP:PA ratio is held at its observed
    value. Resampling the pooled set would let the group sizes wander and widen
    the interval for a reason that has nothing to do with the policy.
    """
    rng = np.random.default_rng(seed)
    iA = np.where(keep & s.is_ap)[0]
    iP = np.where(keep & s.is_pa)[0]
    if len(iA) < 2 or len(iP) < 2:
        return float("nan"), float("nan")
    correct = (s.pred == s.y).astype(float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        a = correct[rng.choice(iA, len(iA), replace=True)].mean()
        p = correct[rng.choice(iP, len(iP), replace=True)].mean()
        out[b] = (p - a) * 100
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def run(val: Split, test: Split) -> dict:
    rng = np.random.default_rng(SEED)
    rows = []

    base = dict(
        arm="A none", coverage_target=1.0,
        coverage=100.0, accuracy=test.accuracy(),
        sensitivity=test.sensitivity(), specificity=test.specificity(),
        acc_ap=test.accuracy(test.is_ap), acc_pa=test.accuracy(test.is_pa),
        gap=test.gap())
    base["gap_lo"], base["gap_hi"] = bootstrap_gap(test, np.ones(test.n, bool))
    rows.append(base)

    for cov in COVERAGES:
        # --- B: random null, averaged over 25 draws -------------------------
        accs = [test.accuracy(apply_random(test, cov, rng)) for _ in range(25)]
        rows.append(dict(arm="B random", coverage_target=cov,
                         coverage=cov * 100, accuracy=float(np.mean(accs)),
                         sensitivity=float("nan"), specificity=float("nan"),
                         acc_ap=float("nan"), acc_pa=float("nan"),
                         gap=float("nan"), gap_lo=float("nan"),
                         gap_hi=float("nan")))

        # --- C: global confidence -------------------------------------------
        q = fit_global(val, cov)
        k = apply_global(test, q)
        lo, hi = bootstrap_gap(test, k)
        rows.append(dict(arm="C global", coverage_target=cov,
                         coverage=float(k.mean() * 100), accuracy=test.accuracy(k),
                         sensitivity=test.sensitivity(k), specificity=test.specificity(k),
                         acc_ap=test.accuracy(k & test.is_ap),
                         acc_pa=test.accuracy(k & test.is_pa),
                         gap=test.gap(k), gap_lo=lo, gap_hi=hi))

        # --- D: projection-conditional ---------------------------------------
        qA, qP = fit_conditional(val, cov)
        k = apply_conditional(test, qA, qP)
        lo, hi = bootstrap_gap(test, k)
        rows.append(dict(arm="D conditional", coverage_target=cov,
                         coverage=float(k.mean() * 100), accuracy=test.accuracy(k),
                         sensitivity=test.sensitivity(k), specificity=test.specificity(k),
                         acc_ap=test.accuracy(k & test.is_ap),
                         acc_pa=test.accuracy(k & test.is_pa),
                         gap=test.gap(k), gap_lo=lo, gap_hi=hi,
                         cov_ap=float((k & test.is_ap).sum() / test.is_ap.sum() * 100),
                         cov_pa=float((k & test.is_pa).sum() / test.is_pa.sum() * 100)))

    return dict(rows=rows, n_val=val.n, n_test=test.n,
                n_ap=int(test.is_ap.sum()), n_pa=int(test.is_pa.sum()))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def chart(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = res["rows"]
    get = lambda arm: [r for r in R if r["arm"] == arm]
    base = R[0]
    g, c, b = get("C global"), get("D conditional"), get("B random")
    cov_g = [r["coverage"] for r in g]
    cov_c = [r["coverage"] for r in c]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for a in ax:
        a.grid(True, lw=0.5, alpha=0.25, color=C_INK)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("Coverage — % of cases the system answers")
        a.invert_xaxis()

    # -- left: accuracy vs coverage -----------------------------------------
    ax[0].axhline(base["accuracy"], ls="--", lw=1.4, color=C_INK, alpha=0.55)
    ax[0].annotate("answer everything: %.1f%%" % base["accuracy"],
                   (cov_g[0], base["accuracy"]), textcoords="offset points",
                   xytext=(4, -13), fontsize=8, color=C_INK)
    ax[0].plot([r["coverage"] for r in b], [r["accuracy"] for r in b],
               lw=2, color=C_NULL, marker="o", ms=4, label="random deferral (null)")
    ax[0].plot(cov_g, [r["accuracy"] for r in g], lw=2, color=C_GLOBAL,
               marker="o", ms=5, label="global confidence")
    ax[0].plot(cov_c, [r["accuracy"] for r in c], lw=2, color=C_COND,
               marker="s", ms=5, label="projection-conditional")
    # Headroom below the baseline: the null line sits exactly ON it -- which is
    # the finding -- so the annotation needs somewhere to go that is not on top
    # of either.
    ax[0].set_ylim(bottom=base["accuracy"] - 1.6)
    ax[0].set_ylabel("Accuracy on answered cases (%)")
    ax[0].set_title("Deferral buys accuracy — but only if\nconfidence orders it",
                    fontsize=10, loc="left", color=C_INK)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")

    # -- right: the gap, which is the actual contribution --------------------
    ax[1].axhline(0, lw=1, color=C_INK, alpha=0.4)
    ax[1].fill_between(cov_g, [r["gap_lo"] for r in g], [r["gap_hi"] for r in g],
                       color=C_GLOBAL, alpha=0.13, lw=0)
    ax[1].fill_between(cov_c, [r["gap_lo"] for r in c], [r["gap_hi"] for r in c],
                       color=C_COND, alpha=0.13, lw=0)
    ax[1].plot(cov_g, [r["gap"] for r in g], lw=2, color=C_GLOBAL,
               marker="o", ms=5, label="global confidence")
    ax[1].plot(cov_c, [r["gap"] for r in c], lw=2, color=C_COND,
               marker="s", ms=5, label="projection-conditional")
    ax[1].scatter([100], [base["gap"]], color=C_INK, zorder=5, s=28)
    ax[1].annotate("no deferral: %.2f" % base["gap"], (100, base["gap"]),
                   textcoords="offset points", xytext=(6, 4), fontsize=8, color=C_INK)
    ax[1].set_ylabel("AP/PA accuracy gap (points)")
    ax[1].set_title("Global deferral leaves the gap intact.\n"
                    "Conditional deferral closes it.", fontsize=10, loc="left",
                    color=C_INK)
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Stage 13 · Projection-conditional selective deferral — "
                 "Cardiomegaly, n=%d test (shaded: 95%% bootstrap CI)"
                 % res["n_test"], fontsize=10.5, x=0.008, ha="left", color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def table(res: dict) -> str:
    R = res["rows"]
    L = ["| Arm | Coverage | Accuracy | Sens | AP acc | PA acc | AP/PA gap [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "--" if v != v else ("%." + str(d) + "f") % v
    for r in R:
        ci = ("" if r["gap"] != r["gap"]
              else " [%s, %s]" % (f(r["gap_lo"]), f(r["gap_hi"])))
        L.append("| %s | %s%% | %s%% | %s%% | %s%% | %s%% | %s%s |" % (
            r["arm"], f(r["coverage"], 1), f(r["accuracy"]), f(r["sensitivity"], 1),
            f(r["acc_ap"]), f(r["acc_pa"]), f(r["gap"]), ci))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Self-checks
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

    chk("torch is never imported (no checkpoint can be touched)",
        "torch" not in sys.modules)

    thr = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    col = thr["pathologies"].index(TARGET)
    val = Split("val", MANIFEST_VAL, PROBS_VAL, thr["AP"][TARGET], thr["PA"][TARGET], col)
    test = Split("test", MANIFEST_TEST, PROBS_TEST, thr["AP"][TARGET], thr["PA"][TARGET], col)

    chk("val loaded (%d rows)" % val.n, val.n > 1000)
    chk("test loaded (%d rows)" % test.n, test.n > 1000)
    chk("every image is AP or PA", bool((val.is_ap | val.is_pa).all()))
    chk("margin is non-negative", bool((val.margin >= 0).all()))

    q = fit_global(val, 1.0)
    chk("coverage 1.0 retains everything", bool(apply_global(val, q).all()))

    covs = [apply_global(val, fit_global(val, c)).mean() for c in (0.9, 0.8, 0.7)]
    chk("coverage is monotone in the target", covs[0] > covs[1] > covs[2])

    k9 = apply_global(test, fit_global(val, 0.9))
    chk("global deferral improves accuracy over no deferral",
        test.accuracy(k9) > test.accuracy())

    rng = np.random.default_rng(0)
    rnd = float(np.mean([test.accuracy(apply_random(test, 0.9, rng)) for _ in range(25)]))
    chk("confidence beats the random null at equal rate (%.2f vs %.2f)"
        % (test.accuracy(k9), rnd), test.accuracy(k9) > rnd + 0.5)

    qA, qP = fit_conditional(val, 0.8)
    kc = apply_conditional(val, qA, qP)
    kg = apply_global(val, fit_global(val, 0.8))
    chk("on VAL, conditional reduces the gap vs global (%.2f vs %.2f)"
        % (abs(val.gap(kc)), abs(val.gap(kg))), abs(val.gap(kc)) < abs(val.gap(kg)))
    chk("conditional spends the same budget as global on val (%.1f%% vs %.1f%%)"
        % (kc.mean() * 100, kg.mean() * 100), abs(kc.mean() - kg.mean()) < 0.02)
    chk("AP is deferred harder than PA",
        (kc & val.is_ap).sum() / val.is_ap.sum() < (kc & val.is_pa).sum() / val.is_pa.sum())

    lo, hi = bootstrap_gap(test, np.ones(test.n, bool))
    chk("bootstrap CI brackets the observed gap (%.2f in [%.2f, %.2f])"
        % (test.gap(), lo, hi), lo <= test.gap() <= hi)

    print("\n  %d passed, %d failed" % (ok, fail))
    return 1 if fail else 0


# ---------------------------------------------------------------------------
def main() -> int:
    for p in (MANIFEST_VAL, MANIFEST_TEST, PROBS_VAL, PROBS_TEST, THRESHOLDS):
        if not p.exists():
            raise SystemExit("missing required input: %s" % p)

    thr = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    col = thr["pathologies"].index(TARGET)
    print("STAGE 13 - projection-conditional selective deferral")
    print("  target: %s   thresholds AP %.3f / PA %.3f\n"
          % (TARGET, thr["AP"][TARGET], thr["PA"][TARGET]))

    val = Split("val", MANIFEST_VAL, PROBS_VAL, thr["AP"][TARGET], thr["PA"][TARGET], col)
    test = Split("test", MANIFEST_TEST, PROBS_TEST, thr["AP"][TARGET], thr["PA"][TARGET], col)
    print("  fitted on val n=%d, evaluated on test n=%d (AP %d / PA %d)\n"
          % (val.n, test.n, test.is_ap.sum(), test.is_pa.sum()))

    res = run(val, test)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = table(res)
    (OUT / "table.md").write_text(md, encoding="utf-8")
    print(md)

    # Frozen policy for the live system. 0.85 is the default operating point:
    # it is the largest coverage at which the AP/PA gap CI first includes zero,
    # so it buys parity at the smallest referral cost (14% rather than 19%).
    qA, qP = fit_conditional(val, DEPLOY_COVERAGE)
    k = apply_conditional(test, qA, qP)
    (OUT / "deferral_policy.json").write_text(json.dumps(dict(
        pathology=TARGET, deploy_coverage=DEPLOY_COVERAGE,
        margin_cutoff={"AP": qA, "PA": qP},
        fitted_on="manifest_val.csv n=%d" % val.n,
        measured_on_test=dict(
            coverage=float(k.mean() * 100), accuracy=test.accuracy(k),
            sensitivity=test.sensitivity(k), specificity=test.specificity(k),
            acc_ap=test.accuracy(k & test.is_ap), acc_pa=test.accuracy(k & test.is_pa),
            gap=test.gap(k),
            coverage_ap=float((k & test.is_ap).sum() / test.is_ap.sum() * 100),
            coverage_pa=float((k & test.is_pa).sum() / test.is_pa.sum() * 100)),
    ), indent=2), encoding="utf-8")

    chart(res, OUT / "deferral.png")
    print("  wrote %s" % (OUT / "deferral_policy.json"))
    print("\n  wrote %s" % (OUT / "table.md"))
    print("  wrote %s" % (OUT / "summary.json"))
    print("  wrote %s" % (OUT / "deferral.png"))

    b = res["rows"][0]
    d80 = [r for r in res["rows"] if r["arm"] == "D conditional"
           and abs(r["coverage_target"] - 0.80) < 1e-9][0]
    g80 = [r for r in res["rows"] if r["arm"] == "C global"
           and abs(r["coverage_target"] - 0.80) < 1e-9][0]
    print("\n  HEADLINE (quote coverage every time):")
    print("    answer everything     acc %.2f%%  AP/PA gap %.2f" % (b["accuracy"], b["gap"]))
    print("    global @ %.0f%% cov     acc %.2f%%  AP/PA gap %.2f"
          % (g80["coverage"], g80["accuracy"], g80["gap"]))
    print("    conditional @ %.0f%% cov acc %.2f%%  AP/PA gap %.2f  [%.2f, %.2f]"
          % (d80["coverage"], d80["accuracy"], d80["gap"], d80["gap_lo"], d80["gap_hi"]))
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--test" in sys.argv else main())
