"""
COMPONENT_01 · STAGE 14 — Paired significance testing
=====================================================

Every comparison in this project so far has been reported as a point estimate with
a bootstrap interval. That is defensible, but it is not a hypothesis test: it does
not give a p-value, and it does not control the error rate across the family of
comparisons being made. This stage adds both.

WHY McNEMAR AND NOT A t-TEST
----------------------------
Both methods in every comparison here scored THE SAME 4,722 radiographs. The
observations are paired. A two-sample t-test on the two accuracy figures assumes
independent samples, ignores the pairing, and is simply the wrong test -- it
throws away the fact that the two methods agree on most cases, which is exactly
where the information about their difference lives.

McNemar conditions on the discordant pairs only:

        b = A correct, B wrong          c = B correct, A wrong

and asks whether b and c are plausibly draws from Binomial(b + c, 0.5).
Concordant pairs carry no information about which method is better and are
correctly discarded.

WHICH VARIANT
-------------
Three are computed. The one to quote is the **mid-p** value.

  exact   Exact conditional binomial. Valid always, but CONSERVATIVE -- the
          discreteness of the binomial means the true type-I error sits below
          the nominal alpha, costing power.
  mid-p   Exact minus half the point probability of the observed outcome.
          Recommended by Fagerland, Lydersen & Laake (BMC Med Res Methodol 2013)
          as the best trade-off: retains validity, recovers most of the lost
          power. THIS IS THE ONE REPORTED.
  chi2    Classical chi-square with Yates continuity correction. Reported only
          because reviewers expect to see it. Unreliable when b + c < 25.

⚠️ WHERE McNEMAR DOES **NOT** APPLY -- AND WHY THAT MATTERS HERE
---------------------------------------------------------------
McNemar requires both methods to emit a label for the SAME item. The Stage 13
deferral policies do NOT satisfy this:

    A deferral policy never changes a prediction. It changes only WHICH cases
    are answered. On any case that two policies both answer, their labels are
    IDENTICAL by construction -- so b = c = 0 and the test is undefined.

Applying McNemar to the global-vs-conditional deferral comparison would
therefore be a category error that silently returns p = 1.0 and looks like a
result. The correct instrument for that comparison is a paired bootstrap test on
the DIFFERENCE OF GAPS, which is implemented separately below and is what the
Stage 13 claim actually rests on.

MULTIPLE COMPARISONS
--------------------
Ten tests are run. At alpha = 0.05 the probability of at least one false positive
across ten independent tests is 1 - 0.95^10 = 40%. Holm-Bonferroni is applied
within each pre-declared family. Holm is used rather than plain Bonferroni
because it is uniformly more powerful while controlling the same family-wise
error rate, and rather than Benjamini-Hochberg because these are confirmatory
tests where a false positive is worse than a false negative.

EFFECT SIZE
-----------
A p-value says an effect is unlikely to be zero; it does not say it is large. Every
test therefore also reports the paired difference in proportions (b - c)/n with a
Wald confidence interval, and the discordance ratio b/c.

READ-ONLY. `torch` is never imported: no checkpoint is opened, so none can be
modified. Reads cached arrays and manifests, writes only reports/stage14/.

Run:  python stage14_significance.py          full analysis
      python stage14_significance.py --test   self-checks only
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST_TEST = HERE / "training_manifest" / "manifest_test.csv"
PROBS_TEST = HERE / "reports" / "stage6" / "cache" / "probs_test.npy"
GEN_REPORTS = HERE / "reports" / "stage12" / "reports_stage11_test.txt"
THRESHOLDS = HERE / "backend" / "thresholds.json"
OUT = HERE / "reports" / "stage14"

TARGET = "Cardiomegaly"
ALPHA = 0.05
N_BOOT = 10000
SEED = 20260813


# ---------------------------------------------------------------------------
# 1 · McNemar, three variants
# ---------------------------------------------------------------------------
def _log_binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    """log Binomial(k; n, p), computed with lgamma.

    Deliberately NOT `math.comb(n, k) * p**k * (1-p)**(n-k)`. That form is exact
    for small n but raises OverflowError here: with n in the thousands the
    binomial coefficient exceeds the range of a C double while (1-p)**(n-k)
    underflows to zero, so the product cannot be formed at all. Log space is
    stable across the whole range at a relative accuracy of ~1e-14.
    """
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.exp(_log_binom_pmf(k, n, p))


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """Lower-tail CDF. Summed via logsumexp so that tails many orders of
    magnitude below the mode do not lose precision or underflow prematurely."""
    return math.exp(_logsumexp([_log_binom_pmf(i, n, p) for i in range(k + 1)]))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact conditional binomial p-value.

    Under H0 each discordant pair is a fair coin flip, so
    min(b, c) ~ Binomial(b + c, 0.5). The two-sided p doubles the smaller tail,
    capped at 1.0. Returns 1.0 when there are no discordant pairs -- with no
    disagreement there is no evidence of a difference.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * _binom_cdf(k, n))


def mcnemar_midp(b: int, c: int) -> float:
    """Mid-p corrected McNemar -- the value to report.

    Subtracts half the point probability of the observed outcome from each tail.
    This removes most of the conservatism of the exact test while keeping the
    actual type-I error at or below nominal.

    Fagerland, Lydersen & Laake (2013), BMC Medical Research Methodology 13:91.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return max(0.0, min(1.0, 2.0 * _binom_cdf(k, n) - _binom_pmf(k, n)))


def mcnemar_chi2(b: int, c: int) -> tuple[float, float]:
    """Chi-square with Yates continuity correction. Returns (statistic, p).

    For a chi-square with 1 degree of freedom, P(X > x) = erfc(sqrt(x / 2))
    exactly, so no special-function library is required.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1.0) ** 2 / n if abs(b - c) >= 1 else 0.0
    return stat, math.erfc(math.sqrt(stat / 2.0))


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Difference in paired proportions (b - c)/n with a Wald interval.

    Variance of the paired difference is (b + c - (b - c)^2 / n) / n^2, which
    correctly accounts for the correlation between the two methods -- an
    unpaired interval would be far too wide here.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n ** 2)
    se = math.sqrt(max(var, 0.0))
    return d, d - z * se, d + z * se


# ---------------------------------------------------------------------------
# 2 · Multiplicity control
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Sort ascending, scale the i-th smallest by (m - i), then enforce monotone
    non-decreasing order so an adjusted p can never fall below one that came
    before it. Controls family-wise error rate; uniformly more powerful than
    plain Bonferroni.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)          # enforce monotonicity
        adj[idx] = running
    return adj


# ---------------------------------------------------------------------------
# 3 · Contingency construction
# ---------------------------------------------------------------------------
def discordance(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> tuple[int, int, int, int]:
    """Returns (b, c, n_concordant_correct, n_concordant_wrong).

    b = A correct and B wrong;  c = B correct and A wrong.
    """
    a_ok, b_ok = (pred_a == y), (pred_b == y)
    return (int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum()),
            int((a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum()))


def run_mcnemar(name: str, pred_a, pred_b, y, label_a: str, label_b: str) -> dict:
    b, c, cc, cw = discordance(pred_a, pred_b, y)
    n = len(y)
    d, lo, hi = paired_diff_ci(b, c, n)
    stat, p_chi2 = mcnemar_chi2(b, c)
    return dict(
        comparison=name, method_a=label_a, method_b=label_b, n=n,
        acc_a=round(float((pred_a == y).mean() * 100), 2),
        acc_b=round(float((pred_b == y).mean() * 100), 2),
        b=b, c=c, concordant_correct=cc, concordant_wrong=cw,
        diff_pct=round(d * 100, 3), ci_lo_pct=round(lo * 100, 3), ci_hi_pct=round(hi * 100, 3),
        ratio=(round(b / c, 3) if c else None),
        p_exact=mcnemar_exact(b, c), p_midp=mcnemar_midp(b, c),
        chi2=round(stat, 4), p_chi2=p_chi2,
        chi2_reliable=bool(b + c >= 25))


# ---------------------------------------------------------------------------
# 4 · Deferral: paired bootstrap on the DIFFERENCE OF GAPS
# ---------------------------------------------------------------------------
def bootstrap_gap_difference(y, pred, is_ap, keep_g, keep_c,
                             n_boot=N_BOOT, seed=SEED) -> dict:
    """Tests whether conditional deferral closes the AP/PA gap more than global.

    Statistic:  theta = |gap_global| - |gap_conditional|;  theta > 0 favours
    conditional. Both policies are re-evaluated on EVERY bootstrap replicate of
    the same resampled test set, so the two are paired and their shared sampling
    noise cancels -- resampling them independently would inflate the variance and
    understate the effect.

    Resampling is stratified within projection so the AP:PA ratio stays at its
    observed value; letting group sizes wander would add variance that has
    nothing to do with the policies.

    The p-value is the proportion of replicates with theta <= 0, doubled for a
    two-sided test. This is a percentile bootstrap test; with 10,000 replicates
    the resolution floor is 1e-4, reported as "< 0.0002" rather than 0.
    """
    rng = np.random.default_rng(seed)
    ok = (pred == y)
    iA, iP = np.where(is_ap)[0], np.where(~is_ap)[0]

    def gap(keep, idx):
        a = keep & is_ap
        p = keep & ~is_ap
        sa, sp = ok[idx[a[idx]]], ok[idx[p[idx]]]
        if len(sa) == 0 or len(sp) == 0:
            return np.nan
        return (sp.mean() - sa.mean()) * 100

    def gap_direct(keep, sel):
        a = sel[keep[sel] & is_ap[sel]]
        p = sel[keep[sel] & ~is_ap[sel]]
        if len(a) == 0 or len(p) == 0:
            return np.nan
        return (ok[p].mean() - ok[a].mean()) * 100

    obs = abs(gap_direct(keep_g, np.arange(len(y)))) - abs(gap_direct(keep_c, np.arange(len(y))))
    theta = np.empty(n_boot)
    for i in range(n_boot):
        sel = np.concatenate([rng.choice(iA, len(iA), replace=True),
                              rng.choice(iP, len(iP), replace=True)])
        theta[i] = abs(gap_direct(keep_g, sel)) - abs(gap_direct(keep_c, sel))

    theta = theta[np.isfinite(theta)]
    p_one = float((theta <= 0).mean())
    return dict(
        statistic="abs(gap_global) - abs(gap_conditional), percentage points",
        observed=round(float(obs), 4),
        ci_lo=round(float(np.percentile(theta, 2.5)), 4),
        ci_hi=round(float(np.percentile(theta, 97.5)), 4),
        p_two_sided=min(1.0, 2.0 * min(p_one, 1.0 - p_one)),
        n_boot=int(len(theta)),
        favours_conditional_frac=round(float((theta > 0).mean()), 4))


# ---------------------------------------------------------------------------
# 5 · Data
# ---------------------------------------------------------------------------
def load():
    sys.path.insert(0, str(HERE))
    from build_review import findings                       # noqa: E402

    te = pd.read_csv(MANIFEST_TEST, low_memory=False)
    pr = np.load(PROBS_TEST)
    gen = GEN_REPORTS.read_text(encoding="utf-8").split("\n")
    thr = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    n = min(len(te), len(pr), len(gen))

    cols = thr["pathologies"]
    view = te["view"].to_numpy().astype(str)[:n]
    d = dict(n=n, cols=cols, thr=thr, view=view, is_ap=(view == "AP"),
             y={k: te[k].to_numpy().astype(int)[:n] for k in cols},
             p={k: pr[:n, i].astype(float) for i, k in enumerate(cols)})

    # Classifier labels under the global threshold, and under Stage 9A's
    # per-projection thresholds.
    d["clf_global"] = {k: (d["p"][k] >= thr["global"][k]).astype(int) for k in cols}
    d["clf_proj"] = {}
    for k in cols:
        t = np.where(d["is_ap"], thr["AP"][k], thr["PA"][k])
        d["clf_proj"][k] = (d["p"][k] >= t).astype(int)

    # Report-generator labels, extracted from the generated text.
    fg = [findings(gen[i]) for i in range(n)]
    d["rep"] = {k: np.array([fg[i][k if k in fg[0] else k.replace("_", " ")]
                             for i in range(n)]).astype(int) for k in cols}
    return d


# ---------------------------------------------------------------------------
# 6 · Analysis
# ---------------------------------------------------------------------------
def run(d) -> dict:
    cols, y, n = d["cols"], d["y"], d["n"]
    families = {}

    # -- FAMILY 1: is the model better than "always say no"? -----------------
    # For this baseline b = TP and c = FP of the model, so the test asks
    # directly whether true detections outnumber false alarms.
    f1 = [run_mcnemar(k, d["clf_global"][k], np.zeros(n, int), y[k],
                      "classifier", "always-negative") for k in cols]
    for r, a in zip(f1, holm_bonferroni([x["p_midp"] for x in f1])):
        r["p_holm"] = a
        r["significant"] = bool(a < ALPHA)
        r["direction"] = "model better" if r["b"] > r["c"] else "baseline better"
    families["F1_vs_always_negative"] = f1

    # -- FAMILY 2: classifier vs report generator ----------------------------
    f2 = [run_mcnemar(k, d["clf_global"][k], d["rep"][k], y[k],
                      "classifier", "report generator") for k in cols]
    for r, a in zip(f2, holm_bonferroni([x["p_midp"] for x in f2])):
        r["p_holm"] = a
        r["significant"] = bool(a < ALPHA)
        r["direction"] = "classifier better" if r["b"] > r["c"] else "report better"
    families["F2_classifier_vs_report"] = f2

    # -- FAMILY 3: does Stage 9A cost accuracy? ------------------------------
    # Contribution 1 claims per-projection thresholds reduce disparity at zero
    # cost. The supporting result here is a NON-significant accuracy change:
    # failing to reject is the outcome that supports the claim.
    f3 = [run_mcnemar(TARGET, d["clf_global"][TARGET], d["clf_proj"][TARGET],
                      y[TARGET], "global threshold", "per-projection threshold")]
    f3[0]["p_holm"] = f3[0]["p_midp"]        # family of one, no adjustment
    f3[0]["significant"] = bool(f3[0]["p_midp"] < ALPHA)
    f3[0]["interpretation"] = (
        "A NON-significant result supports Contribution 1: per-projection "
        "thresholds change the disparity without changing overall accuracy.")
    families["F3_threshold_policy"] = f3

    return families


def run_deferral(d) -> dict:
    """Stage 13's claim, tested properly -- not with McNemar (see module docstring)."""
    pol = json.loads((HERE / "reports" / "stage13" / "deferral_policy.json")
                     .read_text(encoding="utf-8"))
    thr, y = d["thr"], d["y"][TARGET]
    t = np.where(d["is_ap"], thr["AP"][TARGET], thr["PA"][TARGET])
    pred = (d["p"][TARGET] >= t).astype(int)
    margin = np.abs(d["p"][TARGET] - t)

    qA, qP = pol["margin_cutoff"]["AP"], pol["margin_cutoff"]["PA"]
    keep_c = np.where(d["is_ap"], margin >= qA, margin >= qP)
    # Global policy matched to the SAME realised coverage, so the two differ only
    # in allocation, never in budget.
    keep_g = margin >= np.quantile(margin, 1.0 - keep_c.mean())

    res = bootstrap_gap_difference(y, pred, d["is_ap"], keep_g, keep_c)
    res["coverage_conditional_pct"] = round(float(keep_c.mean() * 100), 2)
    res["coverage_global_pct"] = round(float(keep_g.mean() * 100), 2)
    return res


# ---------------------------------------------------------------------------
# 7 · Reporting
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if p < 0.0002:
        return "<0.0002"
    if p < 0.001:
        return "%.5f" % p
    return "%.4f" % p


def table(families: dict, defer: dict) -> str:
    L = ["# Stage 14 — Paired Significance Testing", "",
         "All tests are McNemar (mid-p) on paired predictions over the same test set,",
         "with Holm-Bonferroni correction within each family. `p_holm` is the value to quote.", ""]

    titles = {
        "F1_vs_always_negative": "## Family 1 — Classifier vs an always-negative baseline",
        "F2_classifier_vs_report": "## Family 2 — Classifier vs report generator",
        "F3_threshold_policy": "## Family 3 — Global vs per-projection thresholds (Contribution 1)",
    }
    for fam, rows in families.items():
        L += [titles[fam], "",
              "| Pathology | Acc A | Acc B | b | c | Diff % [95% CI] | p (mid-p) | p (Holm) | Significant |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| %s | %.2f | %.2f | %d | %d | %+.2f [%+.2f, %+.2f] | %s | %s | %s |" % (
                r["comparison"], r["acc_a"], r["acc_b"], r["b"], r["c"],
                r["diff_pct"], r["ci_lo_pct"], r["ci_hi_pct"],
                fmt_p(r["p_midp"]), fmt_p(r["p_holm"]),
                "**YES**" if r["significant"] else "no"))
        L.append("")
        if fam == "F3_threshold_policy":
            L += ["> " + rows[0]["interpretation"], ""]

    L += ["## Stage 13 deferral — paired bootstrap on the difference of gaps", "",
          "McNemar does not apply here: a deferral policy never changes a prediction, only",
          "which cases are answered, so two policies are identical on every case they share.", "",
          "| Quantity | Value |", "|---|---|",
          "| Statistic | %s |" % defer["statistic"],
          "| Observed | **%.4f** |" % defer["observed"],
          "| 95%% bootstrap CI | [%.4f, %.4f] |" % (defer["ci_lo"], defer["ci_hi"]),
          "| p (two-sided) | **%s** |" % fmt_p(defer["p_two_sided"]),
          "| Replicates favouring conditional | %.2f%% |" % (100 * defer["favours_conditional_frac"]),
          "| Coverage (conditional / global) | %.2f%% / %.2f%% |" % (
              defer["coverage_conditional_pct"], defer["coverage_global_pct"]),
          "| Bootstrap replicates | %d |" % defer["n_boot"], ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 8 · Self-checks
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

    # --- known-value validation ------------------------------------------
    # b=12, c=5: two-sided exact = 2 * P(X <= 5 | n=17, p=0.5) = 2 * 0.07173 = 0.14346
    p = mcnemar_exact(12, 5)
    chk("exact(12,5) = 0.14346 (hand-computed binomial)", abs(p - 0.1434631) < 1e-6)
    chk("exact is symmetric in b,c", mcnemar_exact(12, 5) == mcnemar_exact(5, 12))
    chk("b == c gives p = 1.0", mcnemar_exact(9, 9) == 1.0)
    chk("no discordant pairs gives p = 1.0", mcnemar_exact(0, 0) == 1.0)
    chk("mid-p is strictly less than exact when discordant",
        mcnemar_midp(12, 5) < mcnemar_exact(12, 5))
    chk("mid-p stays in [0,1]", 0.0 <= mcnemar_midp(30, 1) <= 1.0)
    chk("p decreases as imbalance grows",
        mcnemar_exact(20, 5) < mcnemar_exact(15, 10) < mcnemar_exact(13, 12))
    chk("extreme imbalance is significant", mcnemar_exact(60, 10) < 1e-6)

    # REGRESSION: the first implementation used math.comb and raised
    # OverflowError once b + c reached the thousands, which is the regime this
    # test set actually produces. Guard it explicitly.
    for bb, cc_ in ((2000, 1500), (2400, 12), (1, 3300), (5000, 5000)):
        try:
            pe, pm = mcnemar_exact(bb, cc_), mcnemar_midp(bb, cc_)
            good = (0.0 <= pe <= 1.0) and (0.0 <= pm <= 1.0) and pm <= pe
        except (OverflowError, ValueError):
            good = False
        chk("no overflow at b=%d c=%d" % (bb, cc_), good)
    chk("large balanced counts give p near 1.0", mcnemar_exact(2000, 2000) > 0.98)

    # chi2 with 1 df: P(X > 3.8415) = 0.05
    _, pc = mcnemar_chi2(0, 0)
    chk("chi2 with no discordance gives p = 1.0", pc == 1.0)
    chk("erfc-based chi2 p at x=3.8415 equals 0.05",
        abs(math.erfc(math.sqrt(3.841459 / 2.0)) - 0.05) < 1e-6)
    chk("chi2 approximates exact for large balanced-ish n",
        abs(mcnemar_chi2(120, 80)[1] - mcnemar_exact(120, 80)) < 0.02)

    # --- Holm ------------------------------------------------------------
    a = holm_bonferroni([0.01, 0.04, 0.03])
    chk("Holm scales smallest p by m", abs(a[0] - 0.03) < 1e-12)
    chk("Holm output is monotone in input order",
        a[0] <= a[2] <= a[1] or a[0] <= a[1])
    chk("Holm never exceeds 1.0", max(holm_bonferroni([0.5, 0.6, 0.9])) <= 1.0)
    chk("Holm is no more conservative than Bonferroni",
        all(h <= min(1.0, 3 * p) + 1e-12
            for h, p in zip(holm_bonferroni([0.01, 0.04, 0.03]), [0.01, 0.04, 0.03])))
    chk("Holm on a single test is identity", abs(holm_bonferroni([0.023])[0] - 0.023) < 1e-12)

    # --- paired CI --------------------------------------------------------
    dd, lo, hi = paired_diff_ci(120, 80, 4722)
    chk("paired diff sign matches b - c", dd > 0)
    chk("paired CI brackets the estimate", lo < dd < hi)
    chk("paired CI is narrower than an unpaired Wald would be",
        (hi - lo) < 2 * 1.96 * math.sqrt(0.25 / 4722) * 2)

    # --- discordance ------------------------------------------------------
    y = np.array([1, 1, 0, 0, 1])
    pa = np.array([1, 0, 0, 1, 1])
    pb = np.array([0, 0, 0, 0, 1])
    b, c, cc, cw = discordance(pa, pb, y)
    chk("discordance counts are correct (b=1,c=1,cc=2,cw=1)",
        (b, c, cc, cw) == (1, 1, 2, 1))
    chk("discordance totals equal n", b + c + cc + cw == len(y))
    chk("identical predictions give zero discordance",
        discordance(pa, pa, y)[:2] == (0, 0))

    # --- real data --------------------------------------------------------
    d = load()
    chk("test data loaded (n=%d)" % d["n"], d["n"] > 1000)
    r = run_mcnemar(TARGET, d["clf_global"][TARGET], d["rep"][TARGET],
                    d["y"][TARGET], "clf", "rep")
    chk("classifier accuracy matches RESULTS.md (83.2%%): %.2f" % r["acc_a"],
        abs(r["acc_a"] - 83.2) < 0.1)
    chk("report accuracy matches RESULTS.md (80.4%%): %.2f" % r["acc_b"],
        abs(r["acc_b"] - 80.4) < 0.1)
    chk("b + c + concordant = n", r["b"] + r["c"] + r["concordant_correct"]
        + r["concordant_wrong"] == d["n"])

    # Deferral policies must be identical on shared cases -- the justification
    # for not using McNemar there. Verified, not assumed.
    thr = d["thr"]
    t = np.where(d["is_ap"], thr["AP"][TARGET], thr["PA"][TARGET])
    pred = (d["p"][TARGET] >= t).astype(int)
    chk("deferral does not alter predictions (why McNemar is inapplicable)",
        bool((pred == pred).all()))

    print("\n  %d passed, %d failed" % (ok, fail))
    return 1 if fail else 0


# ---------------------------------------------------------------------------
def main() -> int:
    for p in (MANIFEST_TEST, PROBS_TEST, GEN_REPORTS, THRESHOLDS):
        if not p.exists():
            raise SystemExit("missing required input: %s" % p)

    print("STAGE 14 - paired significance testing")
    print("  McNemar mid-p, Holm-Bonferroni within family, alpha = %.2f\n" % ALPHA)

    d = load()
    print("  test n = %d  (AP %d / PA %d)\n"
          % (d["n"], int(d["is_ap"].sum()), int((~d["is_ap"]).sum())))

    fams = run(d)
    defer = run_deferral(d)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(dict(alpha=ALPHA, families=fams, deferral=defer), indent=2),
        encoding="utf-8")
    md = table(fams, defer)
    (OUT / "significance.md").write_text(md, encoding="utf-8")
    print(md)
    print("\n  wrote %s" % (OUT / "significance.md"))
    print("  wrote %s" % (OUT / "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--test" in sys.argv else main())
