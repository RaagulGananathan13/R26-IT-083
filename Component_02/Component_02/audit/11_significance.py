"""
11_significance.py — Is the subgroup violation real, or is it noise?

This closes the one hole in the contribution. A panel member will ask:

    "33.3% of 66 patients. How do you know that isn't chance?"

Three independent answers, each stronger than the last:

  A. WILSON CONFIDENCE INTERVALS on every subgroup miss rate. If the whole
     interval sits above the promised alpha, the bound is violated regardless of
     sampling luck.

  B. EXACT ONE-SIDED BINOMIAL TEST of H0: "the true miss rate is <= alpha".
     With HOLM correction, because 23 class-subgroup cells are tested at once and
     an uncorrected p-value across 23 tests is not evidence.

  C. CALIBRATION-DRAW BOOTSTRAP. The threshold itself depends on which patients
     landed in fold 9. Resample the calibration set 2000 times, refit the
     conformal threshold each time, and re-measure the subgroup miss rate on the
     untouched fold 10. This answers "is the violation an artefact of one
     particular calibration draw?" — the conformal analogue of a multi-seed run,
     and the correct test for this layer.

Usage: python -X utf8 audit/11_significance.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import paths                                              # noqa: E402
from src.calibration import TemperatureCalibrator                  # noqa: E402
from src.conformal import _conformal_lower, DEFAULT_ALPHA          # noqa: E402
from src.models import CLASS_NAMES                                 # noqa: E402

CKPT = os.path.join(COMP, "checkpoints")
OUT = os.path.join(COMP, "audit", "results")
os.makedirs(OUT, exist_ok=True)

R, lines = {}, []
def p(s=""):
    print(s, flush=True); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--delta", type=float, default=0.01)
args = ap.parse_args()

val = pd.read_csv(paths.require("val.csv"))
test = pd.read_csv(paths.require("test.csv"))
Yv = val[[f"label_{c}" for c in CLASS_NAMES]].values.astype(int)
Yt = test[[f"label_{c}" for c in CLASS_NAMES]].values.astype(int)
cal = TemperatureCalibrator.load(os.path.join(CKPT, "calibrator.json"))
Pv = cal.predict_proba(np.load(os.path.join(CKPT, "val_logits_seed0.npy")))
Pt = cal.predict_proba(np.load(os.path.join(CKPT, "test_logits_seed0.npy")))


def age_band(a):
    if pd.isna(a) or a >= 300:
        return "unknown"
    return "<50" if a < 50 else ("50-69" if a < 70 else ">=70")


for df in (val, test):
    df["_sex"] = df["sex"].map({0: "male", 1: "female"}).fillna("unknown")
    df["_age"] = df["age"].apply(age_band)

GROUPS = [("sex", g) for g in ("male", "female")] + \
         [("age", g) for g in ("<50", "50-69", ">=70")]


def wilson(k, n, z=1.96):
    """Wilson score interval — correct for small n, unlike the normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def binom_sf(k, n, p0):
    """One-sided exact P(X >= k | Binomial(n, p0))."""
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p0))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj


hdr("0. SETUP")
p(f"  seed 0 · PAC delta = {args.delta} · {args.boot} bootstrap resamples")
p(f"  calibration fold 9 (n={len(val)})   test fold 10 (n={len(test)})")
p(f"  Thresholds fitted MARGINALLY, as standard practice; subgroup miss rates")
p(f"  then measured on the untouched test fold.")

# ═══════════════════════════════════════════════════════════════════════════
hdr("A + B. CONFIDENCE INTERVALS AND EXACT TESTS")

cells = []
for k, c in enumerate(CLASS_NAMES):
    a = DEFAULT_ALPHA[c]
    lam, ok, _ = _conformal_lower(Pv[Yv[:, k] == 1, k], a, args.delta)
    for col, g in GROUPS:
        m = test[f"_{col}"].values == g
        pos = m & (Yt[:, k] == 1)
        n = int(pos.sum())
        if n < 15:
            continue
        miss = int(((Pt[pos, k] < lam)).sum())
        rate = miss / n
        lo, hi = wilson(miss, n)
        pv = binom_sf(miss, n, a)
        cells.append(dict(cls=c, group=f"{col}={g}", alpha=a, n=n, miss=miss,
                          rate=rate, ci_lo=lo, ci_hi=hi, p=pv))

adj = holm(np.array([x["p"] for x in cells]))
for x, q in zip(cells, adj):
    x["p_holm"] = float(q)

p(f"  {'class':<6}{'group':<14}{'alpha':>6}{'miss/n':>10}{'rate':>7}"
  f"{'95% CI':>18}{'p':>10}{'p(Holm)':>10}  verdict")
p("  " + "-" * 96)
sig = []
for x in sorted(cells, key=lambda d: d["p_holm"]):
    strict = x["ci_lo"] > x["alpha"]           # entire CI above the bound
    ok_sig = x["p_holm"] < 0.05
    verdict = ("VIOLATED (CI clears bound)" if strict and ok_sig else
               "violated (p<0.05)" if ok_sig else
               "above bound, n.s." if x["rate"] > x["alpha"] else "within bound")
    if ok_sig:
        sig.append(x)
    p(f"  {x['cls']:<6}{x['group']:<14}{x['alpha']:>6.2f}"
      f"{str(x['miss'])+'/'+str(x['n']):>10}{x['rate']:>7.3f}"
      f"{'['+format(x['ci_lo'],'.3f')+', '+format(x['ci_hi'],'.3f')+']':>18}"
      f"{x['p']:>10.2e}{x['p_holm']:>10.2e}  {verdict}")

R["cells"] = cells
p()
p(f"  Cells tested                                : {len(cells)}")
p(f"  Statistically significant after Holm (<0.05): {len(sig)}")
for x in sig:
    p(f"    {x['cls']} / {x['group']}: promised <= {x['alpha']:.2f}, "
      f"observed {x['rate']:.3f} (95% CI {x['ci_lo']:.3f}-{x['ci_hi']:.3f}), "
      f"p_Holm = {x['p_holm']:.2e}")
if not sig:
    p("    none — the observed excesses are within sampling noise.")

# ═══════════════════════════════════════════════════════════════════════════
hdr("C. CALIBRATION-DRAW BOOTSTRAP")
p("  The threshold depends on WHICH patients landed in fold 9. Resample the")
p("  calibration set, refit the conformal threshold, re-measure on fold 10.")
p("  If the violation survives, it is not an artefact of one calibration draw.")
p()

rng = np.random.default_rng(0)
focus = [x for x in cells if x["p_holm"] < 0.05] or \
        sorted(cells, key=lambda d: -(d["rate"] - d["alpha"]))[:3]

p(f"  {'class':<6}{'group':<14}{'alpha':>6}{'median miss':>13}"
  f"{'2.5%':>8}{'97.5%':>8}{'P(violate)':>12}")
p("  " + "-" * 68)
boot_out = []
for x in focus:
    k = CLASS_NAMES.index(x["cls"])
    col, g = x["group"].split("=", 1)   # ">=70" contains a second '='
    tm = (test[f"_{col}"].values == g) & (Yt[:, k] == 1)
    cal_scores = Pv[Yv[:, k] == 1, k]
    tgt = Pt[tm, k]
    rates = []
    for _ in range(args.boot):
        rs = rng.choice(cal_scores, size=len(cal_scores), replace=True)
        lam, ok, _ = _conformal_lower(rs, x["alpha"], args.delta)
        if not np.isfinite(lam):
            continue
        rates.append(float((tgt < lam).mean()))
    rates = np.array(rates)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    pviol = float((rates > x["alpha"]).mean())
    boot_out.append(dict(cls=x["cls"], group=x["group"], alpha=x["alpha"],
                         median=float(np.median(rates)), lo=float(lo),
                         hi=float(hi), p_violate=pviol))
    p(f"  {x['cls']:<6}{x['group']:<14}{x['alpha']:>6.2f}"
      f"{np.median(rates):>13.3f}{lo:>8.3f}{hi:>8.3f}{pviol:>11.1%}")
R["bootstrap"] = boot_out

p()
p("  P(violate) is the fraction of calibration draws in which the promised bound")
p("  is exceeded. A value near 1.0 means the violation is structural, not luck.")

# ═══════════════════════════════════════════════════════════════════════════
hdr("VERDICT")
strong = [x for x in cells if x["p_holm"] < 0.05 and x["ci_lo"] > x["alpha"]]
if strong:
    p(f"  {len(strong)} subgroup violation(s) survive BOTH a Holm-corrected exact")
    p(f"  test and a Wilson interval that clears the promised bound:")
    for x in strong:
        b = next((b for b in boot_out if b["cls"] == x["cls"]
                  and b["group"] == x["group"]), None)
        p()
        p(f"    {x['cls']} in {x['group']}")
        p(f"      promised    <= {x['alpha']:.0%} missed")
        p(f"      observed       {x['rate']:.1%}  ({x['miss']}/{x['n']} positives)")
        p(f"      95% CI         [{x['ci_lo']:.1%}, {x['ci_hi']:.1%}]  "
          f"-- entirely above the bound")
        p(f"      exact p        {x['p']:.2e}   Holm-adjusted {x['p_holm']:.2e}")
        if b:
            p(f"      bootstrap      violated in {b['p_violate']:.0%} of "
              f"{args.boot} calibration draws")
else:
    p("  No subgroup violation survives correction for multiple testing.")
    p("  Report the observed excesses as suggestive, not established.")

R["summary"] = dict(n_cells=len(cells), n_significant=len(sig),
                    n_strong=len(strong), boot=args.boot)

with open(os.path.join(OUT, "11_significance.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "11_significance.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
