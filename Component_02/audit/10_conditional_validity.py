"""
10_conditional_validity.py — THE CONTRIBUTION EXPERIMENT.

Question
--------
A conformal guarantee is MARGINAL: it holds on average over the whole test
distribution. A clinician does not treat "the whole distribution" — they treat a
72-year-old woman whose ECG is noisy. Does the promised miss-rate bound still
hold for HER?

Two ways the marginal guarantee can be false in practice, both testable on
PTB-XL alone:

  A. SUBGROUP VALIDITY. Conformal controls risk marginally. Nothing stops it
     from concentrating the misses in one demographic. If the MI rule-out bound
     is 5% overall but 12% in women over 70, the system is unsafe for exactly
     the group with atypical presentations.

  B. GATE-INDUCED SHIFT. Our signal-quality gate refuses uninterpretable
     records at inference. The conformal thresholds were calibrated on UNGATED
     data. If quality is correlated with pathology — sick patients move, are in
     distress, have more artefact — then the gate removes a LABEL-DEPENDENT
     subset, exchangeability breaks, and the guarantee is void. The system's own
     safety mechanism silently voids its own guarantee.

Fix tested: Mondrian (group-conditional) conformal prediction — calibrate a
separate threshold per group so validity holds within each stratum by
construction (Vovk et al. 2003; Vovk 2012).

Usage: python -X utf8 Component_02/audit/10_conditional_validity.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import paths, quality as qc, signals                      # noqa: E402
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


# ── data ─────────────────────────────────────────────────────────────────
val = pd.read_csv(paths.find("val.csv"))
test = pd.read_csv(paths.find("test.csv"))
Lv = np.load(os.path.join(CKPT, "val_logits_seed0.npy"))
Lt = np.load(os.path.join(CKPT, "test_logits_seed0.npy"))
cal = TemperatureCalibrator.load(os.path.join(CKPT, "calibrator.json"))
Pv, Pt = cal.predict_proba(Lv), cal.predict_proba(Lt)
Yv = val[[f"label_{c}" for c in CLASS_NAMES]].values.astype(int)
Yt = test[[f"label_{c}" for c in CLASS_NAMES]].values.astype(int)
DELTA = 0.01

hdr("0. SETUP")
p(f"  calibration fold 9 : {len(val)}   test fold 10 : {len(test)}")
p(f"  PAC delta = {DELTA};  alpha = {DEFAULT_ALPHA}")


def fit_lambda(scores_pos, alpha):
    lo, ok, _ = _conformal_lower(np.asarray(scores_pos), alpha, DELTA)
    return lo, ok


def miss_rate(probs, y, k, lam):
    pos = y[:, k] == 1
    if pos.sum() == 0:
        return np.nan, 0
    return float(((probs[pos, k] < lam)).mean()), int(pos.sum())


# ═══════════════════════════════════════════════════════════════════════════
#  A. SUBGROUP VALIDITY OF THE MARGINAL GUARANTEE
# ═══════════════════════════════════════════════════════════════════════════
hdr("A. DOES THE MARGINAL GUARANTEE HOLD WITHIN PATIENT SUBGROUPS?")

def age_band(a):
    if pd.isna(a) or a >= 300:
        return "unknown"
    return "<50" if a < 50 else ("50-69" if a < 70 else ">=70")

for df in (val, test):
    df["_sex"] = df["sex"].map({0: "male", 1: "female"}).fillna("unknown")
    df["_age"] = df["age"].apply(age_band)

groups = ([("sex", g) for g in ["male", "female"]]
          + [("age", g) for g in ["<50", "50-69", ">=70"]])

p("  Thresholds are fitted MARGINALLY on all of fold 9 (the standard approach),")
p("  then the realised miss rate is measured inside each subgroup of fold 10.")
p()
p(f"  {'class':<6} {'alpha':>6} {'overall':>8} | " +
  " ".join(f"{g:>9}" for _, g in groups))
p("  " + "-" * 78)

violations = []
for k, c in enumerate(CLASS_NAMES):
    a = DEFAULT_ALPHA[c]
    lam, ok = fit_lambda(Pv[Yv[:, k] == 1, k], a)
    overall, _ = miss_rate(Pt, Yt, k, lam)
    row = []
    for col, g in groups:
        m = test[f"_{col}"].values == g
        r, n = miss_rate(Pt[m], Yt[m], k, lam)
        row.append((g, r, n))
        if not np.isnan(r) and r > a and n >= 15:
            violations.append(dict(cls=c, group=f"{col}={g}", alpha=a,
                                   miss=float(r), n=int(n),
                                   excess=float(r - a)))
    p(f"  {c:<6} {a:>6.2f} {overall:>8.3f} | " +
      " ".join(f"{r:>6.3f}({n:>3d})" if not np.isnan(r) else f"{'--':>11}"
               for _, r, n in row))

p()
p(f"  Subgroup violations of the promised bound (n>=15): {len(violations)}")
for v in sorted(violations, key=lambda d: -d["excess"]):
    p(f"    {v['cls']:<5} {v['group']:<14} promised <={v['alpha']:.2f}  "
      f"observed {v['miss']:.3f}  (n={v['n']} positives)  "
      f"excess {v['excess']:+.3f}")
R["subgroup_violations"] = violations

# ── Mondrian fix ─────────────────────────────────────────────────────────
p()
p("  MONDRIAN (group-conditional) CALIBRATION — one threshold per subgroup:")
p(f"  {'class':<6} {'group':<14} {'n_cal':>6} {'lambda':>8} {'miss':>7} {'held':>6}")
p("  " + "-" * 58)
mond = []
for k, c in enumerate(CLASS_NAMES):
    a = DEFAULT_ALPHA[c]
    for col, g in groups:
        cm = (val[f"_{col}"].values == g) & (Yv[:, k] == 1)
        tm = test[f"_{col}"].values == g
        if cm.sum() < 20:
            continue
        lam, ok = fit_lambda(Pv[cm, k], a)
        r, n = miss_rate(Pt[tm], Yt[tm], k, lam)
        if np.isnan(r) or n < 15:
            continue
        held = bool(r <= a)
        mond.append(dict(cls=c, group=f"{col}={g}", n_cal=int(cm.sum()),
                         lam=float(lam), miss=float(r), held=held))
        p(f"  {c:<6} {col+'='+g:<14} {int(cm.sum()):>6} {lam:>8.4f} "
          f"{r:>7.3f} {str(held):>6}")
R["mondrian"] = mond
held = sum(1 for m in mond if m["held"])
p()
p(f"  Mondrian held in {held}/{len(mond)} class-group cells "
  f"({held/max(len(mond),1)*100:.0f}%) vs marginal "
  f"{len(mond)-len(violations)}/{len(mond)}")
p("  NOTE: per-group calibration sets are small (see n_cal). Where n_cal is")
p("  below ~1/alpha the PAC bound is infeasible and the cell is skipped —")
p("  that data-hunger is itself a finding: subgroup validity is not free.")

# ═══════════════════════════════════════════════════════════════════════════
#  B. IS SIGNAL QUALITY LABEL-DEPENDENT?
# ═══════════════════════════════════════════════════════════════════════════
hdr("B. IS THE QUALITY GATE LABEL-DEPENDENT? (does it remove sick patients?)")
p("  If SQI is independent of diagnosis, gating is a random filter and")
p("  exchangeability survives. If sick patients have worse signals, the gate")
p("  removes a label-dependent subset and the conformal guarantee is void.")
p()

N = int(os.environ.get("N_SQI", "600"))
sub = test.sample(min(N, len(test)), random_state=0)
fb = dict(zip(test.ecg_id, test.filename_hr))
rows = []
for e in sub.ecg_id:
    s = signals.load(int(e), fb.get(int(e)))
    _, rep = qc.assess(s, 500)
    rows.append(dict(ecg_id=int(e), sqi=rep.sqi, accepted=rep.acceptable,
                     n_flat=len(rep.flat_leads), n_noisy=len(rep.noisy_leads)))
q = pd.DataFrame(rows).merge(test[["ecg_id"] + [f"label_{c}" for c in CLASS_NAMES]],
                             on="ecg_id")

p(f"  n = {len(q)} test records")
p(f"  {'class':<6} {'mean SQI (pos)':>15} {'mean SQI (neg)':>15} {'delta':>8} "
  f"{'noisy leads pos/neg':>21}")
p("  " + "-" * 70)
sqi_rows = []
for c in CLASS_NAMES:
    pos = q[q[f"label_{c}"] == 1]
    neg = q[q[f"label_{c}"] == 0]
    d = pos.sqi.mean() - neg.sqi.mean()
    sqi_rows.append(dict(cls=c, sqi_pos=float(pos.sqi.mean()),
                         sqi_neg=float(neg.sqi.mean()), delta=float(d),
                         noisy_pos=float(pos.n_noisy.mean()),
                         noisy_neg=float(neg.n_noisy.mean())))
    p(f"  {c:<6} {pos.sqi.mean():>15.4f} {neg.sqi.mean():>15.4f} {d:>+8.4f} "
      f"{pos.n_noisy.mean():>10.2f}/{neg.n_noisy.mean():<10.2f}")
R["sqi_by_class"] = sqi_rows

try:
    from scipy.stats import mannwhitneyu
    p()
    p("  Mann-Whitney U on SQI, positive vs negative:")
    for c in CLASS_NAMES:
        pos = q[q[f"label_{c}"] == 1].sqi.values
        neg = q[q[f"label_{c}"] == 0].sqi.values
        if len(pos) > 5 and len(neg) > 5 and (pos.std() + neg.std()) > 0:
            u, pv = mannwhitneyu(pos, neg, alternative="two-sided")
            p(f"    {c:<6} p = {pv:.4g}" + ("   <-- SIGNIFICANT" if pv < 0.05 else ""))
            R.setdefault("sqi_tests", {})[c] = float(pv)
except ImportError:
    pass

# ── gate-induced shift under realistic corruption ────────────────────────
p()
p("  Under realistic deployment corruption (one disconnected lead + an EMG")
p("  burst applied to a random 25% of records):")
rng = np.random.default_rng(0)
keep, corrupt_idx = [], set(rng.choice(len(sub), size=len(sub) // 4, replace=False))
for i, e in enumerate(sub.ecg_id):
    s = signals.load(int(e), fb.get(int(e))).copy()
    if i in corrupt_idx:
        s[:, rng.integers(0, 12)] = 0.0
        k0 = int(rng.integers(0, 4000))
        s[k0:k0 + 900] += rng.normal(0, 0.4, (900, 12)).astype(np.float32)
    _, rep = qc.assess(s, 500)
    keep.append(rep.acceptable)
keep = np.array(keep)
kept = sub[keep]
p(f"    accepted {keep.sum()}/{len(sub)} ({keep.mean()*100:.1f}%)")
p(f"    {'class':<6} {'prevalence all':>15} {'prevalence kept':>16} {'shift':>8}")
shift_rows = []
for c in CLASS_NAMES:
    a_, b_ = sub[f"label_{c}"].mean(), kept[f"label_{c}"].mean()
    shift_rows.append(dict(cls=c, all=float(a_), kept=float(b_), shift=float(b_ - a_)))
    p(f"    {c:<6} {a_:>15.4f} {b_:>16.4f} {b_-a_:>+8.4f}")
R["gate_shift"] = shift_rows
p()
p("  Any non-zero shift means the gated test set is NOT exchangeable with the")
p("  ungated calibration set. The fix is to calibrate THROUGH the same gate.")

# ═══════════════════════════════════════════════════════════════════════════
hdr("SUMMARY")
p(f"  Subgroup violations of the marginal bound : {len(violations)}")
p(f"  Mondrian cells that held                  : {held}/{len(mond)}")
p(f"  Classes whose SQI differs significantly   : "
  f"{sum(1 for v in R.get('sqi_tests', {}).values() if v < 0.05)}/5")
p(f"  Max label shift induced by the gate       : "
  f"{max(abs(r['shift']) for r in shift_rows):.4f}")

with open(os.path.join(OUT, "10_conditional_validity.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "10_conditional_validity.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
