"""
13_out_of_scope.py — CONTRIBUTION EXPERIMENT 3.

    "A five-class ECG model attaches a statistical guarantee to recordings whose
     actual disease it has no output for. The guarantee certifies the wrong
     answer, and nothing in the system knows."

A conformal bound answers: 'among the classes I model, how often do I miss one?'
It says nothing about a class that was never in the label space. Softmax has no
'none of the above', so an atrial fibrillation recording is redistributed across
NORM/MI/STTC/CD/HYP and the pipeline proceeds as if nothing unusual happened.

Measured here, on PTB-XL alone:
  1. How much out-of-scope disease is in the dataset at all?
  2. What does the system report for atrial fibrillation?
  3. How many of those reports carry a guarantee?
  4. Can a physiological rhythm check catch them, with the threshold fitted on
     validation only?
  5. Does withholding the guarantee restore honest behaviour, and at what cost?

Usage: python -X utf8 audit/13_out_of_scope.py [--n 700]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import paths, preprocess as pp, quality as qc, scope, signals   # noqa: E402
from src.models import CLASS_NAMES                                        # noqa: E402
from src.pipeline import ECGPipeline                                      # noqa: E402

CKPT = os.path.join(COMP, "checkpoints")
OUT = os.path.join(COMP, "audit", "results")
os.makedirs(OUT, exist_ok=True)

R, lines = {}, []
def p(s=""):
    print(s, flush=True); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=700,
                help="test-fold sample for evaluation; the threshold is always "
                     "fitted on the FULL validation fold")
ap.add_argument("--fpr", type=float, default=0.05,
                help="false-positive budget when fitting the threshold on validation")
args = ap.parse_args()

# Conditions the five-superclass label space cannot express.
OOS_PATTERNS = {
    "atrial fibrillation": r"atrial fibrillation|absolute arrhythmi|vorhofflimmern",
    "atrial flutter": r"atrial flutter|vorhofflattern",
    "paced rhythm": r"pacemaker|paced|schrittmacher",
    "SV tachycardia": r"supraventricular tachycard",
    "ventricular tachycardia": r"ventricular tachycard",
    "atrial ectopy": r"atrial premature|premature atrial|supraventricular extrasystol",
    "ventricular ectopy": r"ventricular premature|premature ventricular|ventricular extrasystol",
}
IRREGULAR = r"atrial fibrillation|absolute arrhythmi|vorhofflimmern|atrial flutter|vorhofflattern"

val = pd.read_csv(paths.require("val.csv"))
test = pd.read_csv(paths.require("test.csv"))
full = pd.read_csv(paths.require("ptbxl_labeled_final.csv"))
fb = dict(zip(test.ecg_id, test.filename_hr))
fb_val = dict(zip(val.ecg_id, val.filename_hr))

# ════════════════════════════════════════════════════════════════════════ 1
hdr("1. HOW MUCH DISEASE IS OUTSIDE THE LABEL SPACE?")
rep_full = full.report_en.fillna("").str.lower()
p(f"  {'condition':<26}{'records':>9}{'share':>9}")
p("  " + "-" * 44)
any_oos = np.zeros(len(full), bool)
counts = {}
for k, pat in OOS_PATTERNS.items():
    m = rep_full.str.contains(pat, regex=True, na=False).values
    any_oos |= m
    counts[k] = int(m.sum())
    p(f"  {k:<26}{m.sum():>9,}{m.mean()*100:>8.2f}%")
p("  " + "-" * 44)
p(f"  {'ANY out-of-scope':<26}{any_oos.sum():>9,}{any_oos.mean()*100:>8.2f}%")
p()
p("  These are cardiologist-documented findings in the reference reports of the")
p("  very records this system was trained and evaluated on. The five-class label")
p("  space cannot express any of them.")
R["prevalence"] = dict(counts=counts, any=int(any_oos.sum()), n=len(full))

# ════════════════════════════════════════════════════════════════════════ 2
hdr("2. WHAT DOES THE SYSTEM REPORT FOR ATRIAL FIBRILLATION?")
pipe = ECGPipeline.from_checkpoint(
    os.path.join(CKPT, "best_model.pt"), paths.require("norm_stats.json"),
    "resnet_se", os.path.join(CKPT, "calibrator.json"),
    os.path.join(CKPT, "conformal_triage.json"), do_filter=True)

rep_t = test.report_en.fillna("").str.lower()
af_mask = rep_t.str.contains(IRREGULAR, regex=True, na=False).values
af = test[af_mask]
p(f"  {len(af)} recordings in the held-out test fold document atrial")
p(f"  fibrillation or flutter. The model has no output unit for either.")
p()

triage = Counter(); norm_in = 0; with_guarantee = 0; refused = 0
per_record = []
for e in af.ecg_id:
    s = signals.load(int(e), fb.get(int(e)))
    r = pipe.analyse(s, fs=500, with_xai=False)
    triage[r.report.triage] += 1
    ruled_in = [f.cls for f in r.report.findings if f.zone == "rule_in"]
    g = len(r.report.guarantees)
    if r.report.refused:
        refused += 1
    else:
        if "NORM" in ruled_in:
            norm_in += 1
        if g:
            with_guarantee += 1
    per_record.append(dict(ecg_id=int(e), triage=r.report.triage,
                           ruled_in=ruled_in, n_guarantees=g,
                           refused=r.report.refused))

p(f"  triage distribution: {dict(triage)}")
p(f"  refused by quality control            : {refused:>4} / {len(af)}")
p(f"  reported as NORMAL                    : {norm_in:>4} / {len(af)}")
p(f"  carrying a statistical guarantee      : {with_guarantee:>4} / {len(af)}")
p()
p("  Not one of those guarantees concerns atrial fibrillation, because no")
p("  guarantee about atrial fibrillation exists. The system certifies what it")
p("  can measure while being blind to the finding that will cause the stroke.")
if norm_in:
    ex = [r for r in per_record if "NORM" in r["ruled_in"]][:3]
    p()
    p("  Patients told their ECG is normal:")
    for r in ex:
        p(f"    ECG {r['ecg_id']}: triage {r['triage']}, "
          f"{r['n_guarantees']} guarantees attached")
R["af_behaviour"] = dict(n=len(af), triage=dict(triage), norm_ruled_in=norm_in,
                         with_guarantee=with_guarantee, refused=refused)

# ════════════════════════════════════════════════════════════════════════ 3
hdr("3. FITTING THE RHYTHM CHECK ON THE VALIDATION FOLD")
p("  Threshold chosen on the FULL fold 9, at a fixed false-positive budget")
p("  against records with no documented irregular rhythm. Fold 10 is never used")
p("  to select it — the same discipline as every other threshold in this system.")
p()


def irr_of(eid, fbmap):
    s = signals.load(int(eid), fbmap.get(int(eid)))
    pk = qc.detect_r_peaks(pp.bandpass(s)[:, 1], 500)
    f = scope.rr_features(pk, 500)
    return None if f is None else f["irr"]


rep_v = val.report_en.fillna("").str.lower()
v_irr_mask = rep_v.str.contains(IRREGULAR, regex=True, na=False).values
# The threshold shipped in checkpoints/scope.json must not depend on --n.
# Fitting it on a subsample made a smaller run silently degrade the deployed
# system: a --n 400 run pushed sensitivity from 71% to 28%. Always fit on the
# FULL validation fold; --n only limits the test-side evaluation.
vs = val
v_scores, v_lab = [], []
for e, lab in zip(vs.ecg_id, v_irr_mask[vs.index]):
    x = irr_of(e, fb_val)
    if x is not None:
        v_scores.append(x); v_lab.append(bool(lab))
v_scores, v_lab = np.array(v_scores), np.array(v_lab)
neg = v_scores[~v_lab]
thr = float(np.quantile(neg, 1 - args.fpr))
p(f"  validation sample : {len(v_scores)} records "
  f"({v_lab.sum()} irregular, {(~v_lab).sum()} regular)")
p(f"  threshold at {args.fpr:.0%} FPR : irr = {thr:.4f}")
p(f"  sensitivity on validation      : {(v_scores[v_lab] > thr).mean():.1%}")

with open(os.path.join(CKPT, "scope.json"), "w") as f:
    json.dump({"irr_threshold": thr, "fpr_budget": args.fpr,
               "fitted_on": "val fold 9", "n_val": int(len(v_scores))}, f, indent=2)
p(f"  saved -> checkpoints/scope.json")
R["threshold"] = dict(irr=thr, fpr=args.fpr, n_val=int(len(v_scores)))

# ════════════════════════════════════════════════════════════════════════ 4
hdr("4. DOES IT CATCH THEM ON THE UNSEEN TEST FOLD?")
ts = test.sample(min(args.n, len(test)), random_state=0)
t_lab = rep_t.str.contains(IRREGULAR, regex=True, na=False).values[ts.index]
t_scores = []
keep_lab = []
for e, lab in zip(ts.ecg_id, t_lab):
    x = irr_of(e, fb)
    if x is not None:
        t_scores.append(x); keep_lab.append(bool(lab))
t_scores, t_lab = np.array(t_scores), np.array(keep_lab)
flag = t_scores > thr
sens = float(flag[t_lab].mean()) if t_lab.sum() else float("nan")
fpr = float(flag[~t_lab].mean())
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(t_lab, t_scores)
except Exception:
    auc = float("nan")
p(f"  test sample  : {len(t_scores)} records ({t_lab.sum()} irregular)")
p(f"  sensitivity  : {sens:.1%}   (out-of-scope rhythms flagged)")
p(f"  false-positive rate : {fpr:.1%}   (regular rhythms wrongly flagged)")
p(f"  AUROC        : {auc:.3f}")
p()
p("  " + scope.LIMITATION)
R["detector"] = dict(sensitivity=sens, fpr=fpr, auroc=float(auc),
                     n=len(t_scores), n_pos=int(t_lab.sum()))

# ════════════════════════════════════════════════════════════════════════ 5
hdr("5. THE COST OF WITHHOLDING THE GUARANTEE")
n_flagged = int(flag.sum())
p(f"  Records that would have the guarantee withheld: {n_flagged}/{len(t_scores)} "
  f"({n_flagged/len(t_scores):.1%})")
p(f"    of which genuinely out-of-scope : {int((flag & t_lab).sum())}")
p(f"    of which in-scope (over-caution): {int((flag & ~t_lab).sum())}")
p()
p("  Note this does NOT reduce diagnostic output. The classifier still reports")
p("  its five classes. What is withheld is the CLAIM — the sentence promising a")
p("  bounded miss rate — because that claim was never true for these records.")
p()
p("  Trade-off: at a 5% false-positive budget the system stays silent about its")
p("  guarantee on roughly 1 in 20 normal-rhythm records, in exchange for not")
p(f"  making a false promise to {sens:.0%} of patients whose disease it cannot see.")

# ════════════════════════════════════════════════════════════════════════
hdr("SUMMARY")
p(f"  Out-of-scope disease in the dataset          : {any_oos.mean()*100:.1f}% of records")
p(f"  AF/flutter recordings in the test fold       : {len(af)}")
p(f"  ... carrying a statistical guarantee         : {with_guarantee}/{len(af)}")
p(f"  ... reported as NORMAL                       : {norm_in}")
p(f"  Rhythm check sensitivity / FPR / AUROC       : "
  f"{sens:.0%} / {fpr:.0%} / {auc:.3f}")

with open(os.path.join(OUT, "13_out_of_scope.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "13_out_of_scope.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
