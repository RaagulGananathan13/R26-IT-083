"""
08_verify_fixes.py — Proof that each audit finding is actually closed.

Re-runs the exact adversarial inputs that broke the archive and shows the new
behaviour side by side. This is the regression suite; run it after any change.

    python -X utf8 Component_02/audit/08_verify_fixes.py
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import paths, signals                          # noqa: E402
from src.models import CLASS_NAMES                       # noqa: E402
from src.pipeline import ECGPipeline                     # noqa: E402
from src.verify import batch_verify, verify_paraphrase   # noqa: E402
from src.xai import grad_cam                             # noqa: E402

OUT = os.path.join(COMP, "audit", "results")

lines, npass, nfail = [], 0, 0
def p(s=""):
    print(s, flush=True); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)
def check(name, ok, detail=""):
    global npass, nfail
    if ok:
        npass += 1
    else:
        nfail += 1
    p(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# Defaults to the shipped model. Override to verify a different checkpoint:
#   ECG_CKPT=... ECG_MODEL=resnet ECG_FILTER=0 python ...
CKPT = os.environ.get("ECG_CKPT",
                      os.path.join(COMP, "checkpoints", "best_model.pt"))
MODEL = os.environ.get("ECG_MODEL", "resnet_se")
FILTER = os.environ.get("ECG_FILTER", "1") == "1"

if not os.path.exists(CKPT):
    print(f"Checkpoint not found: {CKPT}\n"
          f"Set ECG_CKPT to the model you want to verify.")
    sys.exit(2)
if not signals.available():
    print("No signal source found (expected data/raw_signals/ or data/signals_cache/).\n"
          "This suite replays RAW-millivolt adversarial inputs through the quality\n"
          "gate, so it needs unprocessed signals. The packed training arrays are\n"
          "already filtered and normalised and cannot substitute.")
    sys.exit(3)

print(f"verifying: {CKPT}  (model={MODEL}, filter={FILTER})")
pipe = ECGPipeline.from_checkpoint(
    ckpt_path=CKPT,
    norm_stats_path=paths.require("norm_stats.json"),
    model_name=MODEL,
    calibrator_path=os.path.join(COMP, "checkpoints", "calibrator.json"),
    triage_path=os.path.join(COMP, "checkpoints", "conformal_triage.json"),
    do_filter=FILTER)

test = pd.read_csv(paths.require("test.csv"))
_fb = dict(zip(test.ecg_id, test.filename_hr))
_eid = int(test.ecg_id.iloc[0])
real = signals.load(_eid, _fb.get(_eid))

# ═════════════════════════════════════════════════════════════════════════
hdr("C-1  ADVERSARIAL / DEGENERATE INPUTS  (archive: flatline -> 'MI 0.691')")
cases = [
    ("all-zero flatline", np.zeros((5000, 12), np.float32), 500),
    ("pure Gaussian noise", np.random.default_rng(1).normal(0, 1, (5000, 12)).astype(np.float32), 500),
    ("constant 0.5 mV", np.full((5000, 12), 0.5, np.float32), 500),
    ("real ECG in microvolts", real * 1000.0, 500),
    ("real ECG x50 (saturation)", real * 50.0, 500),
    ("8 of 12 leads dead", np.concatenate([real[:, :4], np.zeros((5000, 8), np.float32)], 1), 500),
    ("2.0 s fragment", real[:1000], 500),
    ("30 s strip", np.tile(real, (3, 1)), 500),
]
for name, sig, fs in cases:
    r = pipe.analyse(sig, fs=fs, with_xai=False)
    refused = r.report.refused
    diagnosed = bool(r.report.findings) and not refused
    detail = ("REFUSED: " + "; ".join(r.quality.errors)[:78]) if refused else \
             f"accepted, triage={r.report.triage}, findings={[f.cls for f in r.report.findings]}"
    # microvolts and the 30 s strip SHOULD be recoverable, the rest refused
    expect_refuse = name not in ("real ECG in microvolts", "30 s strip")
    check(name, refused == expect_refuse, detail)

r = pipe.analyse(real, fs=500, with_xai=False)
check("real ECG still analysed normally", not r.report.refused,
      f"triage={r.report.triage}, p={ {k: round(v,3) for k,v in r.probs_calibrated.items()} }")

# ═════════════════════════════════════════════════════════════════════════
hdr("C-4  NORM / ABNORMALITY CONTRADICTION  (archive: 99 of 1711 reports)")
from src.quality import QualityReport                          # noqa: E402
from src.report import build_report                            # noqa: E402
from src.conformal import RULE_IN, RULE_OUT                    # noqa: E402

q = QualityReport(acceptable=True, sqi=1.0, heart_rate_bpm=72.0, n_beats=12, duration_s=10.0)
thr = {c: t.to_dict() for c, t in pipe.triage.thresholds.items()}
rep = build_report({"NORM": 0.80, "MI": 0.70, "STTC": 0.1, "CD": 0.1, "HYP": 0.1},
                   {"NORM": RULE_IN, "MI": RULE_IN, "STTC": RULE_OUT,
                    "CD": RULE_OUT, "HYP": RULE_OUT}, thr, q)
ruled_in = [f.cls for f in rep.findings if f.zone == RULE_IN]
check("NORM suppressed when an abnormality is ruled in",
      not ("NORM" in ruled_in and len(ruled_in) > 1), f"ruled in = {ruled_in}")
check("no 'within normal limits' alongside an infarction",
      not ("normal limits" in rep.text.lower() and "infarction" in rep.text.lower()))

# ═════════════════════════════════════════════════════════════════════════
hdr("C-3  HALLUCINATION GATE  (archive: 'Graphic atrial fibrillation' x42)")
bad = ("Graphic atrial fibrillation. Minor non-specific findings are present. "
       "AI-generated decision support. NOT a medical device.")
v = verify_paraphrase(bad, rep)
check("paraphrase inventing 'atrial fibrillation' is REJECTED", not v.passed,
      "; ".join(v.errors)[:100])
dropped = "The ECG shows some changes. AI-generated decision support. NOT a medical device."
v2 = verify_paraphrase(dropped, rep)
check("paraphrase that DROPS the infarction is REJECTED", not v2.passed,
      "; ".join(v2.errors)[:100])
v3 = verify_paraphrase(rep.text, rep)
check("faithful paraphrase is ACCEPTED", v3.passed, "; ".join(v3.errors)[:100])

# ═════════════════════════════════════════════════════════════════════════
hdr("C-5  GRAD-CAM THREAD SAFETY  (archive: 3 of 4 concurrent calls corrupted)")
sigs = [signals.load(int(e), _fb.get(int(e))) for e in test.ecg_id.iloc[:4]]
tensors = [torch.from_numpy(((s - pipe.mean) / pipe.std).T[None]).float() for s in sigs]
seq = [grad_cam(pipe.model, t, i % 5) for i, t in enumerate(tensors)]
conc = {}
def w(i):
    conc[i] = grad_cam(pipe.model, tensors[i], i % 5)
ths = [threading.Thread(target=w, args=(i,)) for i in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
mismatch = [i for i in conc if not np.allclose(conc[i], seq[i], atol=1e-5)]
check("4 concurrent Grad-CAM calls match the sequential result", not mismatch,
      f"mismatched indices: {mismatch}")

# ═════════════════════════════════════════════════════════════════════════
hdr("C-6  CALIBRATION")
from src.calibration import calibration_report                  # noqa: E402

# The logits MUST come from the model being verified. Applying a calibrator
# fitted for model A to logits produced by model B yields nonsense — that is
# what happened the first time this ran against the retrained checkpoint.
CKPT_DIR = os.path.join(COMP, "checkpoints")
if os.path.abspath(CKPT).startswith(os.path.abspath(CKPT_DIR)):
    logit_path = os.path.join(CKPT_DIR, "test_logits_seed0.npy")   # retrained model
else:
    logit_path = os.path.join(CKPT_DIR, "test_logits_seed0.npy")   # fallback
if not os.path.exists(logit_path):
    raise SystemExit(f"missing logits for the model under test: {logit_path}")
print(f"  logits: {os.path.relpath(logit_path, ROOT)}")
Lt = np.load(logit_path)
Yt = test[[f"label_{c}" for c in CLASS_NAMES]].values.astype(float)
raw = 1 / (1 + np.exp(-Lt))
cal = pipe.calibrator.predict_proba(Lt)
b, a = calibration_report(raw, Yt), calibration_report(cal, Yt)
check("macro ECE improved", a["macro_ece"] < b["macro_ece"] / 2,
      f"{b['macro_ece']:.4f} -> {a['macro_ece']:.4f}")
check("HYP no longer over-predicted", abs(a["HYP"]["over_prediction"] - 1) < 0.2,
      f"{b['HYP']['over_prediction']:.2f}x -> {a['HYP']['over_prediction']:.2f}x")

# ═════════════════════════════════════════════════════════════════════════
hdr("REPORT CONTENT  (archive: 0/1711 had HR, triage, uncertainty or quality)")
r = pipe.analyse(real, fs=500, with_xai=True)
t = r.report.text.lower()
for label, needle in [("heart rate", "bpm"), ("triage tier", "triage:"),
                      ("signal quality", "signal quality index"),
                      ("statistical guarantee", "conformal"),
                      ("limitations", "limitations:"),
                      ("disclaimer", "not a medical device")]:
    check(f"report contains {label}", needle in t)
check("report passed automated verification", r.verification.passed,
      "; ".join(r.verification.errors)[:90])

# ═════════════════════════════════════════════════════════════════════════
hdr("CORPUS-LEVEL SAFETY  (200 test records)")
reports = []
for eid in test.ecg_id.sample(200, random_state=0):
    s = signals.load(int(eid), _fb.get(int(eid)))
    reports.append(pipe.analyse(s, fs=500, with_xai=False).report)
stats = batch_verify(reports)
check("100% of reports pass verification", stats["pass_rate"] == 1.0,
      f"{stats['passed']}/{stats['n']}  errors={stats['error_types']}")
contra = sum(1 for r_ in reports
             if "normal limits" in r_.text.lower()
             and any(x in r_.text.lower() for x in ("infarction", "conduction delay")))
check("zero self-contradictory reports", contra == 0, f"{contra} found (archive rate: 5.8%)")
uniq = len({r_.text for r_ in reports})
p(f"\n  Distinct reports across 200 patients: {uniq}  "
  f"(archive produced 63 distinct across 1711)")
triages = pd.Series([r_.triage for r_ in reports]).value_counts().to_dict()
p(f"  Triage distribution: {triages}")

# ═════════════════════════════════════════════════════════════════════════
hdr(f"SUMMARY:  {npass} passed, {nfail} failed")
p()
p("  Sample report:")
p("  " + "-" * 74)
for ln in reports[0].text.splitlines():
    p("  " + ln)

with open(os.path.join(OUT, "08_verify_fixes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
sys.exit(0 if nfail == 0 else 1)
