"""
06_leakage_audit.py — Target-leakage proof for the multi-modal fusion model
(_archive/checkpoints/best_model.pt, trained by _archive/training/train.py).

The fusion model consumes a ClinicalBERT [CLS] embedding of `report_en` —
the cardiologist's own free-text report. The 5 superclass labels are derived
from the SCP codes that were assigned to that same report. The text input
therefore contains the answer.

This script proves it by ablation on the SAME test set:
  A. full model            (signal + demographics + report text)
  B. text embedding zeroed (signal + demographics only)
  C. signal zeroed         (demographics + report text only)   <-- the tell
  D. signal AND demographics zeroed (report text only)

If C/D stay near A, the network is reading the report, not the ECG.
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = os.path.join(ROOT, "_archive")
DATA = os.path.join(ARCH, "data")
CACHE = os.path.join(DATA, "signals_cache")
TCACHE = os.path.join(DATA, "text_cache")
CKPT = os.path.join(ARCH, "checkpoints", "best_model.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

R, lines = {}, []
def p(s=""):
    print(s, flush=True); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


class ECGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(12, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(128, 192, 5, padding=2), nn.BatchNorm1d(192), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(192, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.conv_blocks(x).squeeze(-1)


class MultiModalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_encoder = ECGEncoder()
        self.demo_encoder = nn.Module()
        self.demo_encoder.mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.3),
                                              nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3))
        self.text_projector = nn.Module()
        self.text_projector.projector = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128),
                                                      nn.ReLU(), nn.Dropout(0.3))
        self.fusion = nn.Sequential(nn.Linear(448, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, sig, demo, txt):
        return self.fusion(torch.cat([self.ecg_encoder(sig),
                                      self.demo_encoder.mlp(demo),
                                      self.text_projector.projector(txt)], dim=1))


ns = json.load(open(os.path.join(DATA, "norm_stats.json")))
mu = np.array(ns["signal_mean"], np.float32); sd = np.array(ns["signal_std"], np.float32)
d_mu = np.array([ns["demographics"]["age"]["mean"], 0.5,
                 ns["demographics"]["height"]["mean"], ns["demographics"]["weight"]["mean"],
                 0.0, 0.0], np.float32)
d_sd = np.array([ns["demographics"]["age"]["std"], 0.5,
                 ns["demographics"]["height"]["std"], ns["demographics"]["weight"]["std"],
                 1.0, 1.0], np.float32)

state = torch.load(CKPT, map_location="cpu", weights_only=False)
model = MultiModalFusionModel()
model.load_state_dict(state["model_state"])
model.eval()

hdr("0. FUSION CHECKPOINT")
p(f"  epoch {state['epoch']+1}   best val macro-AUROC {state['best_auroc']:.4f}")
p(f"  stored thresholds: {[round(float(t),4) for t in state['optimal_thresholds']]}")
p(f"  INPUTS: 12-lead signal  +  6 demographics  +  ClinicalBERT([CLS], report_en)")
p(f"  The 5 labels were derived from the SCP codes attached to that same report_en.")
thr = list(state["optimal_thresholds"])

test = pd.read_csv(os.path.join(DATA, "test.csv"))
demo_cols = ["age", "sex", "height", "weight", "height_missing", "weight_missing"]


def infer(zero_sig=False, zero_txt=False, zero_demo=False, batch=32):
    P, Y = [], []
    sb, db, tb, yb = [], [], [], []
    rows = list(test.itertuples())
    for i, r in enumerate(rows):
        s = np.load(os.path.join(CACHE, f"{int(r.ecg_id)}.npy")).astype(np.float32)
        s = ((s - mu) / sd).T
        d = np.array([getattr(r, c) for c in demo_cols], np.float32)
        d = (d - d_mu) / d_sd
        t = np.load(os.path.join(TCACHE, f"{int(r.ecg_id)}.npy")).astype(np.float32)
        if zero_sig:  s = np.zeros_like(s)
        if zero_demo: d = np.zeros_like(d)
        if zero_txt:  t = np.zeros_like(t)
        sb.append(s); db.append(d); tb.append(t)
        yb.append([getattr(r, f"label_{c}") for c in CLASSES])
        if len(sb) == batch or i == len(rows) - 1:
            with torch.no_grad():
                P.append(torch.sigmoid(model(torch.from_numpy(np.stack(sb)),
                                             torch.from_numpy(np.stack(db)),
                                             torch.from_numpy(np.stack(tb)))).numpy())
            Y += yb; sb, db, tb, yb = [], [], [], []
    return np.concatenate(P), np.array(Y, float)


def score(P, Y):
    a = [roc_auc_score(Y[:, i], P[:, i]) for i in range(5)]
    f = [f1_score(Y[:, i], (P[:, i] >= thr[i]).astype(int), zero_division=0) for i in range(5)]
    return a, f, float(np.mean(a)), float(np.mean(f))


hdr("1. ABLATION ON THE TEST SET (1711 records)")
configs = [
    ("A  full (signal+demo+TEXT)", dict()),
    ("B  TEXT zeroed             ", dict(zero_txt=True)),
    ("C  SIGNAL zeroed           ", dict(zero_sig=True)),
    ("D  signal+demo zeroed      ", dict(zero_sig=True, zero_demo=True)),
]
res = {}
p(f"  {'config':<28} " + " ".join(f"{c:>7s}" for c in CLASSES) + f" {'mAUROC':>8} {'mF1':>7}")
p("  " + "-" * 84)
for name, kw in configs:
    P, Y = infer(**kw)
    a, f, ma, mf = score(P, Y)
    res[name.strip()] = dict(auroc=a, f1=f, macro_auroc=ma, macro_f1=mf)
    p(f"  {name:<28} " + " ".join(f"{x:>7.4f}" for x in a) + f" {ma:>8.4f} {mf:>7.4f}")
R["ablation"] = res

A = res["A  full (signal+demo+TEXT)"]
B = res["B  TEXT zeroed"]
C = res["C  SIGNAL zeroed"]
D = res["D  signal+demo zeroed"]

hdr("2. VERDICT")
p(f"  Full model macro-AUROC                    : {A['macro_auroc']:.4f}")
p(f"  Remove the ECG, keep the report text      : {C['macro_auroc']:.4f}  "
  f"({C['macro_auroc']/A['macro_auroc']*100:.1f}% of full)")
p(f"  Remove the report text, keep the ECG      : {B['macro_auroc']:.4f}  "
  f"({B['macro_auroc']/A['macro_auroc']*100:.1f}% of full)")
p(f"  Report text ALONE                         : {D['macro_auroc']:.4f}")
p()
p(f"  Drop caused by removing the ECG signal    : {A['macro_auroc']-C['macro_auroc']:+.4f}")
p(f"  Drop caused by removing the report text   : {A['macro_auroc']-B['macro_auroc']:+.4f}")
p()
if C["macro_auroc"] > B["macro_auroc"]:
    p("  >>> The report text carries MORE predictive signal than the entire 12-lead ECG.")
    p("  >>> This is TARGET LEAKAGE, not multi-modal fusion. The 0.9567 macro-AUROC /")
    p("      0.7733 macro-F1 in checkpoints/test_results.json is NOT an ECG result and")
    p("      must not be presented as one. At inference time on a new patient the")
    p("      cardiologist report does not exist — the input is unavailable by construction.")
else:
    p("  >>> Text contributes less than the signal; leakage is present but not dominant.")
R["verdict"] = dict(full=A["macro_auroc"], no_signal=C["macro_auroc"],
                    no_text=B["macro_auroc"], text_only=D["macro_auroc"])

hdr("3. WHICH MODEL IS ACTUALLY DEPLOYED?")
p("  app.py  -> checkpoints_ecg_only/best_model.pt   (ECG only, NO leakage)  GOOD")
p("  README  -> quotes the ECG-only numbers (macro-F1 0.717)                 GOOD")
p("  checkpoints/test_results.json (fusion, leaked) is still in the repo and")
p("  DOCUMENTATION/Project_Complete_Overview.txt calls fusion 'NOVELTY 2'.")
p("  Any thesis claim built on the fusion model must be withdrawn or re-run")
p("  without report_en.")

with open(os.path.join(OUT, "06_leakage_audit.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "06_leakage_audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
