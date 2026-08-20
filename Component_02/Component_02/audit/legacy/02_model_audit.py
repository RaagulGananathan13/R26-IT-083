"""
02_model_audit.py — Independent re-evaluation of the shipped ECG-only ResNet.

Does NOT trust checkpoints_ecg_only/test_results.json. Re-runs inference on
test.csv with the shipped best_model.pt and measures:

  1. AUROC / AUPRC / F1 at 0.5 and at the checkpoint's stored thresholds
  2. Threshold optimism: refit thresholds ON TEST and quantify the gap
  3. Bootstrap 95% CIs (1000 resamples) on macro-F1 and per-class F1
  4. Calibration: Brier score + Expected Calibration Error (10 bins)
  5. Determinism / BatchNorm sensitivity: batch=32 vs batch=1 inference
     (the Flask app serves ONE record at a time — verify it matches)
  6. Sanity: does an all-zero signal produce a confident diagnosis?
  7. Patient-leakage impact: metrics on the subset of test patients that
     also occur in train, vs the clean subset
  8. Report-layer contradiction rate: NORM + abnormality in the same report

Usage: python -X utf8 02_model_audit.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    precision_recall_curve, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = os.path.join(ROOT, "_archive")
DATA = os.path.join(ARCH, "data")
CACHE = os.path.join(DATA, "signals_cache")
CKPT = os.path.join(ARCH, "checkpoints_ecg_only", "best_model.pt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ARCH)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CNN_CHANNELS = [64, 128, 192, 256]
CNN_KERNELS = [15, 7, 5, 3]

R, lines = {}, []
def p(s=""):
    print(s); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, dropout=0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                                      nn.BatchNorm1d(out_ch))

    def forward(self, x):
        r = self.skip(x)
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.dropout(o)
        o = self.bn2(self.conv2(o))
        return F.relu(o + r)


class ECGResNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, in_ch = [], 12
        for i, (oc, ks) in enumerate(zip(CNN_CHANNELS, CNN_KERNELS)):
            blocks.append(ResidualBlock(in_ch, oc, ks, stride=2,
                                        dropout=0.1 if i < 2 else 0.2))
            in_ch = oc
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, x):
        return self.classifier(self.pool(self.backbone(x)).squeeze(-1))


# ── load ────────────────────────────────────────────────────────────────
ns = json.load(open(os.path.join(DATA, "norm_stats.json")))
sig_mean = np.array(ns["signal_mean"], dtype=np.float32)
sig_std = np.array(ns["signal_std"], dtype=np.float32)

state = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ECGResNet()
model.load_state_dict(state["model_state"])
model.eval()
stored_thr = list(state["optimal_thresholds"])

hdr("0. CHECKPOINT PROVENANCE")
p(f"  checkpoint keys      : {list(state.keys())}")
p(f"  best epoch           : {state['epoch'] + 1}")
p(f"  best val macro-AUROC : {state['best_auroc']:.4f}")
p(f"  stored thresholds    : {[round(t,4) for t in stored_thr]}")
p(f"  params               : {sum(p_.numel() for p_ in model.parameters()):,}")
R["ckpt"] = dict(epoch=int(state["epoch"]) + 1, val_auroc=float(state["best_auroc"]),
                 thresholds=[float(t) for t in stored_thr],
                 params=int(sum(p_.numel() for p_ in model.parameters())))

test = pd.read_csv(os.path.join(DATA, "test.csv"))
val = pd.read_csv(os.path.join(DATA, "val.csv"))
train = pd.read_csv(os.path.join(DATA, "train.csv"))


def infer(df, batch=32):
    """Batched inference, exactly as the training script evaluated."""
    probs, labels, ids = [], [], []
    buf = []
    rows = list(df.itertuples())
    for i, row in enumerate(rows):
        s = np.load(os.path.join(CACHE, f"{int(row.ecg_id)}.npy")).astype(np.float32)
        s = ((s - sig_mean) / sig_std).T
        buf.append(s)
        ids.append(int(row.ecg_id))
        labels.append([getattr(row, f"label_{c}") for c in CLASSES])
        if len(buf) == batch or i == len(rows) - 1:
            with torch.no_grad():
                t = torch.from_numpy(np.stack(buf))
                probs.append(torch.sigmoid(model(t)).numpy())
            buf = []
    return np.concatenate(probs), np.array(labels, dtype=float), np.array(ids)


p("\n  Running inference on test set (batch=32) ...")
pt, yt, idt = infer(test, 32)
p("  Running inference on val set (batch=32) ...")
pv, yv, idv = infer(val, 32)


def metrics(probs, y, thr):
    out, f1s, aucs, aps = {}, [], [], []
    for i, c in enumerate(CLASSES):
        auc = roc_auc_score(y[:, i], probs[:, i])
        ap = average_precision_score(y[:, i], probs[:, i])
        f1 = f1_score(y[:, i], (probs[:, i] >= thr[i]).astype(int), zero_division=0)
        out[c] = dict(auroc=auc, auprc=ap, f1=f1, prevalence=float(y[:, i].mean()))
        aucs.append(auc); aps.append(ap); f1s.append(f1)
    out["macro_auroc"] = float(np.mean(aucs))
    out["macro_auprc"] = float(np.mean(aps))
    out["macro_f1"] = float(np.mean(f1s))
    return out


def fit_thr(probs, y):
    thr = []
    for i in range(5):
        pr, rc, t = precision_recall_curve(y[:, i], probs[:, i])
        f = 2 * pr * rc / (pr + rc + 1e-8)
        b = int(np.argmax(f))
        thr.append(float(t[b]) if b < len(t) else 0.5)
    return thr


hdr("1. REPRODUCTION OF PUBLISHED TEST METRICS")
m05 = metrics(pt, yt, [0.5] * 5)
mstored = metrics(pt, yt, stored_thr)
claimed = json.load(open(os.path.join(ARCH, "checkpoints_ecg_only", "test_results.json")))

p(f"  {'Class':<6} {'prev':>6} {'AUROC':>8} {'AUPRC':>8} {'F1@0.5':>8} {'F1@thr':>8} "
  f"{'claimedF1':>10} {'delta':>7}")
p("  " + "-" * 72)
for c in CLASSES:
    cl = claimed["test_optimized"][c]["f1"]
    p(f"  {c:<6} {mstored[c]['prevalence']*100:>5.1f}% {mstored[c]['auroc']:>8.4f} "
      f"{mstored[c]['auprc']:>8.4f} {m05[c]['f1']:>8.4f} {mstored[c]['f1']:>8.4f} "
      f"{cl:>10.4f} {mstored[c]['f1']-cl:>+7.4f}")
p("  " + "-" * 72)
p(f"  {'MACRO':<6} {'':>6} {mstored['macro_auroc']:>8.4f} {mstored['macro_auprc']:>8.4f} "
  f"{m05['macro_f1']:>8.4f} {mstored['macro_f1']:>8.4f} "
  f"{claimed['test_optimized']['macro_f1']:>10.4f} "
  f"{mstored['macro_f1']-claimed['test_optimized']['macro_f1']:>+7.4f}")
p()
p(f"  NOTE: AUPRC (macro {mstored['macro_auprc']:.4f}) is the honest headline for")
p(f"        imbalanced multi-label data; AUROC ({mstored['macro_auroc']:.4f}) flatters it.")
R["reproduced"] = dict(at_0_5=m05, at_stored_thr=mstored)

hdr("2. THRESHOLD OPTIMISM (is the reported F1 tuned on the test set?)")
thr_test = fit_thr(pt, yt)
m_testfit = metrics(pt, yt, thr_test)
p(f"  thresholds fit on VAL  (shipped): {[round(t,3) for t in stored_thr]}")
p(f"  thresholds fit on TEST (oracle) : {[round(t,3) for t in thr_test]}")
p(f"  macro-F1 with val thresholds  : {mstored['macro_f1']:.4f}   <-- honest")
p(f"  macro-F1 with test thresholds : {m_testfit['macro_f1']:.4f}   <-- optimistic/leaky")
p(f"  optimism gap                  : {m_testfit['macro_f1']-mstored['macro_f1']:+.4f}")
R["threshold_optimism"] = dict(val_thr_macro_f1=mstored["macro_f1"],
                               test_thr_macro_f1=m_testfit["macro_f1"],
                               gap=m_testfit["macro_f1"] - mstored["macro_f1"])

hdr("3. BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 resamples)")
rng = np.random.default_rng(0)
n = len(yt)
boot = {c: [] for c in CLASSES}
boot["macro_f1"] = []
boot["macro_auroc"] = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yb, pb = yt[idx], pt[idx]
    f1s, aucs = [], []
    for i, c in enumerate(CLASSES):
        if yb[:, i].sum() == 0 or yb[:, i].sum() == len(yb):
            continue
        f1 = f1_score(yb[:, i], (pb[:, i] >= stored_thr[i]).astype(int), zero_division=0)
        boot[c].append(f1); f1s.append(f1)
        aucs.append(roc_auc_score(yb[:, i], pb[:, i]))
    boot["macro_f1"].append(np.mean(f1s))
    boot["macro_auroc"].append(np.mean(aucs))

p(f"  {'metric':<12} {'point':>8} {'95% CI':>22} {'width':>8}")
p("  " + "-" * 54)
for k in CLASSES + ["macro_f1", "macro_auroc"]:
    arr = np.array(boot[k])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    pt_val = mstored[k]["f1"] if k in CLASSES else mstored[k]
    p(f"  {k:<12} {pt_val:>8.4f} [{lo:>8.4f}, {hi:>8.4f}] {hi-lo:>8.4f}")
    R.setdefault("bootstrap", {})[k] = dict(point=float(pt_val), lo=float(lo), hi=float(hi))
p()
p("  HYP has the widest CI — only ~132 positives in the test set. Any HYP claim")
p("  in the thesis must be reported with this interval, not as a point estimate.")

hdr("4. CALIBRATION (do the probabilities mean anything clinically?)")
p(f"  {'Class':<6} {'Brier':>8} {'ECE':>8} {'mean_p':>8} {'prev':>8} {'ratio':>7}")
p("  " + "-" * 50)
for i, c in enumerate(CLASSES):
    br = brier_score_loss(yt[:, i], pt[:, i])
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pt[:, i], bins) - 1
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(pt[m, i].mean() - yt[m, i].mean())
    mp, pv_ = pt[:, i].mean(), yt[:, i].mean()
    p(f"  {c:<6} {br:>8.4f} {ece:>8.4f} {mp:>8.4f} {pv_:>8.4f} {mp/max(pv_,1e-9):>7.2f}x")
    R.setdefault("calibration", {})[c] = dict(brier=float(br), ece=float(ece),
                                              mean_prob=float(mp), prevalence=float(pv_))
p()
p("  ratio >1 means the model systematically over-predicts that class.")
p("  Cause: WeightedRandomSampler oversampling AND focal-loss alpha weights were")
p("  applied together, so sigmoid outputs are NOT posterior probabilities.")
p("  The report engine prints these as '% confidence' to a nurse -> misleading.")

hdr("5. BATCHNORM / BATCH-SIZE SENSITIVITY (train-time batch=32 vs app batch=1)")
sub = test.head(200)
p1, y1, _ = infer(sub, 1)
p32, y32, _ = infer(sub, 32)
d = np.abs(p1 - p32)
p(f"  max |prob(batch=1) - prob(batch=32)| over 200 records: {d.max():.3e}")
p(f"  mean abs diff: {d.mean():.3e}")
flip = int((( p1 >= np.array(stored_thr)) != (p32 >= np.array(stored_thr))).sum())
p(f"  decision flips caused by batching: {flip}")
R["batch_sensitivity"] = dict(max_diff=float(d.max()), flips=flip)

hdr("6. FAILURE-MODE SANITY CHECKS")
def run_one(sig_raw):
    s = ((sig_raw - sig_mean) / sig_std).T
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(s[None]).float())).numpy()[0]

zeros = run_one(np.zeros((5000, 12), np.float32))
noise = run_one(np.random.default_rng(1).normal(0, 1, (5000, 12)).astype(np.float32))
flat = run_one(np.full((5000, 12), 0.5, np.float32))
real = np.load(os.path.join(CACHE, f"{int(test.ecg_id.iloc[0])}.npy")).astype(np.float32)
scaled10 = run_one(real * 10.0)
inverted = run_one(-real)
realp = run_one(real)

p(f"  {'input':<22} " + " ".join(f"{c:>7s}" for c in CLASSES) + "   detected")
p("  " + "-" * 66)
for name, pr in [("ALL-ZERO (flatline)", zeros), ("pure Gaussian noise", noise),
                 ("constant 0.5 mV", flat), ("real ECG", realp),
                 ("real ECG x10 (gain err)", scaled10), ("real ECG inverted", inverted)]:
    det = [c for i, c in enumerate(CLASSES) if pr[i] >= stored_thr[i]]
    p(f"  {name:<22} " + " ".join(f"{v:>7.3f}" for v in pr) + f"   {det}")
    R.setdefault("sanity", {})[name] = dict(probs=[float(x) for x in pr], detected=det)
p()
p("  A flatline / noise input MUST NOT yield a confident diagnosis. There is no")
p("  signal-quality gate anywhere in app.py -> whatever is uploaded gets a report.")

hdr("7. PATIENT-LEAKAGE IMPACT ON TEST METRICS")
train_pat = set(train.patient_id.dropna().astype(np.int64))
mask = test.patient_id.isin(train_pat).values
p(f"  test records whose patient is also in train: {int(mask.sum())} / {len(test)}")
if mask.sum() >= 30 and (~mask).sum() >= 30:
    for name, mk in [("LEAKED subset", mask), ("CLEAN subset", ~mask)]:
        try:
            mm = metrics(pt[mk], yt[mk], stored_thr)
            p(f"  {name:<14} n={int(mk.sum()):>5}  macro-AUROC={mm['macro_auroc']:.4f}  "
              f"macro-F1={mm['macro_f1']:.4f}")
            R.setdefault("leakage", {})[name] = dict(n=int(mk.sum()),
                                                     macro_auroc=mm["macro_auroc"],
                                                     macro_f1=mm["macro_f1"])
        except ValueError as e:
            p(f"  {name}: {e}")
else:
    p("  (subset too small or empty for a meaningful comparison)")

hdr("8. REPORT-LAYER CONTRADICTION RATE")
det = pt >= np.array(stored_thr)
n_none = int((det.sum(axis=1) == 0).sum())
n_contra = int(((det[:, 0] == 1) & (det[:, 1:].sum(axis=1) > 0)).sum())
p(f"  test records where NOTHING crosses threshold  : {n_none} "
  f"({n_none/len(test)*100:.1f}%)  -> report says 'inconclusive'")
p(f"  test records where NORM *and* an abnormality  : {n_contra} "
  f"({n_contra/len(test)*100:.1f}%)")
p("     -> report_templates.py emits BOTH sentences, producing e.g.")
p("        'The ECG is within normal limits ... consistent with myocardial infarction.'")
p("        There is no mutual-exclusion rule between NORM and the 4 abnormal classes.")
R["report_layer"] = dict(no_detection=n_none, norm_plus_abnormal=n_contra,
                         n=len(test))

# how often does the highest-probability class differ from the reported target
tgt = []
for i in range(len(pt)):
    d_ = [(j, pt[i, j]) for j in range(5) if det[i, j]]
    tgt.append(max(d_, key=lambda x: x[1])[0] if d_ else int(np.argmax(pt[i])))
tgt = np.array(tgt)
p(f"  Grad-CAM target class == argmax(prob): {int((tgt == pt.argmax(1)).mean()*100)}% of records")

np.save(os.path.join(OUT, "test_probs.npy"), pt)
np.save(os.path.join(OUT, "test_labels.npy"), yt)
np.save(os.path.join(OUT, "test_ids.npy"), idt)
with open(os.path.join(OUT, "02_model_audit.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "02_model_audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
