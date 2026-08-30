"""
01_data_audit.py — Independent data-integrity audit of the _archive pipeline.

Checks (no model needed):
  A. Split integrity: patient-level leakage, ecg_id overlap, strat_fold usage
  B. Label integrity: distribution, all-zero rows, NORM+abnormal co-occurrence
  C. Demographic integrity: PTB-XL age=300 sentinel, imputation of height/weight
  D. Signal cache integrity: presence, shape, dtype, all-zero / NaN / flatline
  E. Normalisation-stat provenance
  F. Dataset size vs. official PTB-XL

Writes JSON + human-readable text to Component_02/audit/results/
"""
import os, json, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = os.path.join(ROOT, "_archive")
DATA = os.path.join(ARCH, "data")
CACHE = os.path.join(DATA, "signals_cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
R = {}
lines = []


def p(s=""):
    print(s)
    lines.append(str(s))


def hdr(t):
    p()
    p("=" * 78)
    p(f"  {t}")
    p("=" * 78)


master = pd.read_csv(os.path.join(DATA, "ptbxl_labeled_final.csv"))
train = pd.read_csv(os.path.join(DATA, "train.csv"))
val = pd.read_csv(os.path.join(DATA, "val.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))

# ─────────────────────────────────────────────────────────────── A. SPLITS
hdr("A. SPLIT INTEGRITY")

p(f"  master={len(master):,}  train={len(train):,}  val={len(val):,}  test={len(test):,}")
p(f"  train+val+test = {len(train)+len(val)+len(test):,}")
R["counts"] = dict(master=len(master), train=len(train), val=len(val), test=len(test))

# ecg_id overlap
tr_ids, va_ids, te_ids = set(train.ecg_id), set(val.ecg_id), set(test.ecg_id)
R["ecg_overlap"] = {
    "train_val": len(tr_ids & va_ids),
    "train_test": len(tr_ids & te_ids),
    "val_test": len(va_ids & te_ids),
}
p(f"  ECG-ID overlap  train/val={len(tr_ids & va_ids)}  train/test={len(tr_ids & te_ids)}  val/test={len(va_ids & te_ids)}")

# patient-level leakage
def pset(df):
    return set(df.patient_id.dropna().astype(np.int64))

ptr, pva, pte = pset(train), pset(val), pset(test)
R["patient_overlap"] = {
    "train_val": len(ptr & pva),
    "train_test": len(ptr & pte),
    "val_test": len(pva & pte),
    "n_train_patients": len(ptr),
    "n_test_patients": len(pte),
}
p(f"  PATIENT overlap train/val={len(ptr & pva)}  train/test={len(ptr & pte)}  val/test={len(pva & pte)}")
leaked_test_rows = int(test.patient_id.isin(ptr).sum())
leaked_val_rows = int(val.patient_id.isin(ptr).sum())
p(f"  Test rows whose patient also appears in TRAIN: {leaked_test_rows} / {len(test)} "
  f"({leaked_test_rows/len(test)*100:.2f}%)")
p(f"  Val  rows whose patient also appears in TRAIN: {leaked_val_rows} / {len(val)} "
  f"({leaked_val_rows/len(val)*100:.2f}%)")
R["patient_overlap"]["leaked_test_rows"] = leaked_test_rows
R["patient_overlap"]["leaked_val_rows"] = leaked_val_rows

# strat_fold usage
p()
p("  strat_fold distribution per split (PTB-XL official: fold 9=val, fold 10=test):")
for name, df in [("train", train), ("val", val), ("test", test)]:
    vc = df.strat_fold.value_counts().sort_index().to_dict()
    p(f"    {name:6s}: {vc}")
    R.setdefault("strat_fold", {})[name] = {int(k): int(v) for k, v in vc.items()}

# duplicate ecg_ids inside a split
for name, df in [("train", train), ("val", val), ("test", test)]:
    d = int(df.ecg_id.duplicated().sum())
    if d:
        p(f"  !! {name} has {d} duplicated ecg_id rows")
    R.setdefault("dupes", {})[name] = d

# multiple ECGs per patient (why patient split matters)
recs_per_pt = master.groupby("patient_id").size()
p()
p(f"  Patients in master: {recs_per_pt.shape[0]:,}; records: {len(master):,}")
p(f"  Patients with >1 ECG: {(recs_per_pt > 1).sum():,} "
  f"({(recs_per_pt > 1).mean()*100:.1f}%)  max ECGs/patient: {recs_per_pt.max()}")
R["records_per_patient"] = {
    "n_patients": int(recs_per_pt.shape[0]),
    "n_multi": int((recs_per_pt > 1).sum()),
    "max": int(recs_per_pt.max()),
}

# ─────────────────────────────────────────────────────────────── B. LABELS
hdr("B. LABEL INTEGRITY")

lab_cols = [f"label_{c}" for c in CLASSES]
p(f"  {'split':7s} " + " ".join(f"{c:>7s}" for c in CLASSES) + f"{'  n':>8s}")
for name, df in [("master", master), ("train", train), ("val", val), ("test", test)]:
    row = " ".join(f"{int(df[f'label_{c}'].sum()):>7d}" for c in CLASSES)
    p(f"  {name:7s} {row} {len(df):>8d}")
    R.setdefault("label_counts", {})[name] = {c: int(df[f"label_{c}"].sum()) for c in CLASSES}

p()
p("  Prevalence (%) per split — drift between splits distorts F1 comparisons:")
for name, df in [("train", train), ("val", val), ("test", test)]:
    row = " ".join(f"{df[f'label_{c}'].mean()*100:>6.2f}%" for c in CLASSES)
    p(f"    {name:6s} {row}")

nlab = master[lab_cols].sum(axis=1)
R["label_rows"] = {
    "zero_label_rows": int((nlab == 0).sum()),
    "multi_label_rows": int((nlab > 1).sum()),
    "mean_labels": float(nlab.mean()),
}
p()
p(f"  Rows with ZERO labels : {(nlab == 0).sum()}  (unlabelable → should have been dropped or kept deliberately)")
p(f"  Rows with >1 label    : {(nlab > 1).sum()} ({(nlab>1).mean()*100:.1f}%)")
p(f"  Mean labels/record    : {nlab.mean():.3f}")

norm_and_abn = master[(master.label_NORM == 1) & (master[lab_cols[1:]].sum(axis=1) > 0)]
p(f"  Rows labelled NORM *and* an abnormality: {len(norm_and_abn)} "
  f"({len(norm_and_abn)/len(master)*100:.2f}%)  <-- clinically contradictory ground truth")
R["norm_and_abnormal"] = len(norm_and_abn)

p()
p("  'validated' column (PTB-XL: human-validated report):")
p(f"    {master.validated.value_counts().to_dict()}")
R["validated"] = {int(k): int(v) for k, v in master.validated.value_counts().items()}

# implied pos_weight vs hardcoded POS_WEIGHT in train scripts
p()
p("  POS_WEIGHT hardcoded in train scripts: [1.45, 4.69, 2.78, 2.62, 10.48]")
implied = [float((len(train) - train[f"label_{c}"].sum()) / max(train[f"label_{c}"].sum(), 1)) for c in CLASSES]
p(f"  Recomputed neg/pos ratio on train.csv: {[round(x,2) for x in implied]}")
R["pos_weight_implied"] = implied

# ──────────────────────────────────────────────────────── C. DEMOGRAPHICS
hdr("C. DEMOGRAPHIC INTEGRITY")

for col in ["age", "height", "weight"]:
    s = master[col]
    p(f"  {col:7s} min={s.min():>8.1f} max={s.max():>8.1f} mean={s.mean():>7.2f} "
      f"std={s.std():>7.2f} nan={int(s.isna().sum()):>5d}")
    R.setdefault("demo", {})[col] = dict(min=float(s.min()), max=float(s.max()),
                                         mean=float(s.mean()), std=float(s.std()),
                                         nan=int(s.isna().sum()))

n300 = int((master.age >= 300).sum())
n_over89 = int((master.age > 89).sum())
p()
p(f"  age == 300 sentinel rows (PTB-XL anonymises age>89 as 300): {n300}")
p(f"  age  > 89 rows                                            : {n_over89}")
R["age_300"] = n300
p(f"  -> These inflate age mean/std used in norm_stats.json "
  f"(stored std={json.load(open(os.path.join(DATA,'norm_stats.json')))['demographics']['age']['std']:.2f})")

age_clean = master.age[master.age < 300]
p(f"  Clean age (excl. 300): mean={age_clean.mean():.2f} std={age_clean.std():.2f}")
R["age_clean"] = dict(mean=float(age_clean.mean()), std=float(age_clean.std()))

p()
p(f"  height_missing==1 : {int(master.height_missing.sum()):,} "
  f"({master.height_missing.mean()*100:.1f}%)")
p(f"  weight_missing==1 : {int(master.weight_missing.sum()):,} "
  f"({master.weight_missing.mean()*100:.1f}%)")
p("  (missing values appear imputed — check for a constant fill value below)")
for col in ["height", "weight"]:
    top = master[col].value_counts().head(3)
    p(f"    {col} most common values: {top.to_dict()}")

# ────────────────────────────────────────────────────── D. SIGNAL CACHE
hdr("D. SIGNAL CACHE INTEGRITY")

if not os.path.isdir(CACHE):
    p(f"  !! signals_cache missing at {CACHE}")
else:
    files = [f for f in os.listdir(CACHE) if f.endswith(".npy")]
    p(f"  .npy files in cache: {len(files):,}   master rows: {len(master):,}")
    R["cache_files"] = len(files)

    missing = [int(e) for e in master.ecg_id if not os.path.exists(os.path.join(CACHE, f"{e}.npy"))]
    p(f"  Master ecg_ids with NO cached signal: {len(missing)}")
    R["cache_missing"] = len(missing)

    # scan EVERY cached file used by the three splits
    scan_ids = list(pd.concat([train.ecg_id, val.ecg_id, test.ecg_id]))
    bad_shape, all_zero, has_nan, flat_leads, ok = [], [], [], [], 0
    amp_max = []
    for e in scan_ids:
        fp = os.path.join(CACHE, f"{int(e)}.npy")
        if not os.path.exists(fp):
            continue
        a = np.load(fp, mmap_mode="r")
        if a.shape != (5000, 12):
            bad_shape.append((int(e), tuple(a.shape)))
            continue
        arr = np.asarray(a)
        if not np.isfinite(arr).all():
            has_nan.append(int(e))
        if not np.any(arr):
            all_zero.append(int(e))
            continue
        nflat = int((arr.std(axis=0) < 1e-9).sum())
        if nflat > 0:
            flat_leads.append((int(e), nflat))
        amp_max.append(float(np.abs(arr).max()))
        ok += 1

    p(f"  Scanned {len(scan_ids):,} split records")
    p(f"    OK                    : {ok:,}")
    p(f"    Wrong shape           : {len(bad_shape)}  {bad_shape[:5]}")
    p(f"    ALL-ZERO (download failed, silently zero-filled by prepare_data.py): {len(all_zero)}")
    p(f"      ids: {all_zero[:20]}")
    p(f"    Contains NaN/Inf      : {len(has_nan)}  {has_nan[:10]}")
    p(f"    >=1 completely flat lead : {len(flat_leads)}  {flat_leads[:10]}")
    R["cache_scan"] = dict(ok=ok, bad_shape=len(bad_shape), all_zero=len(all_zero),
                           all_zero_ids=all_zero[:200], nan=len(has_nan),
                           flat_lead_records=len(flat_leads))

    if all_zero:
        z = set(all_zero)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            n = int(df.ecg_id.isin(z).sum())
            p(f"    zero-signal records inside {name}: {n}")
            R["cache_scan"][f"zero_in_{name}"] = n

    if amp_max:
        amp_max = np.array(amp_max)
        p()
        p(f"  |amplitude| max across records: median={np.median(amp_max):.3f} mV  "
          f"p99={np.percentile(amp_max,99):.3f}  max={amp_max.max():.3f}")
        p(f"  Records with |amp|>10 mV (likely artefact/saturation): {int((amp_max>10).sum())}")
        R["amp"] = dict(median=float(np.median(amp_max)), p99=float(np.percentile(amp_max, 99)),
                        max=float(amp_max.max()), gt10mV=int((amp_max > 10).sum()))

# ──────────────────────────────────────────────── E. NORMALISATION STATS
hdr("E. NORMALISATION STATS PROVENANCE")

ns = json.load(open(os.path.join(DATA, "norm_stats.json")))
p(f"  signal_mean (per lead): {[round(x,5) for x in ns['signal_mean']]}")
p(f"  signal_std  (per lead): {[round(x,4) for x in ns['signal_std']]}")
p("  NOTE: prepare_data.py computes these from a 1000-record TRAIN sample -> no test leakage. OK.")
p("  NOTE: stats are GLOBAL per-lead, computed over the time axis. Per-record baseline")
p("        wander / DC offset is therefore NOT removed. No high-pass filter anywhere.")

# ───────────────────────────────────────────────────────── F. VS OFFICIAL
hdr("F. DATASET SIZE VS OFFICIAL PTB-XL")
p(f"  Official PTB-XL v1.0.3 : 21,799 records / 18,869 patients")
p(f"  README claims          : 21,837 records")
p(f"  Actually used (master) : {len(master):,} records / {master.patient_id.nunique():,} patients")
p(f"  Dropped vs official    : ~{21799 - len(master):,} records "
  f"({(21799-len(master))/21799*100:.1f}%) — most likely records with no superclass label")

with open(os.path.join(OUT, "01_data_audit.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "01_data_audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p()
p(f"Saved -> {OUT}")
