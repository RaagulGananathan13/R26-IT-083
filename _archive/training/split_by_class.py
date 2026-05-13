"""
split_by_class.py
─────────────────
Copies PTB-XL test set .dat and .hea files into class-labelled folders.

Output structure:
  data/
    test_by_class/
      NORM/   (707 records)
      MI/     (268 records)
      STTC/   (456 records)
      CD/     (483 records)
      HYP/    (132 records)

Each folder contains pairs of  <ecg_id>_hr.dat  and  <ecg_id>_hr.hea
(copied from the original records500/ tree — originals are NOT deleted).

Usage:
    python split_by_class.py
"""

import os, shutil, pandas as pd
from pathlib import Path

WORK_DIR   = Path(__file__).parent
DATA_DIR   = WORK_DIR / "data"
PTBXL_DIR  = DATA_DIR / "ptb-xl"          # root of the downloaded dataset
TEST_CSV   = DATA_DIR / "test.csv"
OUT_DIR    = DATA_DIR / "test_by_class"

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ── Load test metadata ──────────────────────────────────────────────
df = pd.read_csv(TEST_CSV)
print(f"  Loaded {len(df)} test records from {TEST_CSV.name}\n")

# ── Create output folders ───────────────────────────────────────────
for cls in CLASS_NAMES:
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

# ── Copy files ──────────────────────────────────────────────────────
stats = {cls: {"copied": 0, "missing": 0} for cls in CLASS_NAMES}

for _, row in df.iterrows():
    # filename_hr looks like  "records500/00000/00009_hr"
    rel_path = row["filename_hr"]           # e.g. records500/00000/00009_hr
    record_stem = Path(rel_path).name       # e.g. 00009_hr

    src_base = PTBXL_DIR / rel_path        # .../ptb-xl/records500/00000/00009_hr

    for cls in CLASS_NAMES:
        if row[f"label_{cls}"] != 1:
            continue

        dst_dir = OUT_DIR / cls

        copied_any = False
        for ext in (".dat", ".hea"):
            src = Path(str(src_base) + ext)
            dst = dst_dir / (record_stem + ext)

            if not src.exists():
                # Try without the ptb-xl subfolder (dataset placed directly in data/)
                src_alt = DATA_DIR / rel_path
                src_alt = Path(str(src_alt) + ext)
                if src_alt.exists():
                    src = src_alt

            if src.exists():
                shutil.copy2(src, dst)
                copied_any = True
            else:
                stats[cls]["missing"] += 1
                print(f"  [WARN] Not found: {src}")

        if copied_any:
            stats[cls]["copied"] += 1

# ── Summary ─────────────────────────────────────────────────────────
print("=" * 55)
print(f"  Output: {OUT_DIR}")
print("=" * 55)
print(f"  {'Class':<8}  {'Copied':>8}  {'Missing files':>14}")
print(f"  {'-'*8}  {'-'*8}  {'-'*14}")
for cls in CLASS_NAMES:
    c = stats[cls]["copied"]
    m = stats[cls]["missing"]
    flag = "  ✓" if m == 0 else f"  ⚠ {m} missing"
    print(f"  {cls:<8}  {c:>8}{flag}")
print("=" * 55)
print("  Done.")
