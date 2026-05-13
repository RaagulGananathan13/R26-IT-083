"""
PTB-XL — Organize workspace, split data, report final class distribution.
Creates: data/ folder with train.csv, val.csv, test.csv
Moves scripts to scripts/ folder
Deletes temporary files
"""
import pandas as pd
import numpy as np
import os
import shutil

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # ══════════════════════════════════════════════════════════════
    #  STEP 1: Create organized folder structure
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  STEP 1 -- CREATE FOLDER STRUCTURE")
    print("=" * 70)

    dirs = ['data', 'scripts']
    for d in dirs:
        path = os.path.join(WORK_DIR, d)
        os.makedirs(path, exist_ok=True)
        print(f"  Created: {d}/")

    # ══════════════════════════════════════════════════════════════
    #  STEP 2: Split data into train/val/test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  STEP 2 -- SPLIT DATA INTO TRAIN / VAL / TEST")
    print("=" * 70)

    df = pd.read_csv(os.path.join(WORK_DIR, 'ptbxl_labeled_final.csv'))
    print(f"\n  Loaded {len(df):,} records, {len(df.columns)} columns")

    train = df[df['strat_fold'] <= 8].copy()
    val = df[df['strat_fold'] == 9].copy()
    test = df[df['strat_fold'] == 10].copy()

    # Save splits
    train.to_csv(os.path.join(WORK_DIR, 'data', 'train.csv'), index=False)
    val.to_csv(os.path.join(WORK_DIR, 'data', 'val.csv'), index=False)
    test.to_csv(os.path.join(WORK_DIR, 'data', 'test.csv'), index=False)
    
    # Also keep the master file in data/
    shutil.copy2(
        os.path.join(WORK_DIR, 'ptbxl_labeled_final.csv'),
        os.path.join(WORK_DIR, 'data', 'ptbxl_labeled_final.csv')
    )
    # Copy mapping file
    shutil.copy2(
        os.path.join(WORK_DIR, 'scp_to_superclass_mapping.json'),
        os.path.join(WORK_DIR, 'data', 'scp_to_superclass_mapping.json')
    )

    print(f"\n  Saved to data/:")
    print(f"    train.csv:                {len(train):>8,} records (folds 1-8)")
    print(f"    val.csv:                  {len(val):>8,} records (fold 9)")
    print(f"    test.csv:                 {len(test):>8,} records (fold 10)")
    print(f"    ptbxl_labeled_final.csv:  {len(df):>8,} records (master)")
    print(f"    scp_to_superclass_mapping.json")

    # ══════════════════════════════════════════════════════════════
    #  STEP 3: Class distribution
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  STEP 3 -- CLASS DISTRIBUTION")
    print("=" * 70)

    label_cols = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    print(f"\n  {'':>12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'':>12} {'(1-8)':>8} {'(9)':>8} {'(10)':>8} {'':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for cls in class_names:
        col = f'label_{cls}'
        tr = train[col].sum()
        va = val[col].sum()
        te = test[col].sum()
        tot = tr + va + te
        print(f"  {cls:>12} {tr:>8,} {va:>8,} {te:>8,} {tot:>8,}")

    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Records':>12} {len(train):>8,} {len(val):>8,} {len(test):>8,} {len(df):>8,}")

    # Percentages
    print(f"\n  Class distribution (% of split):")
    print(f"  {'':>12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for cls in class_names:
        col = f'label_{cls}'
        tr_pct = train[col].sum() / len(train) * 100
        va_pct = val[col].sum() / len(val) * 100
        te_pct = test[col].sum() / len(test) * 100
        print(f"  {cls:>12} {tr_pct:>7.1f}% {va_pct:>7.1f}% {te_pct:>7.1f}%")

    # pos_weight reminder
    print(f"\n  PyTorch pos_weight (computed from training set):")
    pw = []
    for cls in class_names:
        pos = train[f'label_{cls}'].sum()
        neg = len(train) - pos
        pw.append(neg / pos)
    print(f"  pos_weight = torch.tensor([{', '.join(f'{w:.2f}' for w in pw)}])")

    # ══════════════════════════════════════════════════════════════
    #  STEP 4: Move scripts to scripts/ folder
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  STEP 4 -- ORGANIZE SCRIPTS")
    print("=" * 70)

    scripts_to_move = [
        'labeling_section1.py',
        'section2_and_3_exploration.py',
        'section4_and_5_analysis.py',
        'section6_7_8_distribution_split_translation.py',
        'translate_reports.py',
        'impute_demographics.py',
        'verify_final.py',
        'verify_sections_9_10_11.py',
    ]

    for script in scripts_to_move:
        src = os.path.join(WORK_DIR, script)
        dst = os.path.join(WORK_DIR, 'scripts', script)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  Moved: {script} -> scripts/{script}")

    # ══════════════════════════════════════════════════════════════
    #  STEP 5: Delete temporary files
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  STEP 5 -- DELETE TEMPORARY FILES")
    print("=" * 70)

    files_to_delete = [
        'translation_progress.json',    # Translation checkpoint (no longer needed)
        'download_ptbxl.py',            # Downloader (not used, we streamed)
        'ptbxl_labeled_final.csv',      # Root copy (now in data/)
        'scp_to_superclass_mapping.json',  # Root copy (now in data/)
    ]

    for f in files_to_delete:
        path = os.path.join(WORK_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted: {f}")

    # ══════════════════════════════════════════════════════════════
    #  FINAL: Show organized structure
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  FINAL WORKSPACE STRUCTURE")
    print("=" * 70)
    print(f"""
  RP-Venu/
  ├── PTB_XL_Dataset_Labeling_Guide.txt   (reference guide)
  ├── organize_and_split.py               (this script)
  │
  ├── data/
  │   ├── ptbxl_labeled_final.csv         (master: 17,221 records, 17 cols)
  │   ├── train.csv                       ({len(train):,} records, folds 1-8)
  │   ├── val.csv                         ({len(val):,} records, fold 9)
  │   ├── test.csv                        ({len(test):,} records, fold 10)
  │   └── scp_to_superclass_mapping.json  (44-code mapping)
  │
  └── scripts/
      ├── labeling_section1.py            (Section 1: label pipeline)
      ├── section2_and_3_exploration.py   (Sections 2-3: file inspection)
      ├── section4_and_5_analysis.py      (Sections 4-5: thresholds)
      ├── section6_7_8_...py              (Sections 6-8: distribution)
      ├── translate_reports.py            (Section 8: translation)
      ├── impute_demographics.py          (Missing data imputation)
      ├── verify_final.py                 (Quick CSV check)
      └── verify_sections_9_10_11.py      (Full sanity checks)
""")
    print("  DONE! Workspace organized and data split.")

if __name__ == "__main__":
    main()
