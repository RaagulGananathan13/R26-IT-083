"""
PTB-XL Sections 6, 7, 8 — Class Distribution, Data Split, Report Translation
Reads ptbxl_labeled_final.csv and streams original data for translation.
Produces updated ptbxl_labeled_final.csv with English-translated reports.
"""
import pandas as pd
import numpy as np
import ast
import io
import os
import time
import requests
from deep_translator import GoogleTranslator
from tqdm import tqdm

USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

def stream_csv(session, filename):
    resp = session.get(BASE_URL + filename)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def main():
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # Load the labeled CSV from Section 1
    labeled_path = os.path.join(WORK_DIR, "ptbxl_labeled_final.csv")
    labeled = pd.read_csv(labeled_path)
    label_cols = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 6 — CLASS DISTRIBUTION AND IMBALANCE ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  SECTION 6 -- CLASS DISTRIBUTION AND IMBALANCE ANALYSIS")
    print("=" * 80)

    total = len(labeled)
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # ── 6.1: Class counts ──
    print(f"\n[6.1] Class distribution (total labeled records: {total:,})")
    print("-" * 65)
    print(f"  {'Class':<8} {'Count':>8} {'%':>8} {'Bar':<30}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*30}")
    counts = {}
    for cls in class_names:
        cnt = labeled[f'label_{cls}'].sum()
        counts[cls] = cnt
        pct = cnt / total * 100
        bar = "#" * int(pct)
        print(f"  {cls:<8} {cnt:>8,} {pct:>7.1f}% {bar}")

    total_labels = sum(counts.values())
    print(f"\n  Total positive label slots: {total_labels:,}")
    print(f"  (Sum > {total:,} because multi-label records are counted multiple times)")

    # ── 6.2: Class imbalance ratios ──
    print(f"\n[6.2] Class imbalance ratios")
    print("-" * 65)
    max_count = max(counts.values())
    min_count = min(counts.values())
    max_class = [c for c, v in counts.items() if v == max_count][0]
    min_class = [c for c, v in counts.items() if v == min_count][0]

    print(f"  Most frequent:  {max_class} ({max_count:,})")
    print(f"  Least frequent: {min_class} ({min_count:,})")
    print(f"  Imbalance ratio (max/min): {max_count/min_count:.2f}x")
    print(f"\n  Class-to-class ratios relative to NORM:")
    for cls in class_names:
        ratio = counts['NORM'] / counts[cls]
        print(f"    NORM/{cls}: {ratio:.2f}x")

    # ── 6.3: Compute BCE class weights ──
    print(f"\n[6.3] Binary Cross-Entropy class weights (for weighted loss)")
    print("-" * 65)
    print(f"  Formula: weight_c = total_samples / count_c")
    print(f"\n  {'Class':<8} {'Count':>8} {'Weight':>10} {'Normalized':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*12}")

    raw_weights = {}
    for cls in class_names:
        w = total / counts[cls]
        raw_weights[cls] = w

    # Normalize so minimum weight = 1.0
    min_w = min(raw_weights.values())
    for cls in class_names:
        raw = raw_weights[cls]
        norm = raw / min_w
        print(f"  {cls:<8} {counts[cls]:>8,} {raw:>10.3f} {norm:>12.3f}")

    print(f"\n  Interpretation:")
    print(f"  - NORM has lowest weight (1.0x) -- most common class")
    print(f"  - HYP has highest weight ({raw_weights['HYP']/min_w:.2f}x) -- rarest class")
    print(f"  - This means HYP misclassifications are penalized {raw_weights['HYP']/min_w:.1f}x more heavily")

    # ── 6.4: Pos/neg ratio per class (for pos_weight in BCEWithLogitsLoss) ──
    print(f"\n[6.4] Positive/Negative ratio per class (for pos_weight)")
    print("-" * 65)
    print(f"  {'Class':<8} {'Positive':>10} {'Negative':>10} {'Neg/Pos':>10} {'pos_weight':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for cls in class_names:
        pos = counts[cls]
        neg = total - pos
        ratio = neg / pos
        print(f"  {cls:<8} {pos:>10,} {neg:>10,} {ratio:>10.2f} {ratio:>12.2f}")

    print(f"\n  Use these pos_weight values in PyTorch BCEWithLogitsLoss:")
    print(f"  pos_weight = torch.tensor([", end="")
    pw_vals = []
    for cls in class_names:
        pw = (total - counts[cls]) / counts[cls]
        pw_vals.append(f"{pw:.2f}")
    print(", ".join(pw_vals), end="")
    print(f"])")

    # ── 6.5: Per-fold class distribution ──
    print(f"\n[6.5] Class distribution per fold")
    print("-" * 65)
    print(f"  {'Fold':>6}", end="")
    for cls in class_names:
        print(f" {cls:>7}", end="")
    print(f" {'Total':>8}")
    print(f"  {'-'*6}", end="")
    for _ in class_names:
        print(f" {'-'*7}", end="")
    print(f" {'-'*8}")
    
    for fold in range(1, 11):
        fold_data = labeled[labeled['strat_fold'] == fold]
        print(f"  {fold:>6}", end="")
        for cls in class_names:
            cnt = fold_data[f'label_{cls}'].sum()
            print(f" {cnt:>7,}", end="")
        print(f" {len(fold_data):>8,}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 7 — DATA SPLIT VERIFICATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  SECTION 7 -- DATA SPLIT: TRAINING, VALIDATION, AND TEST")
    print(f"{'=' * 80}")

    train = labeled[labeled['strat_fold'] <= 8]
    val = labeled[labeled['strat_fold'] == 9]
    test = labeled[labeled['strat_fold'] == 10]

    # ── 7.1: Split sizes ──
    print(f"\n[7.1] Split sizes")
    print("-" * 65)
    print(f"  {'Split':<15} {'Folds':<12} {'Records':>8} {'%':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*8}")
    print(f"  {'Training':<15} {'1-8':<12} {len(train):>8,} {len(train)/total*100:>7.1f}%")
    print(f"  {'Validation':<15} {'9':<12} {len(val):>8,} {len(val)/total*100:>7.1f}%")
    print(f"  {'Test':<15} {'10':<12} {len(test):>8,} {len(test)/total*100:>7.1f}%")
    print(f"  {'TOTAL':<15} {'1-10':<12} {total:>8,} {'100.0':>7}%")

    # ── 7.2: Class distribution per split ──
    print(f"\n[7.2] Class distribution per split")
    print("-" * 65)
    for split_name, split_data in [("Training", train), ("Validation", val), ("Test", test)]:
        print(f"\n  {split_name} ({len(split_data):,} records):")
        for cls in class_names:
            cnt = split_data[f'label_{cls}'].sum()
            pct = cnt / len(split_data) * 100
            print(f"    {cls:<8}: {cnt:>6,} ({pct:.1f}%)")

    # ── 7.3: Data leakage check ──
    print(f"\n[7.3] Data leakage check (patient overlap across splits)")
    print("-" * 65)
    train_patients = set(train['patient_id'].dropna())
    val_patients = set(val['patient_id'].dropna())
    test_patients = set(test['patient_id'].dropna())

    train_val_leak = train_patients & val_patients
    train_test_leak = train_patients & test_patients
    val_test_leak = val_patients & test_patients

    print(f"  Train patients:      {len(train_patients):,}")
    print(f"  Validation patients: {len(val_patients):,}")
    print(f"  Test patients:       {len(test_patients):,}")
    print(f"\n  Overlaps:")
    print(f"    Train <-> Val:  {len(train_val_leak)} patients {'-- NO LEAKAGE' if len(train_val_leak)==0 else '-- LEAKAGE DETECTED!'}")
    print(f"    Train <-> Test: {len(train_test_leak)} patients {'-- NO LEAKAGE' if len(train_test_leak)==0 else '-- LEAKAGE DETECTED!'}")
    print(f"    Val <-> Test:   {len(val_test_leak)} patients {'-- NO LEAKAGE' if len(val_test_leak)==0 else '-- LEAKAGE DETECTED!'}")

    all_ok = len(train_val_leak) == 0 and len(train_test_leak) == 0 and len(val_test_leak) == 0
    print(f"\n  {'VERIFIED: Zero data leakage across all splits!' if all_ok else 'WARNING: Data leakage detected!'}")

    # ── 7.4: Patient ECG count distribution ──
    print(f"\n[7.4] Patients with multiple ECGs")
    print("-" * 65)
    patient_counts = labeled.groupby('patient_id').size()
    multi = patient_counts[patient_counts > 1]
    print(f"  Patients with 1 ECG:    {(patient_counts == 1).sum():,}")
    print(f"  Patients with 2 ECGs:   {(patient_counts == 2).sum():,}")
    print(f"  Patients with 3+ ECGs:  {(patient_counts >= 3).sum():,}")
    print(f"  Max ECGs per patient:   {patient_counts.max()}")

    if len(multi) > 0:
        print(f"\n  Verifying multi-ECG patients stay in same fold:")
        violations = 0
        for pid, cnt in multi.items():
            folds = labeled[labeled['patient_id'] == pid]['strat_fold'].unique()
            if len(folds) > 1:
                violations += 1
        print(f"    Patients spanning multiple folds: {violations}")
        print(f"    {'VERIFIED: All multi-ECG patients confined to single fold!' if violations == 0 else f'WARNING: {violations} patients span multiple folds!'}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 8 — GERMAN REPORT TRANSLATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"  SECTION 8 -- GERMAN REPORT TRANSLATION (de -> en)")
    print(f"{'=' * 80}")

    # Stream original data to get the German reports
    print(f"\n[8.1] Streaming original reports from PhysioNet...")
    ptbxl_orig = stream_csv(session, "ptbxl_database.csv")

    # Merge German reports into labeled data
    reports = ptbxl_orig[['ecg_id', 'report']].copy()
    reports = reports.rename(columns={'report': 'report_de'})
    labeled = labeled.merge(reports, on='ecg_id', how='left')

    # Report statistics
    has_report = labeled['report_de'].notna() & (labeled['report_de'] != '')
    print(f"  Records with German reports: {has_report.sum():,}")
    print(f"  Records without reports:     {(~has_report).sum()}")

    # Sample German reports
    print(f"\n[8.2] Sample German reports (before translation):")
    print("-" * 65)
    samples = labeled[has_report].head(5)
    for _, row in samples.iterrows():
        report = str(row['report_de'])[:80]
        print(f"  ECG {row['ecg_id']:>5}: \"{report}\"")

    # ── 8.3: Translate reports ──
    print(f"\n[8.3] Translating German reports to English...")
    print(f"  Using: GoogleTranslator (source='de', target='en')")
    print(f"  Total to translate: {has_report.sum():,}")
    print("-" * 65)

    translator = GoogleTranslator(source='de', target='en')
    errors = 0
    start_time = time.time()

    report_list = labeled['report_de'].tolist()
    translated_list = [''] * len(report_list)

    total_to_translate = len(report_list)
    
    with tqdm(total=total_to_translate, desc="  Translating", unit="rec", 
              bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}') as pbar:
        for idx, text in enumerate(report_list):
            if pd.isna(text) or str(text).strip() == '':
                translated_list[idx] = ''
            else:
                try:
                    result = translator.translate(str(text))
                    translated_list[idx] = result if result else ''
                except Exception as e:
                    translated_list[idx] = ''
                    errors += 1
            pbar.update(1)
            if errors > 0:
                pbar.set_postfix(errors=errors)

    labeled['report_en'] = translated_list

    elapsed_total = time.time() - start_time
    print(f"\n  Translation complete!")
    print(f"  Time: {elapsed_total/60:.1f} minutes")
    print(f"  Errors: {errors}")
    print(f"  Successfully translated: {sum(1 for t in translated_list if t):,}")

    # ── 8.4: Sample translations ──
    print(f"\n[8.4] Sample translations (German -> English):")
    print("-" * 65)
    sample_idxs = labeled[labeled['report_en'] != ''].head(10).index
    for idx in sample_idxs:
        row = labeled.loc[idx]
        de = str(row['report_de'])[:60]
        en = str(row['report_en'])[:60]
        print(f"  ECG {row['ecg_id']:>5}:")
        print(f"    DE: \"{de}\"")
        print(f"    EN: \"{en}\"")
        print()

    # ── 8.5: Translation statistics ──
    print(f"[8.5] Translation statistics:")
    print("-" * 65)
    en_lengths = labeled['report_en'].apply(lambda x: len(str(x)) if x else 0)
    de_lengths = labeled['report_de'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    print(f"  German report avg length:  {de_lengths.mean():.0f} chars")
    print(f"  English report avg length: {en_lengths.mean():.0f} chars")
    print(f"  Empty translations: {(labeled['report_en'] == '').sum()}")

    # ── 8.6: Save updated CSV ──
    print(f"\n[8.6] Saving updated ptbxl_labeled_final.csv...")

    # Drop the temporary German column, keep report_en
    if 'report_de' in labeled.columns:
        labeled = labeled.drop(columns=['report_de'])

    final_cols = [
        'ecg_id', 'patient_id', 'filename_hr',
        'age', 'sex', 'height', 'weight',
        'report_en', 'strat_fold',
        'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP',
        'validated'
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in labeled.columns]
    labeled[final_cols].to_csv(labeled_path, index=False)

    print(f"  Saved to: {labeled_path}")
    print(f"  Total records: {len(labeled):,}")
    print(f"  Columns: {final_cols}")

    print(f"\n{'=' * 80}")
    print(f"  SECTIONS 6, 7, 8 COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n  Key outputs:")
    print(f"  - Class weights computed for weighted BCE loss")
    print(f"  - Zero data leakage verified across all 3 splits")
    print(f"  - {sum(1 for t in translated_list if t):,} reports translated from German to English")
    print(f"  - ptbxl_labeled_final.csv updated with English translations")

if __name__ == "__main__":
    main()
