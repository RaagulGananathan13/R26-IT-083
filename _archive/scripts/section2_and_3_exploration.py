"""
PTB-XL Sections 2 & 3 — Dataset File Exploration & Complete SCP Mapping
Streams files from PhysioNet and produces a detailed inspection report.
"""
import pandas as pd
import numpy as np
import ast
import io
import requests
import os

# ── PhysioNet credentials ──
USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

def stream_csv(session, filename):
    url = BASE_URL + filename
    resp = session.get(url)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def stream_text(session, filename):
    url = BASE_URL + filename
    resp = session.get(url)
    resp.raise_for_status()
    return resp.text

def main():
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 2 — FILE TYPE 1: ptbxl_database.csv
    # ══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  SECTION 2 -- WHAT FILES EXIST IN THE PTB-XL DATASET")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("  FILE TYPE 1: ptbxl_database.csv  (Master Spreadsheet)")
    print("-" * 80)

    ptbxl = stream_csv(session, "ptbxl_database.csv")
    print(f"\n  Total rows (ECG recordings): {len(ptbxl)}")
    print(f"  Total columns: {len(ptbxl.columns)}")

    print(f"\n  ALL COLUMNS with data types and missing values:")
    print(f"  {'Column':<35} {'Dtype':<12} {'Non-Null':>10} {'Missing':>10} {'% Missing':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for col in ptbxl.columns:
        non_null = ptbxl[col].notna().sum()
        missing = ptbxl[col].isna().sum()
        pct = missing / len(ptbxl) * 100
        print(f"  {col:<35} {str(ptbxl[col].dtype):<12} {non_null:>10,} {missing:>10,} {pct:>9.1f}%")

    # Key columns deep inspection
    print(f"\n  --- ecg_id ---")
    print(f"  Range: {ptbxl['ecg_id'].min()} to {ptbxl['ecg_id'].max()}")
    print(f"  Unique: {ptbxl['ecg_id'].nunique()} (should = {len(ptbxl)})")

    print(f"\n  --- patient_id ---")
    print(f"  Unique patients: {ptbxl['patient_id'].nunique()}")
    multi_ecg = ptbxl.groupby('patient_id').size()
    print(f"  Patients with 1 ECG:  {(multi_ecg == 1).sum()}")
    print(f"  Patients with 2+ ECGs: {(multi_ecg > 1).sum()}")
    print(f"  Max ECGs per patient: {multi_ecg.max()}")

    print(f"\n  --- age ---")
    print(f"  Range: {ptbxl['age'].min()} to {ptbxl['age'].max()} years")
    print(f"  Mean: {ptbxl['age'].mean():.1f}, Median: {ptbxl['age'].median():.1f}")
    print(f"  Missing: {ptbxl['age'].isna().sum()}")

    print(f"\n  --- sex ---")
    sex_counts = ptbxl['sex'].value_counts()
    for val, cnt in sex_counts.items():
        label = "Male" if val == 0 else "Female"
        print(f"  {val} ({label}): {cnt:,} ({cnt/len(ptbxl)*100:.1f}%)")

    print(f"\n  --- height ---")
    print(f"  Range: {ptbxl['height'].min()} to {ptbxl['height'].max()} cm")
    print(f"  Mean: {ptbxl['height'].mean():.1f} cm")
    print(f"  Missing: {ptbxl['height'].isna().sum()} ({ptbxl['height'].isna().sum()/len(ptbxl)*100:.1f}%)")

    print(f"\n  --- weight ---")
    print(f"  Range: {ptbxl['weight'].min()} to {ptbxl['weight'].max()} kg")
    print(f"  Mean: {ptbxl['weight'].mean():.1f} kg")
    print(f"  Missing: {ptbxl['weight'].isna().sum()} ({ptbxl['weight'].isna().sum()/len(ptbxl)*100:.1f}%)")

    print(f"\n  --- scp_codes (THE KEY LABELING COLUMN) ---")
    ptbxl['scp_parsed'] = ptbxl['scp_codes'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
    codes_per_record = ptbxl['scp_parsed'].apply(len)
    print(f"  Min codes per record: {codes_per_record.min()}")
    print(f"  Max codes per record: {codes_per_record.max()}")
    print(f"  Mean codes per record: {codes_per_record.mean():.2f}")
    print(f"  Sample values:")
    for i in [0, 100, 500, 5000, 15000]:
        if i < len(ptbxl):
            print(f"    ECG {ptbxl.iloc[i]['ecg_id']}: {ptbxl.iloc[i]['scp_codes']}")

    # Collect all unique SCP codes used across all records
    all_codes_used = set()
    for codes in ptbxl['scp_parsed']:
        all_codes_used.update(codes.keys())
    print(f"  Total unique SCP codes used across dataset: {len(all_codes_used)}")

    print(f"\n  --- report (German clinical text) ---")
    print(f"  Missing: {ptbxl['report'].isna().sum()}")
    report_lengths = ptbxl['report'].dropna().apply(len)
    print(f"  Report length range: {report_lengths.min()} to {report_lengths.max()} characters")
    print(f"  Mean length: {report_lengths.mean():.0f} characters")
    print(f"  Sample reports:")
    for i in [0, 10, 100]:
        if i < len(ptbxl):
            report = str(ptbxl.iloc[i]['report'])[:80]
            print(f"    ECG {ptbxl.iloc[i]['ecg_id']}: \"{report}...\"")

    print(f"\n  --- validated_by_human ---")
    val_counts = ptbxl['validated_by_human'].value_counts()
    for val, cnt in val_counts.items():
        print(f"  {val}: {cnt:,} ({cnt/len(ptbxl)*100:.1f}%)")

    print(f"\n  --- filename_lr (100Hz signal path) ---")
    print(f"  Sample: {ptbxl['filename_lr'].iloc[0]}")
    print(f"  Missing: {ptbxl['filename_lr'].isna().sum()}")

    print(f"\n  --- filename_hr (500Hz signal path) ---")
    print(f"  Sample: {ptbxl['filename_hr'].iloc[0]}")
    print(f"  Missing: {ptbxl['filename_hr'].isna().sum()}")

    print(f"\n  --- strat_fold (cross-validation fold assignment) ---")
    fold_counts = ptbxl['strat_fold'].value_counts().sort_index()
    for fold, cnt in fold_counts.items():
        role = "TRAIN" if fold <= 8 else ("VAL" if fold == 9 else "TEST")
        print(f"  Fold {fold:>2}: {cnt:>5,} records  [{role}]")
    print(f"  Training (1-8):   {len(ptbxl[ptbxl['strat_fold'] <= 8]):,}")
    print(f"  Validation (9):   {len(ptbxl[ptbxl['strat_fold'] == 9]):,}")
    print(f"  Test (10):        {len(ptbxl[ptbxl['strat_fold'] == 10]):,}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 2 — FILE TYPE 2: scp_statements.csv
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("  FILE TYPE 2: scp_statements.csv  (SCP Code Lookup Table)")
    print("-" * 80)

    scp_df = stream_csv(session, "scp_statements.csv")
    scp_code_col = scp_df.columns[0]
    scp_df = scp_df.rename(columns={scp_code_col: 'scp_code'})

    print(f"\n  Total SCP codes: {len(scp_df)}")
    print(f"  Columns: {list(scp_df.columns)}")
    print(f"\n  ALL COLUMNS:")
    print(f"  {'Column':<30} {'Dtype':<12} {'Non-Null':>10} {'Missing':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
    for col in scp_df.columns:
        non_null = scp_df[col].notna().sum()
        missing = scp_df[col].isna().sum()
        print(f"  {col:<30} {str(scp_df[col].dtype):<12} {non_null:>10,} {missing:>10,}")

    print(f"\n  --- diagnostic column ---")
    diag_counts = scp_df['diagnostic'].value_counts()
    for val, cnt in diag_counts.items():
        print(f"  diagnostic={val}: {cnt} codes")

    print(f"\n  --- Superclass categories ---")
    super_counts = scp_df['diagnostic_class'].value_counts(dropna=False)
    for val, cnt in super_counts.items():
        print(f"  {str(val):>8}: {cnt} codes")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 2 — FILE TYPE 3: Signal Files
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("  FILE TYPE 3: Signal Files (.dat / .hea in records100/ and records500/)")
    print("-" * 80)

    # Stream the RECORDS file to understand the folder structure
    try:
        records_text = stream_text(session, "RECORDS")
        record_lines = [l.strip() for l in records_text.strip().split('\n') if l.strip()]
        print(f"\n  Total signal records listed in RECORDS file: {len(record_lines)}")
        print(f"  Sample paths:")
        for line in record_lines[:5]:
            print(f"    {line}")
        print(f"    ...")
        for line in record_lines[-3:]:
            print(f"    {line}")

        # Count subfolder structure
        folders_100 = set()
        folders_500 = set()
        for line in record_lines:
            parts = line.split('/')
            if len(parts) >= 2:
                if parts[0] == 'records100':
                    folders_100.add(parts[1])
                elif parts[0] == 'records500':
                    folders_500.add(parts[1])

        rec100_count = sum(1 for l in record_lines if l.startswith('records100'))
        rec500_count = sum(1 for l in record_lines if l.startswith('records500'))
        print(f"\n  records100/ entries: {rec100_count} (100Hz, low-resolution)")
        print(f"    Subfolders: {len(folders_100)} (e.g., {sorted(folders_100)[:5]})")
        print(f"  records500/ entries: {rec500_count} (500Hz, high-resolution)")
        print(f"    Subfolders: {len(folders_500)} (e.g., {sorted(folders_500)[:5]})")
        print(f"\n  Each record consists of:")
        print(f"    - .dat file: binary signal data (12-lead waveform values)")
        print(f"    - .hea file: header (format, leads, sampling rate, scale)")
        print(f"  Signal files are NOT modified during labeling.")
    except Exception as e:
        print(f"  Could not stream RECORDS file: {e}")

    # Stream a sample .hea file to show its structure
    try:
        sample_path = ptbxl['filename_hr'].iloc[0]
        hea_url = BASE_URL + sample_path + ".hea"
        resp = session.get(hea_url)
        resp.raise_for_status()
        print(f"\n  Sample .hea header file ({sample_path}.hea):")
        for line in resp.text.strip().split('\n'):
            print(f"    {line}")
    except Exception as e:
        print(f"  Could not fetch sample .hea file: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 3 — THE COMPLETE SCP-TO-SUPERCLASS MAPPING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SECTION 3 -- THE COMPLETE SCP CODE TO SUPERCLASS MAPPING")
    print("=" * 80)

    # Separate diagnostic from non-diagnostic
    diag_df = scp_df[scp_df['diagnostic'] == 1.0].copy()
    nondiag_df = scp_df[scp_df['diagnostic'] != 1.0].copy()

    # Build complete mapping table
    print(f"\n  DIAGNOSTIC CODES: {len(diag_df)} total")

    for superclass in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        subset = diag_df[diag_df['diagnostic_class'] == superclass]
        print(f"\n  {'=' * 72}")
        print(f"  SUPERCLASS: {superclass}")
        print(f"  {'=' * 72}")
        print(f"  Total codes: {len(subset)}")
        print(f"  {'Code':<10} {'Description':<55} {'Used?':>5}")
        print(f"  {'-'*10} {'-'*55} {'-'*5}")
        for _, row in subset.iterrows():
            code = row['scp_code']
            desc = str(row.get('description', 'N/A'))[:55]
            used = "Yes" if code in all_codes_used else "No"
            print(f"  {code:<10} {desc:<55} {used:>5}")

    # Show non-diagnostic codes (discarded)
    print(f"\n  {'=' * 72}")
    print(f"  NON-DIAGNOSTIC CODES (DISCARDED from 5-class labeling)")
    print(f"  {'=' * 72}")
    print(f"  Total non-diagnostic codes: {len(nondiag_df)}")

    # Group by their Statement category
    if 'Statement Category' in nondiag_df.columns:
        cat_col = 'Statement Category'
    elif 'diagnostic_class' in nondiag_df.columns:
        cat_col = 'diagnostic_class'
    else:
        cat_col = None

    print(f"\n  {'Code':<10} {'Description':<45} {'Category':>15} {'Used?':>5}")
    print(f"  {'-'*10} {'-'*45} {'-'*15} {'-'*5}")
    for _, row in nondiag_df.iterrows():
        code = row['scp_code']
        desc = str(row.get('description', 'N/A'))[:45]
        if cat_col and pd.notna(row.get(cat_col)):
            cat = str(row[cat_col])[:15]
        else:
            # Try to identify from other columns
            if row.get('form', 0) == 1:
                cat = 'FORM'
            elif row.get('rhythm', 0) == 1:
                cat = 'RHYTHM'
            else:
                cat = 'OTHER'
        used = "Yes" if code in all_codes_used else "No"
        print(f"  {code:<10} {desc:<45} {cat:>15} {used:>5}")

    # Cross-check: are there codes used in the dataset that are NOT in scp_statements?
    known_codes = set(scp_df['scp_code'].values)
    unknown_used = all_codes_used - known_codes
    if unknown_used:
        print(f"\n  WARNING: {len(unknown_used)} codes used in data but NOT in scp_statements.csv:")
        for code in sorted(unknown_used):
            print(f"    {code}")
    else:
        print(f"\n  VERIFIED: All {len(all_codes_used)} codes used in data are present in scp_statements.csv")

    # Summary mapping dictionary
    print(f"\n  {'=' * 72}")
    print(f"  FINAL MAPPING DICTIONARY (for use in code)")
    print(f"  {'=' * 72}")
    mapping = {}
    for _, row in diag_df.iterrows():
        sc = row.get('diagnostic_class', None)
        if pd.notna(sc):
            mapping[row['scp_code']] = sc
    print(f"  scp_to_superclass = {{")
    for i, (code, sc) in enumerate(sorted(mapping.items())):
        comma = "," if i < len(mapping) - 1 else ""
        print(f"      \"{code}\": \"{sc}\"{comma}")
    print(f"  }}")
    print(f"  Total mapped codes: {len(mapping)}")

    # Confidence distribution across the dataset
    print(f"\n  {'=' * 72}")
    print(f"  CONFIDENCE SCORE DISTRIBUTION (across all records)")
    print(f"  {'=' * 72}")
    all_confidences = []
    for codes in ptbxl['scp_parsed']:
        all_confidences.extend(codes.values())
    conf_series = pd.Series(all_confidences)
    print(f"  Total code-confidence entries: {len(conf_series):,}")
    print(f"  Unique confidence values: {sorted(conf_series.unique())}")
    print(f"\n  Confidence distribution:")
    for val in sorted(conf_series.unique()):
        cnt = (conf_series == val).sum()
        print(f"    {val:>6.1f}%: {cnt:>8,} entries ({cnt/len(conf_series)*100:.1f}%)")

    print(f"\n{'=' * 80}")
    print(f"  SECTIONS 2 & 3 EXPLORATION COMPLETE")
    print(f"{'=' * 80}")

    # ── Save the mapping dictionary to a JSON file for reuse ──
    import json
    mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scp_to_superclass_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  Mapping saved to: {mapping_path}")

if __name__ == "__main__":
    main()
