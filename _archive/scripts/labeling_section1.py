"""
PTB-XL Section 1 — Label Mapping & Transformation
Streams CSV files directly from PhysioNet (no full download required).
Produces: ptbxl_labeled_final.csv
"""
import pandas as pd
import numpy as np
import ast
import io
import requests

# ── PhysioNet credentials ──
USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

def stream_csv(session, filename):
    """Stream a CSV file from PhysioNet directly into a DataFrame."""
    url = BASE_URL + filename
    print(f"  Streaming {filename} from PhysioNet...")
    resp = session.get(url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def main():
    print("=" * 72)
    print("  PTB-XL LABELING — SECTION 1: LABEL MAPPING & TRANSFORMATION")
    print("=" * 72)

    # ── Create authenticated session ──
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — Stream the two critical CSV files
    # ══════════════════════════════════════════════════════════════════════
    print("\n[STEP 1] Streaming dataset files from PhysioNet...\n")
    ptbxl = stream_csv(session, "ptbxl_database.csv")
    scp_df = stream_csv(session, "scp_statements.csv")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — Verify ptbxl_database.csv structure
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 2] Verifying ptbxl_database.csv...")
    print(f"  Total records: {len(ptbxl)}")
    expected_cols = ['ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight',
                     'scp_codes', 'report', 'validated_by_human',
                     'filename_lr', 'filename_hr', 'strat_fold']
    missing = [c for c in expected_cols if c not in ptbxl.columns]
    if missing:
        print(f"  ⚠ Missing columns: {missing}")
    else:
        print(f"  ✓ All expected columns present")
    print(f"  Columns: {list(ptbxl.columns)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — Build the SCP → Superclass mapping dictionary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 3] Building SCP-code → Superclass mapping...")

    # The first column is the SCP code (may be unnamed or named differently)
    scp_code_col = scp_df.columns[0]
    scp_df = scp_df.rename(columns={scp_code_col: 'scp_code'})

    # Keep only diagnostic codes (diagnostic == 1)
    diag_df = scp_df[scp_df['diagnostic'] == 1.0].copy()
    print(f"  Total SCP codes: {len(scp_df)}")
    print(f"  Diagnostic codes (diagnostic=1): {len(diag_df)}")

    # Build mapping: {SCP_CODE: SUPERCLASS}
    # Only include codes that have a valid (non-null) diagnostic_class / superclass
    scp_to_super = {}
    for _, row in diag_df.iterrows():
        code = row['scp_code']
        # Try 'diagnostic_class' first, fall back to 'superclass'
        superclass = row.get('diagnostic_class', row.get('superclass', np.nan))
        if pd.notna(superclass) and superclass != '':
            scp_to_super[code] = superclass

    print(f"  Mapped {len(scp_to_super)} diagnostic codes to superclasses")
    print(f"\n  Mapping preview (first 15):")
    for i, (code, sc) in enumerate(sorted(scp_to_super.items())):
        if i >= 15:
            print(f"    ... and {len(scp_to_super) - 15} more")
            break
        print(f"    {code:8s} → {sc}")

    # Show superclass distribution in the mapping
    from collections import Counter
    sc_counts = Counter(scp_to_super.values())
    print(f"\n  Codes per superclass in mapping:")
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        print(f"    {sc:5s}: {sc_counts.get(sc, 0)} SCP codes")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — Parse the scp_codes column (string → dict)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 4] Parsing scp_codes column (string → dictionary)...")
    ptbxl['scp_codes_dict'] = ptbxl['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )
    sample = ptbxl['scp_codes_dict'].iloc[0]
    print(f"  ✓ Parsed all {len(ptbxl)} rows")
    print(f"  Sample parsed value: {sample}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — Apply confidence threshold (>= 100)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 5] Applying confidence threshold >= 100...")

    def filter_by_confidence(codes_dict, threshold=100.0):
        """Keep only codes with confidence >= threshold."""
        return {code: conf for code, conf in codes_dict.items() if conf >= threshold}

    ptbxl['scp_filtered'] = ptbxl['scp_codes_dict'].apply(filter_by_confidence)

    has_codes = ptbxl['scp_filtered'].apply(len) > 0
    print(f"  Records with >= 1 code at 100% confidence: {has_codes.sum()}")
    print(f"  Records with NO code at 100% confidence:   {(~has_codes).sum()}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 — Map filtered codes to superclasses
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 6] Mapping filtered codes → superclasses...")

    def map_to_superclasses(filtered_codes):
        """Map SCP codes to their superclass set."""
        superclasses = set()
        for code in filtered_codes:
            if code in scp_to_super:
                superclasses.add(scp_to_super[code])
        return superclasses

    ptbxl['superclass_set'] = ptbxl['scp_filtered'].apply(map_to_superclasses)

    has_superclass = ptbxl['superclass_set'].apply(len) > 0
    print(f"  Records mapped to >= 1 superclass: {has_superclass.sum()}")
    print(f"  Records with no superclass match:  {(~has_superclass).sum()}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7 — Apply NORM correction
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 7] Applying NORM correction (NORM removed if pathology present)...")

    pathological = {'MI', 'STTC', 'CD', 'HYP'}
    norm_conflicts = 0

    def correct_norm(sc_set):
        nonlocal norm_conflicts
        if 'NORM' in sc_set and sc_set & pathological:
            norm_conflicts += 1
            return sc_set - {'NORM'}
        return sc_set

    ptbxl['superclass_set'] = ptbxl['superclass_set'].apply(correct_norm)
    print(f"  Records where NORM was removed due to co-occurring pathology: {norm_conflicts}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8 — Exclude records with empty superclass sets
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 8] Filtering out records with no valid superclass label...")
    before = len(ptbxl)
    ptbxl_labeled = ptbxl[ptbxl['superclass_set'].apply(len) > 0].copy()
    excluded = before - len(ptbxl_labeled)
    print(f"  Records before filtering: {before}")
    print(f"  Records excluded (empty labels): {excluded}")
    print(f"  Records remaining (labeled): {len(ptbxl_labeled)}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9 — Build 5 binary label columns
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 9] Building binary label columns [NORM, MI, STTC, CD, HYP]...")

    for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        ptbxl_labeled[f'label_{cls}'] = ptbxl_labeled['superclass_set'].apply(
            lambda s, c=cls: 1 if c in s else 0
        )

    # Print class distribution
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │         CLASS DISTRIBUTION SUMMARY            │")
    print(f"  ├──────────┬──────────┬────────────────────────┤")
    print(f"  │ Class    │  Count   │  % of labeled records  │")
    print(f"  ├──────────┼──────────┼────────────────────────┤")
    total_labeled = len(ptbxl_labeled)
    for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        count = ptbxl_labeled[f'label_{cls}'].sum()
        pct = count / total_labeled * 100
        print(f"  │ {cls:8s} │ {count:>8,} │ {pct:>20.1f}%  │")
    print(f"  ├──────────┼──────────┼────────────────────────┤")
    print(f"  │ TOTAL    │ {total_labeled:>8,} │     (unique records)  │")
    print(f"  └──────────┴──────────┴────────────────────────┘")

    # Multi-label distribution
    ptbxl_labeled['num_labels'] = sum(
        ptbxl_labeled[f'label_{cls}'] for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    )
    print(f"\n  Multi-label distribution:")
    for n in sorted(ptbxl_labeled['num_labels'].unique()):
        cnt = (ptbxl_labeled['num_labels'] == n).sum()
        pct = cnt / total_labeled * 100
        print(f"    {n} label(s): {cnt:>6,} records ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10 — (Report translation skipped for now — separate step)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 10] Report translation placeholder...")
    print(f"  German→English translation will be performed as a separate step.")
    print(f"  Adding 'report_en' column with original text for now.")
    ptbxl_labeled['report_en'] = ptbxl_labeled['report'].fillna('')

    # ══════════════════════════════════════════════════════════════════════
    # STEP 11 — Assemble final CSV columns
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 11] Assembling final labeled CSV...")

    # Create validated column
    ptbxl_labeled['validated'] = ptbxl_labeled['validated_by_human'].apply(
        lambda x: 1 if x == True or x == 'True' else 0
    )

    final_cols = [
        'ecg_id', 'patient_id', 'filename_hr',
        'age', 'sex', 'height', 'weight',
        'report_en', 'strat_fold',
        'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP',
        'validated'
    ]
    final_df = ptbxl_labeled[final_cols].copy()

    # ══════════════════════════════════════════════════════════════════════
    # SANITY CHECKS (Section 11 of the guide)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print(f"  SANITY CHECKS")
    print(f"{'=' * 72}")

    # Check 1 — Class counts
    print(f"\n  CHECK 1 — Class counts vs expected:")
    expected = {'NORM': 9500, 'MI': 5100, 'STTC': 5200, 'CD': 4700, 'HYP': 2600}
    all_ok = True
    for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        actual = final_df[f'label_{cls}'].sum()
        exp = expected[cls]
        diff_pct = abs(actual - exp) / exp * 100
        status = "✓" if diff_pct < 20 else "⚠"
        if diff_pct >= 20:
            all_ok = False
        print(f"    {status} {cls}: {actual:,} (expected ~{exp:,}, diff {diff_pct:.1f}%)")
    print(f"    {'✓ All within tolerance' if all_ok else '⚠ Some counts differ significantly'}")

    # Check 2 — Multi-label sanity
    print(f"\n  CHECK 2 — Multi-label distribution:")
    n_labels = sum(final_df[f'label_{cls}'] for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP'])
    one_label = (n_labels == 1).sum()
    two_label = (n_labels == 2).sum()
    three_plus = (n_labels >= 3).sum()
    zero_label = (n_labels == 0).sum()
    print(f"    1 label:  {one_label:,} ({one_label/len(final_df)*100:.1f}%) — expected 75-80%")
    print(f"    2 labels: {two_label:,} ({two_label/len(final_df)*100:.1f}%) — expected 15-20%")
    print(f"    3+ labels: {three_plus:,} ({three_plus/len(final_df)*100:.1f}%) — expected 2-5%")
    print(f"    0 labels: {zero_label:,} — should be 0")
    print(f"    {'✓ Distribution looks healthy' if zero_label == 0 else '⚠ Found records with 0 labels!'}")

    # Check 3 — Fold distribution
    print(f"\n  CHECK 3 — Fold assignment:")
    print(f"    Training (folds 1-8): {len(final_df[final_df['strat_fold'] <= 8]):,}")
    print(f"    Validation (fold 9):  {len(final_df[final_df['strat_fold'] == 9]):,}")
    print(f"    Test (fold 10):       {len(final_df[final_df['strat_fold'] == 10]):,}")

    # Check 4 — No data leakage
    print(f"\n  CHECK 4 — Data leakage check:")
    train_patients = set(final_df[final_df['strat_fold'] <= 8]['patient_id'])
    test_patients = set(final_df[final_df['strat_fold'] == 10]['patient_id'])
    leak = train_patients & test_patients
    print(f"    Training patients:  {len(train_patients):,}")
    print(f"    Test patients:      {len(test_patients):,}")
    print(f"    Overlapping patients: {len(leak)}")
    print(f"    {'✓ No data leakage detected!' if len(leak) == 0 else f'⚠ LEAKAGE: {len(leak)} patients in both sets!'}")

    # Check 5 — Random manual inspection
    print(f"\n  CHECK 5 — Random manual inspection (5 samples):")
    samples = ptbxl_labeled.sample(5, random_state=42)
    for _, row in samples.iterrows():
        raw = row['scp_codes_dict']
        filtered = row['scp_filtered']
        labels = {cls: row[f'label_{cls}'] for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']}
        active = [cls for cls, v in labels.items() if v == 1]
        print(f"    ECG {row['ecg_id']:>5}: raw={raw}")
        print(f"             filtered(>=100)={filtered}")
        print(f"             labels={labels}  active={active}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 12 & 13 — Save the final file
    # ══════════════════════════════════════════════════════════════════════
    import os
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptbxl_labeled_final.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 72}")
    print(f"  COMPLETE!")
    print(f"{'=' * 72}")
    print(f"  Output file: {output_path}")
    print(f"  Total labeled records: {len(final_df):,}")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"\n  NOTE: 'report_en' currently contains the original German text.")
    print(f"  A separate translation step is needed to convert to English.")
    print(f"{'=' * 72}")

if __name__ == "__main__":
    main()
