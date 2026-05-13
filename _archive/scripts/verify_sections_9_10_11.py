"""
PTB-XL Sections 9, 10, 11 — Final CSV Verification
Validates column structure, exclusion rules, and all 5 sanity checks.
"""
import pandas as pd
import numpy as np
import ast
import io
import requests

USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

def stream_csv(session, filename):
    resp = session.get(BASE_URL + filename)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def main():
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # Load final CSV
    final = pd.read_csv('ptbxl_labeled_final.csv')
    
    # Stream originals for cross-checking
    print("Streaming original data for cross-validation...")
    ptbxl = stream_csv(session, "ptbxl_database.csv")
    scp_df = stream_csv(session, "scp_statements.csv")
    
    # Build mapping
    scp_code_col = scp_df.columns[0]
    scp_df = scp_df.rename(columns={scp_code_col: 'scp_code'})
    diag_df = scp_df[scp_df['diagnostic'] == 1.0]
    scp_to_super = {}
    for _, row in diag_df.iterrows():
        sc = row.get('diagnostic_class', np.nan)
        if pd.notna(sc):
            scp_to_super[row['scp_code']] = sc
    
    ptbxl['scp_parsed'] = ptbxl['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )

    label_cols = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 9 — COLUMN STRUCTURE VERIFICATION
    # ══════════════════════════════════════════════════════════════════
    print("=" * 75)
    print("  SECTION 9 -- FINAL CSV COLUMN STRUCTURE VERIFICATION")
    print("=" * 75)

    required_cols = [
        'ecg_id', 'patient_id', 'filename_hr',
        'age', 'sex', 'height', 'weight',
        'report_en', 'strat_fold',
        'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP',
        'validated'
    ]

    print(f"\n  Expected 15 columns. Found {len(final.columns)} columns.")
    print(f"  Columns in CSV: {list(final.columns)}")

    for col in required_cols:
        check(f"Column '{col}' exists", col in final.columns)

    extra_cols = set(final.columns) - set(required_cols)
    if extra_cols:
        print(f"\n  Extra columns (not in spec): {extra_cols}")
    
    check("Exactly 15 columns", len(final.columns) == 15, 
          f"Found {len(final.columns)}")
    check("One row per ECG", final['ecg_id'].nunique() == len(final),
          f"{final['ecg_id'].nunique()} unique IDs, {len(final)} rows")

    # Column type checks
    print(f"\n  Column types and ranges:")
    check("ecg_id is integer-like", final['ecg_id'].dtype in ['int64', 'int32', 'float64'])
    check("patient_id is numeric", pd.api.types.is_numeric_dtype(final['patient_id']))
    check("filename_hr is string", final['filename_hr'].dtype == 'object')
    check("age is numeric", pd.api.types.is_numeric_dtype(final['age']))
    check("sex is 0/1", set(final['sex'].dropna().unique()) <= {0, 1},
          f"Values: {sorted(final['sex'].dropna().unique())}")
    check("strat_fold is 1-10", set(final['strat_fold'].unique()) <= set(range(1, 11)),
          f"Values: {sorted(final['strat_fold'].unique())}")
    check("report_en is string", final['report_en'].dtype == 'object')
    check("validated is 0/1", set(final['validated'].dropna().unique()) <= {0, 1})
    
    for col in label_cols:
        check(f"{col} is binary (0/1)", set(final[col].unique()) <= {0, 1},
              f"Values: {sorted(final[col].unique())}")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 10 — EXCLUSION RULES VERIFICATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print(f"  SECTION 10 -- EXCLUSION RULES VERIFICATION")
    print(f"{'=' * 75}")

    included_ids = set(final['ecg_id'].values)
    excluded_ids = set(ptbxl['ecg_id'].values) - included_ids

    print(f"\n  Original records: {len(ptbxl):,}")
    print(f"  Included in final CSV: {len(final):,}")
    print(f"  Excluded: {len(excluded_ids):,}")

    # RULE 1 — Excluded records should have ALL codes < 100 confidence
    print(f"\n  RULE 1: Exclude records where ALL codes have confidence < 100")
    rule1_violations = 0
    for _, row in ptbxl[ptbxl['ecg_id'].isin(excluded_ids)].iterrows():
        codes = row['scp_parsed']
        has_100 = any(v >= 100 for v in codes.values())
        has_superclass_at_100 = any(
            v >= 100 and c in scp_to_super 
            for c, v in codes.items()
        )
        if has_superclass_at_100:
            rule1_violations += 1
    check("No excluded record has a diagnostic code at 100% confidence",
          rule1_violations == 0, f"Violations: {rule1_violations}")

    # RULE 2 — Excluded records should have only rhythm/form codes
    print(f"\n  RULE 2: Exclude records where ALL codes are rhythm/form")
    rule2_sample = 0
    for _, row in ptbxl[ptbxl['ecg_id'].isin(excluded_ids)].head(5).iterrows():
        codes = row['scp_parsed']
        diag_codes = [c for c in codes if c in scp_to_super and codes[c] >= 100]
        if not diag_codes:
            rule2_sample += 1
    check("Sample excluded records have no diagnostic codes at 100%",
          rule2_sample == 5 or len(excluded_ids) < 5,
          f"Checked 5 excluded records: {rule2_sample}/5 have no diagnostic codes")

    # RULE 3 — Included records should have at least one superclass
    print(f"\n  RULE 3: Keep records with >= 1 superclass at >= 100")
    num_labels = final[label_cols].sum(axis=1)
    check("All included records have >= 1 label", (num_labels >= 1).all(),
          f"Records with 0 labels: {(num_labels == 0).sum()}")

    # RULE 4 — NORM correction
    print(f"\n  RULE 4: NORM=0 if any pathological class is also 1")
    pathological = ['label_MI', 'label_STTC', 'label_CD', 'label_HYP']
    norm_with_patho = final[
        (final['label_NORM'] == 1) & 
        (final[pathological].sum(axis=1) > 0)
    ]
    check("No record has NORM=1 AND any pathology=1", len(norm_with_patho) == 0,
          f"Violations: {len(norm_with_patho)}")

    # RULE 5 — Validated ratio
    print(f"\n  RULE 5: Track validated_by_human ratio")
    val_count = final['validated'].sum()
    val_pct = val_count / len(final) * 100
    check("Validated column populated", val_count > 0,
          f"Validated: {val_count:,} ({val_pct:.1f}%), Unvalidated: {len(final)-val_count:,} ({100-val_pct:.1f}%)")
    print(f"         For methodology: \"{val_pct:.0f}% of training records were validated")
    print(f"         by a second cardiologist.\"")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 11 — SANITY CHECKS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print(f"  SECTION 11 -- ALL 5 SANITY CHECKS")
    print(f"{'=' * 75}")

    # SANITY CHECK 1 — Class counts
    print(f"\n  SANITY CHECK 1: Class counts (reasonable ranges)")
    for cls, lo, hi in [('NORM',5000,12000), ('MI',2000,7000), ('STTC',3000,7000), 
                         ('CD',3000,7000), ('HYP',1000,4000)]:
        cnt = final[f'label_{cls}'].sum()
        check(f"{cls} count ({cnt:,}) in range [{lo:,}-{hi:,}]",
              lo <= cnt <= hi)

    # SANITY CHECK 2 — Multi-label combinations
    print(f"\n  SANITY CHECK 2: Multi-label distribution")
    n_labels = final[label_cols].sum(axis=1)
    single = (n_labels == 1).sum()
    double = (n_labels == 2).sum()
    triple_plus = (n_labels >= 3).sum()
    zero = (n_labels == 0).sum()
    total = len(final)

    check(f"Single-label: {single/total*100:.1f}% (expect 75-85%)",
          75 <= single/total*100 <= 85)
    check(f"Double-label: {double/total*100:.1f}% (expect 10-25%)",
          10 <= double/total*100 <= 25)
    check(f"Triple+ label: {triple_plus/total*100:.1f}% (expect 1-8%)",
          1 <= triple_plus/total*100 <= 8)
    check(f"Zero-label: {zero} (expect 0)", zero == 0)

    # SANITY CHECK 3 — Random manual inspection (20 records)
    print(f"\n  SANITY CHECK 3: Random manual inspection (20 records)")
    np.random.seed(42)
    sample_ids = np.random.choice(final['ecg_id'].values, 20, replace=False)
    manual_ok = 0
    manual_fail = 0

    for ecg_id in sample_ids:
        # Get original scp_codes
        orig_row = ptbxl[ptbxl['ecg_id'] == ecg_id].iloc[0]
        codes = orig_row['scp_parsed']
        
        # Apply our rules: filter >= 100, map to superclass
        filtered = {c: v for c, v in codes.items() if v >= 100}
        expected_sc = set()
        for c in filtered:
            if c in scp_to_super:
                expected_sc.add(scp_to_super[c])
        
        # NORM correction
        if 'NORM' in expected_sc and expected_sc & {'MI', 'STTC', 'CD', 'HYP'}:
            expected_sc.discard('NORM')
        
        # Build expected label vector
        expected = {f'label_{sc}': 1 for sc in expected_sc}
        
        # Get actual from CSV
        final_row = final[final['ecg_id'] == ecg_id].iloc[0]
        actual = {col: int(final_row[col]) for col in label_cols}
        
        # Compare
        match = all(
            actual.get(f'label_{sc}', 0) == (1 if sc in expected_sc else 0)
            for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        )
        
        if match:
            manual_ok += 1
        else:
            manual_fail += 1
            print(f"    MISMATCH ECG {ecg_id}: raw={codes}")
            print(f"      Expected: {expected_sc}, Got: {actual}")

    check(f"Manual inspection: {manual_ok}/20 correct",
          manual_ok == 20, f"Failures: {manual_fail}")

    # SANITY CHECK 4 — Fold distribution
    print(f"\n  SANITY CHECK 4: Fold distribution (class balance across folds)")
    fold_ok = True
    print(f"  {'Fold':>6}", end="")
    for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        print(f" {cls:>7}", end="")
    print(f" {'Total':>8}")
    print(f"  {'-'*6}", end="")
    for _ in range(5):
        print(f" {'-'*7}", end="")
    print(f" {'-'*8}")

    for fold in range(1, 11):
        fold_data = final[final['strat_fold'] == fold]
        print(f"  {fold:>6}", end="")
        for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
            cnt = fold_data[f'label_{cls}'].sum()
            print(f" {cnt:>7,}", end="")
            # Check no fold has 0 for any class
            if cnt == 0:
                fold_ok = False
        print(f" {len(fold_data):>8,}")
    check("Every fold has > 0 for every class", fold_ok)

    # SANITY CHECK 5 — Data leakage
    print(f"\n  SANITY CHECK 5: No data leakage across splits")
    train_pids = set(final[final['strat_fold'] <= 8]['patient_id'].dropna())
    val_pids = set(final[final['strat_fold'] == 9]['patient_id'].dropna())
    test_pids = set(final[final['strat_fold'] == 10]['patient_id'].dropna())

    check("Train-Val: 0 shared patients", len(train_pids & val_pids) == 0,
          f"Shared: {len(train_pids & val_pids)}")
    check("Train-Test: 0 shared patients", len(train_pids & test_pids) == 0,
          f"Shared: {len(train_pids & test_pids)}")
    check("Val-Test: 0 shared patients", len(val_pids & test_pids) == 0,
          f"Shared: {len(val_pids & test_pids)}")

    # ══════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print(f"  FINAL VERIFICATION SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Total checks: {passed + failed}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    if failed == 0:
        print(f"\n  ALL CHECKS PASSED!")
        print(f"  ptbxl_labeled_final.csv is verified and ready for model training.")
    else:
        print(f"\n  WARNING: {failed} check(s) failed. Review the failures above.")
    print(f"{'=' * 75}")

if __name__ == "__main__":
    main()
