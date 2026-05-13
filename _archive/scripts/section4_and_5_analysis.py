"""
PTB-XL Sections 4 & 5 — Confidence Threshold Analysis & Multi-Label Structure
Streams from PhysioNet. Compares thresholds, validates multi-label encoding.
"""
import pandas as pd
import numpy as np
import ast
import io
import json
import requests
import os
from collections import Counter

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

    print("Streaming data from PhysioNet...")
    ptbxl = stream_csv(session, "ptbxl_database.csv")
    scp_df = stream_csv(session, "scp_statements.csv")

    # Build SCP -> superclass mapping
    scp_code_col = scp_df.columns[0]
    scp_df = scp_df.rename(columns={scp_code_col: 'scp_code'})
    diag_df = scp_df[scp_df['diagnostic'] == 1.0]
    scp_to_super = {}
    for _, row in diag_df.iterrows():
        sc = row.get('diagnostic_class', np.nan)
        if pd.notna(sc):
            scp_to_super[row['scp_code']] = sc

    # Parse scp_codes
    ptbxl['scp_parsed'] = ptbxl['scp_codes'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 4 — CONFIDENCE SCORES AND THRESHOLD ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SECTION 4 -- CONFIDENCE SCORES AND THRESHOLD DECISION")
    print("=" * 80)

    # ── 4.1: Overall confidence distribution ──
    print("\n[4.1] Overall confidence distribution across ALL code entries")
    print("-" * 60)
    all_entries = []
    for _, row in ptbxl.iterrows():
        for code, conf in row['scp_parsed'].items():
            is_diag = code in scp_to_super
            superclass = scp_to_super.get(code, 'N/A')
            all_entries.append({
                'ecg_id': row['ecg_id'],
                'code': code,
                'confidence': conf,
                'is_diagnostic': is_diag,
                'superclass': superclass
            })
    entries_df = pd.DataFrame(all_entries)
    print(f"  Total code-confidence entries: {len(entries_df):,}")
    print(f"  Diagnostic entries: {entries_df['is_diagnostic'].sum():,}")
    print(f"  Non-diagnostic entries: {(~entries_df['is_diagnostic']).sum():,}")

    print(f"\n  Confidence value distribution (ALL codes):")
    print(f"  {'Confidence':>12}  {'Count':>8}  {'%':>7}  {'Cumulative %':>12}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*12}")
    conf_counts = entries_df['confidence'].value_counts().sort_index()
    cumulative = 0
    for val, cnt in conf_counts.items():
        pct = cnt / len(entries_df) * 100
        cumulative += pct
        print(f"  {val:>11.1f}%  {cnt:>8,}  {pct:>6.1f}%  {cumulative:>11.1f}%")

    # ── 4.2: Confidence distribution for DIAGNOSTIC codes only ──
    print(f"\n[4.2] Confidence distribution for DIAGNOSTIC codes only")
    print("-" * 60)
    diag_entries = entries_df[entries_df['is_diagnostic']]
    print(f"  Total diagnostic entries: {len(diag_entries):,}")
    print(f"\n  {'Confidence':>12}  {'Count':>8}  {'%':>7}  Bar")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*30}")
    diag_conf = diag_entries['confidence'].value_counts().sort_index()
    max_cnt = diag_conf.max()
    for val, cnt in diag_conf.items():
        pct = cnt / len(diag_entries) * 100
        bar = "#" * int(cnt / max_cnt * 30)
        print(f"  {val:>11.1f}%  {cnt:>8,}  {pct:>6.1f}%  {bar}")

    # ── 4.3: Per-superclass confidence distribution ──
    print(f"\n[4.3] Confidence distribution PER SUPERCLASS")
    print("-" * 60)
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        sc_entries = diag_entries[diag_entries['superclass'] == sc]
        print(f"\n  {sc} ({len(sc_entries):,} entries):")
        sc_conf = sc_entries['confidence'].value_counts().sort_index()
        for val, cnt in sc_conf.items():
            pct = cnt / len(sc_entries) * 100
            print(f"    {val:>6.1f}%: {cnt:>5,} ({pct:>5.1f}%)")

    # ── 4.4: Threshold comparison ──
    print(f"\n[4.4] THRESHOLD COMPARISON: How many records at each threshold?")
    print("-" * 60)

    thresholds = [0, 15, 50, 80, 100]
    superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    print(f"\n  {'Threshold':>10}", end="")
    for sc in superclasses:
        print(f"  {sc:>7}", end="")
    print(f"  {'Total':>8}  {'Excluded':>8}")
    print(f"  {'-'*10}", end="")
    for _ in superclasses:
        print(f"  {'-'*7}", end="")
    print(f"  {'-'*8}  {'-'*8}")

    for thresh in thresholds:
        counts = {}
        records_with_label = set()
        for _, row in ptbxl.iterrows():
            pos_classes = set()
            for code, conf in row['scp_parsed'].items():
                if conf >= thresh and code in scp_to_super:
                    pos_classes.add(scp_to_super[code])
            # NORM correction
            if 'NORM' in pos_classes and pos_classes & {'MI', 'STTC', 'CD', 'HYP'}:
                pos_classes.discard('NORM')
            if pos_classes:
                records_with_label.add(row['ecg_id'])
            for sc in pos_classes:
                counts[sc] = counts.get(sc, 0) + 1

        total_labeled = len(records_with_label)
        excluded = len(ptbxl) - total_labeled
        print(f"  >= {thresh:>5.0f}%", end="")
        for sc in superclasses:
            print(f"  {counts.get(sc, 0):>7,}", end="")
        print(f"  {total_labeled:>8,}  {excluded:>8,}")

    print(f"\n  INTERPRETATION:")
    print(f"  - >= 100% (CHOSEN): Strictest. Only unanimously agreed diagnoses.")
    print(f"                      Cleanest labels. ~17,000 records. Recommended for training.")
    print(f"  - >= 50%:           Includes moderate-confidence codes. Adds ~2,000+ records")
    print(f"                      but introduces label noise from uncertain diagnoses.")
    print(f"  - >= 0%:            Includes everything including incidental findings.")
    print(f"                      Maximum records but very noisy labels.")

    # ── 4.5: What happens to records at different thresholds ──
    print(f"\n[4.5] Records GAINED/LOST at different thresholds")
    print("-" * 60)

    # Records only available at lower thresholds
    def get_labeled_ids(thresh):
        ids = set()
        for _, row in ptbxl.iterrows():
            for code, conf in row['scp_parsed'].items():
                if conf >= thresh and code in scp_to_super:
                    ids.add(row['ecg_id'])
                    break
        return ids

    ids_100 = get_labeled_ids(100)
    ids_80 = get_labeled_ids(80)
    ids_50 = get_labeled_ids(50)

    print(f"  Records labeled at >= 100:  {len(ids_100):,}")
    print(f"  Records labeled at >= 80:   {len(ids_80):,}  (+{len(ids_80 - ids_100):,} gained)")
    print(f"  Records labeled at >= 50:   {len(ids_50):,}  (+{len(ids_50 - ids_100):,} gained)")

    # Show examples of records that are ONLY available at lower thresholds
    gained_at_80 = ids_80 - ids_100
    if gained_at_80:
        print(f"\n  Sample records that exist only at >= 80 threshold:")
        sample_ids = list(gained_at_80)[:5]
        for eid in sample_ids:
            row = ptbxl[ptbxl['ecg_id'] == eid].iloc[0]
            print(f"    ECG {eid}: {row['scp_codes']}")

    # ── 4.6: Dual threshold strategy for evaluation ──
    print(f"\n[4.6] DUAL THRESHOLD STRATEGY (for methodology)")
    print("-" * 60)
    print(f"  TRAINING:   Use >= 100% confidence (clean, unanimous labels)")
    print(f"  EVALUATION: Report metrics at BOTH >= 100% AND >= 50% thresholds")
    print(f"  This allows comparison with papers that use different thresholds.")
    print(f"\n  To implement in your thesis:")
    print(f"  1. Train on >= 100% labels (17,221 records)")
    print(f"  2. At test time (fold 10), compute predictions for ALL records")
    print(f"  3. Report AUROC/F1 against >= 100% labels (primary metric)")
    print(f"  4. Also report AUROC/F1 against >= 50% labels (secondary metric)")
    print(f"  5. Discuss the difference in your results section")

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 5 — MULTI-LABEL STRUCTURE VERIFICATION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SECTION 5 -- MULTI-LABEL STRUCTURE: BINARY VECTORS")
    print("=" * 80)

    # Load the labeled CSV from Section 1
    labeled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptbxl_labeled_final.csv")
    labeled = pd.read_csv(labeled_path)

    label_cols = ['label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']

    # ── 5.1: Verify binary encoding ──
    print(f"\n[5.1] Binary encoding verification")
    print("-" * 60)
    for col in label_cols:
        unique_vals = sorted(labeled[col].unique())
        print(f"  {col}: unique values = {unique_vals}  (should be [0, 1])")
    all_binary = all(set(labeled[col].unique()) <= {0, 1} for col in label_cols)
    print(f"\n  All columns are strictly binary (0 or 1): {'YES' if all_binary else 'NO -- ERROR!'}")

    # ── 5.2: Label vector examples ──
    print(f"\n[5.2] Sample label vectors [NORM, MI, STTC, CD, HYP]")
    print("-" * 60)

    # Find examples of each common pattern
    patterns = {}
    for _, row in labeled.iterrows():
        vec = tuple(row[col] for col in label_cols)
        if vec not in patterns:
            patterns[vec] = row
    
    # Show the most common patterns
    labeled['label_vec'] = labeled.apply(lambda r: tuple(r[col] for col in label_cols), axis=1)
    vec_counts = labeled['label_vec'].value_counts().head(15)
    
    print(f"  {'Label Vector':<25} {'Count':>8}  {'%':>7}  Meaning")
    print(f"  {'-'*25} {'-'*8}  {'-'*7}  {'-'*35}")
    
    meaning_map = {
        (1,0,0,0,0): "Normal ECG only",
        (0,1,0,0,0): "MI only",
        (0,0,1,0,0): "STTC only",
        (0,0,0,1,0): "CD only",
        (0,0,0,0,1): "HYP only",
        (0,1,1,0,0): "MI + STTC",
        (0,0,1,1,0): "STTC + CD",
        (0,0,0,1,1): "CD + HYP",
        (0,0,1,0,1): "STTC + HYP",
        (0,1,0,1,0): "MI + CD",
        (0,1,1,0,1): "MI + STTC + HYP",
        (0,1,1,1,0): "MI + STTC + CD",
        (0,0,1,1,1): "STTC + CD + HYP",
        (0,1,0,0,1): "MI + HYP",
        (0,1,1,1,1): "MI + STTC + CD + HYP",
    }
    
    for vec, cnt in vec_counts.items():
        pct = cnt / len(labeled) * 100
        meaning = meaning_map.get(vec, "Other combination")
        print(f"  {str(list(vec)):<25} {cnt:>8,}  {pct:>6.1f}%  {meaning}")

    # ── 5.3: Multi-label statistics ──
    print(f"\n[5.3] Multi-label statistics")
    print("-" * 60)
    labeled['num_labels'] = sum(labeled[col] for col in label_cols)

    print(f"  {'# Labels':>10}  {'Count':>8}  {'%':>7}  Bar")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*40}")
    max_n = labeled['num_labels'].value_counts().max()
    for n in sorted(labeled['num_labels'].unique()):
        cnt = (labeled['num_labels'] == n).sum()
        pct = cnt / len(labeled) * 100
        bar = "#" * int(cnt / max_n * 40)
        print(f"  {n:>10}  {cnt:>8,}  {pct:>6.1f}%  {bar}")

    avg_labels = labeled['num_labels'].mean()
    print(f"\n  Average labels per record: {avg_labels:.3f}")
    print(f"  Records with 0 labels: {(labeled['num_labels'] == 0).sum()} (should be 0)")

    # ── 5.4: NORM co-occurrence analysis ──
    print(f"\n[5.4] NORM co-occurrence analysis (NORM correction verification)")
    print("-" * 60)
    norm_records = labeled[labeled['label_NORM'] == 1]
    pathological_cols = ['label_MI', 'label_STTC', 'label_CD', 'label_HYP']
    norm_with_patho = norm_records[pathological_cols].sum(axis=1) > 0

    print(f"  Total NORM=1 records: {len(norm_records):,}")
    print(f"  NORM=1 with ANY pathological class also =1: {norm_with_patho.sum()}")
    print(f"  {'VERIFIED: NORM correction applied correctly!' if norm_with_patho.sum() == 0 else 'WARNING: NORM co-occurs with pathology!'}")

    # Show what NORM records look like
    print(f"\n  NORM records label vectors (should all be [1,0,0,0,0]):")
    norm_vecs = norm_records[label_cols].drop_duplicates()
    for _, row in norm_vecs.iterrows():
        vec = [int(row[c]) for c in label_cols]
        print(f"    {vec}")

    # ── 5.5: Class co-occurrence matrix ──
    print(f"\n[5.5] Class co-occurrence matrix")
    print("-" * 60)
    print(f"  How often each pair of classes appears together:\n")
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print(f"          ", end="")
    for name in class_names:
        print(f" {name:>7}", end="")
    print()
    print(f"          ", end="")
    for _ in class_names:
        print(f" {'-'*7}", end="")
    print()
    
    for i, name_i in enumerate(class_names):
        print(f"  {name_i:>6}  ", end="")
        for j, name_j in enumerate(class_names):
            col_i = f'label_{name_i}'
            col_j = f'label_{name_j}'
            co_count = ((labeled[col_i] == 1) & (labeled[col_j] == 1)).sum()
            print(f" {co_count:>7,}", end="")
        print()

    # ── 5.6: Unique label combinations ──
    print(f"\n[5.6] All unique label combinations found in the dataset")
    print("-" * 60)
    all_combos = labeled['label_vec'].value_counts()
    print(f"  Total unique combinations: {len(all_combos)}")
    print(f"\n  {'#':>3}  {'Vector':<25}  {'Count':>8}  {'%':>7}  Active classes")
    print(f"  {'-'*3}  {'-'*25}  {'-'*8}  {'-'*7}  {'-'*25}")
    for idx, (vec, cnt) in enumerate(all_combos.items(), 1):
        pct = cnt / len(labeled) * 100
        active = [class_names[i] for i, v in enumerate(vec) if v == 1]
        active_str = ", ".join(active) if active else "(none)"
        print(f"  {idx:>3}  {str(list(vec)):<25}  {cnt:>8,}  {pct:>6.1f}%  {active_str}")

    # ── 5.7: Verify against guide expectations ──
    print(f"\n[5.7] Verification against guide expectations")
    print("-" * 60)
    
    single = (labeled['num_labels'] == 1).sum()
    double = (labeled['num_labels'] == 2).sum()
    triple_plus = (labeled['num_labels'] >= 3).sum()
    total = len(labeled)
    
    checks = [
        ("Single-label records", single/total*100, "75-80%", 75, 85),
        ("Double-label records", double/total*100, "15-20%", 10, 25),
        ("Triple+ label records", triple_plus/total*100, "2-5%", 1, 8),
        ("Zero-label records", (labeled['num_labels']==0).sum()/total*100, "0%", 0, 0.1),
    ]
    
    for name, actual, expected, lo, hi in checks:
        status = "PASS" if lo <= actual <= hi else "WARN"
        print(f"  [{status}] {name}: {actual:.1f}% (expected {expected})")

    # ── 5.8: Practical example walkthrough ──
    print(f"\n[5.8] PRACTICAL WALKTHROUGH: Raw codes -> Label vector")
    print("-" * 60)
    
    # Find diverse examples
    examples = []
    # Pure NORM
    for _, row in ptbxl.iterrows():
        codes = row['scp_parsed']
        high_conf = {c: v for c, v in codes.items() if v >= 100}
        mapped = {scp_to_super[c] for c in high_conf if c in scp_to_super}
        if mapped == {'NORM'} and len(examples) < 1:
            examples.append((row, high_conf, mapped, 'Pure NORM'))
        elif 'MI' in mapped and 'STTC' in mapped and len(mapped) == 2 and len(examples) < 2:
            examples.append((row, high_conf, mapped, 'MI + STTC co-occurrence'))
        elif 'CD' in mapped and 'HYP' in mapped and len(examples) < 3:
            examples.append((row, high_conf, mapped, 'CD + HYP co-occurrence'))
        elif len(mapped) >= 3 and len(examples) < 4:
            examples.append((row, high_conf, mapped, f'{len(mapped)}-class co-occurrence'))
        if len(examples) >= 4:
            break

    for row, high_conf, mapped, desc in examples:
        print(f"\n  Example: {desc}")
        print(f"  ECG ID: {row['ecg_id']}")
        print(f"  Raw scp_codes: {row['scp_codes']}")
        print(f"  After >= 100 filter: {high_conf}")
        print(f"  Mapped superclasses: {mapped}")
        
        # Apply NORM correction
        final = mapped.copy()
        if 'NORM' in final and final & {'MI', 'STTC', 'CD', 'HYP'}:
            final.discard('NORM')
            print(f"  After NORM correction: {final}")
        
        vec = [1 if sc in final else 0 for sc in class_names]
        print(f"  Final label vector: {vec}")
        print(f"  -> [NORM={vec[0]}, MI={vec[1]}, STTC={vec[2]}, CD={vec[3]}, HYP={vec[4]}]")

    print(f"\n{'=' * 80}")
    print(f"  SECTIONS 4 & 5 COMPLETE")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
