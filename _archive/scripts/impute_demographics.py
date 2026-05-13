"""
PTB-XL — Impute missing height & weight values.
Strategy: Median imputation + missing-value flag columns.
"""
import pandas as pd
import numpy as np

LABELED_PATH = 'ptbxl_labeled_final.csv'

def main():
    df = pd.read_csv(LABELED_PATH)
    print(f"Loaded {len(df):,} records")

    # ── Before imputation ──
    print(f"\n{'='*65}")
    print(f"  BEFORE IMPUTATION")
    print(f"{'='*65}")
    for col in ['age', 'sex', 'height', 'weight']:
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        if missing > 0:
            print(f"  {col:>8}: {missing:>6,} missing ({pct:.1f}%)  |  median={df[col].median():.1f}  mean={df[col].mean():.1f}  range=[{df[col].min():.0f}-{df[col].max():.0f}]")
        else:
            print(f"  {col:>8}: 0 missing  |  median={df[col].median():.1f}  mean={df[col].mean():.1f}  range=[{df[col].min():.0f}-{df[col].max():.0f}]")

    # ── Sex-stratified medians (more accurate imputation) ──
    print(f"\n  Sex-stratified statistics:")
    for sex_val, sex_label in [(0, 'Male'), (1, 'Female')]:
        subset = df[df['sex'] == sex_val]
        h_med = subset['height'].median()
        w_med = subset['weight'].median()
        print(f"    {sex_label}: height median={h_med:.1f} cm, weight median={w_med:.1f} kg")

    # ── Add missing-value flags ──
    print(f"\n{'='*65}")
    print(f"  IMPUTATION")
    print(f"{'='*65}")

    df['height_missing'] = df['height'].isna().astype(int)
    df['weight_missing'] = df['weight'].isna().astype(int)
    print(f"\n  Added flag columns:")
    print(f"    height_missing: {df['height_missing'].sum():,} records flagged")
    print(f"    weight_missing: {df['weight_missing'].sum():,} records flagged")

    # ── Sex-stratified median imputation ──
    print(f"\n  Applying sex-stratified median imputation...")
    for sex_val, sex_label in [(0, 'Male'), (1, 'Female')]:
        mask = df['sex'] == sex_val
        h_median = df.loc[mask, 'height'].median()
        w_median = df.loc[mask, 'weight'].median()

        h_filled = df.loc[mask, 'height'].isna().sum()
        w_filled = df.loc[mask, 'weight'].isna().sum()

        df.loc[mask, 'height'] = df.loc[mask, 'height'].fillna(h_median)
        df.loc[mask, 'weight'] = df.loc[mask, 'weight'].fillna(w_median)

        print(f"    {sex_label}: filled {h_filled:,} heights with {h_median:.1f} cm, {w_filled:,} weights with {w_median:.1f} kg")

    # Handle any remaining NaN (e.g., if sex itself were missing)
    remaining_h = df['height'].isna().sum()
    remaining_w = df['weight'].isna().sum()
    if remaining_h > 0 or remaining_w > 0:
        overall_h = df['height'].median()
        overall_w = df['weight'].median()
        df['height'] = df['height'].fillna(overall_h)
        df['weight'] = df['weight'].fillna(overall_w)
        print(f"    Fallback: filled {remaining_h} heights, {remaining_w} weights with overall median")

    # ── After imputation ──
    print(f"\n{'='*65}")
    print(f"  AFTER IMPUTATION")
    print(f"{'='*65}")
    for col in ['age', 'sex', 'height', 'weight']:
        missing = df[col].isna().sum()
        print(f"  {col:>8}: {missing} missing  |  median={df[col].median():.1f}  mean={df[col].mean():.1f}  range=[{df[col].min():.0f}-{df[col].max():.0f}]")

    print(f"\n  All demographic inputs now have ZERO missing values.")

    # ── Save updated CSV ──
    final_cols = [
        'ecg_id', 'patient_id', 'filename_hr',
        'age', 'sex', 'height', 'weight',
        'height_missing', 'weight_missing',
        'report_en', 'strat_fold',
        'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP',
        'validated'
    ]
    df[final_cols].to_csv(LABELED_PATH, index=False)

    print(f"\n{'='*65}")
    print(f"  SAVED")
    print(f"{'='*65}")
    print(f"  File: {LABELED_PATH}")
    print(f"  Columns: {len(final_cols)} (added height_missing, weight_missing)")
    print(f"  Total records: {len(df):,}")
    print(f"\n  New columns added:")
    print(f"    height_missing (0/1) -- 1 if height was imputed")
    print(f"    weight_missing (0/1) -- 1 if weight was imputed")
    print(f"\n  For your methodology:")
    print(f'  "Missing height (67.7%) and weight (56.2%) values were imputed')
    print(f'   using sex-stratified median values. Binary indicator columns')
    print(f'   (height_missing, weight_missing) were added to signal imputed')
    print(f'   values to the demographic MLP encoder."')

if __name__ == "__main__":
    main()
