"""Quick verification of the final labeled CSV after translation."""
import pandas as pd

df = pd.read_csv('ptbxl_labeled_final.csv')
print(f"Total records: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nReport translation stats:")
non_empty = (df['report_en'].notna() & (df['report_en'] != '')).sum()
empty = len(df) - non_empty
print(f"  Non-empty English reports: {non_empty:,}")
print(f"  Empty reports: {empty}")

print(f"\nLabel distribution:")
for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
    print(f"  {cls}: {df[f'label_{cls}'].sum():,}")

print(f"\nSplit sizes:")
print(f"  Train (1-8): {len(df[df['strat_fold'] <= 8]):,}")
print(f"  Val (9):     {len(df[df['strat_fold'] == 9]):,}")
print(f"  Test (10):   {len(df[df['strat_fold'] == 10]):,}")

print(f"\n10 sample English reports:")
for _, row in df[df['report_en'].notna() & (df['report_en'] != '')].sample(10, random_state=42).iterrows():
    print(f"  ECG {row['ecg_id']:>5} [{['NORM','MI','STTC','CD','HYP'][[row[f'label_{c}'] for c in ['NORM','MI','STTC','CD','HYP']].index(1)]}]: \"{str(row['report_en'])[:80]}\"")

print(f"\nMissing data:")
for col in ['age', 'sex', 'height', 'weight']:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")
