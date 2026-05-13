"""
PTB-XL Section 8 — German Report Translation (with resume support)
Saves progress every 100 records. Restart safely — already translated records are skipped.
"""
import pandas as pd
import numpy as np
import io
import os
import time
import json
import requests
from deep_translator import GoogleTranslator
from tqdm import tqdm

USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

PROGRESS_FILE = os.path.join(WORK_DIR, "translation_progress.json")
LABELED_PATH = os.path.join(WORK_DIR, "ptbxl_labeled_final.csv")

def stream_csv(session, filename):
    resp = session.get(BASE_URL + filename)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

def load_progress():
    """Load previously translated records."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(translations):
    """Save translations to progress file."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False)

def main():
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # Load labeled CSV
    labeled = pd.read_csv(LABELED_PATH)
    print(f"Loaded {len(labeled):,} records from ptbxl_labeled_final.csv")

    # Stream original German reports
    print("Streaming original German reports from PhysioNet...")
    ptbxl_orig = stream_csv(session, "ptbxl_database.csv")
    reports_map = dict(zip(ptbxl_orig['ecg_id'], ptbxl_orig['report'].fillna('')))
    print(f"Loaded {len(reports_map):,} original reports")

    # Load existing progress
    translations = load_progress()
    already_done = len(translations)
    print(f"Already translated: {already_done:,} records")

    # Identify records that still need translation
    all_ecg_ids = labeled['ecg_id'].tolist()
    to_translate = [(eid, reports_map.get(eid, '')) for eid in all_ecg_ids 
                    if str(eid) not in translations]
    
    print(f"Remaining to translate: {len(to_translate):,} records")

    if len(to_translate) == 0:
        print("\nAll records already translated!")
    else:
        translator = GoogleTranslator(source='de', target='en')
        errors = 0
        save_interval = 100  # Save every 100 records

        with tqdm(total=len(to_translate), desc="Translating", unit="rec",
                  bar_format='{l_bar}{bar:40}{r_bar}',
                  initial=0) as pbar:
            for i, (ecg_id, german_text) in enumerate(to_translate):
                if not german_text or str(german_text).strip() == '' or german_text == 'nan':
                    translations[str(ecg_id)] = ''
                else:
                    try:
                        result = translator.translate(str(german_text))
                        translations[str(ecg_id)] = result if result else ''
                    except Exception as e:
                        translations[str(ecg_id)] = ''
                        errors += 1
                        # If rate limited, wait and retry
                        if 'Too Many Requests' in str(e) or '429' in str(e):
                            time.sleep(2)
                            try:
                                result = translator.translate(str(german_text))
                                translations[str(ecg_id)] = result if result else ''
                                errors -= 1  # Recovered
                            except:
                                pass

                pbar.update(1)
                pbar.set_postfix(done=already_done + i + 1, errors=errors)

                # Save progress periodically
                if (i + 1) % save_interval == 0:
                    save_progress(translations)

        # Final save
        save_progress(translations)

    # Apply translations to the labeled CSV
    print(f"\nApplying {len(translations):,} translations to CSV...")
    labeled['report_en'] = labeled['ecg_id'].apply(
        lambda eid: translations.get(str(eid), '')
    )

    # Statistics
    non_empty = (labeled['report_en'].notna() & (labeled['report_en'] != '')).sum()
    print(f"Non-empty English reports: {non_empty:,}")
    print(f"Empty reports: {len(labeled) - non_empty:,}")

    # Show samples
    print(f"\nSample translations:")
    samples = labeled[labeled['report_en'] != ''].head(5)
    for _, row in samples.iterrows():
        de = reports_map.get(row['ecg_id'], '')[:60]
        en = str(row['report_en'])[:60]
        print(f"  ECG {row['ecg_id']:>5}:")
        print(f"    DE: \"{de}\"")
        print(f"    EN: \"{en}\"")

    # Save updated CSV
    final_cols = [
        'ecg_id', 'patient_id', 'filename_hr',
        'age', 'sex', 'height', 'weight',
        'report_en', 'strat_fold',
        'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP',
        'validated'
    ]
    final_cols = [c for c in final_cols if c in labeled.columns]
    labeled[final_cols].to_csv(LABELED_PATH, index=False)
    print(f"\nSaved to: {LABELED_PATH}")
    print("DONE!")

if __name__ == "__main__":
    main()
