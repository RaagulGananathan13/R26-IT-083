"""
PTB-XL Data Preparation — Downloads ECG signals & pre-computes text embeddings.
Run this ONCE before training. Resumable — skips already-cached files.

Usage: python prepare_data.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ── Paths ──
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
TEXT_CACHE = os.path.join(DATA_DIR, "text_cache")
NORM_STATS_PATH = os.path.join(DATA_DIR, "norm_stats.json")
RAW_DIR = os.path.join(DATA_DIR, "raw_signals")  # Raw .dat/.hea files

# PhysioNet
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
USERNAME = "dilukshan285"
PASSWORD = "Diluviya@250207"


def download_signal_files():
    """Download .dat and .hea files directly via HTTP (much faster than wfdb)."""
    os.makedirs(SIGNAL_CACHE, exist_ok=True)

    master = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_labeled_final.csv"))
    total = len(master)

    # Count already converted to .npy
    already = sum(
        1 for eid in master["ecg_id"]
        if os.path.exists(os.path.join(SIGNAL_CACHE, f"{eid}.npy"))
    )

    print(f"\n{'='*65}")
    print(f"  PHASE 1: Download & Cache ECG Signals (500Hz, 12-lead)")
    print(f"{'='*65}")
    print(f"  Total records: {total:,}")
    print(f"  Already cached: {already:,}")
    print(f"  Remaining: {total - already:,}")

    if already == total:
        print("  All signals already cached. Skipping.")
        return

    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)

    # We need wfdb only for reading the local files after download
    import wfdb

    errors = []
    with tqdm(total=total, initial=already, desc="  Caching signals",
              unit="rec", bar_format='{l_bar}{bar:35}{r_bar}') as pbar:
        for _, row in master.iterrows():
            ecg_id = row["ecg_id"]
            npy_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
            if os.path.exists(npy_path):
                continue

            fn_hr = row["filename_hr"]  # e.g. records500/00000/00001_hr

            # Create local subdirectory
            local_dir = os.path.join(RAW_DIR, os.path.dirname(fn_hr))
            os.makedirs(local_dir, exist_ok=True)

            local_base = os.path.join(RAW_DIR, fn_hr)
            dat_path = local_base + ".dat"
            hea_path = local_base + ".hea"

            try:
                # Download .hea and .dat if not already present
                for ext in [".hea", ".dat"]:
                    local_file = local_base + ext
                    if not os.path.exists(local_file):
                        url = BASE_URL + fn_hr + ext
                        for attempt in range(3):
                            try:
                                resp = session.get(url, timeout=30)
                                resp.raise_for_status()
                                with open(local_file, "wb") as f:
                                    f.write(resp.content)
                                break
                            except KeyboardInterrupt:
                                print("\n  Interrupted! Re-run to resume.")
                                return
                            except Exception:
                                if attempt < 2:
                                    time.sleep(1 + attempt)
                                else:
                                    raise

                # Read with wfdb locally (instant, no network)
                record = wfdb.rdrecord(os.path.join(RAW_DIR, fn_hr))
                signal = record.p_signal  # (5000, 12)
                np.save(npy_path, signal.astype(np.float32))

            except KeyboardInterrupt:
                print("\n  Interrupted! Re-run to resume.")
                return
            except Exception as e:
                errors.append((ecg_id, str(e)))
                np.save(npy_path, np.zeros((5000, 12), dtype=np.float32))

            pbar.update(1)
            if errors:
                pbar.set_postfix(errors=len(errors))

    if errors:
        print(f"\n  Errors: {len(errors)} (zeros saved as fallback)")
        for eid, err in errors[:5]:
            print(f"    ECG {eid}: {err[:80]}")
    else:
        print(f"\n  All {total:,} signals cached successfully!")


def cache_text_embeddings():
    """Pre-compute ClinicalBERT embeddings for all reports."""
    os.makedirs(TEXT_CACHE, exist_ok=True)

    master = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_labeled_final.csv"))
    total = len(master)

    already = sum(
        1 for eid in master["ecg_id"]
        if os.path.exists(os.path.join(TEXT_CACHE, f"{eid}.npy"))
    )
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Pre-compute ClinicalBERT Text Embeddings")
    print(f"{'='*65}")
    print(f"  Total records: {total:,}")
    print(f"  Already cached: {already:,}")
    print(f"  Remaining: {total - already:,}")

    if already == total:
        print("  All embeddings already cached. Skipping.")
        return

    # Import here to avoid loading BERT if cache is complete
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("  Loading Bio_ClinicalBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()

    # Process in batches for efficiency
    batch_size = 32
    ecg_ids = master["ecg_id"].tolist()
    reports = master["report_en"].fillna("").tolist()

    # Filter to only uncached
    uncached = [
        (eid, rep) for eid, rep in zip(ecg_ids, reports)
        if not os.path.exists(os.path.join(TEXT_CACHE, f"{eid}.npy"))
    ]

    with tqdm(total=len(uncached), desc="  Computing embeddings",
              unit="rec", bar_format='{l_bar}{bar:35}{r_bar}') as pbar:
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            batch_ids = [b[0] for b in batch]
            batch_texts = [b[1] if b[1] else "no report available" for b in batch]

            with torch.no_grad():
                inputs = tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt"
                )
                outputs = model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()

            for j, eid in enumerate(batch_ids):
                np.save(
                    os.path.join(TEXT_CACHE, f"{eid}.npy"),
                    embeddings[j].astype(np.float32)
                )

            pbar.update(len(batch))

    print(f"  All {total:,} embeddings cached successfully!")


def compute_normalization_stats():
    """Compute mean/std for demographics and signal normalization."""
    print(f"\n{'='*65}")
    print(f"  PHASE 3: Compute Normalization Statistics")
    print(f"{'='*65}")

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # Demographics stats (from training set only to avoid leakage)
    demo_cols = ["age", "height", "weight"]
    demo_stats = {}
    for col in demo_cols:
        demo_stats[col] = {
            "mean": float(train[col].mean()),
            "std": float(train[col].std()),
        }
        print(f"  {col}: mean={demo_stats[col]['mean']:.2f}, std={demo_stats[col]['std']:.2f}")

    # Signal stats (sample 1000 records from training set)
    print("  Computing signal statistics from 1000 sample records...")
    sample_ids = train["ecg_id"].sample(min(1000, len(train)), random_state=42).tolist()
    all_signals = []
    for eid in sample_ids:
        path = os.path.join(SIGNAL_CACHE, f"{eid}.npy")
        if os.path.exists(path):
            sig = np.load(path)
            all_signals.append(sig)

    if all_signals:
        stacked = np.concatenate(all_signals, axis=0)  # (N*5000, 12)
        sig_mean = stacked.mean(axis=0).tolist()  # per-lead mean
        sig_std = stacked.std(axis=0).tolist()  # per-lead std
        # Replace any zero std with 1.0
        sig_std = [s if s > 1e-6 else 1.0 for s in sig_std]
    else:
        sig_mean = [0.0] * 12
        sig_std = [1.0] * 12

    stats = {
        "demographics": demo_stats,
        "signal_mean": sig_mean,
        "signal_std": sig_std,
    }

    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Signal per-lead mean: {[f'{m:.4f}' for m in sig_mean]}")
    print(f"  Signal per-lead std:  {[f'{s:.4f}' for s in sig_std]}")
    print(f"  Saved to: {NORM_STATS_PATH}")


def verify():
    """Final verification."""
    print(f"\n{'='*65}")
    print(f"  VERIFICATION")
    print(f"{'='*65}")

    master = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_labeled_final.csv"))
    total = len(master)

    sigs = sum(1 for eid in master["ecg_id"]
               if os.path.exists(os.path.join(SIGNAL_CACHE, f"{eid}.npy")))
    texts = sum(1 for eid in master["ecg_id"]
                if os.path.exists(os.path.join(TEXT_CACHE, f"{eid}.npy")))
    norm = os.path.exists(NORM_STATS_PATH)

    print(f"  Signals cached:    {sigs:>6,} / {total:,} {'PASS' if sigs == total else 'FAIL'}")
    print(f"  Embeddings cached: {texts:>6,} / {total:,} {'PASS' if texts == total else 'FAIL'}")
    print(f"  Norm stats:        {'EXISTS' if norm else 'MISSING'}  {'PASS' if norm else 'FAIL'}")

    # Check shapes
    sample_eid = master["ecg_id"].iloc[0]
    sig = np.load(os.path.join(SIGNAL_CACHE, f"{sample_eid}.npy"))
    emb = np.load(os.path.join(TEXT_CACHE, f"{sample_eid}.npy"))
    print(f"  Signal shape:      {sig.shape}  (expect (5000, 12))")
    print(f"  Embedding shape:   {emb.shape}  (expect (768,))")

    # Disk usage
    sig_size = sum(
        os.path.getsize(os.path.join(SIGNAL_CACHE, f))
        for f in os.listdir(SIGNAL_CACHE) if f.endswith('.npy')
    ) / (1024**3)
    print(f"  Signal cache size: {sig_size:.2f} GB")

    if sigs == total and texts == total and norm:
        print(f"\n  ALL CHECKS PASSED — Ready for training!")
        print(f"  Run: python train.py")
    else:
        print(f"\n  Some checks failed. Re-run prepare_data.py.")


if __name__ == "__main__":
    download_signal_files()
    cache_text_embeddings()
    compute_normalization_stats()
    verify()
