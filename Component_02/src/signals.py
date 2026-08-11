"""
signals.py — Load a test-set ECG by id, wherever the data happens to live.

Two sources, tried in order:

  1. signals_cache/<ecg_id>.npy   — the derived float32 cache (240 KB/record,
                                    3.9 GB total). Fastest, but bulky.
  2. raw_signals/<filename_hr>    — the ORIGINAL PTB-XL WFDB record
                                    (int16, 120 KB/record, 1.93 GB total).

Shipping (2) rather than (1) halves the deliverable and hands on the authoritative
data rather than a derived artefact. Reading a WFDB record costs ~4 ms, which is
negligible against the ~6 s an analysis takes.

PTB-XL is distributed under CC-BY 4.0, so redistribution is permitted with
attribution — see Component_02/data/DATASET_LICENSE.txt.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import paths


def raw_signals_dir() -> Optional[str]:
    for base in paths._candidates():
        p = os.path.join(base, "raw_signals")
        if os.path.isdir(p):
            return p
    p = os.path.join(paths.ROOT, "_archive", "data", "raw_signals")
    return p if os.path.isdir(p) else None


def available() -> bool:
    return paths.signals_cache() is not None or raw_signals_dir() is not None


def source_description() -> str:
    if paths.signals_cache():
        return f"cache ({paths.signals_cache()})"
    d = raw_signals_dir()
    return f"raw WFDB ({d})" if d else "NONE"


def load(ecg_id: int, filename_hr: Optional[str] = None) -> np.ndarray:
    """Return the raw 12-lead signal in millivolts, shape (n_samples, 12).

    `filename_hr` is the PTB-XL relative path (e.g. 'records500/00000/00271_hr')
    taken from the metadata CSV. It is required for the raw-WFDB path.
    """
    cache = paths.signals_cache()
    if cache:
        p = os.path.join(cache, f"{int(ecg_id)}.npy")
        if os.path.exists(p):
            return np.load(p).astype(np.float32)

    root = raw_signals_dir()
    if root and filename_hr:
        base = os.path.join(root, *str(filename_hr).split("/"))
        if os.path.exists(base + ".hea"):
            import wfdb
            rec = wfdb.rdrecord(base)
            return np.asarray(rec.p_signal, dtype=np.float32)

    raise FileNotFoundError(
        f"No signal data for ECG {ecg_id}. Provide either signals_cache/ or "
        f"raw_signals/ (see Component_02/README.md).")
