"""
preprocess.py — Deterministic signal conditioning, shared by training and serving.

Fixes audit findings:
  E-5  the archive resampled ANY length to 5000 samples with no fs check,
       compressing a 30 s strip 3x and distorting heart rate silently.
  4.x  no filtering of any kind existed; per-record baseline wander and mains
       hum went straight into the network, and normalisation used global
       per-lead statistics that cannot remove a per-record DC offset.

Filter choice follows the AHA/ACC recommendation for diagnostic ECG:
  high-pass 0.5 Hz (baseline wander; preserves ST level)
  low-pass 40 Hz  (muscle noise)
  notch    50/60 Hz (mains) — applied only when fs allows it

IMPORTANT: training and inference MUST use the identical function. Import it,
do not reimplement it.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
from scipy import signal as sps

from .models import SIGNAL_LENGTH, SAMPLING_RATE

HP_HZ = 0.5
LP_HZ = 40.0
NOTCH_HZ = 50.0     # PTB-XL was recorded in Germany -> 50 Hz mains
NOTCH_Q = 30.0


def resample_to(x: np.ndarray, fs_in: int, fs_out: int = SAMPLING_RATE) -> np.ndarray:
    """Resample along time, preserving duration. Never use this to force a length."""
    if fs_in == fs_out:
        return x.astype(np.float32)
    n_out = int(round(x.shape[0] * fs_out / float(fs_in)))
    return sps.resample(x, n_out, axis=0).astype(np.float32)


def bandpass(x: np.ndarray, fs: int = SAMPLING_RATE,
             hp: float = HP_HZ, lp: float = LP_HZ, notch: bool = True) -> np.ndarray:
    """Zero-phase band-pass + optional mains notch. x is (n_samples, n_leads)."""
    nyq = fs / 2.0
    y = np.asarray(x, dtype=np.float64)
    n = y.shape[0]
    # filtfilt needs > 3 * max(len(a), len(b)) samples
    if n < 60:
        return y.astype(np.float32)

    if hp and hp / nyq < 0.99:
        b, a = sps.butter(3, hp / nyq, btype="high")
        y = sps.filtfilt(b, a, y, axis=0)
    if lp and lp / nyq < 0.99:
        b, a = sps.butter(4, lp / nyq, btype="low")
        y = sps.filtfilt(b, a, y, axis=0)
    if notch and NOTCH_HZ / nyq < 0.99:
        b, a = sps.iirnotch(NOTCH_HZ / nyq, NOTCH_Q)
        y = sps.filtfilt(b, a, y, axis=0)
    return y.astype(np.float32)


def center_or_pad(x: np.ndarray, length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Crop the centre or zero-pad to the model's fixed input length.

    Cropping/padding preserves the time scale. Resampling a 30 s strip to 10 s
    does not — that was E-5.
    """
    n = x.shape[0]
    if n == length:
        return x.astype(np.float32)
    if n > length:
        start = (n - length) // 2
        return x[start:start + length].astype(np.float32)
    pad = length - n
    lo = pad // 2
    return np.pad(x, ((lo, pad - lo), (0, 0)), mode="constant").astype(np.float32)


def normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray,
              per_record: bool = True) -> np.ndarray:
    """Normalise to the model's input space.

    per_record=True additionally removes each lead's own median (a robust DC
    offset removal). The archive used only global statistics, so a record with
    a baseline shift entered the network offset.
    """
    y = np.asarray(x, dtype=np.float32)
    if per_record:
        y = y - np.median(y, axis=0, keepdims=True)
    return ((y - mean) / std).astype(np.float32)


def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        s = json.load(f)
    return (np.asarray(s["signal_mean"], dtype=np.float32),
            np.asarray(s["signal_std"], dtype=np.float32))


def prepare(signal_mv: np.ndarray, fs: int, mean: np.ndarray, std: np.ndarray,
            do_filter: bool = True, per_record: bool = True) -> np.ndarray:
    """Full serving path: mV -> (12, 5000) float32 tensor-ready array."""
    x = resample_to(signal_mv, fs, SAMPLING_RATE)
    if do_filter:
        x = bandpass(x, SAMPLING_RATE)
    x = center_or_pad(x, SIGNAL_LENGTH)
    x = normalise(x, mean, std, per_record=per_record)
    return x.T.astype(np.float32)          # (12, 5000), channels-first


# ── training-time augmentation ───────────────────────────────────────────
def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentations applied to the NORMALISED (12, 5000) array during training.

    Deliberately excludes amplitude scaling beyond a narrow band: the audit
    showed the classifier is highly gain-sensitive (x10 -> HYP 0.998), so wide
    scale augmentation teaches it to ignore voltage criteria, which is exactly
    the evidence HYP depends on.
    """
    if rng.random() < 0.5:
        x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
    if rng.random() < 0.5:
        x = x * np.float32(rng.uniform(0.9, 1.1))
    if rng.random() < 0.3:
        x = np.roll(x, int(rng.integers(-250, 250)), axis=1)
    if rng.random() < 0.25:                                   # lead dropout
        k = int(rng.integers(1, 3))
        for i in rng.choice(12, size=k, replace=False):
            x[i] = 0.0
    if rng.random() < 0.2:                                    # baseline wander
        t = np.linspace(0, 1, x.shape[1], dtype=np.float32)
        w = rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * rng.uniform(0.15, 0.5) * t
                                            + rng.uniform(0, 6.28))
        x = x + w[None, :].astype(np.float32)
    return x.astype(np.float32)
