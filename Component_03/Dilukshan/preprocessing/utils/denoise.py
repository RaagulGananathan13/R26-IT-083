"""
Optional ultrasound-specific denoising.

Echocardiography is corrupted by multiplicative speckle.  These routines are
kept lightweight and OFF by default (frames are cached losslessly), but can be
baked into the cache via CFG.DENOISE, or applied on-the-fly at train time.
"""
from __future__ import annotations
import cv2
import numpy as np


def denoise_frame(frame: np.ndarray, mode: str = "none",
                  median_k: int = 3, nlm_h: float = 7.0) -> np.ndarray:
    """Denoise a single uint8 grayscale frame."""
    if mode == "none":
        return frame
    if mode == "median":
        return cv2.medianBlur(frame, median_k)
    if mode == "nlm":
        # Non-local means: strong speckle suppression, edge preserving.
        return cv2.fastNlMeansDenoising(frame, None, h=nlm_h,
                                        templateWindowSize=7, searchWindowSize=21)
    raise ValueError(f"unknown denoise mode: {mode}")


def denoise_video(vid: np.ndarray, mode: str = "none",
                  median_k: int = 3, nlm_h: float = 7.0) -> np.ndarray:
    """Apply denoise_frame across a (T,H,W) uint8 clip."""
    if mode == "none":
        return vid
    out = np.empty_like(vid)
    for t in range(vid.shape[0]):
        out[t] = denoise_frame(vid[t], mode, median_k, nlm_h)
    return out
