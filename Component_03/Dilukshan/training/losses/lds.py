"""
Label Distribution Smoothing (LDS) for imbalanced REGRESSION.
Yang et al., "Delving into Deep Imbalanced Regression", ICML 2021.

The empirical EF label density is convolved with a Gaussian kernel to obtain an
"effective" density; each sample is weighted by the inverse effective density so
rare EF ranges (very low / very high) contribute comparably to the loss -> lower
worst-region MAE, which directly helps minority-class boundaries.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve1d


def _gaussian_kernel(ks: int, sigma: float) -> np.ndarray:
    half = (ks - 1) // 2
    x = np.arange(-half, half + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def compute_lds_weights(ef, bins: int = 100, vmin: float = 0.0, vmax: float = 100.0,
                        ks: int = 5, sigma: float = 2.0, cap: float = 10.0) -> np.ndarray:
    ef = np.asarray(ef, dtype=np.float64)
    hist, edges = np.histogram(ef, bins=bins, range=(vmin, vmax))
    kernel = _gaussian_kernel(ks, sigma)
    smoothed = convolve1d(hist.astype(np.float64), kernel, mode="reflect")
    smoothed = np.maximum(smoothed, 1e-6)
    bin_idx = np.clip(np.digitize(ef, edges) - 1, 0, bins - 1)
    w = 1.0 / smoothed[bin_idx]
    w = w / w.mean()
    return np.clip(w, None, cap).astype(np.float32)
