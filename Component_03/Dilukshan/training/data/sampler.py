"""
Sampler construction for class-balanced / DRW training.

Phase 1 (epoch < drw_epoch): instance-balanced (uniform) sampling.
Phase 2 (epoch >= drw_epoch): class-balanced sampling via the manifest's
inverse-frequency `sample_weight` -> every EF-severity class is drawn ~equally,
the single biggest lever for minority-class recall.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler


def build_sampler(dataset, cfg, epoch: int):
    """Return a sampler appropriate for the current epoch (DRW-aware)."""
    if not cfg.use_balanced_sampler or epoch < cfg.drw_epoch:
        return RandomSampler(dataset)
    w = torch.as_tensor(dataset.sample_weight, dtype=torch.double)
    w = torch.clamp(w, min=1e-6)
    return WeightedRandomSampler(w, num_samples=len(dataset), replacement=True)


def class_counts(dataset, n_classes: int) -> np.ndarray:
    return np.array([(dataset.ef_class == c).sum() for c in range(n_classes)], dtype=np.float64)


def class_balanced_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Cui-2019 effective-number class weights, normalised to mean 1."""
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    return w / w.mean()
