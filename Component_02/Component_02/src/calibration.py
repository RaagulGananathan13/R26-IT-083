"""
calibration.py — Post-hoc probability calibration.

Fixes audit finding C-6. Measured miscalibration of the shipped model:

    class   ECE     mean predicted p   true prevalence   over-prediction
    NORM    0.130   0.447              0.413             1.08x
    MI      0.156   0.298              0.157             1.90x
    STTC    0.214   0.461              0.267             1.73x
    CD      0.176   0.403              0.282             1.43x
    HYP     0.242   0.320              0.077             4.14x

Root cause: WeightedRandomSampler oversampling AND focal-loss alpha weights were
applied together, correcting class imbalance twice. The sigmoid outputs are not
posterior probabilities, yet the UI printed them to a clinician as "probability %".

Per-class temperature scaling (Guo et al., ICML 2017) is fitted on the
VALIDATION fold only. It is monotone, so AUROC and the conformal guarantees in
conformal.py are unchanged — only the numbers shown to the human become honest.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .models import CLASS_NAMES


class TemperatureCalibrator:
    """One temperature + bias per class, fitted by minimising NLL on validation."""

    def __init__(self, class_names: List[str] | None = None):
        self.class_names = list(class_names or CLASS_NAMES)
        self.temperature = np.ones(len(self.class_names), dtype=np.float64)
        self.bias = np.zeros(len(self.class_names), dtype=np.float64)
        self.fitted = False
        self.fitted_for: Dict = {}

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            max_iter: int = 300, lr: float = 0.02) -> "TemperatureCalibrator":
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        for k in range(len(self.class_names)):
            z = torch.tensor(logits[:, k], dtype=torch.float64)
            y = torch.tensor(labels[:, k], dtype=torch.float64)
            log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([log_t, b], lr=lr, max_iter=max_iter)

            def closure():
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(z / torch.exp(log_t) + b, y)
                loss.backward()
                return loss

            opt.step(closure)
            self.temperature[k] = float(torch.exp(log_t).item())
            self.bias[k] = float(b.item())
        self.fitted = True
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / self.temperature + self.bias

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = self.transform_logits(np.atleast_2d(logits))
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str, fitted_for: Dict | None = None) -> None:
        """`fitted_for` records WHICH model this calibrator belongs to.

        A calibrator is only valid for the exact model whose logits it was fitted
        on. Pairing it with a different model silently produces nonsense — it
        happened twice while building this system. The provenance is written here
        and checked by ECGPipeline at load time.
        """
        with open(path, "w") as f:
            json.dump({"class_names": self.class_names,
                       "temperature": self.temperature.tolist(),
                       "bias": self.bias.tolist(), "fitted": self.fitted,
                       "fitted_for": fitted_for or {}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TemperatureCalibrator":
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["class_names"])
        obj.temperature = np.asarray(d["temperature"], dtype=np.float64)
        obj.bias = np.asarray(d["bias"], dtype=np.float64)
        obj.fitted = bool(d.get("fitted", True))
        obj.fitted_for = d.get("fitted_for", {})
        return obj


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(ece)


def calibration_report(probs: np.ndarray, labels: np.ndarray,
                       class_names: List[str] | None = None) -> Dict[str, Dict]:
    class_names = list(class_names or CLASS_NAMES)
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    out: Dict[str, Dict] = {}
    eces = []
    for k, cls in enumerate(class_names):
        ece = expected_calibration_error(probs[:, k], labels[:, k])
        prev = float(labels[:, k].mean())
        mean_p = float(probs[:, k].mean())
        out[cls] = dict(ece=ece,
                        brier=float(np.mean((probs[:, k] - labels[:, k]) ** 2)),
                        mean_prob=mean_p, prevalence=prev,
                        over_prediction=mean_p / max(prev, 1e-9))
        eces.append(ece)
    out["macro_ece"] = float(np.mean(eces))
    return out
