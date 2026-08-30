"""
pipeline.py — The single inference entry point.

    quality gate -> preprocess -> classify -> calibrate -> conformal triage
                 -> XAI -> grounded report -> verify

Nothing may bypass this order. In particular the quality gate runs BEFORE the
classifier, so a refused record never produces a probability at all (audit
finding C-1: the archive gave an all-zero signal "MI 0.691").

Thread-safe: no mutable module-level state, no shared XAI object (audit C-5).
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from . import preprocess as pp
from . import quality as qc
from .calibration import TemperatureCalibrator
from .conformal import ConformalTriage, RULE_IN, REFER
from .models import CLASS_NAMES, SAMPLING_RATE, SIGNAL_LENGTH, build_model
from .report import ClinicalReport, build_report
from .verify import VerificationResult, verify_report
from .xai import Explanation, explain


@dataclass
class AnalysisResult:
    report: ClinicalReport
    verification: VerificationResult
    quality: qc.QualityReport
    probs_raw: Optional[Dict[str, float]]
    probs_calibrated: Optional[Dict[str, float]]
    zones: Optional[Dict[str, str]]
    explanations: Dict[str, Explanation]
    signal_mv: np.ndarray            # (n, 12) filtered, in mV — for plotting
    r_peaks: np.ndarray

    def to_json(self) -> Dict:
        return {
            "triage": self.report.triage,
            "headline": self.report.headline,
            "refused": self.report.refused,
            "reportText": self.report.text,
            "findings": [f.to_dict() for f in self.report.findings],
            "ruledOut": self.report.ruled_out,
            "referred": self.report.referred,
            "guarantees": self.report.guarantees,
            "limitations": self.report.limitations,
            "quality": self.quality.to_dict(),
            "probabilities": self.probs_calibrated,
            "probabilitiesUncalibrated": self.probs_raw,
            "zones": self.zones,
            "verification": self.verification.to_dict(),
            "leadAttribution": ({c: e.lead_signed for c, e in self.explanations.items()}
                                or None),
            "territory": ({c: e.territory for c, e in self.explanations.items()
                           if e.territory} or None),
        }


class ECGPipeline:
    def __init__(self, model, norm_mean, norm_std,
                 calibrator: Optional[TemperatureCalibrator] = None,
                 triage: Optional[ConformalTriage] = None,
                 class_names: Optional[List[str]] = None,
                 device: str = "cpu", do_filter: bool = True):
        self.model = model.eval().to(device)
        self.device = device
        self.mean = np.asarray(norm_mean, dtype=np.float32)
        self.std = np.asarray(norm_std, dtype=np.float32)
        self.calibrator = calibrator
        self.triage = triage
        self.class_names = list(class_names or CLASS_NAMES)
        self.do_filter = do_filter

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, norm_stats_path: str,
                        model_name: str = "resnet",
                        calibrator_path: Optional[str] = None,
                        triage_path: Optional[str] = None,
                        device: str = "cpu", do_filter: bool = True) -> "ECGPipeline":
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(model_name)
        model.load_state_dict(state["model_state"])
        mean, std = pp.load_norm_stats(norm_stats_path)
        calib = (TemperatureCalibrator.load(calibrator_path)
                 if calibrator_path and os.path.exists(calibrator_path) else None)
        tri = (ConformalTriage.load(triage_path)
               if triage_path and os.path.exists(triage_path) else None)

        # PROVENANCE CHECK. A calibrator and a set of conformal thresholds are
        # only valid for the exact model whose logits produced them. Silently
        # pairing them with a different model destroys both the probabilities and
        # the guarantees while everything still "runs".
        for name, obj in (("calibrator", calib), ("conformal triage", tri)):
            meta = getattr(obj, "fitted_for", None) if obj is not None else None
            if not meta:
                continue
            want = (meta.get("model"), bool(meta.get("filter", False)))
            got = (model_name, bool(do_filter))
            if want != got:
                warnings.warn(
                    f"{name} was fitted for model={want[0]} filter={want[1]}, but the "
                    f"pipeline is loading model={got[0]} filter={got[1]}. Its output "
                    f"is not valid for this model — refit with "
                    f"train/fit_calibration.py.", RuntimeWarning, stacklevel=2)

        return cls(model, mean, std, calib, tri, device=device, do_filter=do_filter)

    # ── core ─────────────────────────────────────────────────────────────
    def logits(self, x_chw: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(x_chw)[None]).float().to(self.device)
        with torch.no_grad():
            return self.model(t).cpu().numpy()[0]

    def analyse(self, signal_raw: np.ndarray, fs: int = SAMPLING_RATE,
                with_xai: bool = True, lead_names: Optional[List[str]] = None
                ) -> AnalysisResult:
        # 1 ── QUALITY GATE, before anything else touches the model
        sig_mv, q = qc.assess(signal_raw, fs, lead_names)
        if not q.acceptable:
            rep = build_report({}, {}, {}, q, {}, self.class_names)
            return AnalysisResult(rep, verify_report(rep), q, None, None, None, {},
                                  sig_mv, np.array([], dtype=int))

        # 2 ── preprocess (filter, resample by rate not by length, normalise)
        x = pp.prepare(sig_mv, fs, self.mean, self.std, do_filter=self.do_filter)
        filtered = pp.center_or_pad(
            pp.bandpass(pp.resample_to(sig_mv, fs, SAMPLING_RATE)) if self.do_filter
            else pp.resample_to(sig_mv, fs, SAMPLING_RATE), SIGNAL_LENGTH)
        r_peaks = qc.detect_r_peaks(filtered[:, 1], SAMPLING_RATE)

        # 3 ── classify
        lg = self.logits(x)
        raw = 1.0 / (1.0 + np.exp(-lg))

        # 4 ── calibrate (monotone: leaves AUROC and the conformal bounds intact)
        cal = (self.calibrator.predict_proba(lg[None])[0] if self.calibrator else raw)

        # 5 ── conformal triage
        if self.triage is not None:
            zones = self.triage.zones_one(cal)
            thresholds = {c: t.to_dict() for c, t in self.triage.thresholds.items()}
        else:                       # no calibration set fitted yet -> refer everything
            zones = {c: REFER for c in self.class_names}
            thresholds = {}

        # 6 ── XAI, only for classes that are actually being reported
        explanations: Dict[str, Explanation] = {}
        if with_xai:
            t = torch.from_numpy(np.ascontiguousarray(x)[None]).float().to(self.device)
            targets = [c for c in self.class_names
                       if zones.get(c) in (RULE_IN, REFER) and c != "NORM"]
            if not targets:
                targets = [self.class_names[int(np.argmax(cal))]]
            for c in targets[:3]:               # bound the cost: 3 explanations max
                k = self.class_names.index(c)
                explanations[c] = explain(self.model, t, k, c,
                                          seconds=SIGNAL_LENGTH / SAMPLING_RATE)

        # 7 ── grounded report
        rep = build_report({c: float(cal[i]) for i, c in enumerate(self.class_names)},
                           zones, thresholds, q, explanations, self.class_names)

        # 8 ── verify before release
        ver = verify_report(rep)
        if not ver.passed:
            rep.text = ("REPORT WITHHELD — the generated text failed automated "
                        "safety verification:\n  - " + "\n  - ".join(ver.errors) +
                        "\n\nManual interpretation is required.\n\n"
                        "AI-generated decision support. NOT a medical device.")
            rep.triage = "REPEAT ECG"

        return AnalysisResult(
            report=rep, verification=ver, quality=q,
            probs_raw={c: float(raw[i]) for i, c in enumerate(self.class_names)},
            probs_calibrated={c: float(cal[i]) for i, c in enumerate(self.class_names)},
            zones=zones, explanations=explanations,
            signal_mv=filtered, r_peaks=r_peaks)

    # ── batch (for evaluation) ───────────────────────────────────────────
    def predict_logits_batch(self, signals_chw: np.ndarray, batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(signals_chw), batch):
            t = torch.from_numpy(signals_chw[i:i + batch]).float().to(self.device)
            with torch.no_grad():
                out.append(self.model(t).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.class_names)))
