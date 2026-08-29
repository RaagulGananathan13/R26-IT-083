"""
Component 04 — unified two-stage inference engine.

Loads every frozen artefact (both learners per stage, the Stage-1 calibrator,
the Stage-2 decision layer, the text embedder) and exposes one entry point that
turns a feature matrix into a four-class decision.

Two composition rules are supported:

  cascade   Stage 1 gates.  p(ACS) < threshold -> No_ACS, otherwise the Stage-2
            subtype.  This mirrors how the system is actually deployed: a
            rule-out screen followed by a subtyping step.

  joint     The two stages are composed into a single four-class distribution
                P = [1 - p,  p*q_UA,  p*q_NSTEMI,  p*q_STEMI]
            and one Constrained Decision Layer is fitted over all four classes
            at once.  This lets the No_ACS recall trade against the three ACS
            recalls explicitly instead of being fixed by a single threshold.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, LABEL_ORDER, MODEL_DIR, SUBTYPE_ORDER, load_json


@dataclass
class ACSPredictor:
    horizon: int
    stage1_xgb: object
    stage1_lgb: object
    stage1_cal: object
    stage1_cfg: Dict
    stage2_xgb: object
    stage2_lgb: object
    stage2_cdl: object
    stage2_cfg: Dict
    features: List[str]
    joint_cdl: object | None = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, horizon: int | None = None) -> "ACSPredictor":
        import xgboost as xgb
        h = CFG.primary_horizon if horizon is None else horizon
        need = os.path.join(MODEL_DIR, f"stage1_config_H{h}.json")
        if not os.path.exists(need):
            raise FileNotFoundError(f"{need} — train Stage 1 for H={h} first")

        s1cfg = load_json(need)
        s2cfg = load_json(os.path.join(MODEL_DIR, f"stage2_config_H{h}.json"))

        m1x = xgb.XGBClassifier(); m1x.load_model(os.path.join(MODEL_DIR, f"stage1_xgb_H{h}.json"))
        m2x = xgb.XGBClassifier(); m2x.load_model(os.path.join(MODEL_DIR, f"stage2_xgb_H{h}.json"))

        jp = os.path.join(MODEL_DIR, f"joint_cdl_H{h}.joblib")
        return cls(
            horizon=h,
            stage1_xgb=m1x,
            stage1_lgb=joblib.load(os.path.join(MODEL_DIR, f"stage1_lgb_H{h}.joblib")),
            stage1_cal=joblib.load(os.path.join(MODEL_DIR, f"stage1_calibrator_H{h}.joblib")),
            stage1_cfg=s1cfg,
            stage2_xgb=m2x,
            stage2_lgb=joblib.load(os.path.join(MODEL_DIR, f"stage2_lgb_H{h}.joblib")),
            stage2_cdl=joblib.load(os.path.join(MODEL_DIR, f"stage2_cdl_H{h}.joblib")),
            stage2_cfg=s2cfg,
            features=list(s1cfg["features"]),
            joint_cdl=joblib.load(jp) if os.path.exists(jp) else None,
        )

    # ------------------------------------------------------------------
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = np.nan
        return X[self.features].astype(np.float32)

    def stage1_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(ACS)."""
        X = self._align(X)
        px = self.stage1_xgb.predict_proba(X)[:, 1]
        pl = self.stage1_lgb.predict_proba(X)[:, 1]
        kind = self.stage1_cfg.get("ensemble", "mean-blend")
        if kind == "xgboost":
            raw = px
        elif kind == "lightgbm":
            raw = pl
        elif kind == "rank-blend":
            from scipy.stats import rankdata
            raw = np.mean([rankdata(p) / len(p) for p in (px, pl)], axis=0)
        else:
            raw = (px + pl) / 2.0
        return np.clip(self.stage1_cal.predict(raw), 0.0, 1.0)

    def stage2_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(UA, NSTEMI, STEMI | ACS) — before the decision layer."""
        X = self._align(X)
        a = float(self.stage2_cfg.get("alpha_xgb", 0.5))
        return a * self.stage2_xgb.predict_proba(X) + \
            (1 - a) * self.stage2_lgb.predict_proba(X)

    # ------------------------------------------------------------------
    def four_class_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Composed four-class distribution [No_ACS, UA, NSTEMI, STEMI]."""
        p = self.stage1_proba(X)
        q = self.stage2_proba(X)
        return np.column_stack([1.0 - p, q * p[:, None]])

    def predict(self, X: pd.DataFrame, mode: str = "cascade") -> np.ndarray:
        if mode == "joint":
            if self.joint_cdl is None:
                raise RuntimeError("joint decision layer not fitted — run evaluate.py")
            return self.joint_cdl.predict(self.four_class_proba(X))
        p = self.stage1_proba(X)
        thr = float(self.stage1_cfg["threshold"])
        sub = self.stage2_cdl.predict(self.stage2_proba(X))
        return np.where(p >= thr, sub + 1, 0)

    # ------------------------------------------------------------------
    def explain_row(self, X: pd.DataFrame, i: int = 0,
                    chief_complaint: str | None = None) -> Dict:
        """Human-readable single-patient output used by the demo."""
        import text_features as TF
        P4 = self.four_class_proba(X.iloc[[i]])[0]
        pred = int(self.predict(X.iloc[[i]])[0])
        out = {
            "prediction": LABEL_ORDER[pred],
            "p_acs": float(1.0 - P4[0]),
            "probabilities": {c: float(v) for c, v in zip(LABEL_ORDER, P4)},
            "subtype_probabilities": {
                c: float(v) for c, v in
                zip(SUBTYPE_ORDER, self.stage2_proba(X.iloc[[i]])[0])},
            "horizon_h": self.horizon,
        }
        if chief_complaint:
            out["text_attribution"] = TF.token_attribution(chief_complaint)
        return out


def load_predictor(horizon: int | None = None) -> ACSPredictor:
    return ACSPredictor.load(horizon)
