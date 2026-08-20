"""
Component 04 adapter -- emergency-department triage record.

Abishnan J (IT22140234). LightGBM/XGBoost over 242 temporally-bounded features
from MIMIC-IV-ED, with a unified four-class model (UM4), a constrained decision
layer and clinician referral. Patient-disjoint test fold, n = 30,452.

WHICH MODEL IS SERVED
---------------------
The component's `inference.ACSPredictor` exposes a two-stage cascade. The
component's own README makes UM4 the deployment configuration, because a
cascade compounds error -- a patient Stage 1 misses can never be recovered by
Stage 2 -- and fitting all four boundaries jointly moved STEMI recall from
58.16 % to 79.82 %. This adapter therefore serves UM4 when its artefacts are
present and falls back to the cascade when they are not, reporting which one
ran in `model.decision_rule`.

THE REFERRAL THRESHOLD
----------------------
UM4's published operating points are stated as a *coverage* (85 % or 65 %),
which is a population-level quantity: `unified4.py` keeps the most confident
fraction of a cohort. A single patient has no cohort, so the coverage is
converted once, at load, into an absolute cut-off on the top-two margin -- the
(1 - coverage) quantile of that margin over the component's persisted
**validation** scores. Validation, not test: the test fold must stay untouched,
and the component's own protocol chooses every threshold on validation.
"""
from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cvxai.adapters.base import ComponentAdapter
from cvxai.core.errors import InferenceFailed, InvalidInput
from cvxai.core.sandbox import ModuleSandbox
from cvxai.schemas.common import Actionability, Envelope, Finding, Reliability
from cvxai.schemas.triage import TriageRequest

LABEL_ORDER = ("No_ACS", "UA", "NSTEMI", "STEMI")

LABEL_DESCRIPTIONS = {
    "No_ACS": "No acute coronary syndrome",
    "UA": "Unstable angina",
    "NSTEMI": "Non-ST-elevation myocardial infarction",
    "STEMI": "ST-elevation myocardial infarction",
}

#: Risk bands and their actions, copied from the component's predict.py so the
#: served wording matches the component's own demo exactly.
RISK_BANDS = (
    (0.80, "CRITICAL", "Immediate cardiology activation. If ST-elevation: cath lab now."),
    (0.50, "HIGH", "Urgent ECG review, serial troponin, admit for monitoring."),
    (0.20, "MODERATE", "Serial troponin at 0/3h, observation, risk stratify."),
    (0.05, "LOW", "Consider alternative diagnoses; single troponin may suffice."),
    (0.00, "MINIMAL", "ACS unlikely on current evidence; pursue other causes."),
)

HORIZON_MEANING = {
    0: "Triage desk -- before any test is ordered",
    6: "ED decision point",
    24: "Workup complete",
}


class TriageAdapter(ComponentAdapter):
    id = "triage"
    name = "Temporally-Safe Explainable ACS Triage"
    owner = "Abishnan J (IT22140234)"
    modality = "Emergency-department triage record (vitals, free text, ECG report, labs)"
    task = "ACS detection and UA / NSTEMI / STEMI subtyping under a temporal contract"
    dataset = "MIMIC-IV-ED + Hosp + ECG (PhysioNet credentialed), patient-disjoint split"
    architecture = "LightGBM + XGBoost, isotonic calibration, unified four-class model, constrained decision layer"
    endpoint = "/api/v1/triage/analyze"

    def __init__(self, settings, root) -> None:
        super().__init__(settings, root)
        self._predict_module = None
        self._text_features = None
        self._predictor = None
        self._embedder = None
        self._um4_models: List[Any] = []
        self._um4_weights = None
        self._um4_coverage: Optional[float] = None
        self._um4_margin_cutoff: Optional[float] = None
        self._horizon = 24

    # ---- capability ---------------------------------------------------
    @property
    def _model_dir(self) -> Optional[Path]:
        return self.root / "artifacts" / "models" if self.root else None

    def required_paths(self) -> List[Path]:
        assert self.root is not None
        horizon = self.settings.triage_horizon
        models = self._model_dir
        return [
            self.root / "src" / "models" / "inference.py",
            self.root / "src" / "predict.py",
            models / ("stage1_config_H%d.json" % horizon),
            models / ("stage1_lgb_H%d.joblib" % horizon),
            models / ("stage1_xgb_H%d.json" % horizon),
            models / ("stage2_cdl_H%d.joblib" % horizon),
        ]

    def build_sandbox(self) -> ModuleSandbox:
        assert self.root is not None
        src = self.root / "src"
        # The component's own modules add these to sys.path themselves; doing it
        # here as well keeps the sandbox's captured set complete and makes the
        # ordering explicit rather than a side effect of import order.
        return ModuleSandbox(
            name="triage",
            roots=[self.root],
            path_entries=[src, src / "core", src / "data", src / "models", src / "analysis"],
        )

    # ---- lifecycle ----------------------------------------------------
    def _load(self) -> None:
        import joblib
        import numpy as np

        self._horizon = int(self.settings.triage_horizon)
        models = self._model_dir
        assert models is not None

        # The component reports progress to stdout on import and on every
        # featurisation. Useful once, noise per request.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            import predict as predict_module                    # type: ignore
            import text_features as text_features_module        # type: ignore
            from inference import ACSPredictor                  # type: ignore

            self.sandbox.verify_owns("config")
            self.sandbox.verify_owns("inference")

            self._predict_module = predict_module
            self._text_features = text_features_module
            self._predictor = ACSPredictor.load(self._horizon)

        embedder_path = models / ("text_embedder_H%d.joblib" % self._horizon)
        if embedder_path.exists():
            self._embedder = joblib.load(embedder_path)
        else:
            self.log.warning("text embedder missing for H=%d; the TF-IDF/SVD text "
                             "columns will be absent", self._horizon)

        self._load_um4(np, joblib, models)

    def _load_um4(self, np, joblib, models: Path) -> None:
        """Load the unified four-class model and derive its referral cut-off."""
        um4_path = models / ("um4_models_H%d.joblib" % self._horizon)
        operating_point = self.settings.triage_operating_point
        decision_path = models / ("um4_decision_%s_H%d.npz"
                                  % (operating_point, self._horizon))
        scores_path = models / ("um4_scores_H%d.npz" % self._horizon)

        if not (um4_path.exists() and decision_path.exists()):
            self.log.warning(
                "UM4 artefacts absent for H=%d (%s); serving the two-stage cascade "
                "instead", self._horizon, operating_point)
            return

        self._um4_models = joblib.load(um4_path)
        decision = np.load(decision_path)
        self._um4_weights = np.asarray(decision["w"], dtype=np.float64)
        self._um4_coverage = float(decision["coverage"])

        if not scores_path.exists():
            self.log.warning(
                "um4_scores_H%d.npz absent, so the referral cut-off cannot be derived "
                "from validation; UM4 will answer every case", self._horizon)
            return

        scores = np.load(scores_path)
        weighted = scores["P_val"] * self._um4_weights
        weighted = weighted / weighted.sum(axis=1, keepdims=True)
        ordered = np.sort(weighted, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        # Coverage c means "answer the most confident c of the population", so
        # the cut-off is the (1 - c) quantile of the confidence distribution.
        self._um4_margin_cutoff = float(np.quantile(margins, 1.0 - self._um4_coverage))
        self.log.info(
            "UM4 %s: coverage %.2f, referral margin cut-off %.4f (validation quantile, "
            "n=%d)", operating_point, self._um4_coverage, self._um4_margin_cutoff,
            len(margins))

    # ---- inference ----------------------------------------------------
    def analyze(self, request: TriageRequest, **_: Any) -> Envelope:
        started = time.perf_counter()
        if request.horizon is not None and int(request.horizon) != self._horizon:
            raise InvalidInput(
                "This service is configured for the H=%d horizon; the request asked "
                "for H=%d. Restart with CVXAI_TRIAGE_HORIZON=%d to serve it."
                % (self._horizon, request.horizon, request.horizon),
                {"configured_horizon": self._horizon})

        self.ensure_loaded()
        payload = request.to_component_dict()
        try:
            with self.sandbox.active():
                features = self._featurise(payload)
                result = self._decide(features, payload)
        except Exception as exc:               # noqa: BLE001
            raise InferenceFailed(
                "ACS triage analysis failed: %s" % exc, {"component": self.id}) from exc
        return self._to_envelope(result, payload, started)

    def _featurise(self, payload: Dict[str, Any]):
        """Flat clinical dictionary -> the model's 242-column feature row."""
        import pandas as pd

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            features = self._predict_module.build_row(payload)
            if self._embedder is not None:
                from config import CFG                          # type: ignore
                normalised = self._text_features.normalise(
                    pd.Series([payload.get("chief_complaint", "")]))
                masked, _ = self._text_features.apply_rdm(
                    normalised, enable=bool(CFG.get("text.rdm_enable", True)))
                embedded = self._embedder.transform(masked)
                embedded.index = features.index
                features = pd.concat([features, embedded], axis=1)
        return features

    def _decide(self, features, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run UM4 where available, otherwise the two-stage cascade."""
        import numpy as np

        aligned = self._predictor._align(features)  # noqa: SLF001 - the component's own API
        stage1 = float(self._predictor.stage1_proba(features)[0])
        stage2 = self._predictor.stage2_proba(features)[0]

        if self._um4_models:
            probabilities = np.mean(
                [model.predict_proba(aligned) for model in self._um4_models], axis=0)
            weighted = probabilities * self._um4_weights
            weighted = weighted / weighted.sum(axis=1, keepdims=True)
            ordered = np.sort(weighted, axis=1)
            margin = float(ordered[0, -1] - ordered[0, -2])
            predicted = int(weighted.argmax(axis=1)[0])
            referred = (self._um4_margin_cutoff is not None
                        and margin < self._um4_margin_cutoff)
            engine = "UM4"
            distribution = weighted[0]
            unweighted = probabilities[0]
        else:
            composed = np.concatenate([[1.0 - stage1], stage2 * stage1])
            threshold = float(self._predictor.stage1_cfg["threshold"])
            subtype = int(self._predictor.stage2_cdl.predict(stage2[None, :])[0])
            predicted = subtype + 1 if stage1 >= threshold else 0
            ordered = np.sort(composed)
            margin = float(ordered[-1] - ordered[-2])
            referred = False
            engine = "cascade"
            distribution = composed
            unweighted = composed

        band, action = self._risk_band(stage1)
        attribution = []
        if payload.get("chief_complaint"):
            with contextlib.redirect_stdout(io.StringIO()):
                attribution = self._text_features.token_attribution(
                    payload["chief_complaint"]) or []

        return {
            "engine": engine,
            "prediction": LABEL_ORDER[predicted],
            "prediction_index": predicted,
            "p_acs": stage1,
            "risk_level": band,
            "recommended_action": action,
            "probabilities": {name: float(distribution[i])
                              for i, name in enumerate(LABEL_ORDER)},
            "probabilities_unweighted": {name: float(unweighted[i])
                                         for i, name in enumerate(LABEL_ORDER)},
            "subtype_probabilities": {name: float(stage2[i]) for i, name
                                      in enumerate(LABEL_ORDER[1:])},
            "confidence_margin": margin,
            "referral_cutoff": self._um4_margin_cutoff,
            "referred": bool(referred),
            "coverage": self._um4_coverage,
            "horizon_h": self._horizon,
            "text_attribution": attribution,
        }

    @staticmethod
    def _risk_band(probability: float):
        for threshold, name, action in RISK_BANDS:
            if probability >= threshold:
                return name, action
        return RISK_BANDS[-1][1], RISK_BANDS[-1][2]

    # ---- translation --------------------------------------------------
    def _to_envelope(self, result: Dict[str, Any], payload: Dict[str, Any],
                     started: float) -> Envelope:
        findings: List[Finding] = [Finding(
            name="Acute coronary syndrome",
            present=result["prediction"] != "No_ACS",
            probability=result["p_acs"],
            threshold=float(self._predictor.stage1_cfg["threshold"]),
            evidence="Rule-out screen tuned to negative predictive value (99.41 % at "
                     "the published safety-first operating point), not to accuracy.",
        )]
        for name, probability in result["probabilities"].items():
            findings.append(Finding(
                name=LABEL_DESCRIPTIONS.get(name, name),
                present=(name == result["prediction"]),
                probability=probability,
                label=name,
            ))

        raw = dict(result)
        raw["horizon_meaning"] = HORIZON_MEANING.get(self._horizon, "")
        # Echoed back so a client can render the attributed terms in place
        # against the text the model actually received, rather than against
        # whatever the user believes they typed.
        raw["chief_complaint"] = payload.get("chief_complaint", "")
        raw["inputs_supplied"] = {
            "vitals": any(payload.get(k) is not None for k in
                          ("heartrate", "sbp", "dbp", "resprate", "o2sat")),
            "chief_complaint": bool(payload.get("chief_complaint")),
            "ecg_report": payload.get("ecg") is not None,
            "troponin_draws": len(payload.get("troponin") or []),
            "bnp": payload.get("bnp") is not None,
        }

        explanation = {
            "text_attribution": result["text_attribution"],
            "modality_attribution_note": (
                "Published SHAP mass by horizon -- H=0: text 31.3 %, ECG 0.1 %, labs "
                "0.0 %; H=6: text 20.2 %, ECG 27.0 %, labs 4.6 %; H=24: text 14.6 %, "
                "ECG 18.1 %, labs 29.6 %. At H=0 the laboratory channel carries exactly "
                "zero attribution, which a pipeline with a temporal leak cannot produce."),
            "referral_rule": (
                "Top-two margin on the reweighted four-class distribution; the cut-off "
                "is the %.0f %% validation quantile matching the published %.0f %% "
                "coverage." % (100 * (1 - (self._um4_coverage or 0)),
                               100 * (self._um4_coverage or 0))
                if self._um4_margin_cutoff is not None else "Not active."),
        }

        return self.envelope(
            headline=self._headline(result),
            findings=findings,
            reliability=self._reliability(result, payload),
            raw=raw,
            started=started,
            explanation=explanation,
            narrative=result["recommended_action"],
            decision_rule="%s at H=%dh, %s operating point (coverage %s)"
                          % (result["engine"], self._horizon,
                             self.settings.triage_operating_point,
                             ("%.0f %%" % (100 * self._um4_coverage))
                             if self._um4_coverage else "n/a"),
        )

    @staticmethod
    def _headline(result: Dict[str, Any]) -> str:
        if result["referred"]:
            return ("Referred to clinician -- evidence does not separate the classes "
                    "(leading: %s, P(ACS)=%.1f %%)"
                    % (result["prediction"], 100 * result["p_acs"]))
        return "%s -- P(ACS) %.1f %%, risk %s" % (
            LABEL_DESCRIPTIONS.get(result["prediction"], result["prediction"]),
            100 * result["p_acs"], result["risk_level"])

    def _reliability(self, result: Dict[str, Any], payload: Dict[str, Any]) -> Reliability:
        """Horizon and referral drive the verdict; both are the component's own.

        Unstable angina is *defined* as ACS with a normal troponin, so it is not
        identifiable before the biomarker returns. The component measures that
        directly: UA recall 37.3 % -> 58.2 % -> 80.0 % across H = 0, 6, 24. A
        result produced at an early horizon is not weaker modelling, it is less
        information, and it is reported as such.
        """
        reasons: List[str] = []
        guarantees: List[str] = []
        coverage = self._um4_coverage

        if result["referred"]:
            actionability = Actionability.DEFERRED
            level = "referred"
            reasons.append(
                "Top-two margin %.4f is below the %.4f referral cut-off, so this case "
                "falls in the %.0f %% of presentations handed back to a clinician at "
                "this operating point."
                % (result["confidence_margin"], self._um4_margin_cutoff or 0.0,
                   100 * (1 - (coverage or 0))))
        else:
            actionability = Actionability.ACTIONABLE
            level = "standard"

        if self._horizon < 24:
            actionability = Actionability.worst([actionability, Actionability.CAUTION])
            level = "early_horizon"
            reasons.append(
                "Served at H=%dh (%s). Unstable angina is defined by a normal troponin "
                "and is not identifiable before the biomarker returns: measured UA "
                "recall is 37.3 %% at H=0, 58.2 %% at H=6 and 80.0 %% at H=24."
                % (self._horizon, HORIZON_MEANING.get(self._horizon, "")))

        if not (payload.get("troponin") or []) and self._horizon >= 6:
            actionability = Actionability.worst([actionability, Actionability.CAUTION])
            reasons.append(
                "No troponin was supplied. The component encodes that as the clinical "
                "fact that no test was ordered rather than imputing a value, but the "
                "published F1 figures come from the biomarker-tested population.")

        if payload.get("charlson_index") is not None:
            reasons.append(
                "A Charlson comorbidity index was supplied. If it derives from the "
                "index admission it is leakage channel L1, which alone moves AUROC "
                "0.9665 -> 0.9889 and invalidates the reported performance.")

        if coverage is not None:
            guarantees.append(
                "Operating point chosen on validation under a hard constraint: maximise "
                "macro-F1 subject to minimum per-class recall >= 0.75. Realised test "
                "min-recall 0.7783 at %.0f %% coverage." % (100 * coverage))
        guarantees.append(
            "Temporal contract: every feature carries a declared availability time and "
            "the featuriser admits none beyond H=%dh." % self._horizon)

        reasons.append(
            "Precision on the rare classes is bounded by prevalence, not by model "
            "quality. At UA prevalence 0.36 %, F1 >= 0.75 at recall 0.75 would require "
            "a positive likelihood ratio above 800; troponin achieves 10-25.")

        return Reliability(
            actionability=actionability, level=level, reasons=reasons,
            guarantees=guarantees, guarantees_void=False, coverage=coverage)

    # ---- documentation ------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        return {
            "test_set_n": 30452,
            "horizon_h": self.settings.triage_horizon,
            "stage1_detection": {
                "auroc_iup": 0.9560, "npv": 0.9941, "sensitivity": 0.9135,
                "auroc_full_ed": 0.9688, "acs_f1_awc": 0.7436,
            },
            "stage2_subtyping": {
                "macro_f1": 0.7448, "balanced_accuracy": 0.7753, "auroc_ovr": 0.8951,
                "recall": {"UA": 0.8000, "NSTEMI": 0.7888, "STEMI": 0.7372},
            },
            "end_to_end_um4": {
                "max_coverage": {"coverage": 0.85, "min_recall": 0.7783,
                                 "balanced_accuracy": 0.8327, "macro_f1": 0.4906},
                "max_macro_f1": {"coverage": 0.65, "min_recall": 0.8105,
                                 "balanced_accuracy": 0.8811, "macro_f1": 0.5720},
            },
            "progressive_horizon_auroc": {"H0": 0.8763, "H6": 0.9121, "H24": 0.9560},
            "leakage_audit": {
                "clean": 0.9665, "with_charlson_mi_column": 0.9889,
                "previously_published_leaky_figure": 0.9841,
            },
        }

    def limitations(self) -> List[str]:
        return [
            "Per-class F1 on a full ED population cannot reach 0.75 and this component "
            "does not claim it. At UA prevalence 0.36 %, F1 >= 0.75 at recall 0.75 "
            "needs a positive likelihood ratio above 800; no instrument reaches it. The "
            "claim made is recall >= 75 % on every class.",
            "STEMI F1 (0.6103) is capped by the modality: MIMIC supplies the ECG cart's "
            "text report, not the waveform, and ST elevation is detectable in only 41 % "
            "of STEMI cases where clinically it is near-universal.",
            "Every selective figure is quoted with its coverage. A selective metric "
            "without coverage is meaningless -- abstaining on 99 % of patients makes "
            "any model look perfect.",
            "Single-centre MIMIC-IV-ED cohort. No external validation.",
            "The previous version of this component reported AUROC 0.9841 from five "
            "leakage channels. The figures here are the post-audit rebuild.",
        ]
