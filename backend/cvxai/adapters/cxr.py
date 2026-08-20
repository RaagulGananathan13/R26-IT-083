"""
Component 01 adapter -- chest radiograph.

Raagul Gananathan (IT22130020). ConvNeXt-Base classifier over 8 pathologies,
Grad-CAM on the last convolutional stage, and a BioBART report generator
sharing the classifier's vision trunk. MIMIC-CXR, patient-disjoint split,
n = 4,722 test images.

The component ships a complete FastAPI service of its own. This adapter drives
its `InferenceService` directly rather than proxying HTTP, so there is one
process, one model load, and no second port to run.

IMPORT HAZARD
-------------
Component 01's package is literally named `backend`, which is also the name of
the directory this service lives in. If the repository root were ever ahead of
the component root on sys.path, `import backend` would resolve to *this*
service and the component would fail in a confusing way. The sandbox prepends
the component root, and `verify_owns` asserts the resolution afterwards, so the
failure mode is a clear error rather than a wrong number.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cvxai.adapters.base import ComponentAdapter
from cvxai.core.errors import InferenceFailed, InvalidInput
from cvxai.core.sandbox import ModuleSandbox
from cvxai.schemas.common import Actionability, Envelope, Finding, Reliability

VALID_VIEWS = ("AP", "PA")


class CxrAdapter(ComponentAdapter):
    id = "cxr"
    name = "Cardiomegaly Detection with XAI and Report Generation"
    owner = "Raagul Gananathan (IT22130020)"
    modality = "Chest radiograph (frontal, AP or PA)"
    task = "Cardiomegaly + 7 co-pathologies, Grad-CAM, draft radiology report"
    dataset = "MIMIC-CXR / MIMIC-CXR-JPG (PhysioNet credentialed)"
    architecture = "ConvNeXt-Base 384x384 + BioBART-v2-base decoder (shared trunk)"
    endpoint = "/api/v1/cxr/analyze"

    def __init__(self, settings, root) -> None:
        super().__init__(settings, root)
        self._service = None

    # ---- capability ---------------------------------------------------
    def required_paths(self) -> List[Path]:
        assert self.root is not None
        return [
            self.root / "backend" / "config.py",
            self.root / "cxr_transforms.py",
            self.root / "checkpoints" / "stage5" / "best.pt",
            self.root / "backend" / "thresholds.json",
        ]

    def notes(self) -> List[str]:
        """Explain the optional asset the component warns about at load.

        Component 01 indexes the ORIGINAL radiologist reports so a bundled test
        image can be shown next to the text a human actually dictated. That CSV
        is MIMIC-CXR derived and credentialed, so it is deliberately not
        distributed. Its absence only blanks `ground_truth_report` for bundled
        test images; every uploaded study is unaffected, because an arbitrary
        upload has no ground truth to look up in the first place.
        """
        if self.root is None:
            return []
        local = self.root / "review_cases" / "cardio_test.csv"
        external = (self.root.parent / "data" / "output"
                    / "cardiomegaly_dataset" / "cardio_test.csv")
        if local.exists() or external.exists():
            return []
        return [
            "Original-report index absent (review_cases/cardio_test.csv), so "
            "`ground_truth_report` is null. Expected: that file is credentialed "
            "MIMIC-CXR derived data and is not distributed. Prediction, Grad-CAM "
            "and report generation are unaffected."
        ]

    def build_sandbox(self) -> ModuleSandbox:
        assert self.root is not None
        return ModuleSandbox(
            name="cxr",
            roots=[self.root],
            # The component root supplies both the `backend` package and the
            # top-level `cxr_transforms` / `stage11_conditioned` modules.
            path_entries=[self.root],
        )

    # ---- lifecycle ----------------------------------------------------
    def _load(self) -> None:
        from backend.services.inference import InferenceService  # type: ignore

        self.sandbox.verify_owns("backend")
        self.sandbox.verify_owns("backend.config")
        self._service = InferenceService()

    # ---- inference ----------------------------------------------------
    def analyze(self, image_bytes: bytes, view: Optional[str] = None,
                filename: Optional[str] = None, **_: Any) -> Envelope:
        started = time.perf_counter()
        view = self._normalise_view(view)
        if not image_bytes:
            raise InvalidInput("The uploaded radiograph is empty.")

        self.ensure_loaded()
        try:
            with self.sandbox.active():
                raw = self._service.predict(image_bytes, view=view, filename=filename)
        except InvalidInput:
            raise
        except Exception as exc:               # noqa: BLE001
            raise InferenceFailed(
                "Chest radiograph analysis failed: %s" % exc,
                {"component": self.id}) from exc

        return self._to_envelope(raw, view, started)

    @staticmethod
    def _normalise_view(view: Optional[str]) -> Optional[str]:
        """AP / PA, or None.

        An unrecognised value becomes None rather than a guess. Guessing PA on
        a bedside film applies the stricter threshold to exactly the patients
        least able to tolerate a missed cardiomegaly.
        """
        if not view:
            return None
        cleaned = view.strip().upper()
        if cleaned in VALID_VIEWS:
            return cleaned
        if cleaned in ("", "UNKNOWN", "NONE", "AUTO"):
            return None
        raise InvalidInput(
            "view must be AP, PA, or omitted (received %r)." % view,
            {"accepted": list(VALID_VIEWS)})

    # ---- translation --------------------------------------------------
    def _to_envelope(self, raw: Dict[str, Any], view: Optional[str],
                     started: float) -> Envelope:
        probability = float(raw.get("probability", 0.0))
        threshold = float(raw.get("threshold", 0.5))
        detected = raw.get("prediction") == "Cardiomegaly"

        findings = [Finding(
            name="Cardiomegaly",
            present=detected,
            probability=probability,
            threshold=threshold,
            evidence="Grad-CAM over the final ConvNeXt stage; see explanation.gradcam_png_base64.",
        )]
        for item in raw.get("copathologies", []):
            findings.append(Finding(
                name=item.get("name", "?"),
                present=item.get("status") == "present",
                probability=item.get("probability"),
                threshold=item.get("threshold"),
            ))

        reliability = self._reliability(raw, view)
        explanation = {
            "gradcam_png_base64": raw.get("gradcam_image") or None,
            "gradcam_target": "Cardiomegaly",
            "gradcam_caveat": (
                "Grad-CAM shows where the model looked, not whether it was right. "
                "Arun et al. (Radiology: AI 2021) measured Grad-CAM repeatability on "
                "chest radiographs at SSIM 0.12. Treat the overlay as a sanity check."),
            "classifier_prompt": raw.get("classifier_prompt") or None,
        }

        return self.envelope(
            headline=self._headline(raw, detected, probability),
            findings=findings,
            reliability=reliability,
            raw=raw,
            started=started,
            explanation=explanation,
            narrative=raw.get("report_text"),
            decision_rule="threshold=%.4f from the %s operating point"
                          % (threshold, raw.get("threshold_source", "global")),
        )

    @staticmethod
    def _headline(raw: Dict[str, Any], detected: bool, probability: float) -> str:
        deferral = raw.get("deferral") or {}
        if deferral.get("defer"):
            return ("Too close to call -- referred for radiologist review "
                    "(cardiomegaly p=%.3f)" % probability)
        return ("Cardiomegaly present (p=%.3f)" % probability if detected
                else "No cardiomegaly (p=%.3f)" % probability)

    def _reliability(self, raw: Dict[str, Any], view: Optional[str]) -> Reliability:
        """Map the component's deferral and projection policy onto the contract.

        Two independent mechanisms, both measured rather than assumed:

        * Selective deferral (Stage 13). The margin cut-off is projection
          specific -- AP 0.2247 against PA 0.0029 -- because AP films carry
          less usable information, so the system must be more reluctant to
          commit on them. Fitted on validation (n=4,474), frozen before test.
        * Per-projection operating points (Stage 9A). AP AUROC 0.8224 against
          PA 0.8864, a gap of 0.0639 that three separate interventions failed
          to close. Reported rather than hidden.
        """
        native = raw.get("reliability") or {}
        deferral = raw.get("deferral") or {}
        level = str(native.get("level", "unknown"))
        reasons: List[str] = []
        guarantees: List[str] = []

        if native.get("message"):
            reasons.append(str(native["message"]))

        if deferral.get("defer"):
            actionability = Actionability.DEFERRED
            reasons.append(str(deferral.get("reason", "Prediction within the deferral margin.")))
        elif level == "reduced":
            actionability = Actionability.CAUTION
        elif level == "unknown":
            actionability = Actionability.CAUTION
            reasons.append(
                "Projection was not supplied, so the global operating point was used. "
                "Sending view=AP or view=PA applies the projection-specific threshold "
                "and reports the measured reliability for that view.")
        else:
            actionability = Actionability.ACTIONABLE

        measured = deferral.get("measured") or {}
        coverage = None
        if deferral.get("active") and measured:
            coverage = measured.get("coverage")
            if coverage is not None:
                coverage = float(coverage) / 100.0
            guarantees.append(
                "Selective deferral fitted on validation and frozen: %.1f%% of studies "
                "answered, %.2f%% accuracy on the answered subset, AP/PA accuracy gap "
                "closed from 6.68 to %.2f points."
                % (measured.get("coverage", 0.0), measured.get("accuracy", 0.0),
                   measured.get("gap", 0.0)))

        return Reliability(
            actionability=actionability,
            level=level,
            reasons=reasons,
            guarantees=guarantees,
            guarantees_void=False,
            coverage=coverage,
        )

    # ---- documentation ------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        return {
            "test_set_n": 4722,
            "classifier": {
                "cardiomegaly_auroc": 0.9189,
                "cardiomegaly_auroc_ci95": [0.9112, 0.9265],
                "cardiomegaly_sensitivity": 0.923,
                "cardiomegaly_specificity": 0.740,
                "mean_auroc_8_labels": 0.8554,
            },
            "report_generator": {
                "chexbert_micro_f1_14": 0.5939,
                "cardiomegaly_report_f1": 0.8287,
                "rouge_l": 0.2896,
                "constant_string_rouge_l_control": 0.2641,
                "fabricated_prior_study_rate": 0.0,
            },
            "acquisition_fairness": {
                "auroc_pa": 0.8864,
                "auroc_ap": 0.8224,
                "gap": 0.0639,
                "gap_ci95": [0.0491, 0.0790],
            },
        }

    def limitations(self) -> List[str]:
        return [
            "Measurably less accurate on AP (bedside) films -- the ones taken of the "
            "sickest patients. The 0.0639 AUROC gap survived per-projection "
            "thresholding, gradient reversal and conditional heads, so it is not "
            "removable at the model level.",
            "ROUGE-L is near-matched by a constant string identical for every patient "
            "(0.2641 against the model's 0.2896) while scoring clinical F1 0.0000. "
            "Clinical-efficacy F1 is the primary report metric.",
            "The split is this project's own, not the official MIMIC-CXR split, and "
            "the test set is cardiomegaly-enriched at 50.4 % prevalence. Figures are "
            "not directly comparable to published MIMIC-CXR results.",
            "Five of the eight pathologies score below an always-negative baseline on "
            "accuracy, an artefact of F1-optimal thresholds on rare disease. Read "
            "AUROC and sensitivity, not accuracy.",
            "Grad-CAM repeatability on chest radiographs is poor (SSIM 0.12, Arun et "
            "al. 2021). Overlays are a sanity check, not localisation evidence.",
        ]
