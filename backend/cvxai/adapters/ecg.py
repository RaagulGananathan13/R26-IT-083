"""
Component 02 adapter -- 12-lead ECG.

Venushan T. 1-D residual CNN with squeeze-excitation over the five PTB-XL
diagnostic superclasses, per-class temperature calibration, PAC conformal
triage, Grad-CAM / integrated-gradients explanation, template-grounded report
and an automated verification gate. Test fold 10, n = 1,711.

TWO OBSTACLES THIS ADAPTER SOLVES
---------------------------------
1. Every path inside the component carries a " (1)" suffix from a zip
   extraction -- `src (1)/pipeline (1).py` -- while the code says
   `from .models import ...`. `SuffixTolerantFinder` maps the clean module path
   onto whichever spelling exists on disk, so the component runs unmodified.

2. The component's own asset resolver (`src/paths.py`) looks for clean asset
   names, so `norm_stats.json` is invisible to it as `norm_stats (1).json`.
   The adapter stages the small assets it needs under clean names in the
   backend cache and points ECG_DATA_DIR there. Nothing in the component tree
   is renamed or written to.

The provenance check in `ECGPipeline.from_checkpoint` is left switched on. A
calibrator and a set of conformal thresholds are valid only for the exact model
whose logits produced them; pairing them with a different model destroys the
guarantee while everything still appears to run.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cvxai.adapters.base import ComponentAdapter
from cvxai.core.errors import InferenceFailed, InvalidInput
from cvxai.core.sandbox import ModuleSandbox, SuffixTolerantFinder
from cvxai.schemas.common import Actionability, Envelope, Finding, Reliability

#: Assets the serving path needs, in the order the component expects them.
_STAGED_ASSETS = ("norm_stats.json",)

CLASS_FULL_NAMES = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction",
    "STTC": "ST/T change",
    "CD": "conduction disturbance",
    "HYP": "ventricular hypertrophy",
}


def _find_variant(directory: Path, name: str) -> Optional[Path]:
    """Locate `name` allowing for the ' (1)' zip-extraction suffix."""
    stem, _, suffix = name.rpartition(".")
    for candidate in (name, "%s (1).%s" % (stem, suffix)):
        path = directory / candidate
        if path.exists():
            return path
    return None


def _find_dir(root: Path, name: str) -> Optional[Path]:
    for candidate in (name, "%s (1)" % name):
        path = root / candidate
        if path.is_dir():
            return path
    return None


class EcgAdapter(ComponentAdapter):
    id = "ecg"
    name = "ECG Abnormality Detection and Cardiac Risk Reporting"
    owner = "Venushan T"
    modality = "12-lead ECG (WFDB .dat + .hea)"
    task = "5 diagnostic superclasses with conformal rule-in / rule-out triage"
    dataset = "PTB-XL (PhysioNet), official fold split; test fold 10, n = 1,711"
    architecture = "1-D residual CNN with squeeze-excitation + temperature scaling + PAC conformal triage"
    endpoint = "/api/v1/ecg/analyze"

    def __init__(self, settings, root) -> None:
        super().__init__(settings, root)
        self._pipeline = None
        self._class_names: List[str] = []
        self._lead_names: List[str] = []
        self._sampling_rate = 500
        self._stage_dir: Optional[Path] = None
        self._render_ecg = None
        self._render_failed = False

    # ---- capability ---------------------------------------------------
    @property
    def _src_dir(self) -> Optional[Path]:
        return _find_dir(self.root, "src") if self.root else None

    @property
    def _ckpt_dir(self) -> Optional[Path]:
        return _find_dir(self.root, "checkpoints") if self.root else None

    @property
    def _csv_dir(self) -> Optional[Path]:
        return _find_dir(self.root, "csv") if self.root else None

    def required_paths(self) -> List[Path]:
        assert self.root is not None
        src, ckpt, csv = self._src_dir, self._ckpt_dir, self._csv_dir
        if src is None or ckpt is None or csv is None:
            missing = self.root / "src|checkpoints|csv"
            return [missing]
        needed: List[Path] = []
        for directory, name in ((src, "pipeline.py"), (ckpt, "best_model.pt"),
                                (ckpt, "calibrator.json"), (ckpt, "conformal_triage.json"),
                                (csv, "norm_stats.json")):
            found = _find_variant(directory, name)
            needed.append(found if found else directory / name)
        return needed

    def _stage_assets(self) -> Path:
        """Copy the small serving assets under clean names into the cache.

        The component tree is never modified. Only files listed in
        `_STAGED_ASSETS` are copied, and only when the staged copy is missing or
        older than the source.
        """
        stage = Path(self.settings.cache_dir) / "ecg_assets"
        stage.mkdir(parents=True, exist_ok=True)
        csv_dir = self._csv_dir
        if csv_dir is None:
            return stage
        for name in _STAGED_ASSETS:
            source = _find_variant(csv_dir, name)
            if source is None:
                continue
            target = stage / name
            if (not target.exists()
                    or target.stat().st_mtime < source.stat().st_mtime):
                shutil.copy2(source, target)
                self.log.info("staged %s -> %s", source.name, target)
        return stage

    def build_sandbox(self) -> ModuleSandbox:
        assert self.root is not None
        src = self._src_dir
        if src is None:
            raise InvalidInput("Component 02 source directory not found under %s" % self.root)
        self._stage_dir = self._stage_assets()
        # The component root supplies the `src` package; its backend directory
        # supplies `server`, whose ECG-paper renderer is reused for the strip.
        entries = [self.root]
        server_dir = _find_dir(self.root, "backend")
        if server_dir is not None:
            entries.append(server_dir)
        return ModuleSandbox(
            name="ecg",
            roots=[self.root],
            path_entries=entries,
            env={"ECG_DATA_DIR": str(self._stage_dir)},
            finders=[lambda: SuffixTolerantFinder("src", src)],
        )

    # ---- lifecycle ----------------------------------------------------
    def _load(self) -> None:
        from src import paths                                   # type: ignore
        from src.models import CLASS_NAMES, LEAD_NAMES, SAMPLING_RATE  # type: ignore
        from src.pipeline import ECGPipeline                    # type: ignore

        self.sandbox.verify_owns("src")
        ckpt = self._ckpt_dir
        assert ckpt is not None

        self._pipeline = ECGPipeline.from_checkpoint(
            ckpt_path=str(_find_variant(ckpt, "best_model.pt")),
            norm_stats_path=paths.require("norm_stats.json"),
            model_name=self.settings.ecg_model,
            calibrator_path=str(_find_variant(ckpt, "calibrator.json")),
            triage_path=str(_find_variant(ckpt, "conformal_triage.json")),
            do_filter=self.settings.ecg_filter,
        )
        if self._pipeline.calibrator is None or self._pipeline.triage is None:
            # Without both, no conformal claim can be made, and the component's
            # whole contribution is about the validity of that claim.
            raise RuntimeError(
                "calibrator or conformal thresholds missing -- the statistical "
                "guarantee cannot be served. Run train/fit_calibration.py.")

        self._class_names = list(CLASS_NAMES)
        self._lead_names = list(LEAD_NAMES)
        self._sampling_rate = int(SAMPLING_RATE)

    # ---- inference ----------------------------------------------------
    def analyze(self, dat_bytes: bytes, hea_bytes: bytes, record_name: str,
                with_xai: bool = True, **_: Any) -> Envelope:
        started = time.perf_counter()
        self.ensure_loaded()
        signal, fs, lead_names, reorder_note = self._read_record(
            dat_bytes, hea_bytes, record_name)
        try:
            with self.sandbox.active():
                result = self._pipeline.analyse(
                    signal, fs=fs, with_xai=with_xai, lead_names=lead_names)
                raw = result.to_json()
        except Exception as exc:               # noqa: BLE001
            raise InferenceFailed(
                "ECG analysis failed: %s" % exc, {"component": self.id}) from exc

        raw["patientId"] = record_name
        raw["uploaded"] = {"fs": fs, "samples": int(signal.shape[0]),
                           "leads": lead_names or self._lead_names}
        if reorder_note:
            raw.setdefault("quality", {}).setdefault("warnings", []).append(reorder_note)
        return self._to_envelope(raw, result, started)

    def _read_record(self, dat_bytes: bytes, hea_bytes: bytes, record_name: str):
        """Decode an uploaded WFDB pair into a (n, 12) millivolt array."""
        import tempfile

        import numpy as np
        import wfdb

        if not dat_bytes or not hea_bytes:
            raise InvalidInput("Both a .dat and a .hea file are required.")

        # WFDB resolves the signal file from the header's CONTENTS, not from the
        # uploaded filename: line 1 declares the record name and line 2 names the
        # .dat file. Writing the upload under its own filename breaks whenever a
        # file has been renamed in transit, which is routine. Both files are
        # therefore staged under the names the header itself declares.
        stem, signal_name = self._header_names(hea_bytes, record_name)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / (stem + ".hea")).write_bytes(hea_bytes)
            (directory / signal_name).write_bytes(dat_bytes)
            try:
                record = wfdb.rdrecord(str(directory / stem))
            except Exception as exc:           # noqa: BLE001
                raise InvalidInput("Could not read the WFDB record: %s" % exc) from exc
            signal = np.asarray(record.p_signal, dtype=np.float32)
            fs = int(record.fs or self._sampling_rate)
            names = [str(s) for s in (record.sig_name or [])]

        if signal.ndim != 2 or signal.shape[1] != 12:
            got = signal.shape[1] if signal.ndim == 2 else "unknown"
            raise InvalidInput(
                "Expected 12 leads, received %s." % got,
                {"expected_leads": self._lead_names})

        note = None
        if names:  # noqa: SIM102 - kept flat for readability below
            found = [n.strip().upper() for n in names]
            expected = [n.upper() for n in self._lead_names]
            if found != expected:
                if sorted(found) == sorted(expected):
                    signal = signal[:, [found.index(lead) for lead in expected]]
                    note = ("Leads were reordered from %s to the standard order."
                            % ", ".join(names))
                else:
                    raise InvalidInput(
                        "Unrecognised lead set: %s" % ", ".join(names),
                        {"expected_leads": self._lead_names})
        return signal, fs, names, note

    @staticmethod
    def _header_names(hea_bytes: bytes, fallback: str):
        """Read the record name and signal filename declared inside the header.

        WFDB header format: the first non-comment line begins with the record
        name; the next non-comment line begins with the signal filename.
        """
        try:
            text = hea_bytes.decode("utf-8", errors="replace")
        except Exception as exc:               # noqa: BLE001
            raise InvalidInput("The .hea file is not readable text: %s" % exc) from exc

        lines = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        if not lines:
            raise InvalidInput("The .hea file contains no header record line.")

        stem = lines[0].split()[0].strip()
        if not stem:
            raise InvalidInput("The .hea file declares an empty record name.")

        signal_name = "%s.dat" % stem
        if len(lines) > 1:
            candidate = lines[1].split()[0].strip()
            # A multi-segment or directory-qualified reference is outside what a
            # two-file upload can satisfy; fall back rather than write outside
            # the temporary directory.
            if candidate and "/" not in candidate and "\\" not in candidate:
                signal_name = candidate
        return stem or fallback, signal_name

    # ---- presentation -------------------------------------------------
    def _renderer(self):
        """Component 02's own ECG-paper renderer, loaded lazily.

        It lives at module level in the component's Flask server, which builds
        its own pipeline on import. That costs about a second for this
        component -- the model is small -- and it is the right trade: the strip
        is drawn on clinical paper (25 mm/s, 10 mm/mV, 3x4 plus a rhythm strip)
        with the calibration pulse, and reimplementing that here would drift
        from what the component's own reviewers checked.

        Failure is not fatal. Without it the analysis still returns; only the
        rendered strip is missing.
        """
        if self._render_ecg is not None or self._render_failed:
            return self._render_ecg
        try:
            with self.sandbox.active():
                import server                            # type: ignore
                self._render_ecg = server.render_ecg
        except Exception as exc:                         # noqa: BLE001
            self._render_failed = True
            self.log.warning("ECG strip rendering unavailable: %s", exc)
        return self._render_ecg

    def _explanation_payload(self, result: Any) -> Dict[str, Any]:
        """Lead attribution and territory, from the pipeline's own Explanation.

        The pipeline returns these as objects; the component's server flattens
        them for the wire and this does the same, so the two surfaces agree.
        """
        payload: Dict[str, Any] = {}
        explanations = getattr(result, "explanations", None) or {}
        if not explanations:
            return payload

        target, explanation = next(iter(explanations.items()))
        payload["target"] = target
        payload["territory"] = explanation.territory
        payload["artery"] = explanation.territory_artery
        payload["territory_score"] = round(float(explanation.territory_score), 2)
        payload["topLeads"] = list(explanation.top_leads or [])
        payload["peaksSeconds"] = [round(float(x), 2) for x in (explanation.cam_peaks_s or [])]
        payload["lead_attribution"] = sorted(
            [
                {
                    "name": name,
                    "signed": round(float(value), 1),
                    "magnitude": round(float(explanation.lead_magnitude.get(name, 0.0)), 1),
                }
                for name, value in (explanation.lead_signed or {}).items()
            ],
            key=lambda item: -abs(item["signed"]),
        )
        return payload

    # ---- translation --------------------------------------------------
    def _to_envelope(self, raw: Dict[str, Any], result: Any, started: float) -> Envelope:
        refused = bool(raw.get("refused"))
        findings: List[Finding] = []
        probabilities = raw.get("probabilities") or {}
        zones = raw.get("zones") or {}
        thresholds = {}
        triage = getattr(self._pipeline, "triage", None)
        if triage is not None:
            thresholds = {name: obj for name, obj in triage.thresholds.items()}

        for name in self._class_names:
            if name not in probabilities:
                continue
            threshold_obj = thresholds.get(name)
            findings.append(Finding(
                name=CLASS_FULL_NAMES.get(name, name),
                present=zones.get(name) == "rule_in",
                probability=float(probabilities[name]),
                zone=zones.get(name),
                threshold=(float(threshold_obj.lambda_in)
                           if threshold_obj is not None else None),
                evidence=self._lead_evidence(raw, name),
            ))

        # The pipeline returns objects, not a wire payload; flatten them the
        # same way the component's own server does.
        explanation = self._explanation_payload(result)

        render = self._renderer()
        if render is not None and getattr(result, "signal_mv", None) is not None:
            cam = None
            if result.explanations:
                cam = next(iter(result.explanations.values())).cam
            try:
                with self.sandbox.active():
                    explanation["ecg_png_base64"] = render(
                        result.signal_mv, cam, result.r_peaks, False)
            except Exception as exc:               # noqa: BLE001
                self.log.warning("strip rendering failed: %s", exc)

        explanation["territory_caveat"] = (
            "Territory localisation is a lead-group heuristic and has not been "
            "clinically validated.")

        return self.envelope(
            headline=str(raw.get("headline") or "ECG analysed"),
            findings=findings,
            reliability=self._reliability(raw),
            raw=raw,
            started=started,
            status="refused" if refused else "ok",
            explanation=explanation,
            narrative=raw.get("reportText"),
            decision_rule="PAC conformal triage, per-class alpha; thresholds fitted on "
                          "validation fold 9 and frozen",
        )

    @staticmethod
    def _lead_evidence(raw: Dict[str, Any], class_name: str) -> Optional[str]:
        explanation = raw.get("explanation") or {}
        if explanation.get("target") != class_name:
            return None
        leads = explanation.get("topLeads") or []
        if not leads:
            return None
        territory = explanation.get("territory")
        text = "Strongest attribution in leads " + ", ".join(str(lead) for lead in leads)
        if territory:
            text += " (%s territory)" % territory
        return text

    def _reliability(self, raw: Dict[str, Any]) -> Reliability:
        """Fold the component's four independent safety signals into one verdict.

        The ordering below is the component's own: a record that fails quality
        control never reaches the classifier, so refusal dominates; a failed
        report verification withholds the text; an electrode reversal or an
        out-of-scope rhythm keeps the probabilities but voids the guarantee,
        because the conformal thresholds were calibrated on correctly-wired
        recordings of in-scope disease.
        """
        quality = raw.get("quality") or {}
        verification = raw.get("verification") or {}
        electrode = raw.get("electrode") or {}
        scope = raw.get("scope") or {}

        reasons: List[str] = []
        guarantees: List[str] = list(raw.get("guarantees") or [])
        void = False

        if raw.get("refused"):
            reasons.extend(str(e) for e in quality.get("errors", []))
            reasons.append(
                "The signal failed quality control, so no probability was produced. "
                "This is not a normal result -- request a repeat ECG.")
            return Reliability(
                actionability=Actionability.WITHHELD,
                level="refused", reasons=reasons, guarantees=[], guarantees_void=True)

        if not verification.get("passed", True):
            reasons.extend(str(e) for e in verification.get("errors", []))
            reasons.append("The generated report failed automated safety verification "
                           "and was withheld.")
            return Reliability(
                actionability=Actionability.WITHHELD,
                level="verification_failed", reasons=reasons,
                guarantees=[], guarantees_void=True)

        if electrode.get("suspected"):
            void = True
            reasons.append(
                "Suspected limb-electrode reversal (%s, confidence %.2f). The recording "
                "is high quality but wired wrongly; the conformal guarantees are "
                "calibrated on correctly-placed recordings and do not apply here."
                % (electrode.get("reversal") or "unspecified",
                   float(electrode.get("confidence") or 0.0)))
            reasons.extend(str(r) for r in electrode.get("reasons", []))

        if scope.get("outOfScope"):
            void = True
            reasons.append(
                "The rhythm appears to lie outside the five modelled superclasses (%s). "
                "No conformal bound covers a class the model has no output unit for, so "
                "an arrhythmia has NOT been excluded."
                % (scope.get("reason") or "irregular R-R"))

        for warning in quality.get("warnings", []) or []:
            reasons.append(str(warning))

        if void:
            return Reliability(
                actionability=Actionability.CAUTION,
                level="guarantees_withdrawn", reasons=reasons,
                guarantees=[], guarantees_void=True)

        zones = raw.get("zones") or {}
        if zones and all(zone == "refer" for zone in zones.values()):
            reasons.append("Every class fell in the refer zone: the evidence supports "
                           "neither ruling in nor ruling out on its own.")
            return Reliability(
                actionability=Actionability.DEFERRED, level="refer",
                reasons=reasons, guarantees=guarantees)

        reasons.append(
            "Marginal conformal validity only. The component's own audit found 9 of 23 "
            "class-subgroup cells violating the advertised bound -- conduction "
            "disturbance in patients under 50 missed 33.3 % against a promised 10 %.")
        return Reliability(
            actionability=Actionability.ACTIONABLE, level="standard",
            reasons=reasons, guarantees=guarantees)

    # ---- documentation ------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        return {
            "test_fold": 10,
            "test_set_n": 1711,
            "macro": {"accuracy": 0.864, "recall": 0.810, "specificity": 0.888,
                      "npv": 0.933, "precision": 0.650, "f1": 0.698},
            "threshold_free_3_seeds": {"macro_auroc": 0.9343, "macro_auroc_sd": 0.0028,
                                       "macro_auprc": 0.8001, "macro_auprc_sd": 0.0029},
            "per_class_npv": {"NORM": 0.868, "MI": 0.967, "STTC": 0.926,
                              "CD": 0.921, "HYP": 0.981},
            "conditional_validity": {
                "cells_satisfying_bound_marginal": "14/23",
                "cells_satisfying_bound_mondrian": "22/23",
                "worst_violation": "CD in patients under 50: 33.3 % missed against a "
                                   "promised 10 % (n=66 positives)",
            },
        }

    def limitations(self) -> List[str]:
        return [
            "Trained on PTB-XL only -- one German cohort, 1989-96. No external validation.",
            "Recognises five superclasses. Atrial fibrillation and other arrhythmias are "
            "not detected; their absence from a report is not evidence of their absence. "
            "14.3 % of the dataset carries a finding the label space cannot express.",
            "The conformal guarantee is marginal. It holds on average over the test "
            "distribution and is violated in 9 of 23 patient subgroups.",
            "The electrode-reversal detector is a physiology rule, not a classifier: "
            "~70 % sensitivity on RA/LA, ~61 % on RA/LL, 4.5 % false positives. LA/LL "
            "reversal leaves aVR unchanged and is essentially undetectable this way.",
            "PR/QRS/QT intervals and QRS axis are not measured; territory localisation "
            "is an unvalidated lead-group heuristic.",
            "Labels used only SCP codes with likelihood == 100, dropping 21 % of PTB-XL. "
            "Results are not directly comparable to published benchmarks.",
        ]
