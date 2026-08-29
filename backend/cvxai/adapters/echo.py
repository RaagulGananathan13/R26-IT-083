"""
Component 03 adapter -- echocardiogram video.

Dilukshan Viyapury (IT22219534). UEF-Net: R(2+1)D-18 spatio-temporal backbone
with four heads (EF regression, ordered-cutpoint ordinal, auxiliary softmax,
log-variance), trained on EchoNet-Dynamic with harmonized CAMUS co-training.
Three-seed ensemble, test split n = 1,277.

THIS COMPONENT SHIPPED NO SERVING PATH
--------------------------------------
Components 01 and 02 both ship an inference service that this backend can drive
directly. Component 03 ships training and dataset-level evaluation only: its
`run_eval.py` and `run_ensemble.py` operate over the cached manifest, not over
one uploaded study. So the single-study path is reproduced here, step for step
from `engine/evaluate.py::run_inference` and `run_ensemble.py`:

    decode to 112x112 grayscale
    -> N deterministic label-free clips spread across the valid window
    -> per clip: grayscale z-score + signed temporal-difference motion channel
    -> forward, then aggregate at study level
         EF        = mean over clips
         P_ord     = mean over clips
         P_class   = mean over clips
         sigma_epi = std over clips          (inter-clip disagreement)
         sigma_ale = mean learned variance   (log-variance head)
    -> average across ensemble members
    -> apply the frozen validation-selected decision rule
    -> split-conformal EF interval

The sampling and motion routines are imported from the component's own
`preprocessing.utils.sampling`, not reimplemented, so there is no skew between
what the model was trained on and what it is served.

Clip sampling is label-free: the annotated ED/ES frames are never consulted,
because they would not exist for a new clinical study.

HONESTY NOTE ON THE DECISION RULE
---------------------------------
The published headline (MAE 3.979, min-recall 0.723) comes from an
ensemble-level calibration that `run_ensemble.py` refits on validation at each
invocation and does not persist. What *is* persisted is each member's own
validation-frozen `thresholds.json`. This adapter therefore applies the
decision rule of the configured member (default `uefnet_v3`) to the ensemble
average, and reports exactly that in `model.decision_rule`. It does not claim
the ensemble-calibrated figure for an individual served study.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cvxai.adapters.base import ComponentAdapter
from cvxai.core.errors import InferenceFailed, InvalidInput
from cvxai.core.sandbox import ModuleSandbox
from cvxai.schemas.common import Actionability, Envelope, Finding, Reliability

VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm")
ARRAY_SUFFIXES = (".npy",)

#: Clinical severity bands, in the component's class order.
CLASS_DESCRIPTIONS = {
    0: "Severe systolic impairment (EF < 30 %)",
    1: "Moderate systolic impairment (EF 30-40 %)",
    2: "Mild systolic impairment (EF 40-55 %)",
    3: "Normal systolic function (EF >= 55 %)",
}


class _Member:
    """One trained seed: its restored config, weights and frozen statistics."""

    def __init__(self, run: str, model: Any, cfg_snapshot: Dict[str, Any],
                 ef_mean: float, ef_std: float, pixel_mean: float, pixel_std: float):
        self.run = run
        self.model = model
        self.cfg_snapshot = cfg_snapshot
        self.ef_mean = ef_mean
        self.ef_std = ef_std
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std


class EchoAdapter(ComponentAdapter):
    id = "echo"
    name = "UEF-Net -- Ejection Fraction Regression and Four-Class Severity Grading"
    owner = "Dilukshan Viyapury (IT22219534)"
    modality = "Echocardiogram video (apical four-chamber)"
    task = "Continuous EF plus Severe / Moderate / Mild / Normal severity grade"
    dataset = "EchoNet-Dynamic (train/val/test) + CAMUS (train only, harmonized)"
    architecture = "R(2+1)D-18, 4 heads (regression, ordered-cutpoint ordinal, class, log-variance)"
    endpoint = "/api/v1/echo/analyze"

    def __init__(self, settings, root) -> None:
        super().__init__(settings, root)
        self._reference_efs: Optional[Dict[str, Dict[str, Any]]] = None
        self._members: List[_Member] = []
        self._calibration: Dict[str, Any] = {}
        self._strategy_name: str = ""
        self._calibration_source: str = "member"
        self._calibration_meta: Dict[str, Any] = {}
        self._cfg = None
        self._device = None
        self._sampling = None       # component's own sampling module
        self._io_utils = None
        self._class_names: List[str] = []

    # ---- capability ---------------------------------------------------
    @property
    def _training_dir(self) -> Optional[Path]:
        return self.root / "training" if self.root else None

    @property
    def _prep_dir(self) -> Optional[Path]:
        return self.root / "preprocessing" if self.root else None

    def _run_dir(self, run: str) -> Path:
        assert self._training_dir is not None
        return self._training_dir / "outputs" / run

    def required_paths(self) -> List[Path]:
        assert self.root is not None
        needed = [
            self._training_dir / "config.py",
            self._training_dir / "models" / "uef_net.py",
            self._prep_dir / "utils" / "sampling.py",
        ]
        decision = self.settings.echo_decision_run
        needed.append(self._run_dir(decision) / "thresholds.json")
        for run in self._serving_runs():
            needed.append(self._run_dir(run) / "best.pt")
            needed.append(self._run_dir(run) / "config.json")
        return needed

    def _serving_runs(self) -> List[str]:
        """Configured runs that actually have weights on disk.

        A missing seed degrades the ensemble rather than failing the component:
        the published progression shows 1 -> 2 -> 3 seeds moving MAE 4.138 ->
        3.994 -> 3.979, so fewer members is a quantified loss, not a breakage.
        """
        runs = [run for run in self.settings.echo_runs
                if (self._run_dir(run) / "best.pt").exists()]
        decision = self.settings.echo_decision_run
        if decision not in runs and (self._run_dir(decision) / "best.pt").exists():
            runs.insert(0, decision)
        return runs or list(self.settings.echo_runs[:1])

    def build_sandbox(self) -> ModuleSandbox:
        assert self.root is not None
        return ModuleSandbox(
            name="echo",
            roots=[self.root],
            # training/ supplies config, models, engine, data, core.
            # the component root supplies the preprocessing package.
            path_entries=[self._training_dir, self.root],
        )

    # ---- lifecycle ----------------------------------------------------
    def _load(self) -> None:
        import torch
        from config import CFG                                   # type: ignore
        from models.uef_net import UEFNet                        # type: ignore
        from preprocessing.utils import sampling                 # type: ignore
        from preprocessing.utils import io_utils                 # type: ignore

        self.sandbox.verify_owns("config")
        self.sandbox.verify_owns("models.uef_net")

        self._sampling = sampling
        self._io_utils = io_utils

        runs = self._serving_runs()
        self.log.info("echo ensemble members: %s", ", ".join(runs))
        device_pref = self.settings.device
        if device_pref == "auto":
            device_pref = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(
            "cuda" if (device_pref == "cuda" and torch.cuda.is_available()) else "cpu")

        for run in runs:
            snapshot = json.loads(
                (self._run_dir(run) / "config.json").read_text(encoding="utf-8"))
            CFG.run_name = run
            CFG.restore_for_evaluation(snapshot)
            CFG.run_name = run
            CFG.pretrained = False            # weights come from the checkpoint
            CFG.n_tta_clips = self.settings.echo_tta_clips
            CFG.device = self._device.type

            norm = dict(snapshot.get("norm_stats") or {})
            frozen = self._run_dir(run) / "norm_stats.json"
            if frozen.exists():
                # The immutable run-local artefact wins over the embedded copy.
                norm.update(json.loads(frozen.read_text(encoding="utf-8")))

            model = UEFNet(CFG)
            checkpoint = torch.load(str(self._run_dir(run) / "best.pt"),
                                    map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            model.load_state_dict(state)
            model.to(self._device).eval()

            self._members.append(_Member(
                run=run, model=model, cfg_snapshot=snapshot,
                ef_mean=float(norm["ef_mean"]), ef_std=float(norm["ef_std"]),
                pixel_mean=float(norm["pixel_mean"]), pixel_std=float(norm["pixel_std"])))
            self.log.info("  %s loaded (epoch %s)", run, checkpoint.get("epoch"))

        self._assert_members_comparable()

        self._load_calibration()

        # Restore a reference run's config so EF_THRESHOLDS and the class names
        # match the calibration being applied. Every member shares these -- the
        # comparability check above already refused to serve members that
        # disagree on preprocessing -- so the configured decision run is used
        # regardless of which calibration level won.
        reference_run = self.settings.echo_decision_run
        snapshot = json.loads(
            (self._run_dir(reference_run) / "config.json").read_text(encoding="utf-8"))
        CFG.run_name = reference_run
        CFG.restore_for_evaluation(snapshot)
        CFG.run_name = reference_run
        CFG.pretrained = False
        CFG.device = self._device.type
        self._cfg = CFG
        self._class_names = list(CFG.CLASS_NAMES)

    def _load_calibration(self) -> None:
        """Prefer the ensemble-level rule; fall back to a member's.

        The published headline (MAE 3.979, min-recall 0.723) comes from a rule
        fitted on the ENSEMBLE's validation predictions, which
        `run_ensemble.py` refits per invocation and never persists. When
        `scripts/freeze_echo_ensemble_calibration.py` has been run, that exact
        rule is on disk and is used. Otherwise the configured member's own
        validation-frozen rule is applied to the ensemble average -- close, but
        not the rule the published numbers describe, which is why the
        distinction is reported in `model.decision_rule` rather than hidden.
        """
        frozen = Path(self.settings.cache_dir) / "echo" / "ensemble_calibration.json"
        if frozen.exists():
            payload = json.loads(frozen.read_text(encoding="utf-8"))
            served = set(member.run for member in self._members)
            recorded = set(payload.get("runs", []))
            if recorded and recorded != served:
                self.log.warning(
                    "frozen ensemble calibration was fitted for %s but this service "
                    "is serving %s; ignoring it and falling back to the member rule",
                    sorted(recorded), sorted(served))
            else:
                self._calibration = payload["calibration"]
                self._strategy_name = str(self._calibration.get("best_strategy", "?"))
                self._calibration_source = "ensemble"
                self._calibration_meta = {
                    "fitted_at": payload.get("fitted_at"),
                    "n_calibration": payload.get("n_calibration"),
                    "n_tta": payload.get("n_tta"),
                    "runs": payload.get("runs"),
                }
                self.log.info("ensemble calibration loaded: strategy=%s, n=%s",
                              self._strategy_name, payload.get("n_calibration"))
                return

        decision_run = self.settings.echo_decision_run
        thresholds = json.loads(
            (self._run_dir(decision_run) / "thresholds.json").read_text(encoding="utf-8"))
        calibration = thresholds.get("calibration")
        if not isinstance(calibration, dict):
            raise RuntimeError(
                "run %r has a schema-v1 thresholds.json with no frozen calibration "
                "block; re-run its calibration step" % decision_run)
        self._calibration = calibration
        self._strategy_name = str(calibration.get("best_strategy", "unknown"))
        self._calibration_source = "member"
        self._calibration_meta = {"run": decision_run}
        self.log.info(
            "member calibration in use (run=%s). Run "
            "scripts/freeze_echo_ensemble_calibration.py to serve the "
            "ensemble-level rule the published figures describe.", decision_run)

    def _assert_members_comparable(self) -> None:
        """Refuse to average members trained under different preprocessing.

        Silently mixing members whose pixel normalisation differs would corrupt
        every prediction while the service still appeared healthy.
        """
        if len(self._members) < 2:
            return
        reference = self._members[0]
        for member in self._members[1:]:
            for field in ("ef_mean", "ef_std", "pixel_mean", "pixel_std"):
                a, b = getattr(reference, field), getattr(member, field)
                if abs(a - b) > 1e-6:
                    raise RuntimeError(
                        "ensemble members %r and %r disagree on %s (%.6f vs %.6f); "
                        "they were trained under different preprocessing and must not "
                        "be averaged" % (reference.run, member.run, field, a, b))

    # ---- inference ----------------------------------------------------
    def analyze(self, video_bytes: bytes, filename: str, **_: Any) -> Envelope:
        started = time.perf_counter()
        self.ensure_loaded()
        frames, source_meta = self._decode(video_bytes, filename)
        source_meta = dict(source_meta or {})
        source_meta["filename"] = filename

        try:
            with self.sandbox.active():
                predictions, per_member = self._predict(frames)
                decision = self._apply_frozen_strategy(predictions)
                # An explanation must never be able to cost a prediction. A
                # failure here loses the map and nothing else.
                try:
                    cam = self._gradcam(frames)
                except Exception as exc:       # noqa: BLE001
                    self.log.warning("Grad-CAM failed: %s", exc)
                    cam = {}
        except InvalidInput:
            raise
        except Exception as exc:               # noqa: BLE001
            raise InferenceFailed(
                "Echocardiogram analysis failed: %s" % exc,
                {"component": self.id}) from exc

        return self._to_envelope(predictions, per_member, decision,
                                 source_meta, started, cam=cam)

    def _decode(self, video_bytes: bytes, filename: str):
        """Uploaded file -> (T, 112, 112) uint8, matching the training cache."""
        import numpy as np

        if not video_bytes:
            raise InvalidInput("The uploaded study is empty.")
        suffix = Path(filename or "").suffix.lower()

        if suffix in ARRAY_SUFFIXES:
            # A cached clip from the component's own preprocessing stage.
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
                handle.write(video_bytes)
                temp_path = Path(handle.name)
            try:
                # allow_pickle stays off: a .npy is untrusted input here, and
                # pickle loading is arbitrary code execution.
                frames = np.load(temp_path, allow_pickle=False)
            except Exception as exc:           # noqa: BLE001
                raise InvalidInput(
                    "Could not read the .npy clip array: %s" % exc) from exc
            finally:
                temp_path.unlink(missing_ok=True)
            if frames.ndim != 3:
                raise InvalidInput(
                    "A .npy study must have shape (frames, height, width); got %s."
                    % (tuple(frames.shape),))
            frames = frames.astype(np.uint8, copy=False)
            meta = {"source": "cached_array", "n_frames": int(frames.shape[0]),
                    "height": int(frames.shape[1]), "width": int(frames.shape[2])}
        elif suffix in VIDEO_SUFFIXES:
            frame_size = int(getattr(self._cfg, "frame_size", 112))
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(video_bytes)
                temp_path = Path(handle.name)
            try:
                with self.sandbox.active():
                    frames, meta = self._io_utils.decode_video(
                        temp_path, size=frame_size, grayscale=True)
            except Exception as exc:           # noqa: BLE001
                raise InvalidInput("Could not decode the video: %s" % exc) from exc
            finally:
                temp_path.unlink(missing_ok=True)
            meta = dict(meta)
            meta["source"] = "decoded_video"
        else:
            raise InvalidInput(
                "Unsupported study format %r." % (suffix or "none"),
                {"accepted": list(VIDEO_SUFFIXES + ARRAY_SUFFIXES)})

        if frames.shape[0] < 2:
            raise InvalidInput(
                "The study has %d frame(s). Ejection fraction is defined by the "
                "end-diastole to end-systole volume change, so a single frame carries "
                "no signal." % frames.shape[0])
        return frames, meta

    def _predict(self, frames):
        """Deterministic multi-clip TTA, then averaging across ensemble members."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from models.uef_net import coral_class_distribution   # type: ignore

        cfg = self._cfg
        n_views = int(self.settings.echo_tta_clips)
        n_frames = int(frames.shape[0])

        per_member: List[Dict[str, Any]] = []
        for member in self._members:
            views = []
            for view_index in range(n_views):
                indices = self._sampling.sample_indices(
                    n_frames, cfg.clip_len, cfg.sampling_period,
                    # Label-free: the annotated ED/ES frames are deliberately not
                    # consulted, because they do not exist for a new study.
                    ed_frame=None, es_frame=None, train=False,
                    rng=np.random.default_rng(view_index),
                    view_index=view_index, n_views=n_views)
                indices = np.clip(indices, 0, n_frames - 1)
                views.append(self._sampling.build_multichannel(
                    np.asarray(frames[indices], dtype=np.uint8),
                    member.pixel_mean, member.pixel_std, cfg.motion_mode))

            batch = torch.from_numpy(
                np.ascontiguousarray(np.stack(views, axis=0))).float().to(self._device)

            with torch.no_grad():
                ef_z, ord_logits, aux = member.model(batch)
                ord_dist = coral_class_distribution(ord_logits.float())
                class_dist = (F.softmax(aux["class_logits"].float(), dim=1)
                              if isinstance(aux, dict) and aux.get("class_logits") is not None
                              else None)
                aleatoric_var = (torch.exp(aux["log_var"].float())
                                 if isinstance(aux, dict) and aux.get("log_var") is not None
                                 else None)

            if not torch.isfinite(ef_z).all() or not torch.isfinite(ord_dist).all():
                raise InferenceFailed(
                    "Non-finite prediction from ensemble member %r." % member.run,
                    {"component": self.id})

            ef_scale = abs(member.ef_std)
            record = {
                "run": member.run,
                # Study-level EF: mean over clips, de-standardised to EF points.
                "ef": float(ef_z.mean().item()) * member.ef_std + member.ef_mean,
                # Epistemic: how much the estimate moves with the cardiac cycle
                # that happened to be captured.
                "ef_epistemic_std": float(ef_z.std(unbiased=False).item()) * ef_scale,
                "ord_dist": ord_dist.mean(dim=0).cpu().numpy(),
                "class_dist": (class_dist.mean(dim=0).cpu().numpy()
                               if class_dist is not None else None),
                "ef_aleatoric_std": (float(aleatoric_var.mean().item()) ** 0.5 * ef_scale
                                     if aleatoric_var is not None else None),
            }
            per_member.append(record)

        ef_pred = float(np.mean([m["ef"] for m in per_member]))
        ef_pred = float(np.clip(ef_pred, 0.0, 100.0))
        ef_epistemic = float(np.mean([m["ef_epistemic_std"] for m in per_member]))
        ord_dist = np.mean([m["ord_dist"] for m in per_member], axis=0)

        class_dists = [m["class_dist"] for m in per_member if m["class_dist"] is not None]
        class_dist = (np.mean(class_dists, axis=0)
                      if len(class_dists) == len(per_member) else None)

        aleatoric = [m["ef_aleatoric_std"] for m in per_member
                     if m["ef_aleatoric_std"] is not None]
        ef_aleatoric = float(np.mean(aleatoric)) if len(aleatoric) == len(per_member) else None

        # Law of total variance, for reporting and for the boundary check.
        ef_total = (float((ef_aleatoric ** 2 + ef_epistemic ** 2) ** 0.5)
                    if ef_aleatoric is not None else ef_epistemic)

        # The conformal interval must be fed the SAME dispersion the calibration
        # was fitted against, which is the inter-clip (epistemic) spread alone:
        # engine/evaluate.py stores `ef_pred_std = view_std * ef_std`, and the
        # learned aleatoric sigma is carried separately and never enters q_hat.
        # Substituting the larger combined sigma here silently rescales every
        # interval -- on this repository it widened a 95 % interval from roughly
        # +/- 7 EF points to +/- 37, which is arithmetically valid and clinically
        # meaningless. Total sigma is reported, but it is not the interval scale.
        predictions: Dict[str, Any] = {
            "ef_pred": np.array([ef_pred], dtype=np.float64),
            "ef_pred_std": np.array([ef_epistemic], dtype=np.float64),
            "ord_dist": ord_dist[None, :],
            "ord_pred": np.array([int(ord_dist.argmax())], dtype=np.int64),
            "_ef_epistemic": ef_epistemic,
            "_ef_aleatoric": ef_aleatoric,
            "_ef_total": ef_total,
            "_n_views": n_views,
        }
        if class_dist is not None:
            predictions["class_dist"] = class_dist[None, :]
            predictions["class_pred"] = np.array([int(class_dist.argmax())], dtype=np.int64)
        return predictions, per_member

    def _apply_frozen_strategy(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the validation-frozen decision rule, using the component's code."""
        from engine.calibrate import apply_frozen_strategy      # type: ignore

        payload = {key: value for key, value in predictions.items()
                   if not key.startswith("_")}
        return apply_frozen_strategy(payload, self._calibration, self._cfg)

    # ---- translation --------------------------------------------------
    def _to_envelope(self, predictions: Dict[str, Any], per_member: List[Dict[str, Any]],
                     decision: Dict[str, Any], source_meta: Dict[str, Any],
                     started: float,
                     cam: Optional[Dict[str, Any]] = None) -> Envelope:
        ef_raw = float(decision["ef_raw"][0])
        ef_calibrated = float(decision["ef_calibrated"][0])
        operational = int(decision["operational_pred"][0])
        clinical = int(decision["clinical_pred"][0])

        interval: Optional[List[float]] = None
        if "interval_low" in decision:
            interval = [round(float(decision["interval_low"][0]), 2),
                        round(float(decision["interval_high"][0]), 2)]

        ordinal = predictions["ord_dist"][0]
        class_probs = (predictions["class_dist"][0]
                       if "class_dist" in predictions else None)

        findings: List[Finding] = [
            Finding(
                name="Left-ventricular ejection fraction",
                value=round(ef_calibrated, 2),
                unit="%",
                interval=interval,
                evidence="Mean over %d deterministic label-free clips, averaged across "
                         "%d ensemble member(s)."
                         % (predictions["_n_views"], len(per_member)),
            ),
            Finding(
                name="Severity grade",
                label=self._class_names[operational],
                probability=float(ordinal[operational]),
                evidence=CLASS_DESCRIPTIONS.get(operational, ""),
            ),
        ]
        for index, name in enumerate(self._class_names):
            findings.append(Finding(
                name=name,
                present=(index == operational),
                probability=float(ordinal[index]),
                label="ordinal head",
            ))

        raw: Dict[str, Any] = {
            "ef_raw": round(ef_raw, 4),
            "ef_calibrated": round(ef_calibrated, 4),
            "ef_interval_95": interval,
            "severity_class_index": operational,
            "severity_class": self._class_names[operational],
            "clinical_reference_class": self._class_names[clinical],
            "clinical_reference_note": (
                "Published boundaries 30/40/55 applied to the raw regression output, "
                "with no post-hoc rule. Reported alongside the operational grade "
                "because the two answer different questions: the clinical reference "
                "scores higher overall accuracy (0.796) while abandoning the minority "
                "classes (min-recall 0.442)."),
            "ordinal_distribution": {name: round(float(ordinal[i]), 4)
                                     for i, name in enumerate(self._class_names)},
            "uncertainty": {
                "aleatoric_ef_std": (round(predictions["_ef_aleatoric"], 3)
                                     if predictions["_ef_aleatoric"] is not None else None),
                "epistemic_ef_std": round(predictions["_ef_epistemic"], 3),
                "total_ef_std": round(predictions["_ef_total"], 3),
                "note": "Aleatoric from the learned log-variance head; epistemic from "
                        "inter-clip disagreement; combined by the law of total variance.",
            },
            "ensemble": [{"run": m["run"], "ef": round(m["ef"], 2)} for m in per_member],
            "tta_clips": predictions["_n_views"],
            "source": source_meta,
        }

        # The measured EF, for bundled EchoNet studies only. Attached beside the
        # estimate rather than folded into it: a reader comparing the two is
        # doing the check the number exists for.
        reference = self._reference_ef(source_meta.get("filename"))
        if reference is not None:
            raw["ground_truth_ef"] = {
                **reference,
                "predicted_ef": round(float(ef_calibrated), 2),
                "absolute_error": round(abs(float(ef_calibrated) - reference["ef"]), 2),
                "within_interval": bool(
                    interval is not None
                    and float(interval[0]) <= reference["ef"] <= float(interval[1])),
            }
        if class_probs is not None:
            raw["auxiliary_class_distribution"] = {
                name: round(float(class_probs[i]), 4)
                for i, name in enumerate(self._class_names)}

        explanation = {
            "method": "Multi-clip test-time augmentation with an explicit motion channel",
            "motion_channel": "Signed temporal difference between consecutive frames, "
                              "giving the network wall motion without extra labels.",
            "clip_sampling": "Label-free: %d clips spread evenly across the valid window. "
                             "Expert ED/ES tracings are used for training-time sampling "
                             "only, never at inference."
                             % predictions["_n_views"],
            "per_clip_ef_spread": round(predictions["_ef_epistemic"], 3),
        }
        if cam:
            explanation["gradcam"] = cam

        return self.envelope(
            headline="EF %.1f %% -- %s" % (ef_calibrated, self._class_names[operational]),
            findings=findings,
            reliability=self._reliability(predictions, ef_calibrated, interval, operational),
            raw=raw,
            started=started,
            explanation=explanation,
            decision_rule=self._describe_decision_rule(len(per_member)),
        )

    # ---- ground truth for bundled studies -------------------------------
    def _reference_ef(self, filename: Optional[str]) -> Optional[Dict[str, Any]]:
        """The measured ejection fraction for a bundled study, if it is one.

        EchoNet-Dynamic ships a human-traced EF per study in `FileList.csv`, and
        the bundled clips keep the EchoNet identifier in their filename --
        `moderate_01_0X7EEA66DBE251854B.npy`. That makes the true value
        recoverable for exactly the studies that came from the benchmark, and
        for nothing else.

        An arbitrary upload has no reference, and returning one would be worse
        than returning none: a number beside a prediction is read as the answer,
        and this must never show a different patient's measurement.

        Read from the demo manifest rather than from `FileList.csv`, so the
        component's own dataset directory is not a runtime dependency of the
        service. Absent manifest, absent reference, and the response simply
        carries no ground truth.
        """
        if not filename:
            return None
        if self._reference_efs is None:
            self._reference_efs = self._load_reference_efs()
        return self._reference_efs.get(str(filename))

    def _load_reference_efs(self) -> Dict[str, Dict[str, Any]]:
        import json
        import re

        table: Dict[str, Dict[str, Any]] = {}
        try:
            here = Path(__file__).resolve()
            manifest = None
            for parent in here.parents:
                candidate = parent / "demo" / "manifest.json"
                if candidate.exists():
                    manifest = candidate
                    break
            if manifest is None:
                return table

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            studies = (payload.get("components", {})
                       .get("echo", {}).get("studies", []) or [])
            for study in studies:
                name = study.get("file")
                truth = str(study.get("ground_truth") or "")
                match = re.search(r"true EF\s*([0-9.]+)", truth)
                if not name or not match:
                    continue
                grade = truth.split("(")[0].strip() or None
                table[str(name)] = {
                    "ef": float(match.group(1)),
                    "grade": grade,
                    "source": "EchoNet-Dynamic FileList.csv (human tracing)",
                }
        except Exception as exc:                       # noqa: BLE001
            self.log.warning("reference EF table unreadable (%s)", exc)
        return table

    # ---- Grad-CAM ---------------------------------------------------------
    def _cam_target_layer(self, model):
        """The last spatiotemporal convolutional block.

        `layer4` by name for both torchvision video backbones this component
        supports, with a search for the final Conv3d as a fallback so a backbone
        swap degrades to no map rather than to a wrong one.
        """
        layer = getattr(getattr(model, "backbone", None), "layer4", None)
        if layer is not None:
            return layer
        last = None
        for module in model.modules():
            if module.__class__.__name__ == "Conv3d":
                last = module
        return last

    def _gradcam(self, frames, top_k: int = 3) -> Dict[str, Any]:
        """Where in the loop the ejection fraction came from.

        WHAT IS BEING EXPLAINED
        -----------------------
        The gradient is taken of `ef_z` -- the continuous regression output, in
        standardised units -- and not of the ordinal head. The grade the report
        shows is a threshold applied to EF, so EF is the quantity the model
        actually estimates and the ordinal logits sit downstream of it. Taking
        the gradient of a threshold crossing would explain the boundary rather
        than the measurement.

        ONE CLIP, NOT THE AVERAGE
        -------------------------
        The reported EF averages several clips and both ensemble members. A
        saliency map averaged the same way would blur across different cardiac
        phases and mean very little, so the map is computed for ONE clip of the
        first member and is labelled as such. It explains a clip, and the
        response says which one; it does not explain the ensemble mean.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        member = self._members[0]
        cfg = self._cfg
        n_frames = int(frames.shape[0])
        n_views = int(self.settings.echo_tta_clips)
        view_index = n_views // 2          # the middle clip, not an edge one

        indices = self._sampling.sample_indices(
            n_frames, cfg.clip_len, cfg.sampling_period,
            ed_frame=None, es_frame=None, train=False,
            rng=np.random.default_rng(view_index),
            view_index=view_index, n_views=n_views)
        indices = np.clip(indices, 0, n_frames - 1)
        clip = self._sampling.build_multichannel(
            np.asarray(frames[indices], dtype=np.uint8),
            member.pixel_mean, member.pixel_std, cfg.motion_mode)

        batch = torch.from_numpy(
            np.ascontiguousarray(clip[None])).float().to(self._device)
        # The parameters may or may not carry requires_grad depending on how the
        # checkpoint was frozen. Making the INPUT require grad guarantees the
        # graph reaches the target layer either way.
        batch.requires_grad_(True)

        target_layer = self._cam_target_layer(member.model)
        if target_layer is None:
            return {}

        captured: Dict[str, Any] = {}

        def _capture(_module, _inputs, output):
            captured["activations"] = output

        handle = target_layer.register_forward_hook(_capture)
        try:
            with torch.enable_grad():
                ef_z, _ordinal, _aux = member.model(batch)
                activations = captured.get("activations")
                if activations is None:
                    return {}
                gradients = torch.autograd.grad(ef_z[0], activations)[0]
        finally:
            handle.remove()
            member.model.zero_grad(set_to_none=True)

        activations = activations.detach()

        # Channel weights are the spatially averaged gradients; the map is the
        # positively contributing part of that weighted sum.
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))

        # Record the resolution the evidence actually has, BEFORE interpolation.
        # On this backbone the map is 4 x 7 x 7 and it is about to be stretched
        # to 32 x 112 x 112 -- a 16-fold spatial upsample. The overlay therefore
        # looks far more precise than the 49 numbers per bin behind it, and a
        # reader who is not told that will over-read the smooth edges.
        native = tuple(int(v) for v in cam.shape[2:])
        native_bins = cam[0, 0].detach().cpu().numpy()
        live_bins = int((native_bins.reshape(native[0], -1).max(axis=1) > 0).sum())

        clip_len, height, width = clip.shape[1], clip.shape[2], clip.shape[3]
        cam = F.interpolate(cam, size=(clip_len, height, width),
                            mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float64)

        ceiling = float(cam.max())
        if not np.isfinite(ceiling) or ceiling <= 0:
            # A flat map is a real outcome, not an error: it means no region of
            # this clip raised the estimate. Saying so beats rendering noise
            # stretched to look like structure.
            return {"degenerate": True,
                    "note": "Grad-CAM was uniformly zero for this clip; no region "
                            "raised the ejection-fraction estimate."}
        cam = cam / ceiling

        per_frame = cam.mean(axis=(1, 2))
        order = list(np.argsort(-per_frame)[:max(1, top_k)])

        # Judge flatness on RELATIVE spread. The absolute mean is small whenever
        # the map is spatially sparse -- most voxels are zero after the ReLU, so
        # a frame average of 0.03 is normal and says nothing about whether the
        # attribution varies over the cycle.
        peak = float(per_frame.max())
        spread = (peak - float(per_frame.min())) / peak if peak > 0 else 0.0

        # The curve is reported normalised to its own peak so it can be drawn;
        # `frame_importance_peak` keeps the absolute height, and `concentration`
        # states how much of the volume carries the mass.
        readable = (per_frame / peak) if peak > 0 else per_frame

        payload: Dict[str, Any] = {
            "frame_importance": [round(float(v), 4) for v in readable],
            "frame_importance_peak": round(peak, 5),
            "frame_importance_spread": round(float(spread), 3),
            "native_resolution": {
                "temporal_bins": native[0],
                "spatial": "%d x %d" % (native[1], native[2]),
                "upsampled_to": "%d x %d x %d" % (clip_len, height, width),
                "temporal_bins_with_signal": live_bins,
                "caveat": ("The map is computed at %d x %d x %d and interpolated up. "
                           "Its real resolution is %d values per time bin across %d "
                           "bins, so smooth edges in the overlay are interpolation, "
                           "not evidence."
                           % (native[0], native[1], native[2],
                              native[1] * native[2], native[0])),
            },
            "concentration": {
                "above_0.5": round(float((cam > 0.5).mean()), 5),
                "above_0.25": round(float((cam > 0.25).mean()), 5),
                "note": ("Share of the clip volume above each level. A very small "
                         "share means the estimate rests on a focal region rather "
                         "than the whole image."),
            },
            "source_frame_indices": [int(i) for i in indices],
            "clip_index": int(view_index),
            "clip_count": int(n_views),
            "member_run": member.run,
            "target": "ef_z (continuous ejection fraction), not the ordinal grade",
            "note": ("Grad-CAM over the last spatiotemporal convolution for ONE of "
                     "the %d clips, from ensemble member %s. The reported EF is the "
                     "mean over every clip and member, so this map explains a clip "
                     "rather than the reported number."
                     % (n_views, member.run)),
        }
        if spread < 0.15:
            payload["flat_attribution"] = True
            payload["flat_attribution_note"] = (
                "Frame importance varies by less than 15 %% of its peak across the "
                "clip, so the estimate does not rest on any particular phase of the "
                "cycle.")
        if live_bins < native[0]:
            payload["partial_temporal_support"] = (
                "%d of the %d temporal bins carry no positive attribution at all. "
                "The estimate draws on part of the clip only."
                % (native[0] - live_bins, native[0]))

        overlays = []
        for rank, frame_index in enumerate(order):
            image = self._overlay_frame(
                np.asarray(frames[indices[int(frame_index)]]), cam[int(frame_index)])
            if image:
                overlays.append({
                    "rank": rank + 1,
                    "clip_frame": int(frame_index),
                    "source_frame": int(indices[int(frame_index)]),
                    "importance": round(float(per_frame[int(frame_index)]), 4),
                    "png_base64": image,
                })
        if overlays:
            payload["frames"] = overlays
        return payload

    @staticmethod
    def _overlay_frame(frame, cam_frame) -> str:
        """One grayscale frame with the map composited over it, as a PNG."""
        import base64
        import io as _io

        import numpy as np

        try:
            from PIL import Image
            from matplotlib import colormaps
        except Exception:                              # noqa: BLE001
            return ""

        try:
            grey = np.asarray(frame)
            if grey.ndim == 3:
                grey = grey.mean(axis=2)
            grey = grey.astype(np.float64)
            spread = float(grey.max() - grey.min())
            grey = (grey - grey.min()) / spread if spread > 0 else np.zeros_like(grey)
            base = np.repeat(grey[:, :, None], 3, axis=2)

            heat = np.asarray(colormaps["inferno"](
                np.clip(cam_frame, 0.0, 1.0)))[:, :, :3]

            # Weight the blend by the map itself, so cold regions stay a clean
            # echo image instead of being tinted dark everywhere.
            alpha = np.clip(cam_frame, 0.0, 1.0)[:, :, None] * 0.65
            blended = np.clip(base * (1.0 - alpha) + heat * alpha, 0.0, 1.0)

            buffer = _io.BytesIO()
            Image.fromarray((blended * 255).astype(np.uint8)).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception:                              # noqa: BLE001
            return ""

    def _describe_decision_rule(self, n_members: int) -> str:
        """State precisely which frozen rule produced this grade."""
        if self._calibration_source == "ensemble":
            return ("%s, fitted on the %d-member ensemble's validation predictions "
                    "(n=%s) and frozen -- the rule the published figures describe"
                    % (self._strategy_name, n_members,
                       self._calibration_meta.get("n_calibration", "?")))
        return ("%s, frozen on validation for run %r and applied to the %d-member "
                "ensemble average. This is a MEMBER-level rule, not the "
                "ensemble-level rule behind the published MAE 3.979 / min-recall "
                "0.723; run scripts/freeze_echo_ensemble_calibration.py to serve that."
                % (self._strategy_name, self.settings.echo_decision_run, n_members))

    def _reliability(self, predictions: Dict[str, Any], ef: float,
                     interval: Optional[List[float]], predicted_class: int) -> Reliability:
        """Trust is driven by boundary proximity and by predictive spread.

        Both are the component's own measurements. 36.7 % of the test cohort
        lies within one MAE of a severity boundary, and the dominant error mode
        (Normal -> Mild, 68 % of all errors) is the crowd at EF = 55. A study
        whose conformal interval straddles a boundary is exactly that case.
        """
        thresholds = list(getattr(self._cfg, "EF_THRESHOLDS", (30.0, 40.0, 55.0)))
        reasons: List[str] = []
        guarantees: List[str] = []

        conformal = self._calibration.get("conformal") or {}
        if interval and conformal:
            alpha = float(conformal.get("alpha", 0.1))
            guarantees.append(
                "Split-conformal prediction interval with %.0f %% marginal coverage, "
                "calibrated on %s validation studies."
                % (100 * (1 - alpha), conformal.get("n_calibration", "?")))

        straddled = [t for t in thresholds
                     if interval and interval[0] <= t <= interval[1]]
        distance = min(abs(ef - t) for t in thresholds)

        actionability = Actionability.ACTIONABLE
        level = "standard"

        if straddled:
            actionability = Actionability.DEFERRED
            level = "boundary_ambiguous"
            reasons.append(
                "The prediction interval %s spans the %s %% severity boundary, so the "
                "grade is not separable from the adjacent class at this confidence. "
                "Grading is deferred; the EF estimate itself still stands."
                % (interval, ", ".join("%g" % t for t in straddled)))
        elif distance <= predictions["_ef_total"]:
            actionability = Actionability.CAUTION
            level = "near_boundary"
            reasons.append(
                "EF %.1f %% lies within one predictive standard deviation (%.1f points) "
                "of a severity boundary. 36.7 %% of the test cohort sits within one MAE "
                "of a threshold, and that crowd produces 68 %% of all misclassifications."
                % (ef, predictions["_ef_total"]))

        if len(self._members) < 3:
            actionability = Actionability.worst([actionability, Actionability.CAUTION])
            reasons.append(
                "Serving %d of 3 trained seeds. The published figures (MAE 3.979, "
                "min-recall 0.723) are for the full three-seed ensemble; fewer members "
                "measured MAE 4.138 at one seed and 3.994 at two."
                % len(self._members))

        reasons.append(
            "MAE 3.979 EF points is at the level of human inter-observer disagreement "
            "(reported 4-5 points), so a difference of this size against a reader is "
            "not evidence of model error.")

        return Reliability(
            actionability=actionability,
            level=level,
            reasons=reasons,
            guarantees=guarantees,
            guarantees_void=False,
        )

    # ---- documentation ------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        return {
            "test_set_n": 1277,
            "ensemble_seeds": 3,
            "regression": {"mae": 3.979, "rmse": 5.211, "r2": 0.818},
            "classification": {
                "overall_accuracy": 0.7298,
                "balanced_accuracy": 0.7366,
                "macro_f1": 0.6844,
                "min_class_recall": 0.7229,
                "within_one_class": 0.9969,
                "catastrophic_errors": 0,
            },
            "per_class_recall": {
                "Severe(<30)": 0.723, "Moderate(30-40)": 0.766,
                "Mild(40-55)": 0.730, "Normal(>=55)": 0.727,
            },
            "clinical_reference_operating_point": {
                "overall_accuracy": 0.796, "min_class_recall": 0.442,
            },
        }

    def limitations(self) -> List[str]:
        return [
            "The >= 0.75-on-all-classes target is not met (recalls 0.723-0.766). Three "
            "measured bounds explain it: min-recall cannot exceed balanced accuracy "
            "(0.737), 36.7 % of studies lie within one MAE of a boundary, and the "
            "label-noise floor is MAE >= 0.8 sigma, i.e. 3.2-4.0 at reported "
            "inter-observer variability.",
            "Selective prediction does NOT repair worst-class recall here. Abstention "
            "raises overall accuracy 0.730 -> 0.929 while min-recall falls, because the "
            "minority classes occupy the boundary region abstention removes.",
            "Severe contains 83 test studies, giving a 95 % Wilson interval of roughly "
            "+/- 9.5 % on its recall. Single-figure comparisons at that size are fragile.",
            "Single-cohort evaluation. CAMUS is used for training only; no external "
            "cohort is held out for testing.",
            "No demographic fairness analysis is possible: EchoNet-Dynamic carries no "
            "age, sex or ethnicity fields. Subgroup robustness is assessed only over "
            "acquisition characteristics.",
            "Precision on the interior classes is low (Moderate 0.434, Mild 0.413), a "
            "direct consequence of optimising worst-class recall under 11:1 imbalance.",
            "Apical four-chamber only. Simpson's biplane uses two views; EchoNet "
            "provides one.",
        ]
