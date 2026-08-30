"""
zoo.py — Run more than one model at once, each with its own safety layer.

WHY THIS EXISTS
───────────────────────────────────────────────────────────────────────────
`ECGPipeline` serves exactly one model, and `backend/server.py` refuses to
start if that model's calibrator and conformal thresholds were fitted for a
different network (correctly — a calibrator is only valid for the logits it was
fitted on). That is the right rule, but it made a second model impossible to
load: the assets live at fixed filenames, so fitting a second model overwrote
the first one's safety layer.

`ModelZoo` keeps one *bundle* per model — checkpoint + calibrator + conformal
thresholds + provenance — so two models can be served side by side without
either borrowing the other's guarantees.

THE TWO-MODEL DECISION RULE
───────────────────────────────────────────────────────────────────────────
With two independently calibrated models the interesting question is not "which
is more accurate" but "what do we do when they disagree". `ModelZoo.consensus`
merges them conservatively:

    a class may be RULED OUT only if EVERY model rules it out;
    any disagreement collapses to REFER.

This is not a heuristic — it preserves the conformal guarantee in the direction
that matters. A true positive is missed by the merged rule only if all models
rule it out, so

    P(merged rules out | Y = 1)  <=  min_m  P(model m rules out | Y = 1)
                                 <=  min_m  alpha_m

i.e. the merged miss rate is no worse than the *best* single model's bound. The
cost is a higher referral rate, which `ModelZoo` reports rather than hides —
the same honest trade the single-model triage already makes.

USAGE
───────────────────────────────────────────────────────────────────────────
    from src.zoo import ModelZoo

    zoo = ModelZoo.discover()                    # finds every bundle on disk
    print(zoo.describe())

    res  = zoo.analyse(signal, fs=500)                    # default model
    res  = zoo.analyse(signal, fs=500, model="resnet")    # a named model
    cons = zoo.consensus(signal, fs=500)                  # every ready model

ASSET LAYOUT
───────────────────────────────────────────────────────────────────────────
Per-model directories are preferred; the flat Progress-2 layout still works so
nothing that exists today breaks:

    checkpoints/best_model.pt                    <- flat (legacy, still read)
    checkpoints/calibrator.json
    checkpoints/conformal_triage.json

    checkpoints/resnet/best_model.pt             <- per-model (preferred)
    checkpoints/resnet/calibrator.json
    checkpoints/resnet/conformal_triage.json
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import paths
from .conformal import REFER, RULE_IN, RULE_OUT
from .models import (CLASS_NAMES, DEFAULT_MODEL, SAMPLING_RATE, ModelSpec,
                     get_spec, resolve_model_name)
from .pipeline import AnalysisResult, ECGPipeline

CHECKPOINT_FILE = "best_model.pt"
CALIBRATOR_FILE = "calibrator.json"
TRIAGE_FILE = "conformal_triage.json"

#: zone pairs that mean the models reached opposite conclusions
_OPPOSITE = frozenset({(RULE_IN, RULE_OUT), (RULE_OUT, RULE_IN)})


def _declared_filter(*artefact_paths: Optional[str]) -> Optional[bool]:
    """`fitted_for.filter` as recorded by train/fit_calibration.py, or None.

    Artefacts written before provenance existed carry no `fitted_for`; those
    return None so the caller falls back to an explicit default.
    """
    for path in artefact_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh).get("fitted_for") or {}
        except (OSError, ValueError):
            continue
        if "filter" in meta:
            return bool(meta["filter"])
    return None


# ══════════════════════════════════════════════════════════════════════════
#  ASSETS
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModelAssets:
    """Where one model's artefacts live, and how that model must be run."""

    name: str
    ckpt: str
    calibrator: Optional[str] = None
    triage: Optional[str] = None
    do_filter: bool = True

    @classmethod
    def from_dir(cls, name: str, directory: str,
                 do_filter: Optional[bool] = None) -> "ModelAssets":
        """Assets in `directory`, with preprocessing read from the artefacts.

        `do_filter` is deliberately per-model, not global: the Progress-1
        baseline was trained on unfiltered signals and the shipped model on
        band-passed ones. Serving either the other's preprocessing is a silent
        train/serve mismatch (it moved macro-ECE 0.183 -> 0.209 during
        development), so the flag is taken from the calibrator's own provenance
        rather than assumed.
        """
        def _opt(fn: str) -> Optional[str]:
            p = os.path.join(directory, fn)
            return p if os.path.exists(p) else None

        calibrator, triage = _opt(CALIBRATOR_FILE), _opt(TRIAGE_FILE)
        resolved = do_filter
        if resolved is None:
            resolved = _declared_filter(calibrator, triage)
        if resolved is None:
            resolved = True                 # matches the shipped model

        return cls(name=name,
                   ckpt=os.path.join(directory, CHECKPOINT_FILE),
                   calibrator=calibrator,
                   triage=triage,
                   do_filter=bool(resolved))

    @property
    def exists(self) -> bool:
        return os.path.exists(self.ckpt)


# ══════════════════════════════════════════════════════════════════════════
#  ONE LOADED MODEL
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ModelBundle:
    """A loaded model plus the safety layer that is valid *for that model*."""

    name: str
    spec: ModelSpec
    pipeline: ECGPipeline
    assets: ModelAssets
    issues: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when the model has a calibrator and thresholds that match it.

        A bundle that is not ready can still be used for research (raw
        probabilities), but it must not be served as a triage decision —
        `ModelZoo.consensus` skips it.
        """
        return not self.issues

    @property
    def has_guarantee(self) -> bool:
        return self.pipeline.triage is not None

    def to_dict(self) -> Dict:
        tri = self.pipeline.triage
        return {
            "name": self.name,
            "description": self.spec.description,
            "isAblation": self.spec.is_ablation,
            "ready": self.ready,
            "issues": list(self.issues),
            "calibrated": self.pipeline.calibrator is not None,
            "guarantee": ({"delta": getattr(tri, "delta", None),
                           "alpha": getattr(tri, "alpha", None)}
                          if tri is not None else None),
            "checkpoint": os.path.basename(self.assets.ckpt),
            "filter": self.assets.do_filter,
        }


def _provenance_issues(obj, label: str, model_name: str,
                       do_filter: bool) -> List[str]:
    """Report — never silently accept — a safety layer fitted for another model."""
    if obj is None:
        return ["%s is missing (run train/fit_calibration.py)" % label]
    meta = getattr(obj, "fitted_for", None) or {}
    if not meta:
        return []                       # pre-provenance artefact; caller decides
    want = (meta.get("model"), bool(meta.get("filter", False)))
    got = (model_name, bool(do_filter))
    if want != got:
        return ["%s was fitted for model=%s filter=%s, not model=%s filter=%s"
                % (label, want[0], want[1], got[0], got[1])]
    return []


def load_bundle(assets: ModelAssets, norm_stats_path: Optional[str] = None,
                device: str = "cpu") -> ModelBundle:
    """Load one model and its safety layer, collecting provenance problems."""
    spec = get_spec(assets.name)
    norm_stats_path = norm_stats_path or paths.require("norm_stats.json")

    # from_checkpoint warns on a provenance mismatch; we record it instead, so
    # the decision can be made per bundle rather than for the whole process.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe = ECGPipeline.from_checkpoint(
            ckpt_path=assets.ckpt,
            norm_stats_path=norm_stats_path,
            model_name=assets.name,
            calibrator_path=assets.calibrator,
            triage_path=assets.triage,
            device=device,
            do_filter=assets.do_filter)

    issues: List[str] = []
    issues += _provenance_issues(pipe.calibrator, "calibrator",
                                 assets.name, assets.do_filter)
    issues += _provenance_issues(pipe.triage, "conformal triage",
                                 assets.name, assets.do_filter)
    return ModelBundle(name=assets.name, spec=spec, pipeline=pipe,
                       assets=assets, issues=issues)


# ══════════════════════════════════════════════════════════════════════════
#  DISAGREEMENT
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ClassDisagreement:
    """One class on which the models did not reach the same triage zone."""

    class_name: str
    zones: Dict[str, str]
    severity: str                       # "major" | "minor"

    @property
    def opposite(self) -> bool:
        """One model ruled it in while another ruled it out."""
        return self.severity == "major"

    def to_dict(self) -> Dict:
        return {"class": self.class_name, "zones": dict(self.zones),
                "severity": self.severity}


@dataclass
class ConsensusResult:
    """What two or more models jointly concluded, and where they diverged."""

    models: List[str]
    results: Dict[str, AnalysisResult]
    zones: Dict[str, Dict[str, str]]            # model -> class -> zone
    probs: Dict[str, Dict[str, float]]          # model -> class -> calibrated p
    consensus_zones: Dict[str, str]
    disagreements: List[ClassDisagreement]
    class_names: List[str]

    @property
    def concordant(self) -> bool:
        return not self.disagreements

    @property
    def escalate(self) -> bool:
        """Any opposite conclusion — the case a single model would have hidden."""
        return any(d.opposite for d in self.disagreements)

    @property
    def primary(self) -> AnalysisResult:
        """The lead model's full result — report, XAI, quality, signal."""
        return self.results[self.models[0]]

    def prob_gap(self) -> Dict[str, float]:
        """Per class, the spread between the models' calibrated probabilities."""
        out = {}
        for c in self.class_names:
            vals = [self.probs[m][c] for m in self.models if c in self.probs[m]]
            out[c] = float(max(vals) - min(vals)) if vals else 0.0
        return out

    def summary(self) -> str:
        if self.concordant:
            return ("%d models concordant on all %d classes."
                    % (len(self.models), len(self.class_names)))
        bits = ["%s (%s)" % (d.class_name, "/".join(sorted(set(d.zones.values()))))
                for d in self.disagreements]
        head = "OPPOSITE CONCLUSIONS" if self.escalate else "partial disagreement"
        return "%s on %d class(es): %s" % (head, len(self.disagreements),
                                           ", ".join(bits))

    def to_dict(self) -> Dict:
        return {
            "models": list(self.models),
            "zones": {m: dict(z) for m, z in self.zones.items()},
            "probs": {m: dict(p) for m, p in self.probs.items()},
            "consensusZones": dict(self.consensus_zones),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "concordant": self.concordant,
            "escalate": self.escalate,
            "probGap": self.prob_gap(),
            "summary": self.summary(),
        }


def _merge_zones(per_model: Dict[str, Dict[str, str]],
                 class_names: Sequence[str]) -> Dict[str, str]:
    """Conservative merge: rule out only on unanimity, otherwise refer.

    See the module docstring — the merged rule-out set is the intersection of
    the individual rule-out sets, so its miss rate is bounded by the tightest
    individual alpha.
    """
    merged = {}
    for c in class_names:
        zones = {z.get(c, REFER) for z in per_model.values()}
        if len(zones) == 1:
            merged[c] = zones.pop()
        elif zones == {RULE_IN, REFER}:
            merged[c] = RULE_IN         # refer is weaker than rule-in; keep the finding
        else:
            merged[c] = REFER           # never resolve a disagreement to RULE_OUT
    return merged


def _disagreements(per_model: Dict[str, Dict[str, str]],
                   class_names: Sequence[str]) -> List[ClassDisagreement]:
    out = []
    for c in class_names:
        zones = {m: z.get(c, REFER) for m, z in per_model.items()}
        distinct = set(zones.values())
        if len(distinct) == 1:
            continue
        severity = "major" if any((a, b) in _OPPOSITE
                                  for a in distinct for b in distinct) else "minor"
        out.append(ClassDisagreement(class_name=c, zones=zones, severity=severity))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  THE ZOO
# ══════════════════════════════════════════════════════════════════════════
class ModelZoo:
    """A named collection of loaded models, each with its own safety layer."""

    def __init__(self, bundles: Sequence[ModelBundle],
                 default: Optional[str] = None,
                 class_names: Optional[Sequence[str]] = None):
        if not bundles:
            raise ValueError("ModelZoo needs at least one bundle")
        self._bundles: Dict[str, ModelBundle] = {b.name: b for b in bundles}
        self.class_names = list(class_names or CLASS_NAMES)
        ready = [b.name for b in bundles if b.ready]
        self.default = default or (ready[0] if ready else bundles[0].name)
        if self.default not in self._bundles:
            raise ValueError("default model %r is not in the zoo" % self.default)

    # ── collection protocol ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._bundles)

    def __contains__(self, name: object) -> bool:
        try:
            return resolve_model_name(str(name)) in self._bundles
        except ValueError:
            return False

    def __iter__(self):
        return iter(self._bundles.values())

    def names(self, ready_only: bool = False) -> List[str]:
        return [n for n, b in self._bundles.items() if b.ready or not ready_only]

    def get(self, name: Optional[str] = None) -> ModelBundle:
        """Bundle for `name` (canonical or alias); the default when omitted."""
        key = self.default if name is None else resolve_model_name(name)
        try:
            return self._bundles[key]
        except KeyError:
            raise KeyError("model %r is not loaded. Loaded: %s"
                           % (key, ", ".join(self._bundles))) from None

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def discover(cls, checkpoints_dir: Optional[str] = None,
                 norm_stats_path: Optional[str] = None,
                 device: str = "cpu", do_filter: Optional[bool] = None,
                 default: Optional[str] = None,
                 extra: Optional[Sequence[ModelAssets]] = None,
                 strict: bool = False) -> "ModelZoo":
        """Load every model found on disk.

        Looks for `checkpoints/<name>/best_model.pt` per-model directories, plus
        the flat `checkpoints/best_model.pt` layout that Progress 2 shipped —
        whose model name is read from the checkpoint itself, not guessed.

        `do_filter=None` (the default) lets each model declare its own
        preprocessing through its calibrator provenance; pass a bool only to
        force one setting on every model.

        `strict=True` raises if any discovered bundle has a provenance problem;
        the default records the problem and marks that bundle not-ready.
        """
        import torch                    # local: keeps torch off `import src.zoo`

        checkpoints_dir = checkpoints_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints")
        found: List[ModelAssets] = []

        flat = os.path.join(checkpoints_dir, CHECKPOINT_FILE)
        if os.path.exists(flat):
            state = torch.load(flat, map_location="cpu", weights_only=False)
            name = state.get("model_name") or os.environ.get("ECG_MODEL",
                                                             DEFAULT_MODEL)
            found.append(ModelAssets.from_dir(resolve_model_name(name),
                                              checkpoints_dir, do_filter))

        for entry in sorted(os.listdir(checkpoints_dir)):
            sub = os.path.join(checkpoints_dir, entry)
            if not os.path.isdir(sub) or not os.path.exists(
                    os.path.join(sub, CHECKPOINT_FILE)):
                continue
            try:
                name = resolve_model_name(entry)
            except ValueError:
                warnings.warn("checkpoints/%s/ is not a registered model name — "
                              "skipping" % entry, RuntimeWarning, stacklevel=2)
                continue
            if any(a.name == name for a in found):
                continue                # a per-model dir never shadows the flat one
            found.append(ModelAssets.from_dir(name, sub, do_filter))

        found += list(extra or [])
        if not found:
            raise SystemExit(
                "No model checkpoints found under %s.\nExpected %s there or in a "
                "per-model subdirectory." % (checkpoints_dir, CHECKPOINT_FILE))

        bundles = [load_bundle(a, norm_stats_path, device)
                   for a in found if a.exists]
        if strict:
            broken = {b.name: b.issues for b in bundles if not b.ready}
            if broken:
                lines = ["  - %s: %s" % (n, "; ".join(v)) for n, v in broken.items()]
                raise SystemExit("Refusing to build the zoo — mismatched safety "
                                 "layers:\n" + "\n".join(lines))
        return cls(bundles, default=default)

    # ── inference ────────────────────────────────────────────────────────
    def analyse(self, signal_raw: np.ndarray, fs: int = SAMPLING_RATE,
                model: Optional[str] = None, **kw) -> AnalysisResult:
        """Full single-model analysis, identical to `ECGPipeline.analyse`."""
        return self.get(model).pipeline.analyse(signal_raw, fs=fs, **kw)

    def consensus(self, signal_raw: np.ndarray, fs: int = SAMPLING_RATE,
                  models: Optional[Sequence[str]] = None,
                  xai_from: str = "primary", **kw) -> ConsensusResult:
        """Run several models on one record and reconcile their triage zones.

        `xai_from` controls the expensive part — integrated gradients and
        Grad-CAM dominate the ~6 s analysis, and a second set of explanations is
        rarely what the clinician needs:
          "primary" (default)  explanations from the lead model only
          "all"                explanations from every model
          "none"               no explanations at all
        """
        if xai_from not in ("primary", "all", "none"):
            raise ValueError("xai_from must be 'primary', 'all' or 'none'")

        chosen = ([self.get(m).name for m in models] if models
                  else self.names(ready_only=True))
        if not chosen:
            raise RuntimeError(
                "No model in the zoo has a matching calibrator and conformal "
                "thresholds. Fit them with train/fit_calibration.py --out-dir.")
        if self.default in chosen:                  # lead model goes first
            chosen = [self.default] + [m for m in chosen if m != self.default]

        results: Dict[str, AnalysisResult] = {}
        for i, name in enumerate(chosen):
            want_xai = (xai_from == "all") or (xai_from == "primary" and i == 0)
            results[name] = self.get(name).pipeline.analyse(
                signal_raw, fs=fs, with_xai=want_xai, **kw)

        zones = {m: dict(r.zones or {}) for m, r in results.items()}
        probs = {m: dict(r.probs_calibrated or {}) for m, r in results.items()}
        return ConsensusResult(
            models=chosen, results=results, zones=zones, probs=probs,
            consensus_zones=_merge_zones(zones, self.class_names),
            disagreements=_disagreements(zones, self.class_names),
            class_names=list(self.class_names))

    # ── reporting ────────────────────────────────────────────────────────
    def describe(self) -> str:
        """Startup banner — what is loaded and whether it may be served."""
        lines = ["  %d model(s) loaded, default = %s"
                 % (len(self._bundles), self.default)]
        for b in self._bundles.values():
            tri = b.pipeline.triage
            guarantee = ("PAC delta=%s" % getattr(tri, "delta", None)
                         if tri is not None else "NO GUARANTEE")
            flag = "ready" if b.ready else "NOT SERVEABLE"
            lines.append("    - %-20s %-18s %s" % (b.name, guarantee, flag))
            for issue in b.issues:
                lines.append("        ! %s" % issue)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {"default": self.default,
                "models": [b.to_dict() for b in self._bundles.values()]}
