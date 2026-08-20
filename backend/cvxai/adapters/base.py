"""
The adapter contract.

An adapter owns exactly three responsibilities for one component:

  1. decide whether the component *can* serve (root, weights, dependencies);
  2. load it once, inside its own module sandbox;
  3. translate one study into the shared `Envelope`.

It must not reimplement any of the component's science. Thresholds, calibration
maps, conformal bounds and decision rules are read from the component's own
frozen artefacts and applied by the component's own code wherever that code
exists. Where a component ships no serving path at all (Component 03), the
adapter reproduces the published evaluation protocol step for step and says so
in its model card.
"""
from __future__ import annotations

import abc
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cvxai.core.errors import ComponentUnavailable
from cvxai.core.logging import capture_stdout, get_logger, get_request_id
from cvxai.core.sandbox import ModuleSandbox
from cvxai.schemas.common import (
    Actionability,
    ComponentStatus,
    Envelope,
    Finding,
    ModelCard,
    Reliability,
)
from cvxai.settings import Settings


class ComponentAdapter(abc.ABC):
    """Base class for the four component adapters."""

    #: Stable URL segment and registry key.
    id: str = ""
    name: str = ""
    owner: str = ""
    modality: str = ""
    task: str = ""
    dataset: str = ""
    architecture: str = ""
    endpoint: str = ""

    def __init__(self, settings: Settings, root: Optional[Path]) -> None:
        self.settings = settings
        self.root = Path(root) if root else None
        self.log = get_logger("cvxai.adapters.%s" % self.id)
        self._sandbox: Optional[ModuleSandbox] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        # Guards the load. Without it two concurrent first-requests both see
        # `_loaded is False` and each builds a full copy of the component --
        # two ConvNeXt/BioBART loads for Component 01, the second silently
        # replacing the first. Easy to trigger by double-clicking a demo.
        self._load_lock = threading.Lock()

    # ---- capability ---------------------------------------------------
    def missing_assets(self) -> List[str]:
        """Paths that must exist before this component can serve.

        Reported in /health so a missing checkpoint is diagnosable without
        reading a stack trace.
        """
        if self.root is None:
            return ["component root directory"]
        return [str(p) for p in self.required_paths() if not p.exists()]

    @abc.abstractmethod
    def required_paths(self) -> List[Path]:
        """Absolute paths whose absence makes the component unservable."""

    def notes(self) -> List[str]:
        """Optional capabilities that are absent on this install.

        Distinct from `missing_assets()`: nothing here stops the component
        serving. It exists so a startup warning about an optional file is
        explained in /health rather than leaving the reader to guess whether
        the service is broken.
        """
        return []

    @property
    def status(self) -> ComponentStatus:
        if self._load_error:
            return ComponentStatus.FAILED
        if self.root is None or self.missing_assets():
            return ComponentStatus.UNAVAILABLE
        return ComponentStatus.READY if self._loaded else ComponentStatus.AVAILABLE

    @property
    def status_detail(self) -> Optional[str]:
        if self._load_error:
            return self._load_error
        if self.root is None:
            return ("Component root not found. Set %s to the component directory."
                    % self.root_env_var)
        missing = self.missing_assets()
        if missing:
            return "Missing required asset(s): " + "; ".join(missing[:4])
        return None

    @property
    def root_env_var(self) -> str:
        return "CVXAI_%s_ROOT" % self.id.upper()

    # ---- lifecycle ----------------------------------------------------
    @abc.abstractmethod
    def build_sandbox(self) -> ModuleSandbox:
        """Construct the import namespace this component runs inside."""

    @abc.abstractmethod
    def _load(self) -> None:
        """Import and instantiate. Called once, already inside the sandbox."""

    @property
    def sandbox(self) -> ModuleSandbox:
        if self._sandbox is None:
            self._sandbox = self.build_sandbox()
        return self._sandbox

    def ensure_loaded(self) -> None:
        """Load on first use. Raises ComponentUnavailable if it cannot serve."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:                   # won the race while blocked
                return
            if self._load_error:
                raise ComponentUnavailable(
                    "%s previously failed to load" % self.name,
                    {"component": self.id, "cause": self._load_error})
            missing = self.missing_assets()
            if self.root is None or missing:
                raise ComponentUnavailable(
                    "%s cannot serve" % self.name,
                    {"component": self.id, "reason": self.status_detail,
                     "missing": missing})

            started = time.perf_counter()
            self.log.info("loading %s from %s", self.name, self.root)
            try:
                # The components report load progress with bare print(); route
                # it through the logger so the startup log has one format.
                with capture_stdout(self.log, prefix="| "), self.sandbox.active():
                    self._load()
            except Exception as exc:           # noqa: BLE001 - reported, not swallowed
                self._load_error = "%s: %s" % (type(exc).__name__, exc)
                self.log.exception("failed to load %s", self.name)
                raise ComponentUnavailable(
                    "%s failed to load" % self.name,
                    {"component": self.id, "cause": self._load_error}) from exc
            self._loaded = True
            self.log.info("loaded %s in %.1fs", self.name,
                          time.perf_counter() - started)

    def warm(self) -> bool:
        """Best-effort eager load. Never raises: a cold component is not fatal."""
        try:
            self.ensure_loaded()
            return True
        except ComponentUnavailable as exc:
            self.log.warning("%s unavailable: %s", self.name, exc.message)
            return False

    # ---- inference ----------------------------------------------------
    @abc.abstractmethod
    def analyze(self, **kwargs: Any) -> Envelope:
        """Run one study and return the shared envelope."""

    # ---- helpers ------------------------------------------------------
    @abc.abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """Published test-set performance, surfaced in every model card."""

    @abc.abstractmethod
    def limitations(self) -> List[str]:
        """The component's own stated limitations. Never omitted."""

    def model_card(self, decision_rule: Optional[str] = None) -> ModelCard:
        return ModelCard(
            component_id=self.id,
            component_name=self.name,
            owner=self.owner,
            modality=self.modality,
            task=self.task,
            dataset=self.dataset,
            architecture=self.architecture,
            metrics=self.metrics(),
            limitations=self.limitations(),
            decision_rule=decision_rule,
        )

    def envelope(
        self,
        headline: str,
        findings: List[Finding],
        reliability: Reliability,
        raw: Dict[str, Any],
        started: float,
        status: str = "ok",
        explanation: Optional[Dict[str, Any]] = None,
        narrative: Optional[str] = None,
        decision_rule: Optional[str] = None,
    ) -> Envelope:
        return Envelope(
            component=self.id,
            status=status,
            headline=headline,
            findings=findings,
            reliability=reliability,
            explanation=explanation or {},
            narrative=narrative,
            model=self.model_card(decision_rule),
            raw=raw,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=get_request_id(),
        )

    @staticmethod
    def unavailable_reliability(reason: str) -> Reliability:
        return Reliability(
            actionability=Actionability.UNAVAILABLE,
            level="unavailable",
            reasons=[reason],
        )

    def __repr__(self) -> str:
        return "<%s id=%s status=%s>" % (type(self).__name__, self.id, self.status.value)
