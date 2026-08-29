"""
Runtime configuration.

Every value is overridable by environment variable, but the defaults are
correct for the repository as laid out on disk, so the service starts with no
configuration at all.

Component roots are *discovered* rather than hard-coded to a single spelling:
the folders in this repository carry artefacts of a zip download (double
nesting, and a trailing " (1)" on every path inside Component_02). Each root is
resolved against an ordered candidate list, so the service keeps working
whether or not those names are ever cleaned up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# backend/cvxai/settings.py -> backend/cvxai -> backend -> <repo root>
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = _env(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_root(env_name: str, candidates: List[str]) -> Optional[Path]:
    """First existing directory: an explicit override wins, else the candidates.

    Returning None rather than raising is deliberate. A missing component must
    degrade to "unavailable" in /health, never take the whole service down —
    the other three still have work to do.
    """
    override = _env(env_name)
    if override:
        path = Path(override).expanduser()
        return path if path.is_dir() else None
    for rel in candidates:
        path = (REPO_ROOT / rel)
        if path.is_dir():
            return path
    return None


@dataclass(frozen=True)
class Settings:
    # ---- service -------------------------------------------------------
    host: str = field(default_factory=lambda: _env("CVXAI_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("CVXAI_PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("CVXAI_LOG_LEVEL", "INFO").upper())
    cors_origins: List[str] = field(default_factory=lambda: _env_list(
        "CVXAI_CORS_ORIGINS",
        ["http://localhost:5173", "http://127.0.0.1:5173",
         "http://localhost:5174", "http://127.0.0.1:5174",
         "http://localhost:3000", "http://127.0.0.1:3000",
         "http://localhost:3001", "http://127.0.0.1:3001"]))

    # ---- inference -----------------------------------------------------
    device: str = field(default_factory=lambda: _env("CVXAI_DEVICE", "auto"))
    eager_load: bool = field(default_factory=lambda: _env_bool("CVXAI_EAGER_LOAD", False))
    max_upload_mb: int = field(default_factory=lambda: _env_int("CVXAI_MAX_UPLOAD_MB", 64))

    # ---- component roots ----------------------------------------------
    cxr_root: Optional[Path] = field(default_factory=lambda: _resolve_root(
        "CVXAI_CXR_ROOT",
        ["Component_01/Component_01", "Component_01", "Component_1"]))
    ecg_root: Optional[Path] = field(default_factory=lambda: _resolve_root(
        "CVXAI_ECG_ROOT",
        ["Component_02/Component_02 (1)", "Component_02/Component_02", "Component_02"]))
    echo_root: Optional[Path] = field(default_factory=lambda: _resolve_root(
        "CVXAI_ECHO_ROOT",
        ["Component_03/Dilukshan", "Component_03"]))
    triage_root: Optional[Path] = field(default_factory=lambda: _resolve_root(
        "CVXAI_TRIAGE_ROOT",
        ["Component_04", "Component_4/Component_04", "Component_4"]))

    # ---- component options --------------------------------------------
    # Component 03: which trained seeds form the serving ensemble, and which
    # run's validation-frozen decision rule is applied to their average.
    echo_runs: List[str] = field(default_factory=lambda: _env_list(
        "CVXAI_ECHO_RUNS", ["uefnet_v3", "uefnet_v3b", "uefnet_v3c"]))
    echo_decision_run: str = field(default_factory=lambda: _env(
        "CVXAI_ECHO_DECISION_RUN", "uefnet_v3"))
    echo_tta_clips: int = field(default_factory=lambda: _env_int("CVXAI_ECHO_TTA_CLIPS", 10))

    # Component 04: disclosure horizon (0 / 6 / 24 h after ED arrival) and the
    # published operating point.
    triage_horizon: int = field(default_factory=lambda: _env_int("CVXAI_TRIAGE_HORIZON", 24))
    triage_operating_point: str = field(default_factory=lambda: _env(
        "CVXAI_TRIAGE_OPERATING_POINT", "max-coverage"))

    # Component 02: model identity must match the calibrator and the conformal
    # thresholds, or the statistical guarantee does not hold.
    ecg_model: str = field(default_factory=lambda: _env("CVXAI_ECG_MODEL", "resnet_se"))
    ecg_filter: bool = field(default_factory=lambda: _env_bool("CVXAI_ECG_FILTER", True))

    #: Serve the fine-tuned Flan-T5 report instead of the deterministic template.
    #:
    #: Off by default, and deliberately so. The template is the component's
    #: shipped generator and every sentence in it traces to a Finding; the
    #: neural model is newer, larger and only worth its cost if its output
    #: survives `verify_paraphrase`. With this off the adapter behaves exactly
    #: as before and the model is never loaded, so nothing pays for a feature
    #: that is not switched on.
    ecg_neural_report: bool = field(
        default_factory=lambda: _env_bool("CVXAI_ECG_NEURAL_REPORT", False))

    # ---- working directory --------------------------------------------
    cache_dir: Path = field(default_factory=lambda: Path(
        _env("CVXAI_CACHE_DIR", str(BACKEND_DIR / ".cache"))).expanduser())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def component_roots(self) -> dict:
        return {"cxr": self.cxr_root, "ecg": self.ecg_root,
                "echo": self.echo_root, "triage": self.triage_root}


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def reset_settings() -> None:
    """Drop the cached settings. Used by the test-suite only."""
    global _settings
    _settings = None
