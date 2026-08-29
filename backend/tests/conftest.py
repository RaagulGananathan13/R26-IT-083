"""Shared fixtures.

The unit tests never load model weights: they exercise the contract, the
routing and the import isolation. The integration tests in
`test_integration.py` do load weights and skip themselves when a component's
assets are not on this machine.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cvxai.core.registry import get_registry, reset_registry   # noqa: E402
from cvxai.main import create_app                              # noqa: E402
from cvxai.settings import get_settings                        # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def registry(settings):
    return get_registry(settings)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    reset_registry()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    reset_registry()


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    """A synthetic grayscale radiograph-shaped image.

    Sufficient to exercise the transform, the network and the response
    contract. It is noise, so the *values* it produces mean nothing and no test
    asserts on them.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(20260818)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((384, 384)) * 255).astype("uint8"), mode="L").save(
        buffer, format="PNG")
    return buffer.getvalue()


def component_ready(registry, component_id: str) -> bool:
    from cvxai.schemas.common import ComponentStatus

    adapter = registry.get(component_id)
    return adapter.status in (ComponentStatus.READY, ComponentStatus.AVAILABLE)


def find_ecg_record(settings):
    """Locate any WFDB pair bundled with Component 02, allowing ' (1)' names."""
    root = settings.ecg_root
    if root is None:
        return None
    for data_dir in (root / "data", root / "data (1)"):
        if not data_dir.is_dir():
            continue
        for dat in data_dir.rglob("*.dat"):
            hea = dat.with_suffix(".hea")
            if hea.exists():
                return dat, hea
            suffixed = dat.parent / (dat.stem + ".hea")
            if suffixed.exists():
                return dat, suffixed
    return None


def find_echo_clip(settings):
    """Any cached clip from Component 03's preprocessing stage."""
    root = settings.echo_root
    if root is None:
        return None
    cache = root / "preprocessing" / "cache" / "videos"
    if not cache.is_dir():
        return None
    return next(iter(sorted(cache.glob("*.npy"))), None)
