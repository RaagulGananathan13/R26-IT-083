"""
Tests for the import isolation.

These are the load-bearing tests of the integration. Without the sandbox,
Components 03 and 04 silently share a `config` module and one of them receives
the other's configuration object -- a defect that produces wrong numbers rather
than an exception, so nothing else in the suite would catch it.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from cvxai.core.sandbox import ModuleSandbox, SuffixTolerantFinder


@pytest.fixture()
def two_components(tmp_path: Path):
    """Two directories that both define a top-level module named `config`."""
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "config.py").write_text("NAME = 'alpha'\nVALUE = 1\n", encoding="utf-8")
    (beta / "config.py").write_text("NAME = 'beta'\nVALUE = 2\n", encoding="utf-8")
    return alpha, beta


def test_colliding_module_names_stay_separate(two_components):
    """The defect this class exists to prevent."""
    alpha, beta = two_components
    sandbox_a = ModuleSandbox("alpha", roots=[alpha], path_entries=[alpha])
    sandbox_b = ModuleSandbox("beta", roots=[beta], path_entries=[beta])

    with sandbox_a.active():
        import config as config_a               # noqa: PLC0415
        assert config_a.NAME == "alpha"

    with sandbox_b.active():
        import config as config_b               # noqa: PLC0415
        # Without isolation this is 'alpha': the first import wins and stays
        # resident in sys.modules for the life of the process.
        assert config_b.NAME == "beta"

    with sandbox_a.active():
        import config as config_a_again         # noqa: PLC0415
        assert config_a_again.NAME == "alpha"


def test_modules_are_removed_from_the_global_namespace(two_components):
    alpha, _ = two_components
    sandbox = ModuleSandbox("alpha", roots=[alpha], path_entries=[alpha])

    assert "config" not in sys.modules
    with sandbox.active():
        import config                           # noqa: F401,PLC0415
        assert "config" in sys.modules
    assert "config" not in sys.modules
    assert "config" in sandbox.imported()


def test_sys_path_is_restored(two_components):
    alpha, _ = two_components
    before = list(sys.path)
    sandbox = ModuleSandbox("alpha", roots=[alpha], path_entries=[alpha])
    with sandbox.active():
        assert str(alpha) in sys.path
    assert sys.path == before


def test_environment_overrides_are_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("CVXAI_TEST_VAR", "original")
    sandbox = ModuleSandbox("env", roots=[tmp_path], path_entries=[tmp_path],
                            env={"CVXAI_TEST_VAR": "sandboxed"})
    import os

    with sandbox.active():
        assert os.environ["CVXAI_TEST_VAR"] == "sandboxed"
    assert os.environ["CVXAI_TEST_VAR"] == "original"


def test_shared_third_party_modules_are_not_captured(tmp_path):
    """A component importing json must not steal it from the rest of the process."""
    component = tmp_path / "shared"
    component.mkdir()
    (component / "uses_json.py").write_text("import json\nDATA = json.dumps({'a': 1})\n",
                                            encoding="utf-8")
    sandbox = ModuleSandbox("shared", roots=[component], path_entries=[component])
    with sandbox.active():
        import uses_json                        # noqa: F401,PLC0415
    assert "json" in sys.modules                # stayed global
    assert "json" not in sandbox.imported()
    assert "uses_json" in sandbox.imported()


def test_nested_activation_is_safe(two_components):
    alpha, _ = two_components
    sandbox = ModuleSandbox("alpha", roots=[alpha], path_entries=[alpha])
    with sandbox.active():
        import config                           # noqa: PLC0415
        with sandbox.active():                  # adapters nest these freely
            assert config.NAME == "alpha"
        # The inner exit must not have detached the modules.
        assert "config" in sys.modules
    assert "config" not in sys.modules


def test_verify_owns_rejects_a_namesake(tmp_path, two_components):
    alpha, beta = two_components
    # Roots deliberately point at beta while the path serves alpha.
    sandbox = ModuleSandbox("mismatched", roots=[beta], path_entries=[alpha])
    with sandbox.active():
        import config                           # noqa: F401,PLC0415
        with pytest.raises(ImportError, match="shadowed"):
            sandbox.verify_owns("config")


class TestSuffixTolerantFinder:
    """Component 02's package is `src (1)/` with `models (1).py` inside it."""

    @pytest.fixture()
    def suffixed_package(self, tmp_path: Path):
        directory = tmp_path / "src (1)"
        directory.mkdir()
        (directory / "__init__ (1).py").write_text("", encoding="utf-8")
        (directory / "models (1).py").write_text("NAMES = ['NORM', 'MI']\n",
                                                 encoding="utf-8")
        (directory / "pipeline (1).py").write_text(
            textwrap.dedent("""
                from .models import NAMES
                def classes():
                    return NAMES
            """), encoding="utf-8")
        return directory

    def test_imports_through_the_suffix(self, tmp_path, suffixed_package):
        sandbox = ModuleSandbox(
            "suffixed", roots=[tmp_path], path_entries=[tmp_path],
            finders=[lambda: SuffixTolerantFinder("src", suffixed_package)])
        with sandbox.active():
            from src.pipeline import classes    # noqa: PLC0415
            # Relative imports inside the package resolve too, which is the
            # part a plain sys.path entry cannot deliver.
            assert classes() == ["NORM", "MI"]

    def test_clean_names_are_preferred(self, tmp_path):
        directory = tmp_path / "src"
        directory.mkdir()
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / "models.py").write_text("SPELLING = 'clean'\n", encoding="utf-8")
        (directory / "models (1).py").write_text("SPELLING = 'suffixed'\n",
                                                 encoding="utf-8")
        sandbox = ModuleSandbox(
            "clean", roots=[tmp_path], path_entries=[tmp_path],
            finders=[lambda: SuffixTolerantFinder("src", directory)])
        with sandbox.active():
            from src.models import SPELLING     # noqa: PLC0415
            assert SPELLING == "clean"

    def test_unknown_module_falls_through(self, tmp_path, suffixed_package):
        sandbox = ModuleSandbox(
            "suffixed", roots=[tmp_path], path_entries=[tmp_path],
            finders=[lambda: SuffixTolerantFinder("src", suffixed_package)])
        with sandbox.active():
            with pytest.raises(ImportError):
                import src.nonexistent          # noqa: F401,PLC0415
