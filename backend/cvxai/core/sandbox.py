"""
Module-namespace isolation for the four research components.

WHY THIS EXISTS
---------------
The components were written independently and never intended to share a Python
process. Their top-level module names collide:

    Component 03 (training/)   config, core, data, models, engine, losses
    Component 04 (src/ + src/core, src/models, src/data, src/analysis)
                               config, core, data, models, preprocess,
                               text_features, inference, utils
    Component 01 (root)        backend, cxr_transforms, stage11_conditioned
    Component 02 (src/)        src

`config`, `core`, `data` and `models` are each claimed by two components. The
failure is silent rather than loud. Measured on this repository:

    sys.path.insert(0, C3_TRAINING); from config import CFG   -> C3 config
    sys.path.insert(0, C4_SRC);      from config import CFG   -> STILL C3

Component 04 receives Component 03's configuration object and the first symptom
is a wrong number, not an exception. Import order would silently decide which
component works, so without isolation neither could be trusted.

WHAT THIS DOES
--------------
Each component gets a ModuleSandbox. Entering it installs that component's
sys.path entries, environment overrides and any custom finders. Leaving it
lifts every module the component owns out of sys.modules into a private store
and restores the previous state. Re-entering puts the private store back, so
each module is imported once and then reused.

Ownership is decided by file location: a module is captured only if its
__file__ lies under one of the sandbox's own roots. Shared third-party packages
(torch, numpy, transformers) are imported once and stay global -- they are
never duplicated per component.

CONCURRENCY
-----------
sys.modules is process-global, so activation is serialised by a single
re-entrant lock. Component inference is therefore one-at-a-time across the
service. That is a deliberate correctness-over-throughput choice: the
components carry their own thread-affinity constraints too (Component 01's
Grad-CAM stores activations on the module; Component 02 documents a previously
fixed shared-XAI defect). A research decision-support service is not a
high-concurrency workload. The lock is re-entrant, so an adapter may nest
active() calls freely.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

# One lock shared by every sandbox: two components must never be active at the
# same time, because both would be mutating the same sys.modules.
_GLOBAL_IMPORT_LOCK = threading.RLock()


def _is_within(path: str, roots: Sequence[Path]) -> bool:
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class ModuleSandbox:
    """An isolated import namespace for one component."""

    def __init__(
        self,
        name: str,
        roots: Iterable[Path],
        path_entries: Iterable[Path],
        env: Optional[Dict[str, str]] = None,
        finders: Sequence[Callable[[], object]] = (),
    ) -> None:
        self.name = name
        self.roots: List[Path] = [Path(r).resolve() for r in roots]
        self.path_entries: List[str] = [str(Path(p).resolve()) for p in path_entries]
        self.env = dict(env or {})
        self._finder_factories = list(finders)
        self._finders: List[object] = []
        self._modules: Dict[str, ModuleType] = {}
        self._depth = 0

    # ------------------------------------------------------------------
    @contextmanager
    def active(self) -> Iterator["ModuleSandbox"]:
        """Make this component's namespace the live one for the block."""
        with _GLOBAL_IMPORT_LOCK:
            if self._depth > 0:            # already active further up the stack
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return

            saved_path = list(sys.path)
            saved_env = {key: os.environ.get(key) for key in self.env}
            baseline = set(sys.modules)

            if not self._finders and self._finder_factories:
                self._finders = [factory() for factory in self._finder_factories]
            for finder in self._finders:
                if finder not in sys.meta_path:
                    sys.meta_path.insert(0, finder)

            for key, value in self.env.items():
                os.environ[key] = value
            for entry in reversed(self.path_entries):
                if entry in sys.path:
                    sys.path.remove(entry)
                sys.path.insert(0, entry)

            sys.modules.update(self._modules)
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                self._detach(baseline)
                for finder in self._finders:
                    if finder in sys.meta_path:
                        sys.meta_path.remove(finder)
                sys.path[:] = saved_path
                for key, previous in saved_env.items():
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous

    # ------------------------------------------------------------------
    def _detach(self, baseline: set) -> None:
        """Move this component's modules out of the global namespace."""
        for name in list(sys.modules):
            if name in self._modules:
                sys.modules.pop(name, None)
                continue
            if name in baseline:
                continue                   # pre-existing: not ours to touch
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None)
            if file and _is_within(file, self.roots):
                self._modules[name] = module
                sys.modules.pop(name, None)
            # Modules imported inside the block but living elsewhere (torch,
            # numpy, transformers) are genuinely shared and stay global.

    # ------------------------------------------------------------------
    def imported(self) -> List[str]:
        """Names currently held in this sandbox's private store."""
        return sorted(self._modules)

    def verify_owns(self, module_name: str) -> None:
        """Assert a module resolved to this component and not to a namesake.

        Called after importing a name that also exists elsewhere in the process
        (`backend`, `config`, `models`). A namesake resolving first is the exact
        failure this class exists to prevent, so it is reported loudly instead
        of being allowed to produce wrong numbers.
        """
        module = sys.modules.get(module_name)
        if module is None:
            raise ImportError("[%s] %r is not imported" % (self.name, module_name))
        file = getattr(module, "__file__", None)
        if file is None or not _is_within(file, self.roots):
            raise ImportError(
                "[%s] %r resolved to %r, which is outside this component (%s). "
                "Another component's module of the same name shadowed it."
                % (self.name, module_name, file,
                   ", ".join(str(root) for root in self.roots)))

    def __repr__(self) -> str:
        return ("ModuleSandbox(name=%r, modules=%d, active=%s)"
                % (self.name, len(self._modules), self._depth > 0))


class SuffixTolerantFinder:
    """Import `pkg` / `pkg.sub` from files that may carry a ' (1)' suffix.

    Component 02 was extracted from a zip archive that de-duplicated every
    name, so on disk the package is `src (1)/` and its modules are
    `models (1).py`, `pipeline (1).py` and so on -- while the code inside still
    says `from .models import ...`. Nothing imports it without help.

    This finder maps the clean module path onto whichever spelling exists, so
    the component runs unmodified today and keeps running unchanged if those
    filenames are ever normalised.
    """

    #: Checked in order; the first spelling found on disk wins.
    _PATTERNS = ("%s.py", "%s (1).py")

    def __init__(self, package: str, directory: Path) -> None:
        self.package = package
        self.directory = Path(directory)

    def _module_file(self, stem: str) -> Optional[str]:
        for pattern in self._PATTERNS:
            candidate = self.directory / (pattern % stem)
            if candidate.is_file():
                return str(candidate)
        return None

    def find_spec(self, fullname: str, path=None, target=None):
        import importlib.util

        if fullname == self.package:
            init = self._module_file("__init__")
            if init is None:
                return None
            return importlib.util.spec_from_file_location(
                fullname, init, submodule_search_locations=[str(self.directory)])

        prefix = self.package + "."
        if not fullname.startswith(prefix):
            return None
        stem = fullname[len(prefix):]
        if "." in stem:                    # only a flat package is supported
            return None
        located = self._module_file(stem)
        if located is None:
            return None
        return importlib.util.spec_from_file_location(fullname, located)

    def __repr__(self) -> str:
        return "SuffixTolerantFinder(%r, %r)" % (self.package, str(self.directory))
