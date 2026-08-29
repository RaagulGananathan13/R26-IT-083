"""
Component 04 — configuration loader.

Resolves every path relative to the Component_04 package root so the
pipeline runs identically from any working directory.
"""
from __future__ import annotations

import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CORE_DIR)          # config.py lives in src/core/
ROOT = os.path.dirname(SRC_DIR)              # .../Component_04
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")

ARTIFACTS = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ARTIFACTS, "data")
MODEL_DIR = os.path.join(ARTIFACTS, "models")
REPORT_DIR = os.path.join(ARTIFACTS, "reports")
FIGURE_DIR = os.path.join(ARTIFACTS, "figures")

for _d in (ARTIFACTS, DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

LABEL_MAP: Dict[int, str] = {0: "No_ACS", 1: "UA", 2: "NSTEMI", 3: "STEMI"}
LABEL_ORDER: List[str] = ["No_ACS", "UA", "NSTEMI", "STEMI"]
SUBTYPE_MAP: Dict[int, str] = {0: "UA", 1: "NSTEMI", 2: "STEMI"}
SUBTYPE_ORDER: List[str] = ["UA", "NSTEMI", "STEMI"]


# --------------------------------------------------------------------------
# Minimal YAML reader.
# The config is a flat-ish mapping of scalars / lists, so we avoid adding a
# PyYAML dependency for something this small and fully under our control.
# --------------------------------------------------------------------------
def _coerce(token: str) -> Any:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        return [_coerce(t) for t in inner.split(",")] if inner else []
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _read_yaml(path: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    # stack of (indent, container)
    stack: List[tuple] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            body = line.strip()
            if "#" in body:  # strip trailing comment (no '#' appears inside our values)
                body = body.split("#", 1)[0].strip()
            if not body:
                continue
            key, _, val = body.partition(":")
            key, val = key.strip(), val.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if val == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _coerce(val)
    return root


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # --- convenience accessors -------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    @property
    def horizons(self) -> List[int]:
        return [int(h) for h in self.get("temporal.horizons_h", [0, 6, 24])]

    @property
    def primary_horizon(self) -> int:
        return int(self.get("temporal.primary_horizon_h", 24))

    @property
    def raw_dir(self) -> str:
        p = str(self.get("paths.raw_dir"))
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(ROOT, p))


def load_config(path: str = CONFIG_PATH) -> Config:
    return Config(_read_yaml(path))


CFG = load_config()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def set_seed(seed: int | None = None) -> int:
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def enable_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
