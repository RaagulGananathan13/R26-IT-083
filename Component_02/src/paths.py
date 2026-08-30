"""
paths.py — Locate data assets so Component_02 works as a self-contained folder.

Search order for every asset:
  1. Component_02/csv/          (bundled — travels with this folder)
  2. Component_02/data/         (packed arrays, if present)
  3. ../_archive/data/          (the original project layout)
  4. $ECG_DATA_DIR              (explicit override, wins if set)

Rationale: the audit scripts legitimately point at `_archive/` because they audit
it. Everything on the SERVING path must not, or the folder cannot be handed to
someone else. This module is the single place that decides.
"""
from __future__ import annotations

import os
from typing import List, Optional

SRC = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(SRC)
ROOT = os.path.dirname(COMP)


def _candidates() -> List[str]:
    out = []
    env = os.environ.get("ECG_DATA_DIR")
    if env:
        out.append(env)
    out += [
        os.path.join(COMP, "csv"),
        os.path.join(COMP, "data"),
        os.path.join(COMP, "assets"),
        os.path.join(ROOT, "_archive", "data"),
    ]
    return out


def find(name: str) -> Optional[str]:
    """Absolute path to `name`, or None if it is nowhere to be found."""
    for base in _candidates():
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def require(name: str) -> str:
    p = find(name)
    if p is None:
        raise SystemExit(
            f"Required asset '{name}' not found.\nLooked in:\n"
            + "\n".join(f"  - {b}" for b in _candidates())
            + "\nSet ECG_DATA_DIR to the directory that contains it.")
    return p


def signals_cache() -> Optional[str]:
    """The 17k cached .npy signals. ~4 GB, deliberately NOT bundled.

    Only needed to browse the built-in test set. Uploading a .dat/.hea pair
    works without it.
    """
    for base in _candidates():
        p = os.path.join(base, "signals_cache")
        if os.path.isdir(p):
            return p
    p = os.path.join(ROOT, "_archive", "data", "signals_cache")
    return p if os.path.isdir(p) else None


def describe() -> dict:
    """What did we find? Printed at startup so misconfiguration is obvious."""
    return {
        "norm_stats.json": find("norm_stats.json"),
        "test.csv": find("test.csv"),
        "val.csv": find("val.csv"),
        "signals_cache": signals_cache(),
    }
