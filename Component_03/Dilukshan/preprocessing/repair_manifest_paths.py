"""
Rewrite manifest cache paths to the portable, PREP_DIR-relative form.

WHY
---
The manifests store `cache_path` as an absolute path recorded at preprocessing
time. Moving the project breaks them. `data/dataset.py::_cache_path` already
anticipates this and documents the fix -- "new manifests store paths relative to
PREP_DIR for portability" -- but the manifests on disk predate that and are
still absolute.

The failure is asymmetric, which is why it went unnoticed:

    EchoNet clips  live in cache/videos/, which IS CFG.CACHE_DIR, so lookup
                   step 1 finds them and the stale absolute path is never used.
    CAMUS clips    live in cache/camus_videos/, which is NOT CACHE_DIR, so
                   step 1 misses, the stale absolute path is tried, and the
                   relocation fallback only re-checks CACHE_DIR -- the wrong
                   directory. Training dies on the first CAMUS clip.

Rewriting both manifests to paths relative to PREP_DIR makes them survive any
future move, and makes lookup step 2 (`PREP_DIR / stored`) succeed for both
caches.

SAFETY
------
Dry run by default. Every rewritten path is verified to exist before anything
is written, and the original is copied to <name>.absolute.bak.

USAGE
-----
    python repair_manifest_paths.py             # report only
    python repair_manifest_paths.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PREP_DIR = Path(__file__).resolve().parent
ARTIFACTS = PREP_DIR / "artifacts"
MANIFESTS = ("manifest.csv", "camus_manifest.csv")


def to_relative(raw: str) -> str | None:
    """Absolute or stale path -> path relative to PREP_DIR, or None if not one.

    Matched on the `preprocessing/` segment rather than on a prefix, so a path
    recorded under any previous project root is handled.
    """
    if not raw or raw != raw:                     # empty or NaN
        return None
    text = str(raw).strip().replace("\\", "/")
    if not text:
        return None
    marker = "/preprocessing/"
    index = text.lower().rfind(marker)
    if index == -1:
        # Already relative, or an unrecognised layout. Leave it alone.
        return None
    return text[index + len(marker):]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    import pandas as pd

    total_changed = 0
    problems: list[str] = []

    for name in MANIFESTS:
        path = ARTIFACTS / name
        if not path.exists():
            print("  %-22s not present, skipping" % name)
            continue

        frame = pd.read_csv(path)
        if "cache_path" not in frame.columns:
            print("  %-22s no cache_path column, skipping" % name)
            continue

        rewritten, unchanged, missing = 0, 0, []
        new_values = []
        for raw in frame["cache_path"]:
            relative = to_relative(raw)
            if relative is None:
                new_values.append(raw)
                unchanged += 1
                continue
            resolved = PREP_DIR / relative
            if not resolved.exists():
                missing.append(relative)
            new_values.append(relative)
            rewritten += 1

        print("  %-22s rows=%-6d rewrite=%-6d keep=%-6d missing-on-disk=%d"
              % (name, len(frame), rewritten, unchanged, len(missing)))
        if missing:
            problems.append("%s: %d cached clips not found, e.g. %s"
                            % (name, len(missing), missing[0]))
            for example in missing[:3]:
                print("      MISSING %s" % example)
            continue

        if args.apply and rewritten:
            backup = path.with_suffix(".csv.absolute.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
                print("      backup -> %s" % backup.name)
            frame["cache_path"] = new_values
            frame.to_csv(path, index=False)
            print("      written")
        total_changed += rewritten

    if problems:
        print("\nREFUSING to treat this as repaired:")
        for problem in problems:
            print("  - %s" % problem)
        return 1

    if not args.apply:
        print("\nDry run. Re-run with --apply to rewrite %d paths." % total_changed)
    else:
        print("\nRewrote %d paths. Manifests are now relocation-safe." % total_changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
