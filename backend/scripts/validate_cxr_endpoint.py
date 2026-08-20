"""
Validate the chest-radiograph endpoint against real, labelled MIMIC-CXR studies.

WHY
---
Every other component can be checked against a bundled study: Component 02 has
PTB-XL records, Component 03 has cached EchoNet clips, Component 04 has vignettes.
Component 01's images are credentialed MIMIC-CXR data and live outside the
repository, so the endpoint had only ever been exercised with synthetic noise --
which proves the pipeline runs but says nothing about whether it is *correct*.

This script draws a stratified sample from `training_manifest/manifest_test.csv`,
posts each image through the live serving path, and compares the served decision
with the radiologist-adjudicated label. It exercises the whole chain: transform,
ConvNeXt, per-projection threshold, deferral policy and Grad-CAM.

The expected figures are Component 01's own published test-set results
(n = 4,722): cardiomegaly accuracy 83.2 %, sensitivity 92.3 %, specificity
74.0 %. A sample of a few hundred will land near those with sampling noise; a
large deviation means the serving path is not reproducing the component.

USAGE
-----
    python scripts/validate_cxr_endpoint.py --n 200
    python scripts/validate_cxr_endpoint.py --n 400 --seed 7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PUBLISHED = {"accuracy": 0.832, "sensitivity": 0.923, "specificity": 0.740,
             "n": 4722}


def wilson(successes: int, total: int, z: float = 1.96):
    """Wilson score interval -- correct for proportions at small counts."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200, help="studies to sample")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--no-view", action="store_true",
                        help="withhold the projection, forcing the global threshold")
    args = parser.parse_args()

    import pandas as pd
    from fastapi.testclient import TestClient

    from cvxai.core.logging import configure_logging
    from cvxai.main import create_app
    from cvxai.settings import get_settings

    configure_logging("WARNING")
    settings = get_settings()
    root = settings.cxr_root
    if root is None:
        print("Component 01 root not found.", file=sys.stderr)
        return 2

    manifest = pd.read_csv(root / "training_manifest" / "manifest_test.csv",
                           low_memory=False)
    image_root = root.parent / "data" / "output" / "cardio_image_384"
    if not image_root.is_dir():
        print("Test images not found at %s\n"
              "They are credentialed MIMIC-CXR data. Link or copy the dataset so "
              "that <Component_01>/data/output/cardio_image_384 resolves."
              % image_root, file=sys.stderr)
        return 2

    manifest = manifest[[(image_root / p).exists() for p in manifest.image_path]]
    # Stratify on the label so a small sample still estimates both arms.
    per_arm = max(1, args.n // 2)
    sample = pd.concat([
        group.sample(min(per_arm, len(group)), random_state=args.seed)
        for _, group in manifest.groupby("Cardiomegaly")
    ]).sample(frac=1.0, random_state=args.seed)

    print("sampling %d of %d available test studies (seed %d)"
          % (len(sample), len(manifest), args.seed))
    print("views: %s\n" % sample["view"].value_counts().to_dict())

    app = create_app()
    tp = tn = fp = fn = 0
    deferred = 0
    by_view: dict = {}
    started = time.perf_counter()

    with TestClient(app) as client:
        for index, (_, row) in enumerate(sample.iterrows(), 1):
            path = image_root / row["image_path"]
            # row["view"] not row.view: `view` is a pandas Series METHOD, so
            # attribute access silently returns the bound method.
            data = {"view": None if args.no_view else str(row["view"])}
            response = client.post(
                "/api/v1/cxr/analyze",
                files={"file": (path.name, path.read_bytes(), "image/png")},
                data={k: v for k, v in data.items() if v})
            if response.status_code != 200:
                print("  %s -> HTTP %d %s" % (path.name, response.status_code,
                                              response.text[:120]))
                continue

            body = response.json()
            finding = next(f for f in body["findings"] if f["name"] == "Cardiomegaly")
            predicted = bool(finding["present"])
            truth = bool(row["Cardiomegaly"])

            if body["reliability"]["actionability"] == "deferred":
                deferred += 1
            if predicted and truth:
                tp += 1
            elif predicted and not truth:
                fp += 1
            elif not predicted and truth:
                fn += 1
            else:
                tn += 1

            bucket = by_view.setdefault(str(row["view"]), [0, 0])
            bucket[1] += 1
            if predicted == truth:
                bucket[0] += 1

            if index % 25 == 0:
                print("  %4d/%d  running accuracy %.3f"
                      % (index, len(sample), (tp + tn) / max(1, tp + tn + fp + fn)))

    total = tp + tn + fp + fn
    if total == 0:
        print("no studies scored", file=sys.stderr)
        return 1

    accuracy = (tp + tn) / total
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)

    print("\n" + "=" * 70)
    print("  Component 01 -- served predictions vs radiologist labels")
    print("=" * 70)
    print("  scored              : %d studies  (%.1f s)"
          % (total, time.perf_counter() - started))
    print("  confusion           : TP %d  FP %d  FN %d  TN %d" % (tp, fp, fn, tn))
    print("  deferred            : %d (%.1f %%)" % (deferred, 100 * deferred / total))
    print()
    for label, value, expected in (
            ("accuracy", accuracy, PUBLISHED["accuracy"]),
            ("sensitivity", sensitivity, PUBLISHED["sensitivity"]),
            ("specificity", specificity, PUBLISHED["specificity"])):
        successes = {"accuracy": tp + tn, "sensitivity": tp, "specificity": tn}[label]
        trials = {"accuracy": total, "sensitivity": tp + fn, "specificity": tn + fp}[label]
        low, high = wilson(successes, trials)
        agrees = low <= expected <= high
        print("  %-12s %.4f   95%% CI [%.3f, %.3f]   published %.3f   %s"
              % (label, value, low, high, expected,
                 "consistent" if agrees else "OUTSIDE INTERVAL"))
    print()
    for view, (correct, seen) in sorted(by_view.items()):
        print("  %-4s accuracy      : %.4f  (%d/%d)" % (view, correct / seen, correct, seen))
    print("=" * 70)
    print("  Published figures are for the full n=4,722 test set. A sample this"
          "\n  size carries the interval shown; read agreement, not equality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
