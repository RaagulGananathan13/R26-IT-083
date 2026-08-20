"""
Measure how far the four components' cohorts actually overlap.

WHY
---
The multi-modal endpoint aggregates rather than fuses, and the stated reason is
that no cohort in this project carries all four modalities for the same patient.
That is a strong claim to leave as an assertion, and a panel is entitled to ask
for the number.

So measure it. Components 01 and 04 are both MIMIC-IV derived and share a
`subject_id` space, so their overlap is computable exactly. Components 02
(PTB-XL, Physikalisch-Technische Bundesanstalt, Germany, 1989-96) and 03
(EchoNet-Dynamic, Stanford; CAMUS, France) come from different institutions,
countries and decades, and carry no identifier that could link to MIMIC or to
each other -- so their overlap is not merely unmeasured, it is zero by
construction.

The result is written to the backend cache and served at /api/v1/cohorts.

USAGE
-----
    python scripts/measure_cohort_overlap.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import pandas as pd

    from cvxai.core.logging import configure_logging, get_logger
    from cvxai.settings import get_settings

    configure_logging("INFO")
    log = get_logger("cohorts")
    settings = get_settings()

    result = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohorts": {},
        "pairs": {},
        "conclusion": "",
    }

    # ---- Component 01: MIMIC-CXR --------------------------------------
    cxr_subjects = set()
    cxr_images = 0
    if settings.cxr_root:
        frames = []
        for split in ("train", "val", "test"):
            path = settings.cxr_root / "training_manifest" / ("manifest_%s.csv" % split)
            if path.exists():
                frames.append(pd.read_csv(path, usecols=["subject_id"], low_memory=False))
        if frames:
            joined = pd.concat(frames)
            cxr_subjects = set(joined.subject_id.astype("int64"))
            cxr_images = len(joined)
    result["cohorts"]["cxr"] = {
        "dataset": "MIMIC-CXR / MIMIC-CXR-JPG",
        "institution": "Beth Israel Deaconess Medical Center, Boston, USA",
        "identifier_space": "MIMIC subject_id",
        "patients": len(cxr_subjects),
        "studies": cxr_images,
    }

    # ---- Component 04: MIMIC-IV-ED ------------------------------------
    ed_subjects = set()
    ed_stays = 0
    if settings.triage_root:
        path = settings.triage_root / "artifacts" / "data" / "split_assignment.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=["subject_id"])
            ed_subjects = set(frame.subject_id.astype("int64"))
            ed_stays = len(frame)
    result["cohorts"]["triage"] = {
        "dataset": "MIMIC-IV-ED / Hosp / ECG",
        "institution": "Beth Israel Deaconess Medical Center, Boston, USA",
        "identifier_space": "MIMIC subject_id",
        "patients": len(ed_subjects),
        "studies": ed_stays,
    }

    # ---- Components 02 and 03: different institutions entirely ---------
    result["cohorts"]["ecg"] = {
        "dataset": "PTB-XL",
        "institution": "Physikalisch-Technische Bundesanstalt, Germany",
        "collected": "1989-1996",
        "identifier_space": "PTB-XL ecg_id / patient_id (unlinkable to MIMIC)",
        "patients": 18885,
        "studies": 21799,
    }
    result["cohorts"]["echo"] = {
        "dataset": "EchoNet-Dynamic (+ CAMUS, train only)",
        "institution": "Stanford Medicine, USA; CHU Saint-Etienne, France",
        "identifier_space": "anonymised study hash (unlinkable to MIMIC)",
        "patients": None,
        "studies": 10030,
    }

    # ---- The one measurable pair --------------------------------------
    shared = cxr_subjects & ed_subjects
    result["pairs"]["cxr+triage"] = {
        "linkable": True,
        "reason": "Both derive from MIMIC-IV and share the subject_id space.",
        "shared_patients": len(shared),
        "share_of_cxr_cohort": (round(len(shared) / len(cxr_subjects), 4)
                                if cxr_subjects else None),
        "share_of_ed_cohort": (round(len(shared) / len(ed_subjects), 4)
                               if ed_subjects else None),
        "paired_study_feasible": len(shared) > 1000,
        "caveat": ("Patient-level overlap is necessary but not sufficient. A valid "
                   "paired study additionally needs temporal linkage -- the "
                   "radiograph must fall inside the ED stay window -- which has "
                   "not been established here."),
    }
    for pair in ("cxr+ecg", "cxr+echo", "ecg+echo", "ecg+triage", "echo+triage"):
        result["pairs"][pair] = {
            "linkable": False,
            "reason": "Different institutions and identifier spaces; no shared "
                      "patient identifier exists, so the overlap is zero by "
                      "construction rather than merely unmeasured.",
            "shared_patients": 0,
            "paired_study_feasible": False,
        }

    result["conclusion"] = (
        "A four-modality cohort does not exist and cannot be constructed from "
        "these datasets: Components 02 and 03 come from different institutions, "
        "countries and decades than Components 01 and 04, with no linkable "
        "identifier. One pair IS linkable -- Components 01 and 04 share %d "
        "patients (%.1f %% of the radiograph cohort) -- so a two-modality paired "
        "study is feasible future work, subject to temporal linkage. No joint "
        "model is trained or claimed here."
        % (len(shared), 100 * len(shared) / len(cxr_subjects) if cxr_subjects else 0))

    out_path = Path(args.out) if args.out else (settings.cache_dir / "cohort_overlap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    log.info("Component 01 (MIMIC-CXR)   : %6d patients, %6d studies",
             len(cxr_subjects), cxr_images)
    log.info("Component 04 (MIMIC-IV-ED) : %6d patients, %6d stays",
             len(ed_subjects), ed_stays)
    log.info("shared (01 n 04)           : %6d patients", len(shared))
    log.info("written: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
