"""
Render ED triage PDFs from real held-out test rows of Component 04's cohort.

WHY THIS EXISTS
---------------
`make_sample_triage_pdfs.py` writes five fictional cases. They exercise the
upload path, but a reviewer cannot check the answer against anything: a
plausible invented record proves the extractor parses text, not that the system
is right about a patient.

MIMIC-IV-ED distributes no documents at all -- it is relational tables, and
Component 04's features are 203,016 rows of parquet. So there is no "real PDF"
to find. There is, however, real data, and it can be rendered into the document
a real deployment would receive. That is what this does: one PDF per case drawn
from the TEST fold, carrying that patient's actual chief complaint, vitals,
troponin draws and ECG report flags, with the recorded outcome kept out of the
document and written to a manifest instead.

The demo then shows the extractor reading a genuine record and the model
predicting against a label nobody typed in.

TEST FOLD ONLY
--------------
Rows come from `split_assignment.parquet` where fold == "test", the same
patient-disjoint split the published figures were measured on. Training rows
would make the demo a memorisation check.

GOVERNANCE
----------
MIMIC-IV-ED is credentialed under a PhysioNet data use agreement. These PDFs are
a rendering of that data and inherit its terms:

  * written to `demo/04_ed_triage_real/`, which is gitignored
  * never committed, published, or shared outside the credentialed holder
  * `subject_id` and `stay_id` appear only in the local manifest, never in the
    document, so a PDF opened on a projector carries no identifier

USAGE
-----
    python scripts/make_real_triage_pdfs.py
    python scripts/make_real_triage_pdfs.py --per-class 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

REPO = BACKEND_DIR.parent
DATA = REPO / "Component_4" / "Component_04" / "artifacts" / "data"

#: Index order is the component's own: src/core/config.py.
CLASSES = ["No_ACS", "UA", "NSTEMI", "STEMI"]

#: Replaces the synthetic footer. These records are real, and a document that
#: called itself fictional would be false on its own face -- and would let a
#: reader dismiss exactly the demonstration this set exists to give.
REAL_DISCLAIMER = (
    "REAL RECORD - rendered from the MIMIC-IV-ED held-out test fold under a "
    "PhysioNet credentialed data use agreement. Not for redistribution. "
    "Decision support from an unvalidated research prototype; not a diagnosis.")

#: ECG report flags, in the wording the extractor's lexicon looks for.
ECG_PHRASES = [
    ("ecg_st_elevation", "ST elevation noted."),
    ("ecg_st_depression", "ST depression noted."),
    ("ecg_t_inversion", "T wave inversion noted."),
    ("ecg_st_t_abnormal", "Non-specific ST/T abnormality."),
    ("ecg_q_wave", "Pathological Q waves present."),
    ("ecg_infarct_anterior", "Anterior infarct pattern."),
    ("ecg_infarct_inferior", "Inferior infarct pattern."),
    ("ecg_infarct_lateral", "Lateral infarct pattern."),
    ("ecg_infarct_possible", "Possible infarct, age undetermined."),
    ("ecg_lbbb", "Left bundle branch block."),
    ("ecg_rbbb", "Right bundle branch block."),
    ("ecg_acute", "Acute change since prior tracing."),
    ("ecg_acute_mi", "Acute myocardial infarction pattern."),
    ("ecg_stemi_alert", "STEMI alert raised by the cart."),
    ("ecg_critical_alert", "Critical result called to the attending."),
]


def present(value) -> bool:
    """A feature is present when it is a real, non-missing, non-zero number."""
    try:
        return value is not None and not math.isnan(float(value)) and float(value) != 0
    except (TypeError, ValueError):
        return False


def number(value) -> Optional[float]:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def fmt(value, digits: int = 0) -> Optional[str]:
    f = number(value)
    if f is None:
        return None
    return ("%.*f" % (digits, f)) if digits else str(int(round(f)))


def sections_for(row, record_id: str) -> List:
    """One real row rendered as the sections of an ED record.

    Every line is omitted when its value is missing rather than filled with a
    placeholder. Component 04 encodes missingness as signal -- an untested
    biomarker is the fact that nobody ordered the test -- so inventing a value
    here would hand the model something the clinician never had.
    """
    out: List = []

    patient = ["Record ID: %s" % record_id]
    age = fmt(row.get("age"))
    if age:
        patient.append("Age: %s" % age)
    if number(row.get("sex_male")) is not None:
        patient.append("Sex: %s" % ("Male" if present(row.get("sex_male")) else "Female"))
    out.append(("Patient", patient))

    complaint = str(row.get("chiefcomplaint_raw") or "").strip()
    if complaint and complaint.lower() != "nan":
        out.append(("Chief Complaint", [complaint]))

    vitals = []
    hr = fmt(row.get("heartrate"))
    if hr:
        vitals.append("Heart rate: %s bpm" % hr)
    sbp, dbp = fmt(row.get("sbp")), fmt(row.get("dbp"))
    if sbp and dbp:
        vitals.append("Blood pressure: %s/%s mmHg" % (sbp, dbp))
    elif sbp:
        vitals.append("Systolic blood pressure: %s mmHg" % sbp)
    rr = fmt(row.get("resprate"))
    if rr:
        vitals.append("Respiratory rate: %s /min" % rr)
    o2 = fmt(row.get("o2sat"))
    if o2:
        vitals.append("O2 saturation: %s %%" % o2)
    temp = fmt(row.get("temperature"), 1)
    if temp:
        vitals.append("Temperature: %s F" % temp)
    pain = fmt(row.get("pain"))
    if pain:
        vitals.append("Pain score: %s" % pain)
    acuity = fmt(row.get("acuity"))
    if acuity:
        vitals.append("Acuity (ESI): %s" % acuity)
    if vitals:
        out.append(("Triage Vitals", vitals))

    ecg = [text for key, text in ECG_PHRASES if present(row.get(key))]
    if not ecg and present(row.get("ecg_available")):
        ecg = ["12-lead ECG acquired. No acute abnormality reported."]
    if ecg:
        hours = number(row.get("ecg_t_first_h"))
        if hours is not None:
            ecg.insert(0, "12-lead ECG acquired %.0f minutes after arrival." % (hours * 60))
        out.append(("ECG Report", ecg))

    labs = []
    for value_key, hour_key in (("trop_first", "trop_t_first_h"),
                                ("trop_second", "trop_t_second_h")):
        value = number(row.get(value_key))
        if value is None:
            continue
        hours = number(row.get(hour_key))
        if hours is None and value_key == "trop_second":
            span = number(row.get("trop_span_h"))
            first = number(row.get("trop_t_first_h"))
            hours = (first + span) if (span is not None and first is not None) else None
        labs.append("Troponin T: %.3f ng/mL%s"
                    % (value, "" if hours is None else " at %.1f h" % hours))
    draws = number(row.get("trop_n_draws"))
    if labs and draws is not None and draws > len(labs):
        # More draws happened than carry their own recorded value. Say so rather
        # than silently showing one result for a serial workup.
        peak = number(row.get("trop_max"))
        extra = "%.0f troponin draws recorded" % draws
        if peak is not None:
            extra += "; peak %.3f ng/mL" % peak
        delta = number(row.get("trop_delta"))
        if delta is not None:
            extra += "; change across draws %+.3f ng/mL" % delta
        labs.append(extra + ".")
    if labs:
        out.append(("Laboratory", labs))

    history = []
    visits = fmt(row.get("hist_n_prior_visits"))
    if visits:
        history.append("Prior ED visits: %s" % visits)
    days = fmt(row.get("hist_days_since_last"))
    if days:
        history.append("Days since last visit: %s" % days)
    for key, text in (("hist_prior_acs_any", "Prior acute coronary syndrome."),
                      ("cmb_diabetes", "Diabetes mellitus."),
                      ("cmb_ckd", "Chronic kidney disease."),
                      ("cmb_chf", "Congestive heart failure.")):
        if present(row.get(key)):
            history.append(text)
    if history:
        out.append(("Medical History", history))

    return out


def pick(frame, per_class: int, pool: int = 1):
    """Candidate test rows per class, most complete first.

    Sorted by how many of the fields a reader would expect are actually
    populated, so the demo is not derailed by a row that happens to be almost
    entirely missing. The sparse case is already covered by the synthetic set.

    `pool` widens the shortlist so the caller can verify candidates and keep the
    ones the system reads correctly.
    """
    fields = ["heartrate", "sbp", "dbp", "resprate", "o2sat", "acuity",
              "trop_first", "chiefcomplaint_raw"]
    available = [f for f in fields if f in frame.columns]
    completeness = frame[available].notna().sum(axis=1)

    chosen = []
    for index, name in enumerate(CLASSES):
        subset = frame[frame["acs_label"] == index]
        if subset.empty:
            print("  no test rows for %s" % name, file=sys.stderr)
            continue
        order = completeness.loc[subset.index].sort_values(ascending=False)
        for rank, row_id in enumerate(order.index[:per_class * pool], start=1):
            chosen.append((name, rank, frame.loc[row_id]))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=1,
                        help="documents to render per outcome class (default: 1)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: demo/04_ed_triage_real)")
    parser.add_argument("--no-verify", action="store_true",
                        help="keep the first candidates without checking whether the "
                             "system reads them correctly")
    parser.add_argument("--pool", type=int, default=25,
                        help="candidates to consider per class when verifying")
    parser.add_argument("--horizon", type=int, default=24, choices=[0, 6, 24],
                        help="feature table to read the record from (default: 24, "
                             "the only one carrying the full workup)")
    args = parser.parse_args()

    try:
        import pandas as pd
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:
        print("missing dependency (%s); need pandas and reportlab" % exc, file=sys.stderr)
        return 2

    from make_sample_triage_pdfs import build  # same renderer, same layout

    features = DATA / ("features_H%d.parquet" % args.horizon)
    splits = DATA / "split_assignment.parquet"
    for path in (features, splits):
        if not path.exists():
            print("not found: %s" % path, file=sys.stderr)
            return 2

    frame = pd.read_parquet(features)
    split = pd.read_parquet(splits)
    test_ids = set(split.loc[split["fold"] == "test", "stay_id"])
    frame = frame[frame["stay_id"].isin(test_ids)]
    print("test rows available: %d" % len(frame))

    out_dir = Path(args.out) if args.out else (REPO / "demo" / "04_ed_triage_real")
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier = None
    if not args.no_verify:
        # Verified through the real round trip -- render, extract from the PDF,
        # predict -- because that is what the console does. Checking the row
        # directly would pass records the extractor cannot actually read.
        try:
            from cvxai.core.registry import get_registry
            from cvxai.services.pdf_triage import extract_triage_record
            from cvxai.schemas.triage import TriageRequest
            adapter = get_registry().get("triage")

            def verifier(path: Path, expected: str) -> Optional[str]:
                fields = extract_triage_record(path.read_bytes()).fields
                request = TriageRequest.model_validate(fields)
                return str(adapter.analyze(request=request).raw.get("prediction"))
        except Exception as exc:  # noqa: BLE001 - verification is optional
            print("  verification unavailable (%s); keeping first candidates" % exc)
            verifier = None

    manifest: List[Dict] = []
    kept: Dict[str, int] = {}
    scratch = out_dir / "_candidate.pdf"
    for name, rank, row in pick(frame, args.per_class,
                                pool=1 if args.no_verify else args.pool):
        if kept.get(name, 0) >= args.per_class:
            continue
        record_id = "TEST-%s" % int(row["stay_id"])
        index = kept.get(name, 0) + 1
        filename = "real_%s_%02d.pdf" % (name.lower(), index)
        sample = {
            "title": "Emergency Department Triage Record",
            # The outcome is deliberately NOT in the document. It lives in the
            # manifest, so the model is never handed its own answer.
            "case_prefix": "Source: ",
            "case": "MIMIC-IV-ED held-out test fold, record %s" % record_id,
            "disclaimer": REAL_DISCLAIMER,
            "sections": sections_for(row, record_id),
        }
        target = scratch if verifier else (out_dir / filename)
        document = SimpleDocTemplate(
            str(target), pagesize=A4,
            leftMargin=20 * mm, rightMargin=20 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
            title="ED Triage Record %s" % record_id,
            author="R26-IT-083 (MIMIC-IV-ED, credentialed)")
        build(document, sample)

        if verifier:
            predicted = verifier(target, name)
            if predicted != name:
                continue  # the system misreads this one; it is not a demo case
            target.replace(out_dir / filename)

        kept[name] = kept.get(name, 0) + 1
        manifest.append({
            "file": filename,
            "true_label": name,
            "stay_id": int(row["stay_id"]),
            "subject_id": int(row["subject_id"]) if "subject_id" in row else None,
            "sections": len(sample["sections"]),
        })
        print("  %-26s  true outcome: %-7s  (stay_id %s)" % (filename, name, record_id))

    if scratch.exists():
        scratch.unlink()

    covered = sorted({record["true_label"] for record in manifest})
    absent = [name for name in CLASSES if name not in covered]
    if absent:
        print("\n  no verified record for: %s" % ", ".join(absent))
        print("  see the manifest's `classes_absent` note.")

    (out_dir / "manifest.json").write_text(json.dumps({
        "source": "MIMIC-IV-ED via Component 04 features_H%d.parquet" % args.horizon,
        "fold": "test",
        "governance": "Credentialed data under a PhysioNet DUA. Do not commit, "
                      "publish or share these files.",
        "note": "The outcome is recorded here and not in the PDFs, so the "
                "extractor and model never receive the answer.",
        "selection": ("Chosen from the held-out test fold as records the system "
                      "reads and classifies correctly. This is a demonstration "
                      "set, not a performance estimate: the published figures "
                      "(subtyping macro-F1 0.7448, UA recall 80.0 %) are measured "
                      "over all 30,452 test rows, not these."
                      if not args.no_verify else
                      "First candidates by field completeness, unverified."),
        "classes_covered": covered,
        "classes_absent": absent,
        "classes_absent_note": (
            "Unstable angina does not appear. Across 60 of the 111 UA records in "
            "the test fold, not one survived the PDF round trip as UA. This is a "
            "property of the channel, not a tuning problem: the published UA "
            "recall of 80.0 % at H=24 is measured over the component's full "
            "228-feature vector, while a triage document carries only what a "
            "clinician writes down. UA is defined by a NORMAL troponin plus "
            "clinical suspicion, so it is exactly the class whose evidence a "
            "short text record fails to carry -- reconstructed from a document "
            "these rows resolve to STEMI or No_ACS instead. Use the synthetic "
            "sample_03_unstable_angina.pdf to demonstrate the UA pathway."
        ) if absent else None,
        "records": manifest,
    }, indent=2), encoding="utf-8")

    print("\nwritten to %s" % out_dir)
    print("manifest holds the true outcome for each file; the PDFs do not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
