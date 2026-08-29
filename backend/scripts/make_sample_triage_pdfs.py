"""
Generate sample ED triage PDFs for the Component 04 upload path.

Five documents covering the four classes plus one deliberately sparse record:

    sample_01_stemi.pdf        anterior STEMI, ECG diagnostic, troponin rising
    sample_02_nstemi.pdf       NSTEMI, ST depression, modest troponin rise
    sample_03_unstable_angina.pdf   exertional pain, normal troponin
    sample_04_non_cardiac.pdf  abdominal pain, no cardiac workup ordered
    sample_05_sparse.pdf       triage-desk note only -- most fields absent

The last one is the important one. Component 04 encodes missingness as signal,
so a record with no biomarker and no ECG is a legitimate input, not a broken
one, and the extractor must report the gaps rather than invent values.

Every patient is fictional. No real record, MRN or identifier appears.

USAGE
-----
    python scripts/make_sample_triage_pdfs.py
    python scripts/make_sample_triage_pdfs.py --out ../frontend/public/samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

Section = Tuple[str, List[str]]

DISCLAIMER = ("SYNTHETIC RECORD - fictional patient, generated for testing the "
              "R26-IT-083 research prototype. Not a real clinical document.")


SAMPLES: List[Dict] = [
    {
        "file": "sample_01_stemi.pdf",
        "title": "Emergency Department Triage Record",
        "case": "Anterior ST-elevation myocardial infarction",
        "expected": "STEMI",
        "sections": [
            ("Patient", [
                "Record ID: SYNTH-0001",
                "Age: 61",
                "Sex: Male",
            ]),
            ("Chief Complaint", [
                "Crushing central chest pain radiating to the left arm, with "
                "diaphoresis and nausea. Onset 45 minutes before arrival, "
                "unrelieved by rest.",
            ]),
            ("Triage Vitals", [
                "Heart rate: 108 bpm",
                "Blood pressure: 92/58 mmHg",
                "Respiratory rate: 24 /min",
                "O2 saturation: 93 %",
                "Temperature: 98.2 F",
                "Pain score: 9",
                "Acuity (ESI): 1",
            ]),
            ("ECG Report", [
                "12-lead ECG acquired 9 minutes after arrival.",
                "ST elevation in leads V2-V4 consistent with acute anterior "
                "infarct. Critical result called to attending.",
                "QRS duration 98 ms. PR interval 156 ms. QTc 430 ms. Axis 12 degrees.",
            ]),
            ("Laboratory", [
                "Troponin I: 1.2 ng/mL at 0.8 h",
                "Troponin I: 6.8 ng/mL at 3.5 h",
            ]),
            ("Home Medications", [
                "Aspirin 81 mg daily",
                "Atorvastatin 40 mg nightly",
            ]),
            ("Medical History", [
                "Hypertension. Former smoker.",
                "Prior ED visits: 1",
            ]),
        ],
    },
    {
        "file": "sample_02_nstemi.pdf",
        "title": "Emergency Department Triage Record",
        "case": "Non-ST-elevation myocardial infarction",
        "expected": "NSTEMI",
        "sections": [
            ("Patient", [
                "Record ID: SYNTH-0002",
                "Age: 74",
                "Sex: Female",
            ]),
            ("Chief Complaint", [
                "Chest pressure with shortness of breath for three hours, "
                "worse on exertion. No radiation.",
            ]),
            ("Triage Vitals", [
                "Heart rate: 92 bpm",
                "Blood pressure: 138/80 mmHg",
                "Respiratory rate: 20 /min",
                "O2 saturation: 96 %",
                "Temperature: 98.6 F",
                "Pain score: 6",
                "Acuity (ESI): 2",
            ]),
            ("ECG Report", [
                "12-lead ECG obtained 24 minutes after arrival.",
                "ST depression in the lateral leads with T wave inversion. "
                "No ST elevation. Infarct of undetermined age cannot be excluded.",
                "QRS duration 104 ms. PR interval 178 ms. QTc 448 ms.",
            ]),
            ("Laboratory", [
                "Troponin I: 0.28 ng/mL at 1.2 h",
                "Troponin I: 0.51 ng/mL at 4.0 h",
                "BNP: 620 pg/mL",
            ]),
            ("Home Medications", [
                "Metoprolol 50 mg twice daily",
                "Lisinopril 10 mg daily",
                "Atorvastatin 20 mg nightly",
            ]),
            ("Medical History", [
                "Type 2 diabetes mellitus. Hypertension.",
                "Prior ED visits: 3",
            ]),
        ],
    },
    {
        "file": "sample_03_unstable_angina.pdf",
        "title": "Emergency Department Triage Record",
        "case": ("Unstable angina - the hardest class. Expect the system to DEFER "
                 "rather than commit"),
        "expected": "deferral (UA recall is 80 % at H=24; this case falls outside it)",
        "sections": [
            ("Patient", [
                "Record ID: SYNTH-0003",
                "Age: 58",
                "Sex: Male",
            ]),
            ("Chief Complaint", [
                "Chest pain on exertion over the past two days, each episode "
                "resolving after a few minutes of rest. Pain free on arrival.",
            ]),
            ("Triage Vitals", [
                "Heart rate: 78 bpm",
                "Blood pressure: 146/88 mmHg",
                "Respiratory rate: 16 /min",
                "O2 saturation: 98 %",
                "Temperature: 98.4 F",
                "Pain score: 5",
                "Acuity (ESI): 2",
            ]),
            ("ECG Report", [
                "12-lead ECG acquired 18 minutes after arrival.",
                "Normal sinus rhythm. No acute ischaemic change.",
                "QRS duration 88 ms. PR interval 162 ms. QTc 412 ms.",
            ]),
            ("Laboratory", [
                "Troponin I: 0.01 ng/mL at 1.5 h",
            ]),
            ("Home Medications", [
                "Aspirin 81 mg daily",
            ]),
            ("Medical History", [
                "Hyperlipidaemia. Family history of coronary disease.",
                "Prior ED visits: 0",
            ]),
        ],
    },
    {
        "file": "sample_04_non_cardiac.pdf",
        "title": "Emergency Department Triage Record",
        "case": "Non-cardiac presentation, no cardiac workup ordered",
        "expected": "No_ACS",
        "sections": [
            ("Patient", [
                "Record ID: SYNTH-0004",
                "Age: 34",
                "Sex: Female",
            ]),
            ("Chief Complaint", [
                "Abdominal pain and nausea since this morning. Denies chest "
                "pain, denies shortness of breath.",
            ]),
            ("Triage Vitals", [
                "Heart rate: 84 bpm",
                "Blood pressure: 118/74 mmHg",
                "Respiratory rate: 16 /min",
                "O2 saturation: 99 %",
                "Temperature: 99.1 F",
                "Pain score: 4",
                "Acuity (ESI): 3",
            ]),
            ("Medical History", [
                "No significant past medical history.",
                "Prior ED visits: 0",
            ]),
            ("Note", [
                "No ECG performed. No cardiac biomarkers ordered.",
                "This absence is itself informative: the model encodes an "
                "unordered test as the clinical decision not to order it, "
                "rather than imputing a population average.",
            ]),
        ],
    },
    {
        "file": "sample_05_sparse.pdf",
        "title": "Emergency Department Triage Note",
        "case": "Triage desk only - most fields absent by design",
        "expected": "sparse input; extraction gaps expected",
        "sections": [
            ("Patient", [
                "Record ID: SYNTH-0005",
                "Age: 67",
                "Sex: Male",
            ]),
            ("Chief Complaint", [
                "Chest tightness, came on while walking. Still present.",
            ]),
            ("Triage Vitals", [
                "Heart rate: 96 bpm",
                "Blood pressure: 154/92 mmHg",
                "Acuity (ESI): 2",
            ]),
        ],
    },
]


def build(document, sample: Dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, Spacer

    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=base["Title"], fontSize=15, spaceAfter=2, alignment=0)
    case_style = ParagraphStyle(
        "case", parent=base["Normal"], fontSize=9.5,
        textColor=colors.HexColor("#4b5563"), spaceAfter=8)
    heading_style = ParagraphStyle(
        "heading", parent=base["Heading2"], fontSize=10.5, spaceBefore=10,
        spaceAfter=3, textColor=colors.HexColor("#111827"))
    body_style = ParagraphStyle(
        "body", parent=base["Normal"], fontSize=10, leading=14.5)
    footer_style = ParagraphStyle(
        "footer", parent=base["Normal"], fontSize=8,
        textColor=colors.HexColor("#9ca3af"), spaceBefore=14)

    # The provenance line and the footer are per-sample rather than fixed:
    # make_real_triage_pdfs.py renders genuine held-out records through this
    # same layout, and stamping "synthetic" on a real record would be a lie on
    # the face of the document.
    story = [
        Paragraph(sample["title"], title_style),
        Paragraph(sample.get("case_prefix", "Synthetic case: ") + sample["case"],
                  case_style),
        HRFlowable(width="100%", thickness=0.7,
                   color=colors.HexColor("#d1d5db"), spaceAfter=4),
    ]
    for heading, lines in sample["sections"]:
        story.append(Paragraph(heading, heading_style))
        for line in lines:
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#e5e7eb")))
    story.append(Paragraph(sample.get("disclaimer", DISCLAIMER), footer_style))
    document.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="output directory (default: samples/triage next to backend)")
    args = parser.parse_args()

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError:
        print("reportlab is required: python -m pip install reportlab", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else (BACKEND_DIR / "samples" / "triage")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        path = out_dir / sample["file"]
        document = SimpleDocTemplate(
            str(path), pagesize=A4,
            leftMargin=20 * mm, rightMargin=20 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
            title=sample["title"], author="R26-IT-083 (synthetic)")
        build(document, sample)
        print("  %-34s  expected: %s" % (sample["file"], sample["expected"]))

    print("\nwritten to %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
