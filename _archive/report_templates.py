"""
report_templates.py — Classifier-Grounded Clinical Report Engine

Pipeline role (Tier 2):
  Classifier probs  →  Threshold gate  →  Template slot-fill  →  Structured report

Key design principle:
  The report can ONLY describe what the ResNet classifier confirmed.
  No language model is involved — hallucination risk is zero at this stage.

Tier 3 (BioBART smoother) is optional and handled in app.py / predict.py.
The smoother uses the structured sentences as its prompt, not the raw signal.
"""

# ── Severity thresholds 
# These are relative to each class's optimal decision threshold.
# We compute how far above threshold the probability is, then bucket.
HIGH_MARGIN   = 0.15   # prob >= threshold + 0.15  → "high confidence" phrasing
MEDIUM_MARGIN = 0.00   # prob >= threshold           → "moderate confidence" phrasing
# Below threshold → not included in report at all.

#  Clinical sentence templates 
# Each class has two severity tiers.
# Sentences are written to match PTB-XL report style.
TEMPLATES = {
    "NORM": {
        "high": (
            "The ECG is within normal limits. "
            "Sinus rhythm is present with no significant abnormalities detected."
        ),
        "medium": (
            "ECG shows predominantly normal features. "
            "Minor non-specific findings are present but do not meet criteria for a defined abnormality."
        ),
    },
    "MI": {
        "high": (
            "ST-segment changes are present and are consistent with myocardial infarction. "
            "Urgent clinical and cardiology review is recommended."
        ),
        "medium": (
            "ST-segment morphology raises concern for possible myocardial infarction. "
            "Clinical correlation and further evaluation are advised."
        ),
    },
    "STTC": {
        "high": (
            "Diffuse ST-segment and T-wave changes are identified. "
            "Ischaemia, metabolic disturbance, or drug effect should be excluded."
        ),
        "medium": (
            "Non-specific ST-segment and T-wave changes are noted. "
            "Clinical context is required for definitive interpretation."
        ),
    },
    "CD": {
        "high": (
            "Conduction delay is detected, consistent with bundle branch block "
            "or intraventricular conduction disturbance. "
            "Cardiology referral is recommended."
        ),
        "medium": (
            "Possible conduction abnormality is present. "
            "Repeat ECG and clinical correlation are recommended."
        ),
    },
    "HYP": {
        "high": (
            "Voltage criteria are consistent with ventricular hypertrophy. "
            "Echocardiographic correlation is advised."
        ),
        "medium": (
            "Borderline voltage findings are present; ventricular hypertrophy cannot be excluded. "
            "Clinical correlation is recommended."
        ),
    },
}

# ── Introductory phrases (used when combining multiple findings) ──────────────
INTRO_NORMAL     = "ECG interpretation:"
INTRO_ABNORMAL   = "ECG interpretation — the following findings were identified:"
CLOSING_NORMAL   = "No further immediate action is required based on ECG findings alone."
CLOSING_ABNORMAL = "Clinical correlation with patient history and examination findings is essential."


def _severity_tier(prob: float, threshold: float) -> str | None:
    """
    HALLUCINATION GATE — determines if a class should appear in the report.
    This is the critical safety check: if prob < threshold, the class is
    completely excluded from the report. No sentence can be generated for it.
    
    Returns:
      "high"   — prob >= threshold + 0.15 (strong detection, urgent language)
      "medium" — prob >= threshold (detected, moderate language)
      None     — prob < threshold (NOT detected, excluded from report entirely)
    """
    if prob < threshold:
        return None  # BELOW THRESHOLD: this class will NOT appear in the report
    margin = prob - threshold
    if margin >= HIGH_MARGIN:
        return "high"    # strong confidence -> urgent clinical language
    return "medium"      # moderate confidence -> cautious clinical language


def build_structured_report(
    probs: list | dict,
    thresholds: list | dict,
    class_names: list[str] = None,
) -> dict:
    """
    CORE HALLUCINATION-FREE REPORT BUILDER (Tier 2).
    
    This function is the heart of the hallucination prevention system.
    It loops through each of the 5 classes, checks if the CNN probability
    exceeds its optimal threshold, and ONLY includes pre-written clinical
    sentences for classes that were actually detected.
    
    WHY IT CANNOT HALLUCINATE:
      - The TEMPLATES dict contains fixed, pre-written medical sentences.
      - A sentence is only selected if _severity_tier() returns non-None.
      - _severity_tier() only returns non-None if prob >= threshold.
      - Therefore, the report can ONLY describe what the CNN confirmed.
      - No language model is involved at this stage.
    
    Args:
        probs       : CNN sigmoid probabilities [NORM, MI, STTC, CD, HYP]
        thresholds  : Per-class optimal decision thresholds (tuned on validation set)
        class_names : Ordered class name list. Default: ["NORM","MI","STTC","CD","HYP"]
    
    Returns dict with:
        detected_labels : list of class names above threshold
        sentences       : one pre-written sentence per detected class
        report_text     : joined paragraph (the final Tier 2 report)
        confidence_map  : {class: {prob, threshold, tier}} for the UI
        has_abnormality : True if any non-NORM class was detected
    """
    if class_names is None:
        class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

    # Normalise inputs to lists
    if isinstance(probs, dict):
        probs = [probs[c] for c in class_names]
    if isinstance(thresholds, dict):
        thresholds = [thresholds[c] for c in class_names]

    detected_labels = []
    sentences       = []
    confidence_map  = {}

    for i, cls in enumerate(class_names):
        p   = float(probs[i])
        thr = float(thresholds[i])
        tier = _severity_tier(p, thr)

        confidence_map[cls] = {
            "prob":      round(p * 100, 1),
            "threshold": round(thr * 100, 1),
            "tier":      tier,   # None if not detected
        }

        if tier is not None:
            detected_labels.append(cls)
            sentences.append(TEMPLATES[cls][tier])

    has_abnormality = any(c != "NORM" for c in detected_labels)

    # If nothing was detected at all, fall back to a conservative statement
    if not detected_labels:
        sentences = [
            "No findings exceeded the diagnostic threshold. "
            "ECG interpretation is inconclusive — manual review by a clinician is required."
        ]
        report_text = sentences[0]
    else:
        # Build paragraph
        if len(sentences) == 1:
            report_text = sentences[0]
        else:
            bullet_sentences = " ".join(sentences)
            report_text = bullet_sentences

        # Append closing
        closing = CLOSING_ABNORMAL if has_abnormality else CLOSING_NORMAL
        report_text = report_text.rstrip(".") + ". " + closing

    return {
        "detected_labels": detected_labels,
        "sentences":       sentences,
        "report_text":     report_text,
        "confidence_map":  confidence_map,
        "has_abnormality": has_abnormality,
    }


def format_smoother_prompt(structured_report: dict) -> str:
    """
    Format the structured report as a prompt for BioBART (Tier 3 smoother).
    The prompt instructs BioBART to ONLY rephrase — not to invent new findings.

    This is a constrained paraphrase task, not free generation.
    The model cannot add information that isn't in the structured_report.
    """
    structured_text = structured_report["report_text"]
    prompt = (
        f"Rephrase the following clinical ECG report in natural medical language. "
        f"Do not add, remove, or change any clinical findings. "
        f"Only improve the phrasing and flow:\n\n{structured_text}"
    )
    return prompt
