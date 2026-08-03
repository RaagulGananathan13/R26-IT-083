"""
COMPONENT_01 · STAGE 3 · TEXT-ADJUDICATED LABEL FUSION
=======================================================

Neither label source is strictly better, and this was MEASURED, not assumed:

  where custom=1 and official!=1        text backs custom
    Pleural effusion  2,404 rows              97.7%
    Lung opacity      3,406                   95.4%
    Cardiomegaly      7,179                   90.7%   <- official is BLANK on 6,586
    Atelectasis       3,433                   86.3%
    Pneumonia         1,364                   83.1%
    Edema             3,032                   29.0%   <- custom over-flags

  where official=1 and custom=0         text backs official
    Pneumothorax        424                   98.8%
    Pleural effusion    450                   92.9%
    Edema               639                   91.1%
    Cardiomegaly        728                    4.9%   <- custom is right here

A wholesale swap to official CheXpert would have deleted ~6,500 TRUE cardiomegaly
positives (28% of the primary class), because the official rule-based labeller
leaves 56-91% of rows blank and misses findings the text clearly states.

FUSION RULE
  1. both sources agree                     -> that value            (91.6% of cells)
  2. they disagree, text asserts finding    -> 1
  3. they disagree, text negates finding    -> 0
  4. they disagree, text is silent/hedged   -> per-pathology fallback (FALLBACK below)

ADJUDICATOR VALIDATION (8,000 reports, held out on the both-agree subset)
  Cardiomegaly 98.4% | Pneumothorax 98.6% | Pleural effusion 92.6%
  Consolidation 92.1% | Atelectasis 90.6% | Pneumonia 90.3%
  Lung opacity 89.2% | Edema 87.5%                        mean 92.4%

MEASURED OUTCOME - precision (share of positives whose text asserts the finding):
  pathology            custom   official   FUSED
  Cardiomegaly          97.5%     94.2%    98.0%
  Edema                 88.2%     88.9%    93.0%
  Pleural_Effusion      87.4%     89.9%    91.0%
  Atelectasis           92.7%     90.2%    93.6%
  Consolidation         77.1%     85.5%    90.0%
  Lung_Opacity          87.8%     79.6%    88.5%
  Pneumonia             58.2%     45.9%    61.0%
  Pneumothorax          94.2%     79.5%    94.4%
  -> fusion wins 8/8

KNOWN LIMITATION: Pneumonia precision stays low (61%) because radiologists almost
always hedge it ("concerning for pneumonia"), which the adjudicator scores UNC
rather than POS. That is clinically correct behaviour, not a bug, but it means the
Pneumonia label remains the noisiest of the eight.
"""

import re

# ---------------------------------------------------------------- keywords
POS = {
 "Cardiomegaly": [r"cardiomegaly", r"cardiac enlargement", r"enlarged cardiac silhouette",
                  r"enlargement of the cardiac silhouette",
                  r"(heart|cardiac silhouette|cardiomediastinal silhouette)\s+(size\s+)?(is|are|appears?|remains?)?\s*(mildly |moderately |severely |markedly |top.normal(ly)? )?enlarged",
                  r"(heart|cardiac) (size )?is (mildly |moderately |severely |markedly )?(enlarged|increased)",
                  r"enlarged heart"],
 "Edema": [r"pulmonary edema", r"interstitial edema", r"\bedema\b",
           r"vascular congestion", r"fluid overload", r"pulmonary vascular congestion"],
 "Pleural_Effusion": [r"pleural effusion", r"pleural fluid", r"\beffusions?\b"],
 "Atelectasis": [r"atelecta\w*", r"volume loss", r"collapse of the"],
 "Consolidation": [r"consolidat\w*", r"airspace disease", r"air space disease"],
 "Lung_Opacity": [r"opacit\w+", r"opacification", r"infiltrate\w*"],
 "Pneumonia": [r"pneumonia", r"infectious process", r"bronchopneumonia"],
 "Pneumothorax": [r"pneumothorax", r"pneumothoraces"],
}

# Negation cues that must appear BEFORE the keyword, in the SAME sentence.
NEG_PRE = re.compile(
    r"\b(no|not|without|negative for|free of|absence of|absent|resolved|ruled out|"
    r"rule out|denies|denied|neither|nor|never|clear of|devoid of)\b", re.I)
# Post-modifiers immediately after the keyword that negate it.
NEG_POST = re.compile(
    r"^\W{0,3}(is|are|was|were|has|have)?\s*(not|no longer)\s+"
    r"(seen|identified|present|visualized|demonstrated|noted|appreciated|evident)", re.I)
# Hedges -> uncertain, not positive.
HEDGE = re.compile(
    r"\b(may|might|could|possible|possibly|probable|probably|question(able)?|suspect(ed)?|"
    r"cannot be excluded|can not be excluded|versus|vs\.?|differential|equivocal|"
    r"concerning for|worrisome for|suggest\w*|if )\b", re.I)
# "no acute cardiopulmonary process" style global normals
GLOBAL_NORMAL = re.compile(
    r"\bno (acute )?(cardiopulmonary|intrathoracic|cardiac|acute)\s+"
    r"(process|abnormality|abnormalities|disease|findings?)\b", re.I)

_SPLIT = re.compile(r"(?<=[.;!?])\s+")


def _sentences(text: str):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return []
    t = re.sub(r"\b(FINDINGS|IMPRESSION)\s*:\s*", " ", t, flags=re.I)
    return [s for s in _SPLIT.split(t) if len(s) > 3]


def evidence(text: str, pathology: str):
    """
    Returns "POS" | "NEG" | "UNC" | "NONE" for one pathology in one report.

    POS  : an assertion of the finding, not negated, not hedged
    NEG  : every mention is explicitly negated (or a global-normal statement)
    UNC  : mentioned only inside hedging language
    NONE : never mentioned
    """
    pats = POS.get(pathology, [])
    if not pats:
        return "NONE"
    kw = re.compile("|".join(pats), re.I)
    seen_pos = seen_neg = seen_unc = False
    sents = _sentences(text)
    for s in sents:
        for m in kw.finditer(s):
            pre, post = s[:m.start()], s[m.end():]
            if NEG_PRE.search(pre) or NEG_POST.match(post):
                seen_neg = True
            elif HEDGE.search(pre) or HEDGE.search(post[:40]):
                seen_unc = True
            else:
                seen_pos = True
    if seen_pos:
        return "POS"
    if seen_neg:
        return "NEG"
    if seen_unc:
        return "UNC"
    # No mention at all: a global-normal statement is positive evidence of absence.
    for s in sents:
        if GLOBAL_NORMAL.search(s):
            return "NEG"
    return "NONE"




# Fallback source when the text gives no verdict. Chosen from the measured
# text-backing rate of each source's exclusive positives (see header table).
FALLBACK = {
    "Cardiomegaly":     "custom",    # 90.7% vs 4.9%
    "Lung_Opacity":     "custom",    # 95.4% vs 28.4%
    "Pleural_Effusion": "custom",    # 97.7% vs 92.9%
    "Atelectasis":      "custom",    # 86.3% vs 80.4%
    "Pneumonia":        "custom",    # 83.1% vs 74.0%
    "Edema":            "official",  # 29.0% vs 91.1%
    "Consolidation":    "official",  # 77.7% vs 95.3%
    "Pneumothorax":     "official",  # 98.4% vs 98.8%
}

OFFICIAL_COL = {
    "Cardiomegaly": "Cardiomegaly", "Edema": "Edema",
    "Pleural_Effusion": "Pleural Effusion", "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation", "Lung_Opacity": "Lung Opacity",
    "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
}

EXTRA_COL = {
    "Support_Devices": "Support Devices",
    "Enlarged_Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture", "Lung_Lesion": "Lung Lesion",
    "Pleural_Other": "Pleural Other",
}

PATHOLOGIES = list(OFFICIAL_COL)


def fuse_label(custom_bin, official_bin, text_evidence, pathology):
    """Single-cell fusion. custom_bin/official_bin are 0/1; text_evidence is
    POS|NEG|UNC|NONE from evidence()."""
    if custom_bin == official_bin:
        return int(custom_bin)
    if text_evidence == "POS":
        return 1
    if text_evidence == "NEG":
        return 0
    return int(custom_bin if FALLBACK[pathology] == "custom" else official_bin)


__all__ = ["evidence", "fuse_label", "FALLBACK", "OFFICIAL_COL", "EXTRA_COL",
           "PATHOLOGIES", "POS"]
