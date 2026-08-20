"""
Component 04 — clinical text pipeline for ED chief complaints.

Three layers, deliberately kept separable so each can be ablated:

  L1  Normalisation      abbreviation expansion, MIMIC de-identification
                         placeholder removal, punctuation/spacing repair.
  L2  RDM                Referral-Diagnosis Masking.  ED chief complaints in
                         MIMIC-IV frequently carry the *referring hospital's*
                         diagnosis ("STEMI, Transfer", "Elevated troponin").
                         Training on those tokens is label leakage dressed up
                         as NLP.  RDM detects and neutralises them, and emits
                         a single binary flag recording that a referral
                         diagnosis was present, so the confound becomes an
                         auditable variable instead of hidden signal.
  L3  Representation     (a) curated clinical lexicon with negation scoping
                         (b) TF-IDF word 1-2 grams + char 3-5 grams -> SVD

Only the training split is ever used to fit the vectorisers.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# --------------------------------------------------------------------------
# L1 — normalisation
# --------------------------------------------------------------------------
_DEID = re.compile(r"_{2,}|\[\*\*.*?\*\*\]")
_ABBREV: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bc/?p\b"), "chest pain"),
    (re.compile(r"\bcp\b"), "chest pain"),
    (re.compile(r"\bsob\b"), "shortness of breath"),
    (re.compile(r"\bdoe\b"), "dyspnea on exertion"),
    (re.compile(r"\bn/?v\b"), "nausea vomiting"),
    (re.compile(r"\bs/?p\b"), "status post"),
    (re.compile(r"\bl\s*arm\b"), "left arm"),
    (re.compile(r"\br\s*arm\b"), "right arm"),
    (re.compile(r"\bams\b"), "altered mental status"),
    (re.compile(r"\bloc\b"), "loss of consciousness"),
    (re.compile(r"\betoh\b"), "alcohol"),
    (re.compile(r"\bhtn\b"), "hypertension"),
    (re.compile(r"\bdm\b"), "diabetes"),
    (re.compile(r"\bchf\b"), "congestive heart failure"),
    (re.compile(r"\bafib\b|\ba\.?\s?fib\b"), "atrial fibrillation"),
    (re.compile(r"\bpalp\b"), "palpitations"),
    (re.compile(r"\bwk?ns\b|\bweak\b"), "weakness"),
    (re.compile(r"\bdiff\s+breathing\b"), "difficulty breathing"),
    (re.compile(r"\bhypotn\b"), "hypotension"),
]
_MULTISPACE = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9/\s'-]")


def normalise(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(_DEID, " ", regex=True)
    s = s.str.replace(_NONWORD, " ", regex=True)
    for pat, rep in _ABBREV:
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()
    return s


# --------------------------------------------------------------------------
# L2 — Referral-Diagnosis Masking
# --------------------------------------------------------------------------
# Terms that ARE the outcome (or a direct restatement of it) rather than a
# presenting symptom.  Their presence means the diagnosis arrived with the
# patient; it is not something a triage model should be credited for.
_RDM_PATTERNS: Dict[str, str] = {
    "stemi":       r"\bstemi\b|\bst\s*elevation\s*mi\b|\bst\s*elevation\s*myocardial\b",
    "nstemi":      r"\bnstemi\b|\bnon\s*st\s*elevation\s*mi\b|\bnstemi\b",
    "mi":          r"\bmyocardial\s+infarct\w*\b|\bheart\s+attack\b|\bmi\b(?!\w)",
    "acs":         r"\bacs\b|\bacute\s+coronary\b|\bunstable\s+angina\b",
    "troponin":    r"\btropon\w*\b|\belevated\s+trop\w*\b|\bpos\w*\s+trop\w*\b",
    "cath":        r"\bcath\s*lab\b|\bcatheter\w*\b|\bpci\b|\bangiopl\w*\b|\bstent\w*\b",
    "cabg":        r"\bcabg\b|\bbypass\s+graft\b",
    "ekg_finding": r"\bst\s*elevation\b|\bst\s*depression\b|\bekg\s*change\w*\b|"
                   r"\babnormal\s+ekg\b|\bekg\s+abnormal\w*\b",
    "ischemia":    r"\bischemi\w*\b|\bcoronary\s+dis\w*\b|\bcad\b",
}
_RDM_COMPILED = {k: re.compile(v) for k, v in _RDM_PATTERNS.items()}
_RDM_ANY = re.compile("|".join(_RDM_PATTERNS.values()))


def apply_rdm(series: pd.Series, enable: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns (masked_text, flags_dataframe).

    flags:  cc_referral_dx        1 if any referral-diagnosis token present
            cc_referral_dx_<k>    per-category indicator (audit only)
            cc_transfer           explicit inter-facility transfer mention
    """
    flags = pd.DataFrame(index=series.index)
    flags["cc_referral_dx"] = series.str.contains(_RDM_ANY, regex=True).astype(np.int8)
    for k, pat in _RDM_COMPILED.items():
        flags[f"cc_rdx_{k}"] = series.str.contains(pat, regex=True).astype(np.int8)
    flags["cc_transfer"] = series.str.contains(
        r"\btransfer\w*\b|\bosh\b|\boutside\s+hospital\b", regex=True
    ).astype(np.int8)

    if not enable:
        return series, flags

    masked = series
    for pat in _RDM_COMPILED.values():
        masked = masked.str.replace(pat, " ", regex=True)
    masked = masked.str.replace(_MULTISPACE, " ", regex=True).str.strip()
    return masked, flags


# --------------------------------------------------------------------------
# L3a — curated clinical lexicon with negation scoping
# --------------------------------------------------------------------------
# Weight = literature-informed contribution to an ACS pre-test probability
# (chest pain quality, radiation, autonomic symptoms, anginal equivalents).
_LEXICON: Dict[str, Tuple[str, float]] = {
    "cc_chest_pain":     (r"\bchest\s*(?:pain|pressure|tightness|discomfort|burning|"
                          r"heaviness|squeez\w*)\b|\bangina\b|\bsubsternal\b|\bprecordial\b", 3.0),
    "cc_chest_any":      (r"\bchest\b", 1.5),
    "cc_dyspnea":        (r"\bshortness of breath\b|\bdyspnea\b|\bdifficulty breathing\b|"
                          r"\bbreathing problem\b|\bcannot breathe\b|\bwinded\b", 1.5),
    "cc_exertional":     (r"\bon exertion\b|\bexertional\b|\bwith activity\b|\bwalking\b", 1.5),
    "cc_radiation":      (r"\bradiat\w*\b|\bleft arm\b|\barm pain\b|\bjaw\b|\bshoulder\b|"
                          r"\bneck pain\b|\bback pain.*chest\b|\binterscapular\b", 2.0),
    "cc_diaphoresis":    (r"\bdiaphor\w*\b|\bsweat\w*\b|\bclammy\b", 2.0),
    "cc_nausea":         (r"\bnausea\b|\bvomit\w*\b|\bemesis\b", 1.0),
    "cc_syncope":        (r"\bsyncop\w*\b|\bfaint\w*\b|\bpass\w* out\b|"
                          r"\bloss of consciousness\b|\bdizz\w*\b|\bpresyncop\w*\b", 1.0),
    "cc_palpitations":   (r"\bpalpitat\w*\b|\bheart racing\b|\birregular heart\b|"
                          r"\bfast heart\b|\batrial fibrillation\b|\bfluttering\b", 1.0),
    "cc_cardiac_arrest": (r"\bcardiac arrest\b|\barrest\b|\bcode\b|\bunresponsive\b|\bcpr\b", 3.0),
    "cc_epigastric":     (r"\bepigastr\w*\b|\bindigestion\b|\bheartburn\b|\breflux\b", 1.0),
    "cc_hypotension":    (r"\bhypotens\w*\b|\blow blood pressure\b|\bshock\b", 2.0),
    "cc_edema":          (r"\bedema\b|\bswelling\b|\bcongestive heart failure\b", 0.5),
    "cc_weakness":       (r"\bweakness\b|\bfatigue\b|\bmalaise\b|\bgenerali\w*\s+weak\w*\b", 0.5),
    "cc_altered":        (r"\baltered mental status\b|\bconfusion\b|\blethargy\b", 0.5),
    # Explicit non-cardiac competing presentations (negative evidence)
    "cc_trauma":         (r"\bfall\b|\bmvc\b|\bassault\w*\b|\bfracture\b|\blaceration\b|"
                          r"\bstab\w*\b|\bgsw\b|\btrauma\b|\bwound\b|\bburn\b", -2.0),
    "cc_psych":          (r"\bsuicidal\b|\bsi\b|\bpsych\w*\b|\bdepress\w*\b|\banxiety\b|"
                          r"\bagitat\w*\b|\boverdose\b|\balcohol\b|\bdetox\b", -2.0),
    "cc_infection":      (r"\bfever\b|\bcough\b|\bflu\b|\bsore throat\b|\burinary\b|"
                          r"\bcellulitis\b|\babscess\b|\bsepsis\b", -1.5),
    "cc_abdominal":      (r"\babd\w*\s*pain\b|\babdominal\b|\bdiarrhea\b|\bconstipat\w*\b|"
                          r"\bbrbpr\b|\bgi bleed\b", -1.5),
    "cc_neuro":          (r"\bheadache\b|\bseizure\b|\bstroke\b|\bnumbness\b|\bvertigo\b", -1.0),
}
_LEX_COMPILED = {k: (re.compile(p), w) for k, (p, w) in _LEXICON.items()}

# Negation: a trigger followed (within 4 tokens) by a concept negates it.
# NOTE the trailing \s* rather than \s+.  Requiring whitespace AFTER each
# captured word silently truncates the scope when the negated phrase ends the
# string: "denies chest pain" captured only "chest ", so `cc_chest_pain` was
# scored as present while `cc_chest_any` was correctly negated.  Measured over
# the full cohort the fix changes 2 lexicon flags in 203,016 records (0.001%),
# so the trained models were not invalidated by it, but the patient-facing
# explanation was visibly wrong and is what this corrects.
_NEG_TRIGGER = re.compile(
    r"\b(no|not|denies|denied|without|negative for|w/?o|neg)\b\s+((?:\w+\s*){0,4})"
)


def _negated_spans(text: str) -> str:
    """Return the concatenated text that falls inside a negation scope."""
    return " ".join(m.group(2) for m in _NEG_TRIGGER.finditer(text))


def lexicon_features(series: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    neg = series.map(_negated_spans)
    for name, (pat, _w) in _LEX_COMPILED.items():
        hit = series.str.contains(pat, regex=True)
        negated = neg.str.contains(pat, regex=True)
        out[name] = (hit & ~negated).astype(np.int8)
    out["cc_negation_present"] = neg.str.len().gt(0).astype(np.int8)
    out["cc_n_tokens"] = series.str.split().map(len).astype(np.int16)
    out["cc_n_chars"] = series.str.len().astype(np.int16)
    out["cc_empty"] = (out["cc_n_tokens"] == 0).astype(np.int8)
    out["cc_n_complaints"] = series.str.count(",").astype(np.int8) + 1

    # Weighted pre-test-probability score (the interpretable NLP summary)
    score = np.zeros(len(series), dtype=np.float32)
    for name, (_p, w) in _LEX_COMPILED.items():
        score += out[name].to_numpy(dtype=np.float32) * w
    out["cc_acs_lexicon_score"] = score
    # Cardiac-vs-noncardiac contrast
    pos = [k for k, (_p, w) in _LEX_COMPILED.items() if w > 0]
    negk = [k for k, (_p, w) in _LEX_COMPILED.items() if w < 0]
    out["cc_cardiac_hits"] = out[pos].sum(axis=1).astype(np.int8)
    out["cc_noncardiac_hits"] = out[negk].sum(axis=1).astype(np.int8)
    return out


CARDIAC_COHORT_RE = re.compile(
    r"\bchest\b|\bangina\b|\bsubsternal\b|\bprecordial\b|\bshortness of breath\b|"
    r"\bdyspnea\b|\bpalpitat\w*\b|\bsyncop\w*\b|\bdiaphor\w*\b|\bcardiac\b|\bheart\b|"
    r"\barrest\b|\bepigastr\w*\b|\bjaw\b|\bleft arm\b|\barm pain\b|\bekg\b|\becg\b|"
    r"\btropon\w*\b|\bstemi\b|\bnstemi\b|\bmyocardial\b|\bhypotens\w*\b|"
    r"\bbradycard\w*\b|\btachycard\w*\b|\bcongestive heart failure\b|\bmi\b"
)


def is_cardiac_presentation(normalised: pd.Series) -> pd.Series:
    """Cohort selector — evaluated on the NORMALISED (pre-RDM) text."""
    return normalised.str.contains(CARDIAC_COHORT_RE, regex=True)


# --------------------------------------------------------------------------
# L3b — TF-IDF (word + char) -> SVD
# --------------------------------------------------------------------------
class TextEmbedder:
    """Fit on train only; transform any split.  Produces `n_components` dims."""

    def __init__(self, n_components: int = 24, word_max: int = 6000,
                 char_max: int = 6000, min_df: int = 3, seed: int = 42):
        self.union = FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  max_features=word_max, min_df=min_df,
                                  sublinear_tf=True, strip_accents="unicode")),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  max_features=char_max, min_df=min_df,
                                  sublinear_tf=True)),
        ])
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.n_components = n_components
        self.fitted = False

    def fit(self, texts: pd.Series) -> "TextEmbedder":
        X = self.union.fit_transform(texts)
        n = min(self.n_components, max(2, X.shape[1] - 1))
        if n != self.n_components:
            self.svd = TruncatedSVD(n_components=n, random_state=self.svd.random_state)
            self.n_components = n
        self.svd.fit(X)
        self.fitted = True
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("TextEmbedder must be fitted on the training split first")
        Z = self.svd.transform(self.union.transform(texts))
        return pd.DataFrame(
            Z.astype(np.float32),
            columns=[f"cc_svd_{i:02d}" for i in range(Z.shape[1])],
            index=texts.index,
        )

    @property
    def explained_variance(self) -> float:
        return float(self.svd.explained_variance_ratio_.sum()) if self.fitted else 0.0

    def top_terms(self, k: int = 12) -> Dict[str, List[str]]:
        """Highest-loading terms per SVD component — used in the report."""
        names = np.array(self.union.get_feature_names_out())
        out = {}
        for i, comp in enumerate(self.svd.components_):
            idx = np.argsort(np.abs(comp))[::-1][:k]
            out[f"cc_svd_{i:02d}"] = [str(names[j]) for j in idx]
        return out


# --------------------------------------------------------------------------
# Token-level attribution for a single prediction (used by the explainer)
# --------------------------------------------------------------------------
def token_attribution(raw_text: str) -> List[Dict]:
    """
    Lexicon-driven token highlighting for one chief complaint.
    Returns [{term, category, weight, negated}] ordered by |weight|.
    """
    norm = normalise(pd.Series([raw_text])).iloc[0]
    neg_span = _negated_spans(norm)
    out: List[Dict] = []
    for name, (pat, w) in _LEX_COMPILED.items():
        for m in pat.finditer(norm):
            negated = bool(pat.search(neg_span))
            out.append({
                "term": m.group(0),
                "category": name.replace("cc_", ""),
                "weight": 0.0 if negated else float(w),
                "negated": negated,
            })
    # de-duplicate by (term, category)
    seen, uniq = set(), []
    for d in out:
        key = (d["term"], d["category"])
        if key not in seen:
            seen.add(key); uniq.append(d)
    return sorted(uniq, key=lambda d: -abs(d["weight"]))
