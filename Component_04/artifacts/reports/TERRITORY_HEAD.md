# STEMI territory: anterior vs inferior

**Question.** Component 04 predicts four classes. ICD codes the *wall* a STEMI
involves. Is that recoverable from the same ED-triage feature vector?

**Answer.** Yes for the two territories that have enough cases, and the honest
figure depends entirely on whether the ECG cart's own printed interpretation is
allowed as an input.

---

## Where the label comes from

The extraction that built `data/processed/` collapsed every acute-coronary ICD
code into a four-value `acs_label`. Wall location survives in neither the
processed tables nor the feature matrices, so the label was rebuilt from
`hosp/diagnoses_icd.csv.gz`.

Two things make that easy to get wrong, and both were got wrong on the first
attempt:

**WHO ICD-10 and ICD-10-CM are different code sets.** The WHO browser lists
I21.0 anterior and I21.1 inferior. MIMIC uses the US clinical modification,
which subdivides by *culprit artery* instead — I21.02 left anterior descending,
I21.09 other anterior artery, I21.11 right coronary, I21.19 other inferior
artery. Matching the WHO codes returns **zero** anterior and **zero** inferior
cases, which reads as missing data and is really a wrong list.

**This cohort straddles the ICD-9 transition.** 467 of the 941 STEMI stays are
coded in ICD-10 and the other 474 in ICD-9 — almost exactly half. Excluding
ICD-9 "to avoid mixing vocabularies" discards half the labels, and 410.x
carries wall location in its fourth digit at least as richly.

| Territory | ICD-10-CM | ICD-9 | Stays |
|---|---|---|---|
| Anterior | I21.01, I21.02, I21.09 | 410.0, 410.1 | 311 |
| Inferior | I21.11, I21.19 | 410.2, 410.3, 410.4 | 373 |
| Other site | I21.21, I21.29 | 410.5, 410.6, 410.8 | 66 |
| Unspecified | I21.3, I21.9 | 410.9 | 199 |

With both vocabularies every STEMI stay is coded and none is lost.

**676 stays are cleanly one of anterior or inferior** (3 coded as both were
excluded as ambiguous rather than assigned to whichever appeared first).

---

## Protocol

The existing patient-grouped fold assignment is used unchanged — no
`subject_id` appears in more than one fold.

| Fold | Anterior | Inferior | Ratio |
|---|---|---|---|
| train | 224 | 251 | 1 : 1.12 |
| val | 40 | 57 | 1 : 1.43 |
| test | 43 | 61 | 1 : 1.42 |

LightGBM + XGBoost, seven seeds, averaged. The operating threshold is chosen
from **grouped out-of-fold predictions over train + val**, never from test; the
test fold is scored once with that threshold frozen.

The threshold maximises the **worst of all four class metrics** — both recalls
and both precisions. Maximising the worse recall alone is the more usual choice
and was the first one made here, but it lets precision drift: the cut that
recalls anterior generously pays for it in anterior precision, and on a
two-class problem that trade is invisible unless both are watched. Optimising
the worst metric puts the operating point where no per-class number is quietly
worse than the headline.

---

## Results

### FULL — every feature

Threshold 0.515. Test n = 104.

| Class | Recall | Precision | Specificity | F1 | Support |
|---|---|---|---|---|---|
| **Anterior** | 0.8605 | 0.7551 | 0.8033 | 0.8043 | 43 |
| **Inferior** | 0.8033 | 0.8909 | 0.8605 | 0.8448 | 61 |

| Overall | Value |
|---|---|
| AUROC | **0.9074** |
| Accuracy | **0.8269** |
| Balanced accuracy | **0.8319** |
| Macro precision | **0.8230** |
| Macro F1 | **0.8246** |
| Minimum class metric | **0.7551** |

95 % Wilson intervals — anterior recall [0.727, 0.934] (37/43), anterior
precision [0.619, 0.854] (37/49); inferior recall [0.687, 0.884] (49/61),
inferior precision [0.782, 0.949] (49/55).

Confusion matrix, rows = truth:

| | predicted anterior | predicted inferior |
|---|---|---|
| **anterior** | 37 | 6 |
| **inferior** | 12 | 49 |

### PHYSIOLOGY — the same, minus three features

Threshold 0.520. Test n = 104.

| Class | Recall | Precision | Specificity | F1 | Support |
|---|---|---|---|---|---|
| Anterior | 0.7442 | 0.6038 | 0.6557 | 0.6667 | 43 |
| Inferior | 0.6557 | 0.7843 | 0.7442 | 0.7143 | 61 |

| Overall | Value |
|---|---|
| AUROC | 0.7743 |
| Accuracy | 0.6923 |
| Balanced accuracy | 0.7000 |
| Macro precision | 0.6940 |
| Macro F1 | 0.6905 |
| Minimum class metric | 0.6038 |

Confusion matrix, rows = truth:

| | predicted anterior | predicted inferior |
|---|---|---|
| anterior | 32 | 11 |
| inferior | 21 | 40 |

---

## The three features, and why the second table exists

`ecg_infarct_anterior`, `ecg_infarct_inferior` and `ecg_territory_count` are
parsed from the ECG cart's own printed report. Removing them costs **0.133
AUROC and 0.135 accuracy**, and on their own they reach AUROC 0.841 — nearly
the whole of the full model's discrimination from three columns. Every
per-class number falls with them: anterior precision 0.7551 → 0.6038,
inferior recall 0.8033 → 0.6557.

This is not temporal leakage. Those features exist at triage, before any
discharge code. But the ICD coder read the same ECG when assigning the code, so
the feature and the label share a source: a model built on them is largely
**transcribing an interpretation that is already in the record** rather than
deriving one from physiology.

Both tables therefore belong in any write-up. Reporting only the first would be
the same class of mistake as the temporal leak this component already retracted
an AUROC for.

---

## What can and cannot be claimed

**Can:** anterior versus inferior is recoverable with **every per-class metric
at or above 0.75** — the weakest is anterior precision at 0.7551 — on a target
balanced 1 : 1.2, against 1 : 210 for STEMI versus No_ACS.

**Cannot:** the wider split. This was measured rather than assumed —
`src/models/territory_multiclass.py` trains all three configurations:

| Configuration | Anterior | Inferior | Other site | Unspecified | Accuracy |
|---|---|---|---|---|---|
| 2-class | **0.8636** | **0.8033** | — | — | **0.8286** |
| 3-class | 0.7955 | 0.7049 | **0.0833** | — | 0.6752 |
| 4-class | 0.7045 | 0.6557 | 0.0833 | **0.4167** | 0.5816 |

*Other site* is recalled **1 case in 12**. It is a real anatomical territory —
circumflex, lateral, posterior — but at 64 stays the model learns to never
predict it, and class weighting does not rescue a class that thin.

*Unspecified* reaches 0.4167, and even that number means little: I21.3, I21.9
and 410.9 are the coder recording an infarct **without naming a wall**. A model
asked to predict it is being asked to predict whether the documentation was
complete, not where the infarct was.

The decisive part is the first two columns. Adding classes does not merely add
a weak one — it **degrades the two that worked**: anterior falls 0.8636 →
0.7955 → 0.7045 and inferior 0.8033 → 0.7049 → 0.6557, because probability mass
that belonged to them is now spread across classes the model cannot separate.
The binary head is not a simplification for convenience; it is the only
configuration the labels support.

**Read with care:** 104 test cases. Anterior precision is 37/49 = 0.7551 with
an interval of [0.619, 0.854] — **two cases either way moves it across 0.75**.
Every number here clears the bar, and several clear it narrowly enough that the
interval matters more than the point estimate.

**Not served.** This head is a research result and is wired into neither the
backend nor the console. Serving a model whose discrimination rests mostly on
the ECG cart's own printed read would present a transcription as a prediction.

**Method note.** Two configurations were run. The second added class weighting,
tighter capacity for 475 rows against 219 features, and a grouped out-of-fold
threshold — motivated by the prior shift visible between the train and
validation folds (1 : 1.12 against 1 : 1.43), not by test performance. The test
fold was scored once per configuration. Stated because the distinction between
those two things is the whole difference between a result and a number.

---

## Reproduce

```bash
cd Component_04
python src/data/icd_subtypes.py            # label availability
python src/models/train_territory.py --seeds 7
```

Needs `hosp/diagnoses_icd.csv.gz` and `hosp/d_icd_diagnoses.csv.gz` in
`Component_04/data/mimic_icd/`. Both are credentialed MIMIC-IV data under a
PhysioNet DUA; `Component_04/data/` is gitignored in full.
