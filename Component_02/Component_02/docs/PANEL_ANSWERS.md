# Progress Presentation 2 — Answers

**Venushan T** · XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting
Component 02 · PTB-XL · 12-lead · 5 diagnostic superclasses

Every number below is reproducible from `analysis/` and `audit/`. Nothing is
estimated or quoted from memory.

---

## 1. HEADLINE RESULT — every class meets the target

**Test fold 10 (n = 1,711), never used to select anything.**

| Class | Accuracy | **Recall** | Specificity | **NPV** | Precision | F1 |
|---|---|---|---|---|---|---|
| NORM | 0.883 | **0.796** | 0.943 | 0.868 | 0.908 | 0.849 |
| MI | 0.884 | **0.836** | 0.893 | **0.967** | 0.591 | 0.692 |
| STTC | 0.868 | **0.803** | 0.892 | 0.926 | 0.729 | 0.764 |
| CD | 0.869 | **0.805** | 0.894 | 0.921 | 0.750 | 0.776 |
| HYP | 0.817 | **0.811** | 0.818 | **0.981** | 0.271 | 0.406 |
| **MACRO** | **0.864** | **0.810** | 0.888 | **0.933** | 0.650 | 0.698 |

> ✅ **Accuracy ≥ 0.75 for all 5 classes**
> ✅ **Recall (sensitivity) ≥ 0.75 for all 5 classes**
> ✅ **NPV ≥ 0.86 for all 5 classes — macro 0.933**

Discrimination (threshold-free), 3 seeds: **macro-AUROC 0.9343 ± 0.0028**,
**macro-AUPRC 0.8001 ± 0.0029**.

Reproduce: `python -X utf8 analysis/02_operating_point.py`

---

## 2. "Why is your F1 low on HYP?" — answer this before they ask

**Because F1 is the wrong metric for a rule-out system, and hypertrophy is the
hardest class in this dataset for four measurable reasons.**

F1 weights a missed infarction and an unnecessary review **equally**. No
cardiology pathway does that — the ESC 0/1h hs-troponin algorithm and HEART are
governed by **sensitivity and NPV**. Our HYP F1 is 0.41; our HYP **NPV is 0.981**.
The second number is the one that decides whether you can rule hypertrophy out.

Why HYP is hard — measured, from `analysis/01_dataset_deep_audit.py`:

| Reason | Evidence |
|---|---|
| **Scarcity** | 1,468 positives, 8.5% prevalence, **only 132 in the test fold** |
| **Entanglement** | of HYP records, **63.8% also carry STTC**, 35.7% also CD, **0% are NORM** |
| **Voltage = amplitude** | LVH is diagnosed by QRS amplitude (Sokolow-Lyon, Cornell); any amplitude normalisation destroys the evidence — which is why our augmentation is capped at 0.9–1.1×, not 0.8–1.2× |
| **Label noise** | ECG criteria for LVH have known low sensitivity against echocardiography, the true reference. The label is a proxy. |

**Published ceiling for HYP F1 on PTB-XL is ≈ 0.54.** Our HYP AUPRC improved
**0.5405 → 0.5918** over the audited baseline — a real gain on the hardest class.
A claim of HYP F1 ≥ 0.75 would require AUPRC > 0.8 and is not supported by any
published result on this dataset.

---

## 3. THE RESEARCH GAP

Conformal prediction gives a classifier a **provable** bound: *"when I rule this
out, I am wrong at most α of the time."* It has been applied to ECG (2025) and to
medical imaging.

**But every published result validates the guarantee MARGINALLY — averaged over
the whole population.** A cardiologist never treats the average. Nobody has asked
whether the guarantee still holds **for the individual patient in front of them**.

> **Gap: is a conformal ECG guarantee valid *conditionally on the patient*, or
> only on average?**

---

## 4. THE CONTRIBUTION

> ### A conformal ECG system can satisfy its advertised guarantee exactly — and still be unsafe for identifiable groups of patients.

Thresholds fitted the standard (marginal) way on fold 9; miss rate then measured
**inside each subgroup** of fold 10.

| Class | Promised α | **Overall** | <50 | 50–69 | ≥70 |
|---|---|---|---|---|---|
| NORM | 0.20 | **0.190 ✓** | 0.103 | 0.228 | **0.330** |
| MI | 0.05 | **0.015 ✓** | 0.000 | 0.011 | 0.019 |
| STTC | 0.10 | **0.092 ✓** | 0.128 | 0.080 | 0.093 |
| CD | 0.10 | **0.099 ✓** | **0.333** | 0.099 | 0.042 |
| HYP | 0.15 | **0.121 ✓** | 0.444¹ | 0.159 | 0.066 |

¹ n = 9 — indicative only.

**Every class passes overall. Two violations survive rigorous statistical testing.**

23 class–subgroup cells were tested. Each was assessed with a Wilson confidence
interval, an exact one-sided binomial test, **Holm correction for 23
comparisons**, and a 2,000-draw calibration bootstrap. Only two survive — and
they survive overwhelmingly:

| Cell | Promised | Observed | 95% CI | Holm-adj. p | Bootstrap |
|---|---|---|---|---|---|
| **CD, age < 50** | ≤ 10% | **33.3%** (22/66) | **[23.2%, 45.3%]** | **5.1×10⁻⁶** | violated in **100%** of 2,000 draws |
| **NORM, age ≥ 70** | ≤ 20% | **33.0%** (34/103) | **[24.7%, 42.6%]** | **2.9×10⁻²** | violated in **100%** of 2,000 draws |

Both confidence intervals lie **entirely above** the promised bound. The
bootstrap resamples the calibration fold and refits the threshold each time, so
these are not artefacts of one lucky calibration draw — they are structural.

The other 7 apparent excesses **did not survive multiple-testing correction** and
are reported as noise. Saying so is part of the result.

> The overall CD figure — 9.9%, comfortably inside the promised 10% — gives no
> hint that a third of under-50 patients with conduction disturbance are missed.

Reproduce: `python -X utf8 audit/11_significance.py`

**The fix:** Mondrian (group-conditional) calibration — one threshold per subgroup.

| | Cells satisfying the bound |
|---|---|
| Marginal calibration (standard practice) | 14 / 23 |
| **Mondrian (group-conditional)** | **22 / 23 (96%)** |

**And its cost — a second finding:** subgroup calibration needs enough positives
*per group*. ST/T change in under-50s had 42 calibration positives, the PAC bound
became infeasible, and the system returned λ = −∞: **it can never rule out ST/T
change in a young patient.** Conditional validity costs data, and the groups
needing it most have the least.

Reproduce: `python -X utf8 audit/10_conditional_validity.py`

---

## 4b. THE SECOND DEMONSTRATION — same principle, different axis

The subgroup result showed a guarantee failing **conditional on the patient**.
The same failure exists **conditional on the recording**, and it is invisible.

### Electrode reversal is invisible to signal-quality assessment

A swapped cable produces a perfectly clean signal — correct amplitude, correct
duration, no noise, no flat leads. Measured on 600 test records:

| | Quality gate accepts | Mean quality index |
|---|---|---|
| Correct placement | 589/600 | 1.000 |
| RA/LA reversal | **587/600** | **1.000** |
| RA/LL reversal | **589/600** | **1.000** |
| LA/LL reversal | **589/600** | **1.000** |

### It changes the diagnosis

| Reversal | Any label flips | MI flips |
|---|---|---|
| RA/LA | **86.8%** | 56.2% |
| RA/LL | **87.8%** | 57.7% |
| LA/LL | 46.5% | 21.3% |

### And it voids the guarantee — the question nobody has asked

| Class | Promised | Correct | RA/LA | RA/LL | LA/LL |
|---|---|---|---|---|---|
| NORM | ≤20% | 16.1% | **99.6%** ❗ | **100%** ❗ | 19.3% |
| MI | ≤5% | 1.0% | 0.0% | 1.0% | **14.4%** ❗ |
| STTC | ≤10% | 11.4% | **70.9%** ❗ | **33.5%** ❗ | 10.1% ❗ |
| HYP | ≤15% | 11.4% | **25.0%** ❗ | **47.7%** ❗ | 4.5% |

**Nine guarantees voided by a cable.** Under RA/LA reversal the system fails to
recognise 99.6% of normal ECGs while still printing its promise.

Prior work on lead reversal does two things: **detect** it (ML classifiers) and
**measure accuracy loss** under it. Neither asks what it does to a *statistical
guarantee*. That is the gap.

### The unification — this is the point

At a realistic reversal prevalence of 0.4–4%, the **population-level** guarantee
still survives, because 96–99.6% of records are correctly placed. A hospital
auditing across all its ECGs would see nothing wrong. But for **the individual
patient whose electrodes were swapped**, the promise is void — and nothing tells
anyone.

| Marginal validity holds | Conditional validity fails |
|---|---|
| across the whole population | within a patient subgroup (age) |
| across all recordings | within a mis-acquired recording |

> **One principle, two independent demonstrations: a conformal guarantee averaged
> over a population says nothing about the patient in front of you.**

### Implemented in the system

`src/electrodes.py` — exact reversal simulators plus a **physiology-based**
detector (aVR polarity + lead I inversion). No machine learning; every decision
is a stated clinical rule. When a reversal is suspected the system:

1. shows it in the UI and in the report's technical line
2. **withdraws its statistical guarantees for that record**, replacing them with
   *"GUARANTEES SUSPENDED — the conformal bounds are calibrated on
   correctly-placed recordings and do not hold here"*
3. still analyses the record, rather than refusing it silently

Measured: **70% sensitivity on RA/LA, 61% on RA/LL, 4.5% false-positive rate.**
Refusing detections restored only 1 of 9 guarantees — so detection alone is
**not** the fix, and I report that rather than claiming it is.

Reproduce: `python -X utf8 audit/12_electrode_reversal.py`

---

## 4c. THE THIRD DEMONSTRATION — the guarantee certifies the wrong answer

The first two showed guarantees failing conditional on the **patient** and on the
**recording**. The third is the one a cardiologist reacts to hardest: the
guarantee fails when the patient's actual disease is **not in the label space at
all**.

### The model has five output units and no way to abstain

NORM, MI, STTC, CD, HYP. There is no unit for atrial fibrillation, flutter, paced
rhythm or ventricular tachycardia. Softmax has no "none of the above", so an AF
recording is redistributed across the five classes that do exist, the conformal
layer converts that into a decision, and the report prints a bounded miss rate
beside it.

### How much disease is outside the label space?

Cardiologist-documented findings in the reference reports of the very records
this system was trained and evaluated on:

| Condition | Records | Share |
|---|---|---|
| Atrial fibrillation | 1,225 | 7.11% |
| Ventricular ectopy | 1,127 | 6.54% |
| Atrial ectopy | 617 | 3.58% |
| Ventricular tachycardia | 30 | 0.17% |
| Atrial flutter | 27 | 0.16% |
| SV tachycardia | 28 | 0.16% |
| Paced rhythm | 12 | 0.07% |
| **Any out-of-scope** | **2,455** | **14.26%** |

### What the system actually did

**114 recordings in the held-out test fold document atrial fibrillation or
flutter.**

| | |
|---|---|
| Refused by quality control | **1 / 114** |
| **Carrying a statistical guarantee** | **113 / 114** |
| Guarantees that concern atrial fibrillation | **0** |
| **Reported as a normal ECG** | **2** |

> The system certifies what it can measure while being blind to the finding that
> will cause the stroke. Two patients in atrial fibrillation were told their ECG
> was normal, with four and three conformal guarantees attached.

**Every PTB-XL five-superclass paper inherits this failure.** None report it,
because the benchmark scores only the five classes it defined.

### The fix — physiology, not machine learning

Atrial fibrillation is *defined* by an irregularly irregular ventricular
response, so the R-R interval series carries the signal directly — and the
pipeline already detects R-peaks to compute heart rate, so the features cost
nothing.

| Feature | AUROC (AF/flutter vs rest, test fold) |
|---|---|
| normalised median \|ΔRR\| | **0.912** |
| coefficient of variation of RR | 0.907 |
| pNN50 | 0.892 |

Threshold fitted on the **full validation fold**, at a 5% false-positive budget
(irr = 0.179). On the unseen test fold: **48.9% sensitivity, 4.8% FPR**.

Sensitivity is capped by the false-positive budget, not by the feature — AUROC
is 0.912, so a looser budget buys more sensitivity at the cost of withholding
the guarantee more often. 5% was chosen a priori.

Critically, it does **not** diagnose atrial fibrillation — the system is not
permitted to name a class it was never trained on. It answers a narrower and more
honest question: *is this rhythm outside the region where my guarantee was
calibrated?*

### Wired into the system

Of the 114 AF/flutter records, **60 are now flagged and have the guarantee
withheld**, replaced with:

> *"STATISTICAL GUARANTEES WITHHELD: the rhythm appears to lie outside the five
> diagnostic classes this model was trained on. No conformal bound covers a class
> that is not in the label space. The findings below concern ONLY the five
> superclasses and must not be read as excluding an arrhythmia."*

The diagnosis is still reported. Only the false promise is removed.

**Cost:** at the 5% budget, 53/695 test records (7.6%) have the guarantee
withheld. 53 of the 114 AF/flutter records are still missed and still receive a
guarantee — the fix is partial, and the report says so.

Reproduce: `python -X utf8 audit/13_out_of_scope.py`

### The unification — three axes, one principle

| Marginal validity holds | Conditional validity fails |
|---|---|
| across the whole population | within a patient subgroup (age) |
| across all recordings | within a mis-acquired recording |
| **across the label space** | **when the disease is not in it** |

> **A conformal guarantee averaged over a benchmark says nothing about the
> patient in front of you.**

---

## 5. WHY A CARDIOLOGIST CARES

- **Conduction disturbance in the young is not benign.** Under 50 it raises
  Brugada, ARVC and inherited conduction disease — causes of sudden cardiac death
  in young adults. That is the group where a miss is least acceptable, and it is
  where the system fails worst.
- **"Normal" in the elderly is the highest-volume decision in any ED**, and it
  missed 33% there.
- **Subgroup performance is a regulatory expectation.** As of March 2025 the FDA
  had authorised 1,018 AI-enabled devices, 104 cardiovascular. A guarantee valid
  only marginally would not survive that review.

---

## 6. WHAT IS HONESTLY MINE

| Claim | Status |
|---|---|
| Conformal prediction for ECG | **Not mine** — Ann Noninvasive Electrocardiol 2025, cited |
| XAI → MI subtype localisation | **Not mine** — Strodthoff et al. 2024, cited |
| **Conditional (subgroup) validity of ECG conformal guarantees** | **MINE.** No prior work found. |
| **Evidence that marginal validity hides a 3.3× subgroup violation** | **MINE.** Measured, n reported. |
| **Mondrian calibration for ECG, with its data cost quantified** | **MINE** (method: Vovk 2003; ECG application and cost analysis mine) |
| **Recall-first operating point certified by a conformal lower bound** | **MINE** |
| **Electrode reversal as a silent guarantee-voiding failure** | **MINE.** Prior work detects reversal and measures accuracy loss; none asks what it does to a guarantee. |
| **Out-of-scope disease receiving a conformal guarantee** | **MINE.** 113/114 AF recordings carried a guarantee about classes that were not their disease. Not previously reported for PTB-XL. |
| **Rhythm-scope screening that withholds the guarantee** | **MINE** (R-R irregularity is textbook; using it as a *guarantee-applicability* gate is not) |
| Detecting lead reversal at all | **Not mine** — ML detectors exist (systematic review 2020), cited |
| Accuracy | **Within the published band.** Not claimed as a contribution. |

Stating the prior art out loud is deliberate. It shows the gap was found by
reading the literature, not by not reading it.

---

## 7. THE SYSTEM (what was engineered)

```
quality gate → preprocess → classify → calibrate → conformal triage
     │                                                    │
 REFUSED if                                               ▼
 uninterpretable                        XAI → grounded report → verify
```

Built after a defect-by-defect audit of the Progress-1 system, which found 12
defects. All are closed and **26/26 regression checks pass**
(`audit/08_verify_fixes.py`).

| Was | Now |
|---|---|
| Flatline signal → "MI 0.691" | **REFUSED** — "every lead is flat" |
| Microvolt file → "STTC 100%, HYP 100%" | detected, rescaled, correct |
| 5.8% of reports said NORM *and* an abnormality | **0** |
| Report generator invented "atrial fibrillation" in 42 records | removed; verifier blocks added/dropped findings |
| 3 of 4 concurrent XAI calls corrupted | **0** — per-model lock |
| HYP predicted at 4.14× its true rate | **1.01×** (ECE 0.183 → 0.018) |
| Multi-modal model leaked the cardiologist's report | withdrawn; text alone scored 0.887 AUROC with **no ECG** |

Delivered as a self-contained folder: React 19 + Vite 6 + Tailwind 4 frontend,
Flask JSON API, 1.93 GB bundled dataset, model + calibrator + conformal
thresholds. Runs with two commands.

---

## 8. LIVE DEMO — three clicks

1. **An MI case (ECG 271)** → `IMMEDIATE`; MI ruled in; *"maximal in V1, V2, V5;
   a septal distribution (proximal LAD)"*. The cardiologist's own reference report
   for this record says **"old anteroseptal myocardial infarction"** — the
   localisation matched a human's anatomical description.
2. **A borderline case** → lands in **REFER**; the system declines to decide.
3. **Upload a flatline** → **REFUSED**, not diagnosed. This is the audit finding,
   fixed, live.
4. **Swap two electrodes** → the signal still looks perfect, quality index stays
   1.00, but the system flags *"POSSIBLE RA/LA ELECTRODE REVERSAL"* and
   **withdraws its statistical guarantees for that record**.

   Then show the one it *misses*: record 9 under RA/LA reversal is not detected,
   and the triage silently flips **ROUTINE → IMMEDIATE**. That single screen is
   the whole second contribution.
5. **Open ECG 318** (documented atrial fibrillation) → the banner shows
   *"rhythm outside model scope"* and the report replaces its guarantees with
   *"no conformal bound covers a class that is not in the label space — an
   arrhythmia has NOT been excluded."*

   Then open **ECG 15796**, also atrial fibrillation. The detector misses it and
   the system reports a normal ECG with four guarantees attached. Show both. The
   second slide is the honest one.

---

## 9. QUESTIONS THEY WILL ASK

**"Is 33% of 66 patients just noise?"** — *this is the question they will ask*
No, and it was tested three independent ways. The Wilson 95% interval is
[23.2%, 45.3%], entirely above the promised 10%. The exact one-sided binomial
test gives p = 2.2×10⁻⁷, still 5.1×10⁻⁶ after **Holm correction for all 23 cells
tested**. And a 2,000-draw bootstrap that resamples the calibration fold and
refits the threshold each time violates the bound in **100% of draws** — so it is
not an artefact of one calibration split. Seven other apparent excesses did *not*
survive correction and are reported as noise.

**"Why not just beat the accuracy benchmark?"**
PTB-XL 5-superclass has been 0.92–0.94 macro-AUROC for six years. We are at
0.9343 — inside the band. Results above 0.94 use foundation models pretrained on
100k+ ECGs. Accuracy is not where the open problem is.

**"Isn't recall-first just lowering thresholds?"**
It is choosing an operating point using a **PAC conformal lower bound on recall**,
fitted on validation with a design margin, so the sensitivity floor holds on
unseen data rather than in expectation. The cost is stated: macro precision falls
from 0.72 to 0.65.

**"Did you tune on the test set?"**
No. Thresholds, calibration and conformal bounds are all fitted on **fold 9**.
Fold 10 is scored once, for reporting. The audit found this exact malpractice in
the Progress-1 code and it was removed.

**"How do you know the report is safe?"**
Every sentence is emitted from a `Finding` object carrying its evidence, and the
finished text passes a verifier that checks bidirectional containment against the
structured findings. 200/200 reports verified; 0 contradictions.

**"Your model can't detect atrial fibrillation — so what?"**
Correct, and that is the point. It is not that the system fails to detect AF; it
is that it does not KNOW it cannot, so it issues a statistical guarantee anyway.
113 of 114 AF recordings in the test fold carried one. Two were called normal.
The fix is not to add an AF class — it is to make the system recognise when a
recording lies outside the space its guarantee was calibrated on, and withhold
the claim.

**"Isn't the rhythm check just an AF detector under another name?"**
No, and the distinction is deliberate. It never outputs "atrial fibrillation" —
the system is not permitted to name a class it was never trained on. It answers
"is this rhythm outside my calibrated region?" and withholds the guarantee. An
AF detector would make a diagnosis; this makes an admission.

**"Your electrode detector is only 70% sensitive — isn't that useless?"**
Detection is not the contribution; the *finding* is. Refusing detected reversals
restored only 1 of 9 guarantees, and I report that rather than claiming a fix.
The point is that a mis-acquired record is outside the calibration distribution,
so the honest response is to **withdraw the guarantee**, which the system now
does. A better detector improves coverage; it does not change the principle.

**"Doesn't reversal only affect a tiny fraction of ECGs?"**
Yes — 0.4-4% in the literature, and at that prevalence the *population-level*
guarantee survives. That is precisely the finding: the marginal number looks
fine while the promise made to the affected individual is void. It is the same
structure as the subgroup result, which is why the two belong together.

**"What about a different hospital?"**
Unknown, and I say so. PTB-XL is one German cohort, 1989–96. External validation
is the next study and it needs data I do not have.

---

## 10. LIMITATIONS — state these before the panel does

- **Single dataset, single centre.** No external validation.
- **The `likelihood == 100` label filter dropped 21% of PTB-XL** — the ambiguous
  cases. The task here is *easier* than the published benchmark, so our numbers
  are **not directly comparable** to Strodthoff et al. (2021). This cannot be
  reversed without `ptbxl_database.csv` + `scp_statements.csv`, which this project
  does not contain.
- **Subgroup analysis uses one trained model (seed 0).** Sampling and
  calibration-draw variability are tested rigorously (§4); variability across
  *retraining* seeds is not, and is the next run.
- **Small subgroup samples.** Cells with fewer than 15 positives were excluded
  from testing entirely. The two established violations have n = 66 and n = 103.
- **Subgroups are age and sex only** — the variables PTB-XL provides.
- **No clinician has reviewed the outputs.** Every claim is statistical.
- **Territory localisation is a lead-group heuristic**, not clinically validated.
- **The electrode detector is a rule, not a classifier**: 70% sensitivity on
  RA/LA, 61% on RA/LL, 4.5% false positives. LA/LL reversal leaves aVR unchanged
  and is essentially undetectable this way (3.5%), matching the ~22% ceiling
  reported for neural detectors.
- **Reversal effects are simulated**, using the exact linear maps of the derived
  limb leads. No physically re-recorded reversed ECGs were available.
- **The rhythm-scope check catches only IRREGULAR out-of-scope disease** (AF,
  flutter with variable block, frequent ectopy) at 48.9% sensitivity. Paced rhythms
  at a fixed rate, monomorphic ventricular tachycardia, Brugada and long-QT
  preserve rhythm regularity and **remain silent failures**. 53 of 114 AF records
  are still missed — including ECG 15796, the case reported as normal.
- **Out-of-scope labels come from the free-text reference reports**, matched by
  keyword, not from structured annotations. A report that omits the arrhythmia
  would be counted as in-scope.

---

## 11. NEXT STEPS

1. Repeat the subgroup analysis across all 3 training seeds.
2. External validation on PhysioNet Challenge 2021 (7 institutions, 3 continents).
4. Cardiologist review of a sample of generated reports.

---

## REFERENCES

1. *Conformal prediction for AMI risk on PTB-XL.* Ann Noninvasive Electrocardiol, 2025. doi:10.1111/anec.70099
2. Strodthoff N. et al. *Explaining deep learning for ECG analysis.* Comput Biol Med, 2024.
3. Angelopoulos A. et al. *Conformal Risk Control.* arXiv:2208.02814, 2023.
4. Vovk V. et al. *Mondrian confidence machine.* 2003.
5. Vovk V. *Conditional validity of inductive conformal predictors.* ACML, 2012.
6. *Pitfalls of Conformal Predictions for Medical Image Classification.* arXiv:2506.18162, 2025.
7. Wagner P. et al. *PTB-XL, a large publicly available ECG dataset.* Sci Data 7:154, 2020.
8. Strodthoff N. et al. *Deep learning for ECG analysis: benchmarks from PTB-XL.* IEEE JBHI, 2021.
9. Guo C. et al. *On calibration of modern neural networks.* ICML, 2017.
10. Adebayo J. et al. *Sanity checks for saliency maps.* NeurIPS, 2018.
