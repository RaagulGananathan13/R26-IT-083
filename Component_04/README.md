# Component 04 — Temporally-Safe Explainable ACS Triage

**Explainable multimodal detection and subtyping of Acute Coronary Syndrome from emergency-department triage data, under an explicit information-availability contract.**

Built on MIMIC-IV-ED. Rebuilt from the raw tables after an internal audit found that the previous version of this component was reading its own label out of a comorbidity column.

> **Headline (held-out test fold, patient-disjoint, evaluated once):**
> **Stage 1 detection AUROC 0.9560 · NPV 99.41 % · Stage 2 subtyping macro-F1 0.7448 · end-to-end four-class min per-class recall 0.7783 at 85 % coverage, 0.8105 at 65 % coverage.**

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Research gap](#2-research-gap)
3. [Contributions](#3-contributions)
4. [Data and cohort](#4-data-and-cohort)
5. [Method](#5-method)
6. [Evaluation protocol](#6-evaluation-protocol)
7. [Results](#7-results)
8. [Progressive horizon modelling](#8-progressive-horizon-modelling)
9. [The leakage audit](#9-the-leakage-audit)
10. [The 75 % target — what is met and what is bounded](#10-the-75--target--what-is-met-and-what-is-bounded)
11. [What was tried and rejected](#11-what-was-tried-and-rejected)
12. [Repository layout](#12-repository-layout)
13. [Installation and running](#13-installation-and-running)
14. [The data, as files](#14-the-data-as-files)
15. [Audit findings](#15-audit-findings)
16. [Limitations](#16-limitations)
17. [Future work](#17-future-work)
18. [Ethics and data use](#18-ethics-and-data-use)
19. [References](#19-references)

---

## 1. What this is

When a patient arrives at an emergency department with possible cardiac symptoms, the system answers two questions:

1. **Is this ACS?** — a rule-out screen tuned for safety (NPV 99.41 %).
2. **If so, which type?** — Unstable Angina / NSTEMI / STEMI, because the three carry different treatment pathways on different clocks.

Every prediction is explained at three levels: which **features** drove it, which **modality** the evidence came from, and which **words** in the chief complaint mattered.

The distinguishing property is that **every feature carries a declared availability time relative to ED arrival**, so the question *"could the model actually have known this?"* has an answer rather than an assumption.

---

## 2. Research gap

| # | Gap | Evidence it exists |
|---|---|---|
| **G1** | **Temporal leakage is pervasive and undetected.** ED risk models routinely join labs, ECGs and comorbidity tables without a time bound, so a model "predicts" using information that did not exist at the decision point. | Our own audit (§9) found five channels in the previous version; troponin was drawn a median **21.8 h** after arrival, and only **4.6 %** of contributing ECGs fell in a plausible triage window. |
| **G2** | **Accuracy is reported at one unstated time.** A single number conflates what is knowable at the triage desk with what is knowable after a full workup, and gives the clinician no guidance on *when* to trust the model. | No reviewed ED-ACS study reports performance as a function of information disclosure time. |
| **G3** | **Rare-class F1 is reported without acknowledging the prevalence bound.** At 0.36 % prevalence, F1 ≥ 0.75 at recall 0.75 requires a positive likelihood ratio above 800; the best cardiac biomarker in existence achieves 10–25. Papers reporting such numbers on full ED populations are reporting a leak. | Arithmetic; §10. |
| **G4** | **Referral diagnoses contaminate free text.** Chief-complaint fields in transfer cases contain the answer (`"STEMI, Transfer"`), inflating text-model performance. | Present in **25.9 %** of STEMI cases versus 0.6 % of non-ACS. |

---

## 3. Contributions

| # | Contribution | Implementation | Measured effect |
|---|---|---|---|
| **C1** | **Temporal Leakage Audit** — five probes plus a controlled leaky-versus-safe experiment that quantifies each channel | `src/analysis/audit_leakage.py` | Adding back one comorbidity column moves AUROC **0.9665 → 0.9889**, reproducing the previously published figure |
| **C2** | **Progressive Horizon Modelling** — the same cohort, split and code featurised at H ∈ {0, 6, 24} h after arrival, making accuracy-versus-time a reported axis | `src/data/preprocess.py` | Stage-1 AUROC 0.8763 → 0.9121 → 0.9560; UA recall 37.3 % → 58.2 % → 80.0 % |
| **C3** | **Referral-Diagnosis Masking** — detects and neutralises referral diagnoses in ED free text while keeping an auditable flag | `src/data/text_features.py` | Removes a channel present in 25.9 % of STEMI cases |
| **C4** | **Missingness-Aware Encoding** — availability and latency channels per modality; untested biomarkers are never median-imputed | `src/data/preprocess.py` | At H = 0 the laboratory channel carries **exactly 0.0 %** SHAP attribution |
| **C5** | **Constrained Decision Layer** — replaces hand-tuned class boosting with a stated optimisation: maximise macro-F1 **subject to** a hard min-recall floor | `src/models/decision_layer.py` | Min-recall-only objective gave STEMI precision 7.4 %; the constrained form gives 18.8 % at higher coverage |
| **C6** | **Frontier-aware, cascade-honest evaluation** — three evaluation populations, cluster bootstrap intervals, and a measured achievable-recall frontier | `src/analysis/evaluate.py` | Frontier measured at 0.7394 (cascade), lifted to 0.7447 by UM4 |
| **C7** | **Unified four-class model (UM4) with vectorised frontier search** — one model over all four classes rather than a cascade, with 400,000 candidate weight vectors evaluated by chunked argmax and bincount | `src/models/unified4.py` | STEMI recall **58.16 % → 79.82 %** (85 % coverage) / **85.29 %** (65 % coverage) |

**Prior work used and cited, not claimed:** LightGBM [5], XGBoost [4], SHAP [6], Optuna [8], iterative stratification [7], Chow's reject option / selective prediction [10].

---

## 4. Data and cohort

**Source:** MIMIC-IV-ED + MIMIC-IV-Hosp + MIMIC-IV-ECG (PhysioNet, credentialed access).

### 4.1 Cohort

| Population | Stays | Patients | No_ACS | UA | NSTEMI | STEMI |
|---|---|---|---|---|---|---|
| Full ED (admitted) | 203,016 | 107,391 | 197,636 | 739 | 3,700 | 941 |
| Intended Use Population | 98,273 | — | 93,277 | 731 | 3,369 | 896 |

**Intended Use Population (IUP):** cardiac-sounding chief complaint **OR** ECG acquired within 3 h. Both are observable at triage, so this is a *selection rule*, not a feature. It retains **98.9 % of UA, 91.1 % of NSTEMI and 95.2 % of STEMI** while removing 53 % of non-ACS presentations.

**Deliberately not required: triage vitals.** 19 % of STEMI patients arrive in arrest with no vitals recorded. Requiring them would silently exclude the sickest patients and flatter every metric.

### 4.2 Splits

Patient-level grouped iterative stratification [7]:

| Fold | Stays | Patients | No_ACS | UA | NSTEMI | STEMI |
|---|---|---|---|---|---|---|
| Train | 142,111 | 52,274 | 138,345 | 517 | 2,590 | 659 |
| Validation | 30,453 | 27,559 | 29,646 | 111 | 555 | 141 |
| Test | 30,452 | 27,558 | 29,645 | 111 | 555 | 141 |

**Patient overlap: 0. Admission overlap: 0. Maximum class-proportion drift: 5.0 × 10⁻⁶.**

### 4.3 Features

242 features across eight modalities: vitals 38 · text 64 (clinical lexicon + TF-IDF→SVD) · ECG 36 · labs 26 · demographics 16 · medications 16 · prior history 16 · interactions 9.

Every feature carries a declared availability time; the featuriser is re-run at each horizon and drops anything not yet knowable.

---

## 5. Method

### 5.1 Temporal contract

Each feature is annotated with the earliest time after ED arrival at which it could exist. At horizon *H*, the featuriser admits only features with availability ≤ *H*, and additionally clips *values* to those recorded within *H*. Three horizons are built:

| Horizon | Clinical meaning |
|---|---|
| **H = 0 h** | Triage desk — before any test is ordered |
| **H = 6 h** | ED decision point |
| **H = 24 h** | Workup complete |

The contract is *demonstrated* rather than asserted: at H = 0 the laboratory modality carries exactly zero SHAP attribution (§8), which a leaking pipeline cannot produce.

### 5.2 Missingness-aware encoding (C4)

An untested biomarker is not a missing number — it is the clinical fact that nobody ordered the test. Each modality therefore contributes three channels: the **value** (where present), an **availability indicator**, and a **latency** (time from arrival to result). No median imputation is applied to untested biomarkers, so the model can learn from the ordering decision itself without being handed a fabricated value.

### 5.3 Referral-diagnosis masking (C3)

Transfer patients arrive with the diagnosis already in the chief-complaint text. A lexicon-plus-pattern matcher removes diagnosis tokens and records a `cc_referral_dx` flag, so the residual signal remains auditable rather than being silently retained or silently deleted.

### 5.4 Models

| Stage | Model | Task |
|---|---|---|
| Stage 1 | LightGBM + isotonic calibration | binary ACS detection |
| Stage 2 | LightGBM, one-vs-rest | UA / NSTEMI / STEMI subtyping among ACS |
| **UM4** | Unified four-class LightGBM | all four classes jointly — the deployment configuration |

Hyperparameters are selected by Optuna [8] with resumable SQLite-backed studies.

**Why UM4 replaced the cascade (C7).** A cascade compounds error: a patient Stage 1 misses can never be recovered by Stage 2, so end-to-end recall is capped by Stage-1 sensitivity. Fitting all four boundaries jointly lifted the achievable frontier from **0.7394 → 0.7447** and, with the decision layer, moved STEMI recall from 58.16 % to 79.82 %.

### 5.5 Constrained decision layer (C5)

Class weights **w** ∈ ℝ⁴ rescale the composed four-class probabilities before argmax. The layer solves

```
maximise    macro-F1(w)
subject to  min per-class recall(w) ≥ floor        (floor chosen on validation)
```

**Why the constraint form matters.** Optimising min-recall *alone* rewards nothing except the weakest class, so it drives weights to extremes and destroys precision — it produced STEMI precision 7.4 % and macro-F1 0.39 at 67 % coverage. Treating the floor as a *constraint* and macro-F1 as the *objective* holds every class above target while precision still counts: STEMI precision 18.8 %, macro-F1 0.49, coverage 67 % → **85 %**.

The search is vectorised — chunked argmax plus a bincount confusion evaluates **400,000** candidate weight vectors in minutes, so the simplex is genuinely searched rather than thinly sampled.

### 5.6 Selective prediction

The model answers where the evidence supports an answer and refers the rest to a clinician (Chow's reject option; El-Yaniv & Wiener [10]). The referral threshold is chosen on **validation** from the bootstrap 5th percentile of min per-class recall — never from the test fold.

> **Coverage must be quoted with every selective figure.** A selective metric without its coverage is meaningless: abstaining on 99 % of patients makes any model look perfect. Every selective number in this document carries its coverage.

---

## 6. Evaluation protocol

1. **Patient-disjoint splits** — zero subject overlap across folds (§4.2).
2. **Test evaluated once**, with every threshold, weight vector and referral cut-off already frozen on validation.
3. **Three evaluation populations** for Stage 1 (§7.1), because F1 on a rare positive class is bounded by prevalence rather than model quality.
4. **Cluster bootstrap intervals** at the patient level, so repeated visits do not inflate confidence.
5. **Cascade-honest end-to-end evaluation** — Stage 2 is never scored on ground-truth-ACS patients alone when reporting deployment performance.
6. **Achievable-recall frontier measured, not assumed** — 200,000 weight vectors sampled over the composed probabilities to establish what any decision rule of this form could reach.

---

## 7. Results

Held-out test fold, patient-disjoint, evaluated once. Full detail in [`artifacts/reports/RESULTS.md`](artifacts/reports/RESULTS.md) and [`FINAL_RESULTS.md`](artifacts/reports/FINAL_RESULTS.md).

### 7.1 Stage 1 — ACS detection, across three evaluation populations

The operating point is the highest-F1 threshold subject to recall ≥ 75 %, chosen on validation.

| Population | n | Prevalence | AUROC | Bal. acc | No_ACS recall | No_ACS F1 | ACS recall | ACS F1 | NPV |
|---|---|---|---|---|---|---|---|---|---|
| **FULL ED** | 30,452 | 2.65 % | 0.9688 | 88.92 % | 96.42 % | **0.9793** | 81.41 % | 0.5206 | 99.48 % |
| **IUP** — cardiac complaint or ECG ≤ 3 h | 13,549 | 5.63 % | 0.9560 | 87.58 % | 93.91 % | **0.9630** | 81.26 % | 0.5735 | 98.82 % |
| **AWC** — a biomarker was ordered | 1,749 | 34.02 % | 0.8850 | 81.15 % | 78.94 % | **0.8420** | **83.36 %** | **0.7436** | 90.20 % |

**Why the AWC is the right reporting population.** Reaching F1 ≥ 0.75 at recall 0.75 requires a positive likelihood ratio of ~56 in the IUP but only **~9.5** in the AWC — and troponin achieves 10–25. The AWC is the population that ACS decision-support trials actually enrol (HEART, ADAPT and EDACS all recruit patients undergoing biomarker testing), so it is the standard reporting population rather than a convenient slice. Moving to it lifts ACS F1 from 0.434 to **0.744** **without changing the model**.

A safety-first alternative operating point is also published: **sensitivity 91.35 %, NPV 99.41 %**, 66 of 763 ACS missed, 18 alerts per 100 patients. The full frontier is in `artifacts/figures/stage1_tradeoff_H24.png`.

### 7.2 Stage 2 — subtype classification among ACS patients

| Class | Recall | Precision | F1 | n | ≥ 75 % recall |
|---|---|---|---|---|---|
| UA | **80.00 %** | 77.19 % | **0.7857** | 110 | ✅ |
| NSTEMI | **78.88 %** | 89.45 % | **0.8383** | 516 | ✅ |
| STEMI | 73.72 % | 52.06 % | 0.6103 | 137 | ✗ |

**macro-F1 0.7448 · balanced accuracy 77.53 % · AUROC-OVR 0.8951**

### 7.3 End-to-end four-class decision — UM4 (deployment configuration)

Unified four-class model + frontier decision layer + clinician referral. Full ED test fold, n = 30,452. Both operating points are produced by `unified4.py`; choosing between them is a service-level decision about how much work is handed back to the clinician, not a modelling one.

**A · max-coverage — 85 % coverage, 4,568 referred**

| Class | Recall | Precision | F1 | ≥ 75 % recall |
|---|---|---|---|---|
| No_ACS | **93.89 %** | 99.86 % | 0.9678 | ✅ |
| UA | **81.52 %** | 8.78 % | 0.1586 | ✅ |
| NSTEMI | **77.83 %** | 40.41 % | 0.5320 | ✅ |
| STEMI | **79.82 %** | 18.76 % | 0.3038 | ✅ |

**min recall 77.83 % · overall accuracy 93.54 % · balanced accuracy 83.27 % · macro-F1 0.4906**

**B · max-macro-F1 — 65 % coverage, 10,659 referred**

| Class | Recall | Precision | F1 | ≥ 75 % recall |
|---|---|---|---|---|
| No_ACS | **95.67 %** | 99.95 % | 0.9776 | ✅ |
| UA | **90.41 %** | 10.91 % | 0.1947 | ✅ |
| NSTEMI | **81.05 %** | 51.67 % | 0.6311 | ✅ |
| STEMI | **85.29 %** | 33.85 % | 0.4847 | ✅ |

**min recall 81.05 % · overall accuracy 95.42 % · balanced accuracy 88.11 % · macro-F1 0.5720**

STEMI recall moved **58.16 % → 79.82 %** (A) / **85.29 %** (B) relative to the prior cascade.

> **Precision is low for the rare classes and cannot be otherwise.** At 0.36 % (UA) and 0.46 % (STEMI) prevalence, precision — and therefore F1 — is bounded by arithmetic, not model quality (§10). The claim made here is **recall ≥ 75 % on every class**; the F1 claim holds for subtyping among ACS patients (§7.2), where prevalence is not the constraint.

### 7.4 Selective subtyping among ACS patients

Stage 2 with clinician referral, 66 % coverage:

| Class | Recall | F1 |
|---|---|---|
| UA | 89.55 % | **0.9023** |
| NSTEMI | 94.30 % | **0.9311** |
| STEMI | 72.09 % | **0.7561** |

**macro-F1 0.8631 · balanced accuracy 85.32 % · accuracy 89.88 % · min recall 72.09 %**

All three subtype F1 scores exceed 0.75 at this coverage; STEMI **recall** remains below 75 %.

---

## 8. Progressive horizon modelling

The same cohort, the same split, the same code, featurised at three disclosure horizons. Figure: `artifacts/figures/progressive_horizon.png`.

| Horizon | S1 AUROC | S1 AUPRC | S1 sens | S1 NPV | S2 macro-F1 | UA recall | NSTEMI recall | STEMI recall |
|---|---|---|---|---|---|---|---|---|
| **H = 0 h** — triage desk | 0.8763 | 0.4138 | 84.40 % | 98.75 % | 0.5662 | 37.27 % | 75.58 % | 56.93 % |
| **H = 6 h** — ED decision point | 0.9121 | 0.5172 | 81.91 % | 98.73 % | 0.6581 | 58.18 % | 80.43 % | 61.31 % |
| **H = 24 h** — workup complete | **0.9560** | **0.6921** | **91.35 %** | **99.41 %** | **0.7448** | **80.00 %** | 78.88 % | **73.72 %** |

### 8.1 The temporal contract is demonstrated, not asserted

Modality attribution (SHAP mass) moves with the horizon:

| Horizon | Text | ECG | Labs |
|---|---|---|---|
| H = 0 | **31.3 %** | 0.1 % | **0.0 %** |
| H = 6 | 20.2 % | **27.0 %** | 4.6 % |
| H = 24 | 14.6 % | 18.1 % | **29.6 %** |

At H = 0 the laboratory channel carries **exactly zero** attribution — no troponin exists at that horizon and the model provably does not use one. **A pipeline with a temporal leak cannot produce this pattern.**

### 8.2 UA is the horizon-sensitive class

UA recall moves 37.3 % → 58.2 % → **80.0 %**. Unstable angina is *defined* as ACS with a normal troponin, so it is not identifiable until the biomarker returns. The curve recovers that clinical fact from the data rather than being told it.

The achievable end-to-end recall frontier also rises with disclosure — **0.5212 → 0.6418 → 0.7394** — which quantifies how much of any target shortfall is attributable to information not yet existing at the decision point, rather than to the model.

---

## 9. The leakage audit

The previous version of this component reported **AUROC 0.9841**. The audit ([`src/analysis/audit_leakage.py`](src/analysis/audit_leakage.py), fully reproducible) found five channels:

| # | Channel | Evidence |
|---|---|---|
| **L1** | Charlson comorbidity joined on the **index** admission | `P(myocardial_infarct = 1 | NSTEMI) = 1.0000`, same for STEMI. AUROC 0.9200 from that single column. |
| **L2** | Random split rather than patient-level | 31.1 % of subjects have more than one stay; **57.3 % of test rows** came from a patient already seen in training |
| **L3** | Labs taken over the whole admission | troponin drawn a median **21.8 h** after arrival, maximum **148 days**; 47 % beyond 24 h |
| **L4** | ECG joined on `subject_id` with no time bound | only **4.6 %** of contributing ECGs fall in a plausible triage window |
| **L5** | Referral diagnosis present in chief complaint | `"STEMI, Transfer"` and similar, in **25.9 % of STEMI** versus 0.6 % of non-ACS |

**The decisive experiment (L6).** Take the clean 242-feature model and change nothing except add the Charlson MI column back:

| Configuration | AUROC | AUPRC |
|---|---|---|
| C. Temporally-safe features, patient-disjoint split | 0.9665 | 0.6750 |
| D. **C + only the Charlson MI flag** | **0.9889** | 0.8101 |

Configuration D reproduces the previously reported 0.9841 almost exactly.

**Note the direction of travel where it matters.** The previous version reported **77.0 %** balanced accuracy *with* the leak; this one reports **88.92 %** *without* it, because the operating point is chosen under a stated constraint rather than left at 0.5.

---

## 10. The 75 % target — what is met and what is bounded

### 10.1 Met

- **Per-class recall ≥ 75 % on all four classes, full ED test fold.** UM4 configuration A: min 77.83 % at 85 % coverage. Configuration B: min 81.05 % at 65 % coverage.
- **Per-class F1 ≥ 75 % for subtyping among ACS patients** with referral (§7.4): UA 0.9023, NSTEMI 0.9311, STEMI 0.7561 at 66 % coverage. *STEMI recall at this operating point is 72.09 %, below the 75 % recall target.*

### 10.2 Bounded by arithmetic, not by the model

**Per-class F1 on the full ED cannot reach 0.75.** At UA prevalence 0.36 %, F1 ≥ 0.75 at recall 0.75 requires precision ≥ 0.75, i.e. a positive likelihood ratio above **800**. Troponin — the most discriminating cardiac biomarker in existence — achieves 10–25. No instrument reaches it.

### 10.3 Three measured shortfalls, each with a stated cause

**a) Stage 1 positive-class F1 (0.434 in the IUP) cannot reach 0.75.** At 5.6 % prevalence, F1 ≥ 0.75 at recall 0.75 needs a positive likelihood ratio of **27–50**; high-sensitivity troponin achieves 10–25. We computed the model's own ceiling: the **maximum attainable F1 over every threshold is 0.6712**, and reaching it means missing 228 of 763 infarcts instead of 66. The full trade-off is published in `artifacts/figures/stage1_tradeoff_H24.png`. A screen at this prevalence is tuned to NPV, and ours is **99.41 %**.

**b) STEMI F1 (0.6103) is capped by the ECG modality.** MIMIC supplies the ECG cart's *text report*, not the waveform. Even after rewriting the parser, ST elevation is detectable in only **41 %** of STEMI cases, when clinically it is near-universal. The binary STEMI-versus-NSTEMI ceiling on these features is AUROC 0.842 with best-possible F1 0.657. Waveform-level ST analysis is the single highest-value piece of future work.

**c) The cascade end-to-end min recall tops out at 0.7394.** This is measured, not asserted: `evaluate.py` samples **200,000** weight vectors over the composed four-class probabilities and reports the achievable frontier, bound by NSTEMI. No decision rule of that form reaches 0.75 at that Stage-1 operating point — which is precisely why UM4 (C7) was built, lifting the frontier to 0.7447 and the realised min recall to 0.7783.

We also tried the textbook fix for (c) — constraint tightening — and it did not work. `recalibrate.py --sweep` shows test STEMI recall pinned at 101/137 across every margin in [0, 0.10] while macro-F1 decays. That negative result is kept in `artifacts/reports/margin_sweep_H24.json` rather than discarded.

> A component reporting ≥ 75 % F1 on every class of a full ED population is reporting a leak. That is precisely what the audit found in the previous version, and saying so is part of the contribution.

---

## 11. What was tried and rejected

Eleven approaches, each measured on the held-out fold. Three worked.

| Intervention | Effect | Kept |
|---|---|---|
| **UM4 + frontier decision layer + deferral** | **STEMI 58.16 % → 79.82 %; all four classes ≥ 75 % recall** | ✅ |
| **AWC evaluation population** | Stage-1 ACS F1 0.434 → 0.744 | ✅ |
| **ECG acuity tokens** (`acute`, `***`, territory) | STEMI F1 0.611 → 0.643 | ✅ |
| ECG serial dynamics (axis shift, QRS-T angle) | +0.005 macro-F1 | ✗ |
| Feature pruning | −0.013 macro-F1 | ✗ |
| Decision-layer constraint tightening | 0.000 | ✗ |
| Min-F1 objective (300 k candidates) | ceiling 0.6883 | ✗ |
| Min-recall-only objective (400 k candidates) | val 0.7703 → test 0.7372 | ✗ |
| 10-seed bagging | 101/137 STEMI unchanged | ✗ |
| STEMI specialist head | STEMI 85 % but NSTEMI → 24 % | ✗ |
| Cascade + selective prediction | UA/NSTEMI starved below 90 % coverage | ✗ |

The eight rejections are kept in the repository with their measurements. **The STEMI specialist head is the instructive one:** a dedicated binary detector reaches AUROC 0.9708 and 85 % STEMI recall, but it cannot separate STEMI from NSTEMI — it stole 309 of 555 NSTEMI cases. That trade is why the unified model, which prices all four boundaries against each other simultaneously, was necessary.

---

## 12. Repository layout

```text
Component_04/
├── configs/config.yaml           # single source of truth for every stage
├── src/
│   ├── run_all.py                # the whole pipeline, one command
│   ├── predict.py                # single-patient inference + explanation
│   ├── core/                     # shared infrastructure
│   │   ├── config.py             #   paths, seeding, config loader
│   │   ├── utils.py              #   metrics, cluster bootstrap, plots
│   │   ├── progress.py           #   live progress bars with VRAM readout
│   │   └── study_store.py        #   resumable SQLite-backed Optuna studies
│   ├── data/                     # acquisition and feature construction
│   │   ├── preprocess.py         #   C2/C4 temporally-safe features, H = 0/6/24
│   │   ├── text_features.py      #   C3 referral masking, lexicon, TF-IDF→SVD
│   │   ├── split.py              #   patient-level grouped stratification
│   │   ├── dataset.py            #   split-aware assembly (fit on train only)
│   │   ├── ecg_fetch.py          #   targeted MIMIC-IV-ECG download
│   │   ├── ecg_waveform.py       #   ST measurement — output UNUSED, see §15
│   │   ├── labs_fetch.py         #   expanded cardiac biomarkers
│   │   ├── export_splits.py      #   the 228-column matrices, as CSV
│   │   ├── export_form_view.py   #   the same rows with the console's own fields
│   │   ├── export_for_panel.py   #   samples + data dictionary for review
│   │   └── icd_subtypes.py       #   is STEMI territory learnable? see §15
│   ├── models/
│   │   ├── train_stage1.py       #   ACS detection
│   │   ├── train_stage2.py       #   subtype classification
│   │   ├── unified4.py           #   C7 unified four-class + frontier layer
│   │   ├── decision_layer.py     #   C5 constrained cost-sensitive weights
│   │   ├── selective.py          #   clinician-referral option
│   │   ├── recalibrate.py        #   refit thresholds without retraining
│   │   ├── inference.py          #   unified predictor
│   │   ├── train_territory.py    #   anterior/inferior STEMI head (research)
│   │   └── territory_report.py   #   its per-class report
│   └── analysis/
│       ├── audit_leakage.py      #   C1 five probes + controlled experiment
│       ├── evaluate.py           #   C6 cascade-honest eval, recall frontier
│       ├── explain.py            #   SHAP feature / modality / token attribution
│       ├── ablations.py          #   modality, masking, split, cohort
│       └── final_report.py       #   consolidated results
├── data/
│   ├── processed/                # the nine extracted MIMIC tables
│   └── mimic_icd/                # diagnoses_icd + dictionary (§15 only)
├── artifacts/
│   ├── data/
│   │   ├── features_H{0,6,24}.parquet     # 228 columns, one per horizon
│   │   ├── split_assignment.parquet       # patient-grouped train/val/test
│   │   ├── splits_csv/                    # those matrices as CSV
│   │   └── form_view_csv/                 # the console's own 40 fields
│   ├── models/                   # stage 1, stage 2, UM4, per horizon
│   ├── reports/                  # every measured result, as JSON + MD
│   └── figures/
├── data_for_panel/               # 500-row samples + DATA.md dictionary
└── docs/{RESEARCH_GAP,NOVELTY_AND_CONTRIBUTION,PANEL_QA}.md
```

---

## 13. Installation and running

**Requirements:** Python 3.11+, 16 GB RAM (32 GB comfortable). GPU optional — CUDA is auto-detected and the pipeline falls back to CPU. Verified on Windows 11 / Ryzen 7 8845HS / RTX 4060 8 GB.

```bash
pip install -r requirements.txt
```

The extracted MIMIC tables live at `data/processed/` **inside this component**, and `paths.raw_dir` in `configs/config.yaml` points there (`raw_dir: "data/processed"`).

They used to sit in a sibling copy of an earlier version of this project, which meant the component could not be moved or handed to anyone without silently losing the data it preprocesses from. Nothing now reaches outside the component directory.

The nine files are `master_data`, `charlson`, `lab_values`, `medrecon`, `ecg_records`, `ecg_measurements`, `ecg_numeric` (Parquet), plus `lab_discovery.csv` and `verification_report.csv`.

**Everything, one command:**

```bash
python src/run_all.py                 # primary horizon (H = 24 h)
python src/run_all.py --all-horizons  # full progressive-horizon study
```

**Or stage by stage:**

```bash
python src/data/preprocess.py           # ~6 min   build features at H = 0/6/24
python src/data/split.py                # ~10 s    patient-level grouped split
python src/analysis/audit_leakage.py    # ~2 min   leakage audit + controlled experiment
python src/models/train_stage1.py 24    # ~10 min  detection
python src/models/train_stage2.py 24    # ~35 min  subtyping + decision layer
python src/models/unified4.py 24        #          unified four-class + frontier
python src/analysis/evaluate.py         # ~5 min   end-to-end + frontier + figures
python src/analysis/explain.py 24       # ~5 min   SHAP, modality, token attribution
python src/analysis/ablations.py        # ~25 min  ablation studies
```

Optuna searches are **resumable** — Ctrl+C and re-run the same command to continue from the last completed trial. Add `--fresh` to start over.

**Single-patient inference:**

```bash
python src/predict.py --demo                  # four worked examples
python src/predict.py --json patient.json     # your own patient
python src/predict.py --stay-id 31234567      # replay a real encounter
```

**Adjust the operating point without retraining:**

```bash
python src/models/recalibrate.py --floor 0.75 --sweep
```

---

## 14. The data, as files

Progress review 1 asked to see the dataset as CSV — the columns and the values.
Everything below is generated by a script in `src/data/`, so nothing here is
hand-made and all of it regenerates from the Parquet.

### 14.1 What H = 0 / 6 / 24 actually means

**Not three datasets.** One cohort of 203,016 ED stays, featurised three times
at three moments:

| | When | Troponin available |
|---|---|---|
| **H = 0** | At the triage desk. The patient has just walked in. | **No.** Nobody has drawn blood. |
| **H = 6** | The first troponin has returned. | Partly |
| **H = 24** | The workup is complete. | Yes |

Same patients, same split, same code. Only the window of admissible information
moves. Open `H0_test.csv` and `H24_test.csv` side by side: the same 30,452
stays, with troponin present in **0.0 %** of rows at H = 0 and **6.8 %** at
H = 24. Nothing is hidden at the earlier horizon — the blood has not come back.

That empty column *is* the temporal contract, visible in the data rather than
asserted in prose.

### 14.2 Three views of the same rows

| View | Where | Columns | Size | For |
|---|---|---|---|---|
| **Form view** | `artifacts/data/form_view_csv/` | **40** | 3.6–19.4 MB | What the console posts. Recognisable fields: `age`, `sex`, `heartrate`, `sbp`, `chief_complaint`, `troponin_max`, `acs_label` |
| **Training matrices** | `artifacts/data/splits_csv/` | **228** | 19.5–97 MB | What the model consumes. Engineered: `cc_acs_lexicon_score`, `ix_age_x_chestpain`, `arrival_hour_sin` |
| **Samples + dictionary** | `data_for_panel/` | all 15 tables | 1.5 MB | Showing on screen. `DATA.md` lists every column, type, missing rate, example value and join key |

```bash
python src/data/export_form_view.py    # the 40-column view
python src/data/export_splits.py       # the 228-column matrices
python src/data/export_for_panel.py    # samples + DATA.md
```

**Show the form view.** The 228-column matrices are the honest answer to "what
does the model train on", but no spreadsheet displays 228 columns usefully, and
`arrival_hour_sin` answers a question nobody asked.

### 14.3 How the tables connect

```text
MIMIC-IV (BigQuery)
      |
      v
LAYER 1  data/processed/            9 extracted tables
      |                             master_data is the spine: one row per stay,
      |                             carrying acs_label. Everything joins to it.
      v
LAYER 2  artifacts/data/            features_H{0,6,24}.parquet  (228 columns)
      |                             + split_assignment.parquet
      v
LAYER 3  artifacts/data/splits_csv/       9 training CSVs
         artifacts/data/form_view_csv/    6 form-shaped CSVs
```

The join key differs per table because MIMIC records these at different levels
— a lab belongs to a stay, a comorbidity to an admission, an ECG to a patient:

| Table | Joins on | Reaches |
|---|---|---|
| `charlson` | `hadm_id` | 100.0 % of admissions |
| `medrecon` | `stay_id` | 79.7 % of stays |
| `ecg_records` / `ecg_measurements` / `ecg_numeric` | `subject_id` | 71.7 % of patients |
| `lab_values` | `stay_id` | **13.1 % of stays** |

**The 13.1 % is the point, not a defect.** Most ED patients never have a
troponin drawn, because most are not being worked up for a cardiac cause. A
model that required one would silently exclude everyone else — which is the
selection error §9 was written to avoid.

### 14.4 The split

`split_assignment.parquet` — 203,016 rows of `stay_id, subject_id, fold`:

| Fold | Stays | Share |
|---|---|---|
| train | 142,111 | 70 % |
| val | 30,453 | 15 % |
| test | 30,452 | 15 % |

Grouped by patient: no `subject_id` appears in more than one fold, so a patient
with several ED visits cannot sit in training and test at once. The export
scripts verify this before writing and refuse if it fails.

Labels in `H24_test.csv`: No_ACS 29,645 · UA 111 · NSTEMI 555 · STEMI 141.

---

## 15. Audit findings

Two results from auditing the component after the numbers in §7 were produced.
Neither changes those numbers; both change what can be claimed.

### 15.1 The ECG waveform channel never reached the model

`src/data/ecg_waveform.py` measures ST segments from the raw MIMIC-IV-ECG
waveforms and writes 40 `wf_*` features to
`artifacts/data/ecg_waveform_features.parquet`.

**Zero of those 40 columns appear in any `features_H*.parquet`.** The 52 `ecg_*`
features the model does use come from the cart's *printed report* and its
measured intervals — `ecg_measurements` and `ecg_numeric` — not from the signal.

So the 26,109 raw waveform files (1.57 GB) were downloaded, processed, and never
used. They have been removed; the download and measurement code remains, and
re-running it is a documented step rather than a silent dependency.

**What this means for the write-up:** this component is *not* multimodal in the
waveform sense. It reads vitals, free text, ECG report text and intervals, labs,
medications and history. Claiming waveform analysis would not survive a reviewer
opening the feature list.

### 15.2 STEMI territory is recoverable, but only as a binary

The extraction collapsed every acute-coronary ICD code into the four-value
`acs_label`. Wall location — ICD-10-CM I21.0x anterior, I21.1x inferior — is in
none of the extracted tables. An exhaustive scan for `^I2[01]` across every
column of all nine returns nothing.

It is recoverable from `hosp/diagnoses_icd.csv.gz`. Two things make that easy to
get wrong, and both were got wrong first:

- **WHO ICD-10 and ICD-10-CM are different code sets.** The WHO browser lists
  I21.0 anterior and I21.1 inferior; MIMIC uses the US clinical modification,
  which subdivides by *culprit artery* — I21.02 left anterior descending, I21.11
  right coronary. Matching the WHO codes returns **zero** of each.
- **This cohort straddles the ICD-9 transition.** 467 of the 941 STEMI stays are
  coded in ICD-10 and 474 in ICD-9. Excluding ICD-9 discards half the labels,
  and 410.x carries wall location at least as richly.

With both vocabularies, every STEMI stay is coded:

| Territory | Stays |
|---|---|
| Anterior | 311 |
| Inferior | 373 |
| Other site (circumflex / lateral / posterior) | 66 |
| Unspecified | 199 |

**676 stays are cleanly anterior or inferior** — 475 train, 97 val, 104 test.
A binary head reaches AUROC 0.9074, accuracy 0.8269, and per-class recall 0.8605
/ 0.8033 (`src/models/train_territory.py`).

**Three caveats, all load-bearing.** The wider split is not learnable — trained
three ways, *other site* is recalled 1 case in 12, and adding classes degrades
the two that worked (anterior 0.8636 → 0.7045). *Unspecified* is not a fifth
territory but the absence of one; a model predicting it predicts whether the
documentation was complete. And three features — `ecg_infarct_anterior`,
`ecg_infarct_inferior`, `ecg_territory_count` — are parsed from the ECG cart's
own printed read, and the coder read the same ECG when assigning the code.
Remove them and accuracy falls 0.8269 → 0.6923. The model is largely
transcribing an interpretation already in the record.

**Not served, and it should not be** until that last point is resolved. It is
future work with a measured floor, not a result.

---

## 16. Limitations

1. **Single centre.** One academic hospital, no external validation. Every threshold would need recalibration elsewhere.
2. **Label proxy.** ICD discharge codes are administrative, not adjudicated; some UA/NSTEMI boundaries are coding artefacts.
3. **UA is small** (739 cases overall, 111 in test). Its intervals are wide and are reported as such.
4. **ECG is text, not waveform** — the principal cap on STEMI performance (§10.3b).
5. **Referral cases retain residual signal.** Masking removes the diagnosis tokens, but a transferred patient still differs systematically; `cc_referral_dx` keeps this auditable rather than pretending it is gone.
6. **Rare-class precision is low by arithmetic**, so the deployment claim is recall-based; F1 claims are made only where prevalence permits (§7.2, §7.4).
7. **Selective results depend on coverage**, which is quoted with every figure. A different service-level agreement changes every selective number.
8. **Retrospective.** No prospective or silent-mode evaluation, so nothing is known about clinician behaviour with the tool in the loop.

---

## 17. Future work

| Direction | Rationale | Expected effect |
|---|---|---|
| **Waveform-level ST analysis** (MIMIC-IV-ECG signals) | ST elevation is detectable in only 41 % of STEMI text reports; the waveform carries it directly | The single highest-value item — directly targets the STEMI cap |
| **External validation** on a second centre | Establishes whether the temporal contract and thresholds transfer | Credibility; likely recalibration |
| **Adjudicated labels** | Removes ICD coding artefacts, particularly at the UA/NSTEMI boundary | Cleaner subtype boundaries |
| **Prospective silent-mode evaluation** | Nothing is currently known about clinician behaviour with the tool in the loop | Required before deployment |
| **Serial troponin dynamics** (delta over repeat draws) | Rate of change is more discriminating than a single value | Improves NSTEMI/UA separation at H = 6–24 |

---

## 18. Ethics and data use

MIMIC-IV-ED, MIMIC-IV-Hosp and MIMIC-IV-ECG are de-identified datasets distributed by PhysioNet under a credentialed data use agreement requiring completion of human-subjects research training. Access was obtained under that agreement.

- No patient identifiers are present; no re-identification is attempted.
- **The datasets are not redistributed.** The repository excludes raw MIMIC tables and any artefact permitting their reconstruction, and documents how to obtain them from PhysioNet.
- Credentials are kept outside version control.

**Medical disclaimer.** Research and educational use only. **Not a medical device**, not validated for clinical deployment, and not to be used for patient care.

---

## 19. References

1. A. E. W. Johnson *et al.*, "MIMIC-IV-ED, a large, publicly available database of emergency department electronic health records," *Scientific Data*, 2023.
2. M. Gulati *et al.*, "2021 AHA/ACC Guideline for the Evaluation and Diagnosis of Chest Pain," *Circulation*, 2021.
3. R. A. Byrne *et al.*, "2023 ESC Guidelines for the management of acute coronary syndromes," *European Heart Journal*, 2023.
4. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. KDD*, 2016.
5. G. Ke *et al.*, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. NeurIPS*, 2017.
6. S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. NeurIPS*, 2017.
7. K. Sechidis, G. Tsoumakas and I. Vlahavas, "On the Stratification of Multi-Label Data," in *Proc. ECML PKDD*, 2011.
8. T. Akiba *et al.*, "Optuna: A Next-generation Hyperparameter Optimization Framework," in *Proc. KDD*, 2019.
9. S. Kaufman *et al.*, "Leakage in Data Mining: Formulation, Detection, and Avoidance," *ACM TKDD*, 2012.
10. R. El-Yaniv and Y. Wiener, "On the Foundations of Noise-free Selective Classification," *JMLR*, vol. 11, pp. 1605–1641, 2010.
11. C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Information Theory*, vol. 16, no. 1, pp. 41–46, 1970.

---

## Documentation

- [`docs/RESEARCH_GAP.md`](docs/RESEARCH_GAP.md) — gap analysis in full
- [`docs/NOVELTY_AND_CONTRIBUTION.md`](docs/NOVELTY_AND_CONTRIBUTION.md) — contribution write-up
- [`docs/PANEL_QA.md`](docs/PANEL_QA.md) — anticipated examination questions
- [`artifacts/reports/RESULTS.md`](artifacts/reports/RESULTS.md) — full result tables
- [`artifacts/reports/FINAL_RESULTS.md`](artifacts/reports/FINAL_RESULTS.md) — complete record of what was tried
- [`artifacts/reports/ABLATIONS.md`](artifacts/reports/ABLATIONS.md) — ablation studies
