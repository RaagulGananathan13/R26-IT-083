# Component 04 — Complete System Reference

**Project:** Explainable AI-Based Acute Coronary Syndrome (ACS) Detection and Type Classification  
**Component:** 04 — Temporally-Safe Explainable ACS Triage  
**Repository:** [R26-IT-083](https://github.com/RaagulGananathan13/R26-IT-083)  
**Component owner:** Abishnan (`abis029`)

> **Research software only — not a medical device, diagnosis, or clinical triage instruction.**
> The recorded results are retrospective research findings. They are not evidence of
> prospective clinical validity. A qualified clinician must make patient-care decisions.

This document is a practical **system-level companion** to [README.md](README.md).
It explains the checked-in Component 04 architecture, source files, data flow,
experiments, results, execution commands, and known limitations. Unlike the older
`Component_4/Component_4` ZIP layout, the **current GitHub component** lives in
`Component_04/` and uses a time-aware LightGBM/XGBoost research pipeline. Do not
mix training commands or model files from those two versions.

**Evidence scope:** Source code, configuration, and reported results in the GitHub
repository as reviewed on **14 October 2026**. The numbers below are **reported in
repository documentation**; they have not been independently reproduced by running
training in this documentation change. References to files describe the repository
layout, not a claim that all large datasets/artifacts are available on every clone.

## Contents

1. [Purpose and boundaries](#1-purpose-and-boundaries)
2. [Architecture and prediction flow](#2-architecture-and-prediction-flow)
3. [Repository map](#3-repository-map)
4. [Data, cohorts, and labels](#4-data-cohorts-and-labels)
5. [Temporal and leakage-safety contract](#5-temporal-and-leakage-safety-contract)
6. [Feature-engineering pipeline](#6-feature-engineering-pipeline)
7. [Training and model specification](#7-training-and-model-specification)
8. [Decision policies and clinician referral](#8-decision-policies-and-clinician-referral)
9. [Explainability](#9-explainability)
10. [Evaluation protocol and reported results](#10-evaluation-protocol-and-reported-results)
11. [Source files by responsibility](#11-source-files-by-responsibility)
12. [Reproducing and running the project](#12-reproducing-and-running-the-project)
13. [Single-patient input and output](#13-single-patient-input-and-output)
14. [Audits, known gaps, and unresolved distinctions](#14-audits-known-gaps-and-unresolved-distinctions)
15. [Safe development and testing checklist](#15-safe-development-and-testing-checklist)
16. [Medical and data-use restrictions](#16-medical-and-data-use-restrictions)
17. [Related documentation](#17-related-documentation)

---

## 1. Purpose and boundaries

Acute Coronary Syndrome (ACS) is a group of time-sensitive coronary conditions.
This component estimates (1) whether ACS is present and (2) its subtype, given
emergency-department observations **available by an explicitly specified time**.

| Output label | Meaning |
|---|---|
| `No_ACS` | No acute coronary syndrome according to this retrospective label scheme |
| `UA` | Unstable angina |
| `NSTEMI` | Non-ST-elevation myocardial infarction |
| `STEMI` | ST-elevation myocardial infarction |

The technical focus is *temporal data availability*, rare-class evaluation,
explanation, and safe expression of uncertainty—not stand-alone diagnosis.
Clinical labels are proxies derived from administrative records; they are not
independently adjudicated diagnoses.

### Research contributions represented in the code

- **Leakage audit:** identifies admission-level comorbidity, random-split,
  unbounded laboratory/ECG timing, and referral-text leakage routes.
- **Progressive horizons:** evaluates the same cohort at 0, 6, and 24 hours
  after emergency-department arrival.
- **Referral-diagnosis masking:** excludes explicit diagnosis clues from complaint
  text and retains an audit flag.
- **Missingness-aware features:** records that a test was not available instead
  of inventing a normal laboratory value.
- **Constrained decision layer:** chooses class-weighted decisions against an
  explicit per-class recall constraint.
- **Unified four-class model (UM4):** avoids irrecoverable errors introduced by
  gating the subtype model behind a binary ACS stage.
- **Explainability:** feature-level SHAP, modality-level attribution, and
  token-level chief-complaint highlighting.

## 2. Architecture and prediction flow

```text
MIMIC-IV-ED + MIMIC-IV-Hosp + MIMIC-IV-ECG derived tables
                             |
                             v
              data/processed/ (source extracts)
                             |
                             v
           src/data/preprocess.py + text_features.py
             anchor T0 = ED arrival; choose H
                    H in {0, 6, 24}
                             |
                             v
               artifacts/data/features_H*.parquet
                             |
                  subject-level train/val/test split
                             |
              +--------------+--------------+
              |                             |
       Two-stage research path        Unified UM4 research path
              |                             |
     Stage 1: ACS / No_ACS          Four classes jointly
              |                             |
     Stage 2: UA / NSTEMI / STEMI   Cost-sensitive decision search
              |                             |
              +--------------+--------------+
                             |
               evaluation + referral policies
                             |
            SHAP + modality + clinical-token context
                             |
                JSON/Markdown reports and CLI
```

**Important runtime distinction:** `src/models/unified4.py` trains and evaluates
UM4, which the repository describes as its headline four-class configuration.
However, `src/predict.py` loads `ACSPredictor`, and
`src/models/inference.py` uses **`mode="cascade"` by default** when calling
`ACSPredictor.predict()`. Do not assume the CLI's prediction labels are UM4
predictions without a separately verified UM4 inference path. This should be
resolved or explicitly described in any system-integration demonstration.

**Application boundary:** This directory provides a Python research pipeline
and a command-line patient-inference interface. It is **not** the same as the
older ZIP's FastAPI + React demonstration. A verified HTTP API or UI integration
for the current pipeline is not established by the files described here.

## 3. Repository map

All paths in this section are relative to `Component_04/`.

```text
Component_04/
├── README.md                         # Research narrative and results
├── SYSTEM_README.md                  # This system-level technical reference
├── requirements.txt                 # Python dependencies
├── configs/config.yaml              # Central research configuration
├── src/
│   ├── run_all.py                    # Orchestrates the full pipeline
│   ├── predict.py                    # Single-patient CLI and worked examples
│   ├── core/                         # Configuration, shared utilities
│   ├── data/                         # Extraction, preprocessing, splits, text
│   ├── models/                       # Two-stage, UM4, decision/referral models
│   └── analysis/                     # Leakage, evaluation, SHAP, ablations
├── data/processed/                   # Credentialed source-derived extracts
├── artifacts/
│   ├── data/                        # Feature tables and split assignments
│   ├── models/                      # Fitted classifiers and policy artifacts
│   ├── reports/                     # Evaluation tables, JSON, and reports
│   └── figures/                     # Evaluation and attribution plots
├── data_for_panel/                  # Selected demonstrations and data dictionary
└── docs/                            # Research gaps, novelty, panel questions
```

Paths under `data/` and `artifacts/` may contain restricted or locally generated
material. Their mention here **does not authorize uploading or redistributing**
private MIMIC data, credential files, or derived patient-level records.

## 4. Data, cohorts, and labels

The repository describes a combined retrospective cohort using MIMIC-IV-ED,
MIMIC-IV-Hosp, and MIMIC-IV-ECG under PhysioNet credentialed-access rules.

### Reported cohort counts

| Population | ED stays | No_ACS | UA | NSTEMI | STEMI |
|---|---:|---:|---:|---:|---:|
| Full admitted ED cohort | 203,016 | 197,636 | 739 | 3,700 | 941 |
| Intended Use Population (IUP) | 98,273 | 93,277 | 731 | 3,369 | 896 |

The IUP is selected using a cardiac-sounding chief complaint **or** an ECG
within three hours, as described in the main README. The selection is intended
to reflect suspected-ACS presentations, not every ED visit. It is not equivalent
to filtering for known positive cases.

### Patient-disjoint split (reported)

| Partition | Stays | Purpose |
|---|---:|---|
| Training | 142,111 | Fit models and feature transformations |
| Validation | 30,453 | Tune thresholds, weights, and referral policies |
| Held-out test | 30,452 | Final evaluation only |

`src/data/split.py` is responsible for patient-level partitioning. No patient
(`subject_id`) should appear in multiple partitions; otherwise estimates are
inflated by repeated encounters from the same person.

### Source table responsibilities

| Table | Role |
|---|---|
| `master_data.parquet` | ED-stay spine and retrospective label information |
| `charlson.parquet` | Comorbidity source; index-admission leakage must be excluded |
| `lab_values.parquet` | Timed laboratory measurements |
| `medrecon.parquet` | Medication reconciliation |
| `ecg_records.parquet` | ECG record metadata and timestamps |
| `ecg_measurements.parquet` | ECG report-derived measurements |
| `ecg_numeric.parquet` | Further numeric ECG descriptors |
| `lab_discovery.csv` | Extracted laboratory-code discovery |
| `verification_report.csv` | Extraction verification information |

## 5. Temporal and leakage-safety contract

The reference time **T0** is arrival at the emergency department. A feature
used at horizon **H** must be known by `T0 + H`, not merely appear somewhere in
the same admission. `configs/config.yaml` sets:

| Setting | Checked-in value | Meaning |
|---|---|---|
| `temporal.horizons_h` | `[0, 6, 24]` | Progressive information windows |
| `temporal.primary_horizon_h` | `24` | Default research horizon |
| `temporal.ecg_lookback_h` | `1.0` | Permitted pre-arrival ECG lookback |
| `cohort.ecg_within_h` | `3.0` | IUP selection window |
| `cohort.require_triage_vitals` | `false` | Do not exclude patients missing triage vitals |
| `split.group_col` | `subject_id` | Split by patient identity, not stay |

At **H = 0**, post-arrival blood results must be absent. At **H = 6**, only
results available in the first six hours can be considered. At **H = 24**,
workup data up to 24 hours may be included. The reported SHAP lab-modality
attribution at H = 0 is 0.0%; this is an observed audit result, not proof that
all conceivable leakage has been eliminated.

### Leakage pathways previously investigated

| ID | Issue | Safeguard or investigation |
|---|---|---|
| L1 | Index-admission MI comorbidity encodes the outcome | Restrict history to genuinely earlier encounters |
| L2 | Random row split repeats a patient's history across folds | Grouped patient-level partition |
| L3 | Troponin from later in the admission | Bound results by T0 + H |
| L4 | Closest ECG joined without an arrival-time bound | Restrict ECG record timestamps |
| L5 | Chief complaint literally names the referral diagnosis | Mask diagnosis terms; retain audit indicator |

The controlled experiment reported in the repository shows AUROC rising from
**0.9665 to 0.9889** after restoring a single leaked Charlson MI feature.
The larger number is **not** a valid model improvement.

## 6. Feature-engineering pipeline

`src/data/preprocess.py` coordinates the feature construction for each horizon.
The main feature families are:

1. **Vitals:** heart rate, blood pressure, oxygen saturation, respiratory rate,
   pain/acuity, and derived measures with physiological plausibility checks.
2. **Demographics:** age, sex and time-related contextual fields.
3. **Complaint text:** cleaned clinical terms, negation-aware lexicon, and
   text-derived components built by `src/data/text_features.py`.
4. **Home medications:** medication reconciliation at an appropriate time.
5. **Prior history:** earlier events only, not the current admission outcome.
6. **ECG report/interval features:** timed report text and printed measurements.
7. **Laboratories:** troponin, natriuretic peptides, timing and availability.
8. **Interactions:** clinically motivated cross-modality features.

Where appropriate, a modality includes `*_available` flags and `*_t_first_h`
latency features. A biomarker that was not measured is not silently treated as
normal; model-specific missing-value handling retains that distinction.

**Feature-count note:** The research README describes a 242-feature engineering
inventory and exported training/form views with different column counts.
Therefore, **do not equate “242 engineered fields” with the exact number of
columns loaded by any individual fitted classifier**. The actual input vector
must be inspected in the corresponding model configuration/feature list.

## 7. Training and model specification

### Stage 1 — ACS detection

`src/models/train_stage1.py` trains a binary detection model using gradient-
boosted trees. The inference implementation (`src/models/inference.py`) loads
both XGBoost and LightGBM estimators, blends the scores according to the saved
configuration, and applies a fitted calibrator to produce `P(ACS)`.

The operating threshold is fitted on validation and recorded in the Stage 1
model configuration, rather than implicitly being 0.5.

### Stage 2 — subtype classification

`src/models/train_stage2.py` predicts **UA / NSTEMI / STEMI conditional on ACS**.
The inference implementation blends XGBoost and LightGBM subtype distributions;
a fitted constrained decision layer chooses the final subtype.

### Combined cascade

For `p = P(ACS)` and conditional subtype scores
`q = [P(UA|ACS), P(NSTEMI|ACS), P(STEMI|ACS)]`, the inference engine can form:

```text
P(No_ACS) = 1 - p
P(UA)     = p × q_UA
P(NSTEMI) = p × q_NSTEMI
P(STEMI)  = p × q_STEMI
```

In the default **cascade decision**, patients below the saved Stage 1 threshold
are labelled `No_ACS`, regardless of Stage 2. This means Stage 1 false negatives
cannot be corrected by Stage 2.

### UM4 — unified four-class research model

`src/models/unified4.py` fits all four labels together instead of using the
binary gate. It includes a vectorised class-weight search and decision policy.
This model underpins the repository's **end-to-end selective results**, but is
not automatically the predictor used by the default `src/predict.py` CLI.

### Training artifacts

The current inference loader expects files with horizon suffixes, including:

```text
artifacts/models/stage1_config_H24.json
artifacts/models/stage1_xgb_H24.json
artifacts/models/stage1_lgb_H24.joblib
artifacts/models/stage1_calibrator_H24.joblib
artifacts/models/stage2_config_H24.json
artifacts/models/stage2_xgb_H24.json
artifacts/models/stage2_lgb_H24.joblib
artifacts/models/stage2_cdl_H24.joblib
artifacts/models/text_embedder_H24.joblib
```

Some paths are created by training; do not assume a fresh checkout has them.
An optional `joint_cdl_H24.joblib` is used by the two-stage **joint** decision
mode and should not be confused with UM4 weights.

## 8. Decision policies and clinician referral

`src/models/decision_layer.py` implements the constrained cost-sensitive
layer. Its objective is to maximise macro-F1 while imposing a required minimum
per-class recall, with a documented fallback if the requested floor cannot be
met. It chooses weights using **validation**, not test labels.

`src/models/selective.py` handles referral/abstention: if evidence is not
sufficient for a confident automated classification, the system may **refer**
instead of emitting a definitive class. The fraction of all cases for which
it still issues a prediction is **coverage**.

Any selective metric must always include its coverage and evaluation population.
For example, a result at 65% coverage refers about 35% of evaluated cases and
must not be presented as performance on 100% of patients.

## 9. Explainability

`src/analysis/explain.py` contains three complementary views:

| Level | Method | Clinician-facing question |
|---|---|---|
| Feature | SHAP / TreeSHAP | Which specific model inputs affected this score? |
| Modality | Aggregate attribution across feature groups | Were the important signals laboratory, ECG report, text, vitals, etc.? |
| Text | Clinical token highlighting with negation handling | Which complaint terms and negations were recognised? |

**Interpretation limits:** An SHAP value explains the behaviour of a fitted
model—not causality, underlying pathophysiology, or medical certainty.
Complaint-token highlights from a clinical lexicon are not interchangeable with
model-faithful per-token SHAP attributions.

The currently reported ECG modality is primarily **device-generated ECG text,
measurements, and intervals**. The repository audit states that raw waveform-
derived `wf_*` columns were **not included** in trained feature matrices. Avoid
claiming that deployed predictions directly interpret 12-lead waveforms.

## 10. Evaluation protocol and reported results

The main README reports patient-disjoint evaluation, validation-frozen
thresholds/decision weights, patient-cluster bootstrapping, and evaluation of
the complete cascade rather than only ground-truth-positive subtype cases.

### Reported headline results

| Experiment / population | Reported metric | Value |
|---|---|---:|
| Stage 1, IUP H=24 | AUROC | **0.9560** |
| Stage 1, H=24 safety-first operating point | Negative predictive value | **99.41%** |
| Stage 2, ACS subset H=24 | Macro-F1 | **0.7448** |
| UM4, full ED, 85% coverage | Minimum per-class recall | **77.83%** |
| UM4, full ED, 65% coverage | Minimum per-class recall | **81.05%** |
| UM4, full ED, 85% coverage | Macro-F1 | **0.4906** |
| UM4, full ED, 65% coverage | Macro-F1 | **0.5720** |

**Do not mix metrics from different evaluation populations or operating points.**
For example, the Stage 1 IUP operating-point NPV reported elsewhere in the README
is **98.82%**, not 99.41%. The latter belongs to a separate safety-first
operating point and should not be paired with IUP precision/recall as if they
were one confusion matrix.

### Progressive horizon results (reported)

| Horizon | Stage 1 AUROC | Stage 2 macro-F1 | UA recall |
|---|---:|---:|---:|
| H=0 (arrival) | 0.8763 | 0.5662 | 37.27% |
| H=6 (ED workup) | 0.9121 | 0.6581 | 58.18% |
| H=24 (later workup) | 0.9560 | 0.7448 | 80.00% |

Detailed values, thresholds, confidence bounds, and experiment provenance are
recorded in the main [README.md](README.md) and
`artifacts/reports/{RESULTS,FINAL_RESULTS,ABLATIONS}.md` where available.

### What these numbers do *not* mean

- Overall accuracy on a highly imbalanced ED cohort is not a substitute for
  sensitivity, precision, NPV, or class-wise recall.
- High NPV at low ACS prevalence does not mean all suspected ACS is ruled out.
- Selective min-recall above 75% does **not** imply subtype precision or F1
  exceeds 75% on the full ED population.
- Results are retrospective internal test-set figures, **not** clinical validation.

## 11. Source files by responsibility

| Path | Main responsibility |
|---|---|
| `configs/config.yaml` | Paths, cohorts, horizons, split, training settings |
| `src/core/config.py` | Loads configuration and resolves component-relative paths |
| `src/run_all.py` | Runs pipeline stages; offers skip/from/force controls |
| `src/predict.py` | Constructs one patient row, loads models, prints probabilities and notes |
| `src/data/preprocess.py` | Builds horizon-safe modality features |
| `src/data/text_features.py` | Normalisation, referral diagnosis masking, text features |
| `src/data/split.py` | Grouped train/validation/test assignment |
| `src/data/export_form_view.py` | Human-readable feature/label export |
| `src/data/export_splits.py` | Training-table CSV export |
| `src/data/export_for_panel.py` | Demonstration samples and data dictionary |
| `src/models/train_stage1.py` | Stage 1 detection fitting and evaluation |
| `src/models/train_stage2.py` | Conditional ACS-subtype fitting |
| `src/models/inference.py` | Loads fitted artifacts; cascade and joint prediction APIs |
| `src/models/decision_layer.py` | Constrained probability weighting |
| `src/models/unified4.py` | Unified four-class experiments and search |
| `src/models/selective.py` | Abstention/referral experiments |
| `src/models/recalibrate.py` | Refit operating points without retraining core models |
| `src/models/train_territory.py` | Exploratory STEMI territory classifier (not served) |
| `src/analysis/audit_leakage.py` | Leakage audits and controlled comparisons |
| `src/analysis/evaluate.py` | End-to-end metrics and recall frontier |
| `src/analysis/explain.py` | SHAP, modality, and token explanations |
| `src/analysis/ablations.py` | Controlled ablation studies |
| `src/analysis/final_report.py` | Consolidated experiment reporting |

Consult source files directly for exact function signatures and CLI flags;
this table is an architectural index, not an API specification.

## 12. Reproducing and running the project

### Prerequisites

- Python **3.11+**, with a local virtual environment recommended.
- Sufficient memory and disk for MIMIC extracts; GPU is optional in the
  documented research setup.
- Proper authorized access to any MIMIC-IV datasets used.
- Generated feature matrices and fitted model artifacts for inference.

From the **`Component_04/`** directory:

```bash
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Linux/macOS alternative:
# python3 -m venv .venv
# source .venv/bin/activate
# python -m pip install -r requirements.txt
```

### Run stages

```bash
python src/run_all.py                  # Default primary-horizon pipeline
python src/run_all.py --all-horizons   # Horizon comparison study
python src/run_all.py --skip audit,ablations
python src/run_all.py --from train_stage1
```

Individual commands (after data preparation, with needed artifacts present):

```bash
python src/data/preprocess.py
python src/data/split.py
python src/analysis/audit_leakage.py
python src/models/train_stage1.py 24
python src/models/train_stage2.py 24
python src/models/unified4.py 24
python src/analysis/evaluate.py
python src/analysis/explain.py 24
```

**Caution:** `src/run_all.py` will train and write artifacts. It is **not** a
read-only health check. Do not execute full training or export patient-level
files simply to verify this documentation.

### Demo or new patient from JSON

```bash
python src/predict.py --demo
python src/predict.py --json patient.json
python src/predict.py --stay-id 31234567
```

A `--stay-id` replay reads a processed encounter. It is not a suitable
public-facing feature without dataset access controls and privacy review.

## 13. Single-patient input and output

`src/predict.py` accepts a flat JSON dictionary. Example **synthetic input**:

```json
{
  "age": 61,
  "sex": "M",
  "heartrate": 108,
  "sbp": 92,
  "dbp": 58,
  "resprate": 24,
  "o2sat": 93,
  "pain": 9,
  "acuity": 1,
  "chief_complaint": "Chest pain radiating to the arm",
  "troponin": [1.2, 6.8],
  "troponin_hours": [0.8, 3.5],
  "ecg": {
    "st_elevation": true,
    "acute": true,
    "hours_after_arrival": 0.15
  },
  "home_medications": ["aspirin"],
  "prior_ed_visits": 1
}
```

The CLI displays an estimated `P(ACS)`, a four-class probability vector,
conditional subtype scores, a prediction label, a rule-based risk band/action,
and recognised chief-complaint terms. Its risk/action strings are **research
presentation logic**, not validated clinical recommendations.

**Important:** Direct single-row inference builds features from the provided
JSON and aligns missing model columns to `NaN` through `ACSPredictor._align()`.
It does not prove those fields were recorded at the stated horizon; calling
code must still enforce the clinical information-availability contract.

## 14. Audits, known gaps, and unresolved distinctions

The following are **limitations or review items**, not claimed fixes:

1. **Current vs legacy version.** The older uploaded `Component_4/Component_4`
   project used a React/FastAPI demonstration and separate JSON XGBoost models.
   This GitHub `Component_04` has a different feature schema, artifacts, and
   training/inference pathway. Treat them as different project snapshots.
2. **UM4 versus CLI default.** The repository reports UM4 results, while
   `src/models/inference.py` defaults to a cascade. A future change should
   expose an explicit, testable UM4 inference option or clearly label CLI
   outputs as cascade results.
3. **Waveform boundary.** Raw-waveform-derived `wf_*` columns were reportedly
   generated but never incorporated into model input matrices. The reported ECG
   evidence mainly comes from cart text and measured intervals.
4. **Unserved territory model.** Anterior/inferior STEMI territory was explored,
   but the report notes potential interpretation-to-label leakage and does not
   recommend serving it as an independent clinical decision.
5. **Model performance changes by population and threshold.** AUROC, recall,
   F1, NPV, and referral coverage must all state the correct cohort/setting.
6. **No external or prospective validation reported.** Do not describe the
   system as production-ready for clinical decision-making.
7. **Documentation-only verification.** Creating this file neither retrains
   models nor independently validates the reported numbers. Follow up with
   automated regressions, feature/horizon contract tests, and clean-room runs.

## 15. Safe development and testing checklist

For future updates, prefer small feature branches and reviewable commits:

- [ ] Confirm the intended GitHub branch and current artifact versions.
- [ ] Run syntax/format checks on touched Python modules.
- [ ] Add a regression case for `T0 + H` time-bound enforcement.
- [ ] Test that a patient cannot be in more than one split.
- [ ] Test referral-diagnosis masking, negation, and empty complaint fields.
- [ ] Test absent ECG and unmeasured troponin without invented values.
- [ ] Test feature list alignment against saved model configuration.
- [ ] Test stage/cascade vs UM4 routing and report the mode in output.
- [ ] Confirm thresholds are chosen on validation, never retuned on test.
- [ ] Reproduce key metrics with fixed versions and saved seeds, when data and
      artifacts can be accessed under the research-use agreement.
- [ ] Scan the Git diff for patient data, keys, tokens, secrets, and large models.
- [ ] Request review before merging changes into the shared `main` branch.

## 16. Medical and data-use restrictions

The MIMIC datasets require appropriate credentialed access and compliance with
PhysioNet's data use agreement. Do not commit extracted patient-level tables,
waveforms, authentication material, or other restricted records to GitHub.
Use synthetic cases for demos and public tests. Research metrics are not
clinical clearance; no live patient-care use is established by this repository.

## 17. Related documentation

- [Component 04 research README](README.md) — research question, cohort, all
  reported evaluation results, limitations, references, and background.
- [`docs/RESEARCH_GAP.md`](docs/RESEARCH_GAP.md) — evidence and gap framing.
- [`docs/NOVELTY_AND_CONTRIBUTION.md`](docs/NOVELTY_AND_CONTRIBUTION.md) —
  contributions and their measured effects.
- [`docs/PANEL_QA.md`](docs/PANEL_QA.md) — panel/viva preparation.
- [`artifacts/reports/RESULTS.md`](artifacts/reports/RESULTS.md) — reported
  performance detail where present.
- [`artifacts/reports/FINAL_RESULTS.md`](artifacts/reports/FINAL_RESULTS.md) —
  experiment history and negative results.
- [`artifacts/reports/ABLATIONS.md`](artifacts/reports/ABLATIONS.md) — model
  and modality ablations.

---

**Maintenance note:** Update this reference whenever a model, feature schema,
operating point, inference mode, or reported metric changes. Keep each claim
traceable to source code, configuration, or a named experimental artifact.
