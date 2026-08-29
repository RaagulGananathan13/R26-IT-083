# Component Write-Up: Temporally-Safe Explainable ACS Triage (Component 04)

**Repository root:** `c:\Users\dviya\Desktop\Component_4\Component_04`
**Dossier compiled from:** source (7,446 lines of Python across 25 modules), `configs/config.yaml`, `artifacts/reports/*.json`, `docs/*.md`, `README.md`

**Author:** Abishnan *(stated by the project team; see §23 — no in-repository evidence supports or contradicts this)*

> **⚠️ AUTHORSHIP CAVEAT.** This directory is **not a git repository** (`git rev-parse` → *fatal: not a git repository*). No commit history, no author metadata, and no ownership file exists. The attribution to **Abishnan** above was supplied verbally by the project team, **not extracted from the codebase**. §23 records this distinction; a formal author-contributions note still needs the author's own statement.

---

## 0. Component Abstract

Emergency-department risk models for Acute Coronary Syndrome (ACS) routinely join laboratory, ECG and comorbidity tables without a time bound, so they "predict" using information that did not exist at the decision point (`docs/RESEARCH_GAP.md`; `src/analysis/audit_leakage.py`). This component rebuilds ACS detection and subtyping from MIMIC-IV-ED raw tables under an explicit **information-availability contract**: every feature carries a declared availability time relative to ED arrival, and the same cohort is featurised at three disclosure horizons (H = 0, 6, 24 h). It combines a LightGBM detection stage, a unified four-class model with a constrained decision layer, and clinician-referral abstention. On the held-out patient-disjoint test fold the system reaches **Stage-1 AUROC 0.9560 with NPV 99.41 %**, **Stage-2 subtyping macro-F1 0.7448**, and end-to-end **minimum per-class recall 0.7783 at 85 % coverage** (`artifacts/reports/stage1_metrics_H24.json`, `stage2_metrics_H24.json`, `um4_H24.json`). A controlled leakage experiment shows that re-adding a single comorbidity column moves AUROC 0.9665 → 0.9889, reproducing the component's own previously published figure.

**Keywords:** Acute Coronary Syndrome; Emergency Department Triage; Temporal Data Leakage; Explainable AI; Class Imbalance; Selective Prediction

---

## 1. Role in the Overall System

**FACT (`README.md` §1):** the component answers two questions for a patient arriving with possible cardiac symptoms — *"Is this ACS?"* (rule-out screen tuned for safety) and *"If so, which type?"* (Unstable Angina / NSTEMI / STEMI, "because the three carry different treatment pathways on different clocks").

**Plain-language paragraph:**
A patient arrives at the emergency department with chest pain. This component reads what is known about them *at that moment* — their complaint text, vital signs, any ECG already taken, any bloods already back — and estimates whether they are having a heart attack, and if so which kind. It explains each answer three ways: which measurements mattered, which *type* of evidence (text, ECG, labs) mattered, and which words in the complaint mattered. Its distinguishing feature is that it will not use information that did not yet exist when the decision had to be made, and it can prove this: at the triage-desk horizon the laboratory evidence contributes exactly zero.

**⚠️ NOT FOUND IN CODEBASE — needs input from author:** there is **no code linking this component to Components 01/02/03**. `configs/config.yaml` points `paths.raw_dir` at `../Component_4/Component_4/data/processed`, i.e. a sibling data folder, not another component's API. Integration is **[inferred]** design intent only.

---

## 2. Problem Statement & Motivation

**The specific problem:**
- Detect ACS among ED arrivals (`src/models/train_stage1.py`) and, for detected cases, assign one of three subtypes (`src/models/train_stage2.py`), using only information available within a declared time window of arrival (`src/data/preprocess.py`).
- FACT — cohort scale (`README.md` §4.1): 203,016 admitted ED stays, 107,391 patients; No_ACS 197,636 / UA 739 / NSTEMI 3,700 / STEMI 941.

**Why it matters (FACT, `README.md`, `configs/config.yaml`):**
- Prevalence is extreme: ACS is **2.65 %** of the full ED test fold, UA is **0.36 %** (`README.md` §7.1, §7.3). A model can be highly accurate and clinically useless.
- The three subtypes "carry different treatment pathways on different clocks" (`README.md` §1) — STEMI is time-critical.
- Missing an infarct is the consequential error, so the screen is tuned to **NPV** (99.41 %, `stage1_metrics_H24.json`), not to accuracy.
- **A specific failure already occurred in this project:** an audit found the previous version of this component was reading its own label out of a comorbidity column (`README.md` §9). The motivation is therefore partly corrective.

---

## 3. The Gap

**FACT — stated in `docs/RESEARCH_GAP.md` and `README.md` §2, with measured evidence:**

| # | Gap | Measured evidence in repo |
|---|---|---|
| **G1** | Temporal leakage is pervasive and undetected in ED risk models | `leakage_audit.json` L3: troponin drawn median **21.75 h** after arrival, max **3,559 h**, **47.0 %** beyond 24 h; L4: only **4.56 %** of 2,546,849 ECG–stay pairs fall in a plausible window |
| **G2** | Accuracy is reported at a single unstated time | The component's own progressive-horizon table shows S1 AUROC moving 0.8763 → 0.9121 → 0.9560 across H = 0/6/24 (`README.md` §8) |
| **G3** | Rare-class F1 is reported without acknowledging the prevalence bound | `README.md` §10.2: at UA prevalence 0.36 %, F1 ≥ 0.75 at recall 0.75 requires a positive likelihood ratio > 800; troponin achieves 10–25 |
| **G4** | Referral diagnoses contaminate ED free text | `leakage_audit.json` L5: referral diagnosis present in **15.7 % of NSTEMI** and **0.58 % of No_ACS** (README quotes 25.9 % for STEMI) |

**⚠️ Note:** these gaps are argued from the component's *own* measurements, which is strong internal evidence. **No external literature survey artifact exists** (no BibTeX, no notes file). The claim that published work *does not* address these requires author literature research.

---

## 4. Research Question(s) This Component Answers

**⚠️ FACT: no file states research questions explicitly.** No `RQ1`/`RQ2` string exists in the codebase. The following are **[inferred]** from what the code measures and **require author confirmation**:

- **RQ1 [inferred]:** How much of a published ED-ACS model's apparent performance is attributable to temporal leakage rather than signal? — *measured by* `src/analysis/audit_leakage.py`, configurations A–D in `leakage_audit.json`.
- **RQ2 [inferred]:** How does achievable ACS detection and subtyping performance vary with the amount of information disclosed since arrival? — *measured by* re-running the identical pipeline at H ∈ {0, 6, 24} (`config.yaml temporal.horizons_h`), reported in `evaluation_H0/H6/H24.json`.
- **RQ3 [inferred]:** Can a decision layer hold every class above a recall floor without destroying precision, under 11:1-plus imbalance? — *measured by* `src/models/decision_layer.py` and the objective comparison in `README.md` §5.5.
- **RQ4 [inferred]:** Does a unified four-class model outperform a two-stage cascade end-to-end? — *measured by* the frontier comparison 0.7394 → 0.7447 (`README.md` §5.4, `um4_H24.json`).
- **RQ5 [inferred]:** Which modalities carry the evidence, and does that change with horizon? — *measured by* SHAP modality attribution (`src/analysis/explain.py`, `explainability_H*.json`).

---

## 5. Contribution Bullets & Novelty

1. **We audit temporal leakage with five probes and a controlled leaky-vs-safe experiment.** *(14 words)*
   → **Novel (as an instrument).** Leakage as a concept is established (Kaufman et al. 2012, `README.md` ref [9]); the contribution is a *reproducible domain-specific audit* that quantifies each channel and isolates one column's effect. `src/analysis/audit_leakage.py`.

2. **We featurise one cohort at three disclosure horizons, making accuracy-versus-time a reported axis.** *(14 words)*
   → **Novel.** What makes it different: standard practice reports one number at one unstated time. Here `config.yaml temporal.horizons_h: [0, 6, 24]` drives identical code over identical splits. `src/data/preprocess.py`.

3. **We optimise class weights for macro-F1 subject to a hard per-class recall floor.** *(13 words)*
   → **Adapted.** Cost-sensitive class weighting is standard; the adaptation is the *constrained* form (floor as constraint, macro-F1 as objective) replacing a hand-tuned boost. `src/models/decision_layer.py`.

4. **We mask referral diagnoses in ED free text while retaining an auditable flag.** *(12 words)*
   → **Adapted.** Text sanitisation is standard practice; the contribution is domain-specific detection plus the decision to keep `cc_referral_dx` auditable rather than silently deleting. `src/data/text_features.py`.

**Also present but explicitly Engineering, not novelty:** LightGBM/XGBoost training, Optuna search, SHAP attribution, isotonic calibration, patient-level grouped stratification, cluster bootstrap, selective prediction (Chow's rule). `README.md` §3 lists these as "prior work used and cited, not claimed" — that framing is correct and should be preserved.

**Conservative statement:** bullets 3 and 4 are **adaptations**. Bullet 2 is the strongest genuine novelty claim. Bullet 1 is novel as an *artifact* rather than as a method.

---

## 6. Contribution → Evidence Traceability Table

| Contribution | Implemented where | Evaluated where | Evidence status |
|---|---|---|---|
| **1.** Temporal leakage audit | `src/analysis/audit_leakage.py` (328 lines) | `artifacts/reports/leakage_audit.json` — L1 single-feature AUROC 0.9200; L2 contaminated test fraction 0.5730; L3 median 21.75 h; L4 4.56 % in-window; L6 configurations A–D | ✅ **STRONG — controlled experiment isolates one column (0.9665 → 0.9889)** |
| **2.** Progressive horizon modelling | `src/data/preprocess.py` (1,095 lines); `config.yaml temporal.horizons_h` | `evaluation_H0.json`, `evaluation_H6.json`, `evaluation_H24.json`; `explainability_H*.json`; `figures/progressive_horizon.png` | ✅ **STRONG — three complete runs; modality attribution shifts 0.0 % → 4.6 % → 29.6 % labs** |
| **3.** Constrained decision layer | `src/models/decision_layer.py` (246 lines); `config.yaml decision.*` | `um4_H24.json` (two operating points); `margin_sweep_H24.json` (negative result for constraint tightening) | ✅ **STRONG — includes a documented failed alternative** |
| **4.** Referral-diagnosis masking | `src/data/text_features.py` (297 lines); `config.yaml text.rdm_enable` | `ablations.json` → `rdm`: ON macro-F1 0.7688 / STEMI recall 0.7007 vs OFF 0.7633 / 0.6934 | ⚠️ **WEAK EFFECT — the ablation shows RDM ON is only +0.0055 macro-F1; the leakage it removes is real (L5) but the measured performance impact is small** |
| *(supporting)* UM4 vs cascade | `src/models/unified4.py` (305 lines) | `um4_H24.json` frontier 0.7447 vs cascade 0.7394 | ✅ Measured |
| *(supporting)* Selective deferral | `src/models/selective.py` (184 lines) | `selective_H24.json` — 66 % coverage, macro-F1 0.8631 | ✅ Measured, coverage reported |

**⚠️ Paper-writing risk:** contribution 4 (RDM) is *justified by a leakage argument* but its measured performance benefit is **+0.006 macro-F1** — within noise for this sample size. State it as a **methodological-integrity** measure, not a performance contribution, or a reviewer will do it for you.

---

## 7. Related Work / Prior Approaches Referenced

**FACT — 9 references in `README.md` §10 (original file) / §17 (revised):**

| Approach | Key idea | Limitation / role here | Citation |
|---|---|---|---|
| MIMIC-IV-ED | Source EHR database | Single centre; ICD labels are administrative | Johnson et al., *Scientific Data*, 2023 |
| AHA/ACC chest-pain guideline | Clinical definitions of ACS and pathways | Domain grounding, not a baseline | Gulati et al., *Circulation*, 2021 |
| ESC ACS guideline | Subtype definitions, treatment clocks | Domain grounding | Byrne et al., *European Heart Journal*, 2023 |
| XGBoost | Scalable boosted trees | Used as model, not compared against | Chen & Guestrin, KDD 2016 |
| LightGBM | Efficient histogram GBDT | Used as model | Ke et al., NeurIPS 2017 |
| SHAP | Unified feature attribution | Used for explanation (C-explainability) | Lundberg & Lee, NeurIPS 2017 |
| Iterative stratification | Multi-label stratified splitting | Used for patient-level grouped split | Sechidis, Tsoumakas & Vlahavas, ECML PKDD 2011 |
| Optuna | Hyperparameter search framework | Used for tuning | Akiba et al., KDD 2019 |
| Leakage in data mining | Formulation, detection, avoidance of leakage | **The conceptual basis for C1** | Kaufman et al., *ACM TKDD*, 2012 |
| Selective classification | Reject option / abstention theory | Basis for the referral mechanism | El-Yaniv & Wiener, *JMLR* 11:1605-1641, 2010 (added in revised README) |
| Chow's rule | Optimum error/reject tradeoff | Basis for the referral mechanism | Chow, *IEEE T-IT* 16(1):41-46, 1970 (added in revised README) |

**⚠️ Gap:** **no competing ED-ACS risk model is cited or compared against** — not HEART, TIMI, GRACE, EDACS, or any published ML model. `README.md` §7.1 mentions "HEART, ADAPT and EDACS all recruit patients undergoing biomarker testing" as a *cohort-definition* argument, but none is implemented or benchmarked. **This is the largest related-work gap and requires author literature research.**

---

## 8. Domain-Specific Structuring Fit

**Best fit: Motivating-Example → Generalize**, with a strong secondary **Build-up Ablation** character.

**Why (FACT):** the codebase and README open with a *concrete failure case* — the previous version of this component leaked its label through a comorbidity column — and then generalise to a five-probe audit instrument and a temporal contract. `README.md` opens: *"Rebuilt from the raw tables after an audit found that the previous version of this component was reading its own label out of a comorbidity column."* The decisive experiment (L6) is a controlled reproduction of the failure.

**Secondary fit:** `artifacts/reports/ablations.json` contains four systematic ablation families (modality, split, cohort, RDM), and `FINAL_RESULTS.md` records eleven interventions with three kept and eight rejected — a build-up-ablation structure.

**What this implies should be captured:**
- ✅ Already captured: the motivating failure (§9 of README), the general instrument (`audit_leakage.py`), the controlled experiment.
- ⚠️ **Should be made explicit in the paper:** the narrative order *failure → instrument → contract → results*, which the code follows but the current README states only implicitly.
- ⚠️ **Missing:** no external competing method is used as the "naive approach" to generalise against (§7).

---

## 9. Method / Design

### 9.1 Architecture (words → diagram)

```
MIMIC-IV-ED / Hosp / ECG parquet  (paths.raw_dir)
        │
   [ ecg_fetch.py · labs_fetch.py ]        targeted acquisition
        │
   [ preprocess.py ]                       C2 temporal contract, C4 missingness
        │   emits features at H = 0, 6, 24
   [ text_features.py ]                    C3 referral masking, lexicon, TF-IDF→SVD
        │
   [ split.py ]                            patient-level grouped stratification
        │
   [ dataset.py ]                          split-aware assembly (fit on train only)
        ├──────────────┬───────────────────┐
        ▼              ▼                   ▼
 train_stage1.py   train_stage2.py    unified4.py
 (ACS detection)   (UA/NSTEMI/STEMI)  (all four classes, C7)
        │              │                   │
        └──────────────┴───────────────────┘
                       ▼
              [ decision_layer.py ]         C5 constrained weights
                       ▼
              [ selective.py ]              clinician referral
                       ▼
   [ evaluate.py · explain.py · ablations.py · final_report.py ]
                       ▼
              artifacts/{reports,figures}
```

### 9.2 Key algorithms

**(a) Constrained decision layer** — `src/models/decision_layer.py`, `config.yaml decision.*`:
```
maximise    macro_F1(w)                       objective: "macro_f1"
subject to  min per-class recall(w) >= floor  min_recall_floor: 0.75
w searched on VALIDATION, then frozen before test
```
Relaxation ladder if infeasible: `floor_relaxation: [0.75, 0.74, 0.73, 0.72, 0.70]`.

**(b) Vectorised frontier search** — `src/models/unified4.py`: chunked argmax plus bincount confusion evaluates **400,000** candidate weight vectors (`README.md` §5.5), so the simplex is searched rather than sampled.

**(c) Progressive horizon featurisation** — `src/data/preprocess.py`: at horizon *H*, admit only features whose declared availability ≤ *H*, and clip values to those recorded within *H*.

**(d) Selective deferral** — `src/models/selective.py`: threshold chosen on validation from the **bootstrap 5th percentile of min per-class recall** (`README.md` §5.6), never from test.

### 9.3 Key design decisions with rationale (quoted from source)

| Decision | Rationale as written | Source |
|---|---|---|
| `require_triage_vitals: false` | "MUST stay false: 19% of STEMI patients arrive in arrest and have NO triage vitals recorded. Requiring vitals silently excludes the sickest patients and biases the cohort." | `configs/config.yaml` |
| `floor_margin: 0.0` | "constraint tightening is the textbook fix… Here it does not help: `recalibrate.py --sweep` shows test STEMI recall pinned at 101/137 for every margin in [0, 0.10] while macro-F1 decays monotonically, because the shortfall is a 2-case sampling gap on 137 STEMIs, not a mis-set threshold." | `configs/config.yaml` |
| IUP selection rule | "Both criteria are observable AT triage (a chief complaint, and the act of ordering an ECG), so this is a selection rule, not a feature leak." | `configs/config.yaml` |
| `max_bin: 4096`, `n_parallel_trials: 8` | "A single fit on 4k x 221 floats leaves an RTX 4060 idle between kernel launches. Concurrency is what fills it; VRAM follows from concurrency and histogram width, not from the (tiny) dataset." | `configs/config.yaml` |
| `gpu_oom_backoff: true` | "halve max_bin and retry instead of dying" | `configs/config.yaml` |
| Objective = macro-F1 under floor, not min-recall | "Optimising min-recall by itself rewards nothing except the weakest class… it produced STEMI precision 7.4% and macro-F1 0.39 at 67% coverage." | `README.md` §5.5 |
| UM4 replaces cascade | "A cascade compounds error: a patient Stage 1 misses can never be recovered by Stage 2, so end-to-end recall is capped by Stage-1 sensitivity." | `README.md` §5.4 |

### 9.4 Novel vs. standard/reused — module by module

| Module | Lines | Assessment |
|---|---|---|
| `src/analysis/audit_leakage.py` | 328 | **Novel instrument** — five domain-specific probes + controlled experiment |
| `src/data/preprocess.py` | 1,095 | **Novel** temporal-contract logic (C2/C4); standard pandas feature engineering underneath |
| `src/data/text_features.py` | 297 | **Adapted** — RDM detection is custom; TF-IDF→SVD is standard scikit-learn |
| `src/models/decision_layer.py` | 246 | **Adapted** — constrained cost-sensitive weighting |
| `src/models/unified4.py` | 305 | **Adapted/Engineering** — the vectorised frontier search is a performance optimisation of exhaustive search, not a new algorithm |
| `src/models/selective.py` | 184 | **Standard** — Chow's rule / El-Yaniv & Wiener |
| `src/models/train_stage1.py`, `train_stage2.py` | 325 + 298 | **Engineering** — LightGBM/XGBoost + Optuna, standard patterns |
| `src/data/split.py` | 199 | **Standard** — iterative stratification (Sechidis et al.) applied with patient grouping |
| `src/analysis/explain.py` | 297 | **Standard** — wraps SHAP |
| `src/analysis/evaluate.py` | 480 | **Adapted** — cluster bootstrap is standard; the "achievable recall frontier" sampling is custom |
| `src/data/ecg_waveform.py` | 358 | **Adapted** — ST-segment measurement to ESC/AHA criteria |
| `src/core/progress.py`, `study_store.py` | 141 + 84 | **Engineering** — progress display, resumable Optuna |

### 9.5 Notation table

| Symbol | Meaning | Source |
|---|---|---|
| `T0` | ED arrival time (`edstays.intime`) | `config.yaml temporal` |
| `H` | Disclosure horizon in hours after T0; ∈ {0, 6, 24} | `config.yaml temporal.horizons_h` |
| `w` | Per-class weight vector applied before argmax | `decision_layer.py` |
| `floor` | Hard minimum per-class recall constraint = 0.75 | `config.yaml decision.min_recall_floor` |
| NPV | Negative predictive value — the screen's primary safety metric | `stage1_metrics_H24.json` |
| LR+ | Positive likelihood ratio; used in the prevalence-bound argument | `README.md` §10 |
| IUP | Intended Use Population — cardiac complaint OR ECG ≤ 3 h | `config.yaml cohort` |
| AWC | "A biomarker was ordered" evaluation population | `README.md` §7.1 |
| Coverage | Fraction of patients the system grades rather than defers | `selective.py` |

---

## 10. Algorithmic Complexity Analysis

**Applicable — for the decision-layer frontier search only.** Model training is standard GBDT complexity and is not original algorithmic work.

**`unified4.py` frontier search:**
- Candidate weight vectors: **W = 400,000** (`README.md` §5.5).
- Each candidate requires an argmax over an `N × K` probability matrix (N test rows, K = 4 classes) → **O(N·K)**, followed by a bincount confusion → **O(N)**.
- Naïvely that is **O(W·N·K)** = 400,000 × 30,452 × 4 ≈ 4.9 × 10¹⁰ elementary operations, which is why the vectorised chunked form matters: it amortises the per-candidate Python overhead, leaving the same asymptotic cost but a far smaller constant.
- **Space: O(C·N·K)** where C is the chunk size — the whole candidate set is never materialised at once; chunking is what bounds memory.
- **`evaluate.py` frontier:** the same structure with **W = 200,000** (`README.md` §10.3c).

**`decision_layer.py`:** identical structure with the relaxation ladder adding at most `|floor_relaxation| = 5` outer passes → **O(5·W·N·K)** worst case, **O(W·N·K)** when the first floor is feasible.

**Not applicable** to: `preprocess.py`, `train_stage*.py`, `explain.py`, `split.py` — feature engineering, library training calls, and standard SHAP evaluation.

---

## 11. Experimental Setup

### Hardware

**FACT (`requirements.txt` header, `README.md` §13):** "Python 3.11 / Windows 11 / RTX 4060 8GB (CUDA 12.8)"; "16 GB RAM (32 GB comfortable). GPU optional — CUDA is auto-detected and the pipeline falls back to CPU." `README.md` §13 also states "Verified on Windows 11 / Ryzen 7 8845HS / RTX 4060 8 GB."

### Software

**FACT (`requirements.txt`):**

| Library | Constraint | Purpose (from file comments) |
|---|---|---|
| numpy | ≥ 2.0 | |
| pandas | ≥ 2.2 | |
| pyarrow | ≥ 15.0 | parquet I/O |
| scipy | ≥ 1.13 | |
| scikit-learn | ≥ 1.5 | |
| xgboost | ≥ 3.0 | "GPU histogram trees (device=cuda)" |
| lightgbm | ≥ 4.3 | |
| optuna | ≥ 4.0 | |
| shap | ≥ 0.46 | |
| matplotlib | ≥ 3.8 | |
| joblib | ≥ 1.4 | |
| google-cloud-bigquery, pydata-google-auth, db-dtypes | **commented out** | "Optional — only needed to re-extract the raw tables from BigQuery" |

**⚠️ Versions are lower bounds only — no lockfile, no exact pinned versions.** The precise versions used for the reported results are **NOT FOUND IN CODEBASE — needs input from author**.

### Datasets

**FACT (`README.md` §4, `configs/config.yaml`):** MIMIC-IV-ED + MIMIC-IV-Hosp + MIMIC-IV-ECG (PhysioNet, credentialed). Raw tables expected at `paths.raw_dir` as parquet: `master_data`, `lab_values`, `medrecon`, `ecg_records`, `ecg_measurements`, `ecg_numeric`, `charlson`.

**Cohort (FACT, `README.md` §4.1):** 203,016 stays / 107,391 patients; No_ACS 197,636, UA 739, NSTEMI 3,700, STEMI 941. IUP: 98,273 stays.

**Splits (FACT, `artifacts/reports/split_report.json`):**

| Fold | Stays | Patients | No_ACS | UA | NSTEMI | STEMI |
|---|---|---|---|---|---|---|
| train | 142,111 | 52,274 | 138,345 | 517 | 2,590 | 659 |
| val | 30,453 | 27,559 | 29,646 | 111 | 555 | 141 |
| test | 30,452 | 27,558 | 29,645 | 111 | 555 | 141 |

`patient_overlap: 0`, `hadm_overlap: 0`, `max_proportion_drift: 5.017 × 10⁻⁶`.

**Features (FACT, `README.md` §4.3):** 242 across eight modalities — vitals 38, text 64, ECG 36, labs 26, demographics 16, medications 16, prior history 16, interactions 9.

**Licence (FACT, `README.md` §16):** PhysioNet credentialed data use agreement; de-identified; not redistributed.

**⚠️ CONFLICT:** `ablations.json` → `modality` reports `n_features: 242` for "ALL modalities", consistent with the README. However `configs/config.yaml` model comment references "4k x 221 floats" — **221 vs 242 features**. [inferred] 221 may be post-selection or a stale comment; **needs author confirmation**.

### Environment

- **FACT:** Windows 11; Python 3.11; no Dockerfile, no `environment.yml`, no `pyproject.toml`, no `setup.py` present.
- **FACT:** `configs/config.yaml` is the single configuration source; `src/core/config.py` (180 lines) loads it.
- **FACT:** `.credentials` directory exists at `../Component_4/Component_4/.credentials` (sibling folder) — **[inferred]** BigQuery service-account credentials, correctly kept outside the component tree.

### Compute Cost

- **FACT (`README.md` §13):** per-stage estimates — preprocess ~6 min, split ~10 s, audit ~2 min, stage1 ~10 min, stage2 ~35 min, evaluate ~5 min, explain ~5 min, ablations ~25 min. **Total ≈ 88 min for one horizon.**
- **FACT:** wall-clock instrumentation exists (`src/core/utils.py` L46-49 prints `"{label} done in {t:.1f}s"`; `src/core/progress.py` shows elapsed time and VRAM).
- **⚠️ NOT LOGGED to file:** no artifact records actual elapsed time, peak VRAM, or total compute across the three horizons. Timings are printed to stdout only. **Worth persisting.**

---

## 12. Parameters / Configuration

**FACT — all from `configs/config.yaml`:**

| Group | Parameter | Value |
|---|---|---|
| global | `seed` | 42 |
| paths | `raw_dir` | `../Component_4/Component_4/data/processed` |
| paths | `artifacts_dir` | `artifacts` |
| temporal | `horizons_h` | `[0, 6, 24]` |
| temporal | `primary_horizon_h` | 24 |
| temporal | `ecg_lookback_h` | 1.0 |
| temporal | `max_ed_los_h` | 168 |
| cohort | `enable` | true |
| cohort | `ecg_within_h` | 3.0 |
| cohort | `require_triage_vitals` | **false** (deliberate; see §9.3) |
| split | `train / val / test` | 0.70 / 0.15 / 0.15 |
| split | `group_col` | `subject_id` |
| split | `stratify_col` | `acs_label` |
| text | `rdm_enable` | true |
| text | `svd_components` | 24 |
| text | `tfidf_word_max_features` | 6000 |
| text | `tfidf_char_max_features` | 6000 |
| text | `min_df` | 3 |
| model | `device` | `cuda` (auto-fallback to CPU) |
| model | `n_parallel_trials` | 8 |
| model | `max_bin` | 4096 |
| model | `gpu_oom_backoff` | true |
| model | `progress` | true |
| model.stage1 | `n_trials` | 40 |
| model.stage1 | `n_estimators` | 3000 |
| model.stage1 | `early_stopping_rounds` | 100 |
| model.stage2 | `n_trials` | 60 |
| model.stage2 | `n_estimators` | 2000 |
| model.stage2 | `early_stopping_rounds` | 80 |
| model.stage2 | `cv_folds` | 5 |
| decision | `min_recall_floor` | 0.75 |
| decision | `floor_margin` | 0.0 |
| decision | `floor_relaxation` | `[0.75, 0.74, 0.73, 0.72, 0.70]` |
| decision | `objective` | `macro_f1` |
| decision | `stage1_target_sensitivity` | 0.80 |
| evaluation | `bootstrap_n` | 1000 |
| evaluation | `bootstrap_alpha` | 0.05 |

**⚠️ CONFLICT:** `config.yaml evaluation.bootstrap_n: 1000` and `evaluate.py L281` reads that value, **but `evaluate.py L240` hardcodes `n_bootstrap=30`** for a different call. Two different bootstrap sizes are in use; the 30-resample one should be identified and justified or raised. **Needs author confirmation.**

**⚠️ Learned model hyperparameters** (tree depth, learning rate, etc.) are Optuna-selected and stored in `artifacts/models/stage1_config_H*.json` — not enumerated here; they are outputs, not inputs.

---

## 13. Baseline(s) Compared Against

**FACT — internal baselines that exist as measured comparisons:**

| Comparison | Result | Source |
|---|---|---|
| Leaky features + random split (A) | AUROC 0.9580, AUPRC 0.7188 | `leakage_audit.json` L6 |
| Leaky features + patient-disjoint split (B) | AUROC 0.9577, AUPRC 0.7590 | " |
| **Safe features + patient-disjoint split (C)** | **AUROC 0.9665, AUPRC 0.6750** | " |
| C + only the Charlson MI flag (D) | **AUROC 0.9889, AUPRC 0.8101** | " |
| Random stratified split | AUROC 0.9538, 5,804 patients in both folds, 7,627 contaminated test rows | `ablations.json` → `split` |
| **Patient-level grouped split (ours)** | AUROC 0.9558, **0** shared patients, **0** contaminated rows | " |
| Cascade (two-stage) | frontier 0.7394 | `README.md` §5.4 |
| **UM4 (unified four-class)** | frontier 0.7447 | `um4_H24.json` |

**⚠️ MAJOR GAP — no external baseline.** No published ED-ACS risk score (HEART, TIMI, GRACE, EDACS) and no published ML model is implemented or compared against. All comparisons are internal (this system versus earlier versions of itself). `FINAL_RESULTS.md` records eleven interventions, but these are **variants of this system**, not competing methods. **This is the single most likely reviewer objection and requires author action.**

---

## 14. Evaluation Metrics

**FACT — computed in `src/core/utils.py` and `src/analysis/evaluate.py`:**

| Metric | Type | Why it fits | Answers RQ |
|---|---|---|---|
| AUROC | Ranking | Threshold-free detection quality under imbalance | RQ1, RQ2 |
| AUPRC | Ranking | More informative than AUROC at 2.65 % prevalence | RQ1, RQ2 |
| **NPV** | Screening | **Primary safety metric** for a rule-out screen — 0.9941 test | RQ3 |
| PPV | Screening | Alert burden; 0.2844 test | RQ3 |
| Sensitivity / specificity | Screening | Operating-point description | RQ3 |
| Balanced accuracy | Classification | Removes majority-class advantage | RQ2, RQ4 |
| Per-class recall | Classification | The ≥75 % floor is stated on recall | RQ3, RQ4 |
| Per-class precision / F1 | Classification | Bounded by prevalence for rare classes (§10 of README) | RQ3 |
| macro-F1 | Classification | Decision-layer objective | RQ3, RQ4 |
| min per-class recall | Classification | Decision-layer constraint | RQ3, RQ4 |
| `n_missed_acs` | Clinical | 66 of 763 at the safety operating point | RQ3 |
| `alerts_per_100` | Operational | 18.09 per 100 patients — workload measure | RQ3 |
| Coverage | Selective | Fraction graded vs deferred; quoted with every selective figure | RQ3 |
| Cluster bootstrap CI | Statistical | Patient-level resampling so repeat visits do not inflate confidence | all |
| SHAP feature / modality / token attribution | Explainability | Three-level explanation; modality shift proves the temporal contract | RQ5 |
| Achievable recall frontier | Diagnostic | Establishes what *any* rule of this form could reach | RQ3, RQ4 |

---

## 15. Experimental Repetition & Statistical Robustness

**FACT:**
- **Single fixed seed: 42** (`config.yaml seed: 42`). **No multi-seed repetition** — the pipeline is run once per horizon.
- **Cluster bootstrap confidence intervals ARE computed** at patient level: `config.yaml evaluation.bootstrap_n: 1000`, `bootstrap_alpha: 0.05`; implemented `src/core/utils.py bootstrap_ci`, invoked `evaluate.py L280-281`. Reported in `stage1_metrics_H24.json → test_ci`, e.g. ACS recall mean 0.9137, lo 0.8921.
- **Selective threshold uses the bootstrap 5th percentile** of min per-class recall on validation (`README.md` §5.6) — bootstrap used for *decision-making*, not only reporting.
- **10-seed bagging was tried and rejected** (`README.md` §11: "101/137 STEMI unchanged").

**⚠️ WEAKNESSES:**
1. **No formal significance testing.** No t-test, McNemar, ANOVA, or p-value anywhere in `src/`. Grep for `ttest|mannwhitney|mcnemar|p_value` returns only bootstrap-CI matches.
2. **No multi-seed variance.** All headline numbers come from a single seed-42 run; there is no run-to-run standard deviation for any metric.
3. **⚠️ CONFLICT:** `evaluate.py L240` uses `n_bootstrap=30` while `L281` uses the configured 1000. Thirty resamples is too few for a stable 95 % interval; the two call sites should be reconciled.
4. Comparisons between configurations (e.g. UM4 vs cascade, RDM on/off) are reported as **point differences with no interval**, so it is not established that e.g. the +0.0055 macro-F1 from RDM exceeds noise.

---

## 16. Ablation Studies

**FACT — four ablation families exist in `artifacts/reports/ablations.json`:**

**(a) Modality ablation** (leave-one-modality-out, 242 features baseline):

| Configuration | n_features | S1 AUROC | S1 AUPRC | ΔS1 AUPRC | S2 macro-F1 | ΔS2 macro-F1 |
|---|---|---|---|---|---|---|
| ALL modalities | 242 | 0.9562 | 0.7179 | — | 0.7593 | — |
| − labs | 216 | 0.9530 | 0.6859 | **−0.0320** | 0.7419 | −0.0174 |
| − text | 178 | 0.9503 | 0.6930 | −0.0249 | 0.7428 | −0.0165 |
| − ecg | 190 | 0.9525 | 0.7044 | −0.0135 | 0.7399 | **−0.0194** |
| − vitals | 204 | 0.9545 | 0.7076 | −0.0103 | 0.7494 | −0.0099 |

**(b) Split protocol ablation:** random stratified → AUROC 0.9538, 5,804 shared patients, 7,627 contaminated test rows; patient-level grouped → AUROC 0.9558, 0 shared, 0 contaminated.

**(c) Cohort ablation:** IUP (n 13,549, prevalence 5.63 %) AUROC 0.9562 / AUPRC 0.7179; full ED (n 30,452, prevalence 2.65 %) AUROC 0.9713 / AUPRC 0.6946.

**(d) RDM ablation:** ON → S1 AUROC 0.9558, S2 macro-F1 0.7688, STEMI recall 0.7007; OFF → 0.9562, 0.7633, 0.6934.

**FACT — additionally, eleven interventions with keep/reject decisions** are recorded in `README.md` §11 / `FINAL_RESULTS.md`, including three kept (UM4+frontier+deferral, AWC population, ECG acuity tokens) and eight rejected with their measurements.

**Assessment:** ablation coverage here is **strong** — notably stronger than most student projects, and the retention of eight *rejected* approaches with measurements is good practice.

**⚠️ Still missing:** no ablation of the **horizon mechanism itself** (i.e. train at H=24 but evaluate features as if H=0), and no ablation isolating the **constrained** objective from the **unified model** — the UM4 and decision-layer changes were adopted together (`README.md` §5.4-5.5).

---

## 17. Existing Figures / Visual Assets Inventory

**FACT — 51 PNG files in `artifacts/figures/`:**

| Path pattern | Count | What it shows |
|---|---|---|
| `stage1_curves_H{0,6,24}.png` | 3 | ROC / PR curves for detection at each horizon |
| `stage1_confusion_H*.png` | 3 | Stage-1 confusion matrices |
| `stage1_tradeoff_H*.png` | 3 | Sensitivity/NPV/alert-burden frontier — **cited in README §10.3a** |
| `stage2_confusion_H*.png` | 3 | Subtype confusion matrices |
| `stage2_per_class_H*.png` | 3 | Per-class subtype metrics |
| `stage2_true_acs_confusion_H*.png` | 3 | Subtyping restricted to true-ACS patients |
| `e2e_confusion_H*.png` | 3 | End-to-end four-class confusion |
| `e2e_per_class_H*.png` | 3 | End-to-end per-class metrics |
| `shap_stage1_beeswarm_H*.png` | 3 | Stage-1 SHAP beeswarm |
| `shap_stage1_top_H*.png` | 3 | Stage-1 top features |
| `shap_stage2_top_H*.png` | 3 | Stage-2 top features |
| `shap_stage2_{UA,NSTEMI,STEMI}_H*.png` | 9 | Per-subtype SHAP attribution |
| `modality_attribution_H*.png` | 3 | **SHAP mass by modality — the temporal-contract evidence** |
| `progressive_horizon.png` | 1 | **Accuracy-vs-disclosure-time — the C2 headline figure** |
| `um4_confusion_H24.png`, `um4_final_confusion_H24.png` | 2 | UM4 confusion matrices |
| `um4_per_class_H24.png`, `um4_final_per_class_H24.png` | 2 | UM4 per-class metrics |
| `um4_coverage_curve_H24.png` | 1 | Coverage vs performance for selective deferral |

**Directly reusable in the paper:** `progressive_horizon.png` (the C2 result), `modality_attribution_H*.png` (the leakage-freedom evidence), `stage1_tradeoff_H24.png` (the NPV/F1 argument), `um4_coverage_curve_H24.png` (selective operating points).

---

## 18. Results Found in Repo (facts only)

### 18.1 Stage 1 — ACS detection at H = 24 (`stage1_metrics_H24.json`)

| Metric | Validation | Test |
|---|---|---|
| AUROC | 0.9455 | **0.9560** |
| AUPRC | 0.6593 | **0.6921** |
| Balanced accuracy | 0.8778 | **0.8882** |
| Accuracy | 0.8627 | 0.8657 |
| macro-F1 | 0.6735 | 0.6788 |
| min recall | 0.8608 | 0.8628 |
| min F1 | 0.4250 | 0.4337 |
| Sensitivity | 0.8949 | **0.9135** |
| Specificity | 0.8608 | 0.8628 |
| **NPV** | 0.9927 | **0.9941** |
| PPV | 0.2786 | 0.2844 |
| n missed ACS | 80 | **66** |
| Alerts per 100 | 18.21 | 18.09 |

Bootstrap CIs present in `test_ci`, e.g. ACS recall mean 0.9137 [0.8921, —]; No_ACS F1 0.9238 [0.9202, 0.9272].

### 18.2 Stage 1 across evaluation populations (`README.md` §7.1)

| Population | n | Prevalence | AUROC | Bal. acc | ACS F1 | NPV |
|---|---|---|---|---|---|---|
| FULL ED | 30,452 | 2.65 % | 0.9688 | 88.92 % | 0.5206 | 99.48 % |
| IUP | 13,549 | 5.63 % | 0.9560 | 87.58 % | 0.5735 | 98.82 % |
| AWC | 1,749 | 34.02 % | 0.8850 | 81.15 % | **0.7436** | 90.20 % |

### 18.3 Stage 2 — subtyping (`stage2_metrics_H24.json`, `test`)

| Class | Recall | Precision | F1 | n |
|---|---|---|---|---|
| UA | 0.8000 | 0.7719 | 0.7857 | 110 |
| NSTEMI | 0.7888 | 0.8945 | 0.8383 | 516 |
| STEMI | 0.7372 | 0.5206 | 0.6103 | 137 |

macro-F1 **0.7448** · balanced accuracy **0.7753**
*(`test_argmax` variant, same file: macro-F1 0.7613, bal. acc 0.7688 — a different decision rule.)*

### 18.4 End-to-end UM4 (`um4_H24.json`)

| Operating point | Coverage | Accuracy | Balanced | macro-F1 | min recall | No_ACS | UA | NSTEMI | STEMI |
|---|---|---|---|---|---|---|---|---|---|
| max-coverage | 85 % | 0.9354 | 0.8327 | 0.4906 | **0.7783** | 0.9389 | 0.8152 | 0.7783 | 0.7982 |
| max-macro-F1 | 65 % | 0.9542 | 0.8811 | 0.5720 | **0.8105** | 0.9567 | 0.9041 | 0.8105 | 0.8529 |

**⚠️ CONFLICT:** `um4_final_H24.json` records a **third** configuration — also 85 % coverage but different weights, objective `"max macro-F1 s.t. val min-recall>=0.78"` — with test balanced 0.8355, accuracy 0.9344, macro-F1 0.4983, **min recall 0.7619** and per-class No_ACS 0.9381 / UA 0.8454 / NSTEMI 0.7619 / STEMI 0.7965. **This does not match either operating point in `um4_H24.json`.** The README quotes `um4_H24.json`. **Which is the intended deployment configuration requires author confirmation.**

### 18.5 Selective subtyping (`selective_H24.json`)

Coverage 0.6606 — UA recall 0.8955 / F1 0.9023; NSTEMI 0.9430 / 0.9311; STEMI 0.7209 / 0.7561; macro-F1 0.8631, balanced accuracy 0.8532, accuracy 0.8988, min recall 0.7209.

### 18.6 Leakage audit (`leakage_audit.json`)

- **L1:** `P(myocardial_infarct = 1)` = **1.0000** for both NSTEMI and STEMI; 0.0613 for No_ACS. Single-feature AUROC **0.9200**.
- **L2:** 203,016 stays / 107,391 patients; 31.06 % multi-visit; 19,936 shared patients; **contaminated test fraction 0.5730**.
- **L3:** troponin median **21.75 h** post-arrival, max **3,559.08 h**, **47.00 %** beyond 24 h. Within 1 h: 2.97 %; 3 h: 5.46 %; 6 h: 9.57 %; 24 h: 52.996 %.
- **L4:** 2,546,849 ECG–stay pairs; 36.35 % after 24 h; 56.99 % before 24 h; **4.56 % in window**.
- **L5:** referral diagnosis present — No_ACS 0.58 %, UA 3.65 %, NSTEMI 15.68 %.
- **L6:** A 0.9580/0.7188 · B 0.9577/0.7590 · C **0.9665/0.6750** · D **0.9889/0.8101**.

### 18.7 Progressive horizon (`README.md` §8, from `evaluation_H*.json`)

| Horizon | S1 AUROC | S1 AUPRC | S1 sens | S1 NPV | S2 macro-F1 | UA rec | NSTEMI rec | STEMI rec |
|---|---|---|---|---|---|---|---|---|
| H = 0 | 0.8763 | 0.4138 | 84.40 % | 98.75 % | 0.5662 | 37.27 % | 75.58 % | 56.93 % |
| H = 6 | 0.9121 | 0.5172 | 81.91 % | 98.73 % | 0.6581 | 58.18 % | 80.43 % | 61.31 % |
| H = 24 | 0.9560 | 0.6921 | 91.35 % | 99.41 % | 0.7448 | 80.00 % | 78.88 % | 73.72 % |

Modality SHAP mass: H=0 text 31.3 % / ECG 0.1 % / **labs 0.0 %**; H=6 20.2 / 27.0 / 4.6; H=24 14.6 / 18.1 / 29.6.

### 18.8 Ablations

See §16 — all four families have complete numeric results.

---

## 19. Interpretation Notes

**Interpretations already written by the author, quoted with source:**

- **On the AWC population** (`README.md` §7.1): "Reaching F1 ≥ 0.75 at recall 0.75 requires a positive likelihood ratio of ~56 in the IUP but only ~9.5 in the AWC — and troponin achieves 10–25… Moving to it lifts ACS F1 from 0.434 to 0.744 without changing the model."
- **On the temporal contract** (`README.md` §8.1): "At H = 0 the laboratory channel carries exactly zero attribution — no troponin exists at that horizon and the model provably does not use one. A pipeline with a temporal leak cannot produce this pattern."
- **On UA's horizon sensitivity** (`README.md` §8.2): "Unstable angina is defined as ACS with a normal troponin, so it is not identifiable until the biomarker returns. The curve recovers that clinical fact from the data rather than being told it."
- **On the decision-layer objective** (`README.md` §5.5): "Optimising min-recall by itself rewards nothing except the weakest class, so it drives weights to extremes and destroys precision."
- **On why the cascade was replaced** (`README.md` §5.4): "A cascade compounds error: a patient Stage 1 misses can never be recovered by Stage 2."
- **On the STEMI specialist head** (`README.md` §11): "a dedicated binary detector reaches AUROC 0.9708 and 85 % STEMI recall, but it cannot separate STEMI from NSTEMI — it stole 309 of 555 NSTEMI cases."
- **On honest reporting** (`README.md` §10.3): "A component reporting ≥ 75 % F1 on every class of a full ED population is reporting a leak. That is precisely what the audit found in the previous version, and saying so is part of the contribution."
- **On the STEMI cap** (`README.md` §10.3b): "MIMIC supplies the ECG cart's text report, not the waveform… ST elevation is detectable in only 41 % of STEMI cases, when clinically it is near-universal."

**Requires author interpretation — human judgment call, not extractable from code:**
- Whether the leakage audit is framed as the paper's *primary* contribution or as methodology.
- Which UM4 operating point is the deployment recommendation (§18.4 conflict).
- Clinical acceptability of 18 alerts per 100 patients and 66 missed ACS.
- Whether the negative RDM performance result (+0.006 macro-F1) is reported as such.

---

## 20. Limitations & Threats to Validity

**FACT — stated by the author (`README.md` §14), 8 items:** single centre, no external validation; ICD labels are administrative not adjudicated; UA is small (739 cases, 111 in test); ECG is text not waveform; referral cases retain residual signal; rare-class precision bounded by arithmetic; selective results depend on coverage; retrospective only.

**ADDITIONAL threats visible in code, not listed by the author:**

| Threat | Evidence |
|---|---|
| **Single seed (42)** | `config.yaml seed: 42`; no repetition, so no run-to-run variance for any reported number |
| **`n_bootstrap=30` at one call site** | `evaluate.py L240` — too few resamples for a stable interval; conflicts with configured 1000 |
| **Hardcoded relaxation ladder** | `floor_relaxation: [0.75 … 0.70]` — if the floor is infeasible the system silently reports a lower one; the achieved floor must be read from output, not assumed |
| **Feature-count discrepancy** | 242 (README, ablations) vs "221 floats" (config comment) — unresolved |
| **Three UM4 configurations exist** | `um4_H24.json` (two) and `um4_final_H24.json` (one more) with different weights and results; deployment intent ambiguous |
| **`raw_dir` is a relative path to a sibling folder** | `../Component_4/Component_4/data/processed` — the pipeline is not runnable from a clean clone without that external directory |
| **No lockfile** | `requirements.txt` gives only lower bounds; exact reproduction not guaranteed |
| **No test suite** | No `tests/` directory anywhere in the component |
| **Not a git repository** | No version history; provenance of any result cannot be traced to a code state |

---

## 21. Ethical & Societal Considerations

- **Data privacy — Applicable.** MIMIC-IV-ED is de-identified patient EHR data under a PhysioNet credentialed DUA requiring human-subjects training (`README.md` §16). No re-identification is attempted. **FACT:** a `.credentials` directory exists in the *sibling* folder, correctly outside this component tree; **[inferred]** BigQuery service-account keys.
- **Data redistribution — Applicable and handled.** `README.md` §16: raw MIMIC tables are not redistributed. **⚠️ HOWEVER:** `artifacts/data/ecg_waveforms/` contains **real patient ECG waveform files** (`.dat`/`.hea`, e.g. `41068472/41068472.dat`) inside the component directory. These are de-identified MIMIC-IV-ECG records, but **they must not be committed to any public repository**. There is **no `.gitignore` in this component** (and no git repository), so nothing currently prevents this. **⚠️ ACTION REQUIRED before any push.**
- **Potential misuse — Applicable.** A triage aid used autonomously could withhold workup from a patient the model scores low. `README.md` §16: "Not a medical device, not validated for clinical deployment, and not to be used for patient care."
- **Fairness / bias — Applicable but NOT ANALYSED.** The component makes clinical decisions about people, and its feature set explicitly includes **16 demographic features** (`README.md` §4.3). **No subgroup performance breakdown exists** in any report. Unlike Component 03 (whose dataset carries no demographics), here the data *is* available — so this is a genuine, closable gap. **⚠️ Recommend a subgroup analysis by age/sex before submission.**
- **Environmental / compute cost — Applicable, low.** ≈ 88 min per horizon × 3 horizons on a single laptop GPU. Not measured in kWh.
- **Dataset licensing — Applicable and handled.** PhysioNet credentialed DUA; cited (`README.md` ref [1]).

---

## 22. Reproducibility / How to Run

**FACT — from `README.md` §13:**

```bash
pip install -r requirements.txt
# then point paths.raw_dir in configs/config.yaml at the extracted MIMIC parquet files
```

**Everything, one command:**
```bash
python src/run_all.py                 # primary horizon (H = 24 h)
python src/run_all.py --all-horizons  # full progressive-horizon study
```

**Stage by stage (with the author's timing estimates):**
```bash
python src/data/preprocess.py           # ~6 min
python src/data/split.py                # ~10 s
python src/analysis/audit_leakage.py    # ~2 min
python src/models/train_stage1.py 24    # ~10 min
python src/models/train_stage2.py 24    # ~35 min
python src/models/unified4.py 24
python src/analysis/evaluate.py         # ~5 min
python src/analysis/explain.py 24       # ~5 min
python src/analysis/ablations.py        # ~25 min
```

**Single-patient inference:** `python src/predict.py --demo | --json patient.json | --stay-id 31234567`
**Re-tune the operating point without retraining:** `python src/models/recalibrate.py --floor 0.75 --sweep`

**Resumability (FACT, `README.md` §13):** "Optuna searches are resumable — Ctrl+C and re-run the same command to continue from the last completed trial. Add `--fresh` to start over." Implemented via `src/core/study_store.py` (SQLite-backed studies).

**Entry points:** `src/run_all.py`, `src/predict.py`, plus the nine stage scripts above.

**Artifact status:** **Not currently a shareable artifact.** It is not a git repository, has no lockfile, and `paths.raw_dir` points outside the component tree. A third party could not reproduce it without (a) PhysioNet credentials, (b) the BigQuery extraction step whose dependencies are commented out of `requirements.txt`, and (c) the sibling `data/processed` directory.

---

## 23. My Individual Role / Contribution Statement

**⚠️ CANNOT BE DETERMINED FROM THE CODEBASE.**

- **FACT:** this directory is **not a git repository** — `git rev-parse --is-inside-work-tree` returns *"fatal: not a git repository (or any of the parent directories)"*. There is no commit history, no author metadata, and no `.git` directory at any level above it within the project tree.
- **FACT:** no `AUTHORS`, `CONTRIBUTORS`, or ownership file exists.
- **FACT:** in the Component 03 proposal (`Project_Proposal_IT22219534_Dilukshan_Viyapury.docx`, Table 8.1), Component 04 is assigned to a **team member**, not to Dilukshan Viyapury.
- **SUPPLIED BY THE TEAM (not codebase-derived):** this component is authored by **Abishnan**. This dossier was compiled by a teammate (Dilukshan Viyapury) for merging into the shared group paper.

**Still NOT FOUND IN CODEBASE — needs input from Abishnan.** Required for the group paper's author-contributions note:
1. Which modules were written solo vs jointly.
2. Whether any code was adapted from another source or teammate.
3. Confirmation of the attribution above, since no repository evidence exists either way.

**Recommendation:** initialise a git repository now (`git init`) **with a `.gitignore` excluding `artifacts/data/ecg_waveforms/` and `.credentials`** — see §21 — so future work is attributable.

---

## 24. Key Terms / Mini-Glossary

- **ACS (Acute Coronary Syndrome)** — an umbrella term for conditions where blood flow to the heart is suddenly reduced; covers unstable angina, NSTEMI and STEMI.
- **UA / NSTEMI / STEMI** — the three ACS subtypes, in increasing severity of confirmed heart-muscle damage; they carry different treatment urgency.
- **Temporal leakage** — using information in a model that did not exist yet at the moment the prediction was supposed to be made.
- **Disclosure horizon (H)** — how many hours after arrival the model is allowed to look; here 0, 6 or 24.
- **Intended Use Population (IUP)** — the subset of patients the tool is meant for (cardiac complaint or an early ECG), chosen using only information visible at triage.
- **NPV (negative predictive value)** — of the patients the model calls negative, the share who really are negative; the key safety number for a rule-out screen.
- **Positive likelihood ratio (LR+)** — how much a positive test multiplies the odds of disease; used here to show some F1 targets are arithmetically unreachable.
- **Referral-Diagnosis Masking (RDM)** — stripping the already-known diagnosis out of transfer patients' complaint text so the model cannot simply read the answer.
- **Constrained decision layer** — choosing class weights to maximise one metric while forcing every class to stay above a minimum recall.
- **Coverage** — the fraction of patients the system answers for, rather than deferring to a clinician; must always be quoted alongside selective metrics.
- **Cluster bootstrap** — resampling by *patient* rather than by row, so patients with several visits do not make results look more certain than they are.
- **Ablation** — a test where you remove one part of a system to see if it still works well.

---

## 25. Gaps & Open Questions

### A. Requires literature research from the author

| # | Item |
|---|---|
| 1 | **No external baseline is cited or implemented.** No HEART, TIMI, GRACE, EDACS, or published ML ED-ACS model appears anywhere in the codebase. All comparisons are internal (this system vs earlier versions of itself). **This is the single largest reviewer risk.** |
| 2 | **Gap claims (§3) are argued from internal measurements only.** No literature-survey artifact exists to substantiate that published work does not address temporal leakage / horizon reporting / prevalence bounds. |
| 3 | **The "41 % of STEMI show ST elevation in the text report" figure** (`README.md` §10.3b) has no source in the repo — is it measured here or quoted? |

### B. Requires author decision / authoring

| # | Item |
|---|---|
| 4 | **No research questions stated anywhere.** RQ1–RQ5 in §4 are inferred and need confirmation or rewriting. |
| 5 | **Which UM4 configuration is the deployment recommendation** — `um4_H24.json` has two, `um4_final_H24.json` has a third (§18.4). |
| 6 | **How to frame the RDM result** — its leakage justification is sound but its measured benefit is +0.006 macro-F1 (§6). |
| 7 | **Clinical acceptability** of 18 alerts per 100 patients and 66 missed ACS — needs clinician input. |
| 8 | **§23 authorship** — who built this component; not determinable without git history. |

### C. Missing evidence / experiments

| # | Item |
|---|---|
| 9 | **No external baseline comparison** (see A1). |
| 10 | **No multi-seed repetition.** Single `seed: 42`; no run-to-run variance for any metric. |
| 11 | **No significance testing.** No t-test / McNemar / p-value anywhere; configuration differences are reported as bare point estimates. |
| 12 | **No fairness/subgroup analysis** — and unlike Component 03, **the data supports it**: 16 demographic features are in the feature set. Closable gap. |
| 13 | **No horizon-mechanism ablation** (train at H=24, evaluate as if H=0). |
| 14 | **UM4 and constrained decision layer were adopted together**; their individual contributions are not separated. |
| 15 | **Compute cost not persisted** — timings are printed to stdout (`utils.py` L46-49) but no artifact records elapsed time, peak VRAM, or total compute. |
| 16 | **No test suite** anywhere in the component. |

### D. Conflicts / inconsistencies (must be resolved)

| # | Item |
|---|---|
| 17 | **Bootstrap size conflict:** `config.yaml evaluation.bootstrap_n: 1000` and `evaluate.py L281` use 1000, but **`evaluate.py L240` hardcodes `n_bootstrap=30`**. Thirty resamples cannot support a stable 95 % interval. |
| 18 | **Feature count conflict:** 242 features (`README.md` §4.3, `ablations.json`) vs "4k x 221 floats" (`config.yaml` model comment). |
| 19 | **Three different UM4 operating points exist** across `um4_H24.json` and `um4_final_H24.json` with different weights, objectives and results (§18.4). |
| 20 | **Two Stage-2 decision rules reported** in `stage2_metrics_H24.json`: `test` (macro-F1 0.7448) and `test_argmax` (macro-F1 0.7613). The README quotes the former; the difference should be explained. |
| 21 | **`paths.raw_dir` points outside the component** (`../Component_4/Component_4/data/processed`), so the component is not self-contained. |

### E. ⚠️ Immediate action required (not a paper issue — a data-governance issue)

| # | Item |
|---|---|
| 22 | **`artifacts/data/ecg_waveforms/` contains real patient ECG waveform files** (`.dat`/`.hea`). The component has **no `.gitignore`** and is **not a git repository**. Before this is ever committed or shared, add a `.gitignore` excluding `artifacts/data/`, `.credentials`, and `artifacts/models/`. MIMIC's DUA prohibits redistribution. |
