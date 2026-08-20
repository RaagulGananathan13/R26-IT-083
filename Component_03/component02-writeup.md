# Component Write-Up: XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting (Component 02)

> **Document type:** evidence-based technical dossier, not paper prose.
> **Extracted from:** `C:\Users\Venushan\Desktop\RP-Venu\Component_02` (entire tree, excluding `frontend/node_modules/`).
> **Convention:** every claim cites its source file. Statements not directly present in a file are marked `[inferred]`. Missing information is marked `NOT FOUND IN CODEBASE — needs input from author`.
> **Component name source:** `README.md` L1 — "Component 02 — ECG Abnormality Detection & Cardiac Risk Reporting"; the "XAI-Based" prefix is from the project title used across `docs/PANEL_ANSWERS.md`.

---

## 0. Component Abstract

**Mini-abstract (facts only, sources in brackets):**

Automated 12-lead ECG classifiers report a probability but not a *bound on how often they are wrong*, and the statistical guarantees that do exist are marginal — averaged over the whole population — so they can fail silently for individual patient subgroups, for miswired recordings, and for diseases outside the label space [`docs/CONTRIBUTION_FINAL.md`, `audit/10_conditional_validity.py` docstring, `audit/13_out_of_scope.py` L4-11]. This component implements an end-to-end pipeline — quality gate → band-pass/normalise → 1-D residual CNN with squeeze-excitation → per-class temperature calibration → PAC (training-conditional) conformal triage → Grad-CAM/integrated-gradients explanation → template-grounded clinical report → automated text-verification gate — over PTB-XL, and then audits the *validity of its own guarantee* under three stressors [`src/pipeline.py`, `src/conformal.py`, `src/report.py`, `src/verify.py`]. On the untouched test fold (n = 1,711) the shipped recall-first operating point achieves macro accuracy 0.864, macro recall 0.810 and macro NPV 0.933, with every class ≥ 0.75 on both accuracy and recall [`README.md` L124-133; `analysis/results/02_operating_point.txt`]. The three audits show the marginal guarantee is violated in 9 of 23 patient subgroups (2 surviving Holm correction), that a simulated limb-electrode reversal changes up to 87% of diagnoses and voids 7 guarantees, and that 113 of 114 test-fold atrial-fibrillation recordings receive a statistical guarantee for a disease the model has no output unit for [`audit/results/10_conditional_validity.txt`, `11_significance.txt`, `12_electrode_reversal.txt`, `13_out_of_scope.txt`].

**Keywords (ACM/IEEE style):**
`Electrocardiography` · `Conformal prediction` · `Conditional validity` · `Explainable AI` · `Clinical decision support` · `Uncertainty calibration`

---

## 1. Role in the Overall System

**FACT — stated ownership and scope**
- `README.md` L1-8: "Component 02 — ECG Abnormality Detection & Cardiac Risk Reporting … **Venushan T** · part of the Explainable AI System for Cardiovascular Disease Detection and Diagnosis."
- Stated function: "Takes a 12-lead ECG, classifies it into 5 diagnostic superclasses, explains the decision, and produces a verified clinical report with a statistical guarantee on missed cases."

**FACT — the integration contract**
- The component ships as a **JSON-only HTTP API** with no HTML/templates, explicitly so other components can call it [`README.md` L51: "The backend is **JSON only** — no HTML, no templates. Point any client at it."].
- Endpoints [`backend/server.py` L339, L364, L380, L405, L413]:

| Method | Route | Handler | Purpose (per `README.md` L53-59) |
|---|---|---|---|
| `GET` | `/api/health` | `health()` | model info, class list, thresholds, readiness |
| `GET` | `/api/patients/<class_name>` | `patients()` | browse the built-in test set |
| `POST` | `/api/analyze/<int:ecg_id>` | `analyze_by_id()` | analyse a bundled record |
| `POST` | `/api/demo` | `demo()` | random record |
| `POST` | `/api/predict` | `predict()` | **analyse an uploaded `.dat` + `.hea` pair** — "the integration point" |

- Error handlers registered for HTTP 413 and 500 [`backend/server.py` L463, L468].
- `src/` is described as an importable library: "`src/` library — import this to embed the pipeline" [`README.md` L186].

**Plain-language paragraph for a non-expert teammate:**
This component is the ECG-reading engine. You hand it the two files a hospital ECG machine produces (`.dat` = the raw voltages, `.hea` = the header describing them). It first checks the recording is actually usable — right length, right units, no dead leads, no excessive noise — and refuses outright if it is not. If it passes, it filters and normalises the signal, runs it through a trained neural network, and produces five probabilities: normal, myocardial infarction, ST/T change, conduction disturbance, hypertrophy. It then does three things most classifiers do not: it converts each probability into a three-way decision (*ruled out* / *refer* / *ruled in*) carrying a proven bound on how often that decision is wrong; it highlights which leads and which moments in the trace drove the decision; and it writes a plain clinical report that is machine-checked, sentence by sentence, against the numbers that produced it, so the text cannot say anything the model did not actually compute. Other components in the group project call it over HTTP and get back a single JSON object containing the probabilities, the decision zones, the guarantee sentences, the explanation and a rendered ECG image.

**`[inferred]`** — The specific division of labour between this component and the other three teammates' components (what they consume from `/api/predict`, whether a shared orchestrator exists) is **NOT FOUND IN CODEBASE — needs input from author**. Nothing inside `Component_02/` imports from, or references by path, any sibling component.

---

## 2. Problem Statement & Motivation

**The specific problem this component addresses (FACT, from code docstrings and docs):**

1. **A probability is not a guarantee.** `src/conformal.py` L133-147 states the design intent directly: a marginal split-conformal bound "only controls the miss rate *in expectation over repeated calibration draws*", and the audit of that marginal version "found it violated on this single test realisation for CD (0.122 vs alpha 0.10) and HYP (0.174 vs 0.15), because those classes have few calibration positives."
2. **Generated clinical text can assert things the model never computed.** `src/verify.py` implements `_hallucinated_terms()`, `asserted_classes()`, `verify_report()`, `verify_paraphrase()` and `safe_paraphrase()` — an explicit gate between generated text and display.
3. **A classifier will answer even when the input is uninterpretable.** `src/quality.py::assess()` runs before inference; `README.md` L155-157: "The quality gate runs **before** the classifier, so a refused record never produces a probability."
4. **A five-class label space cannot express the disease that is actually present.** `audit/13_out_of_scope.py` L8-11: "Softmax has no 'none of the above', so an atrial fibrillation recording is redistributed across NORM/MI/STTC/CD/HYP and the pipeline proceeds as if nothing unusual happened."

**Why it matters (FACT, quoted from the codebase):**
- `analysis/results/02_operating_point.txt`: "For a rule-out system that is the correct trade: a false alarm costs a cardiologist's review, a false negative can cost a life."
- `audit/13_out_of_scope.py` L137-139: "The system certifies what it can measure while being blind to the finding that will cause the stroke."
- `README.md` L264-266: "**Atrial fibrillation and other arrhythmias are not detected** — their absence from a report is not evidence of their absence. 14.3% of the dataset carries a documented finding the label space cannot express."

---

## 3. The Gap

**FACT — the gap as articulated inside the repo**

`audit/10_conditional_validity.py` L28 cites Vovk et al. 2003 and Vovk 2012 for group-conditional (Mondrian) construction. `docs/CONTRIBUTION_FINAL.md` L16 positions Angelopoulos et al. 2023 Conformal Risk Control as "Method I use". `docs/PANEL_ANSWERS.md` L336 explicitly disclaims one candidate novelty: "XAI → MI subtype localisation | **Not mine** — Strodthoff et al. 2024, cited".

**The stated opening (FACT, `docs/CONTRIBUTION_FINAL.md`, `docs/RESEARCH_CONTRIBUTION.md` L133):**
- Conformal risk control exists as a method (Angelopoulos et al. 2023).
- Conformal prediction has already been applied to PTB-XL (Ann Noninvasive Electrocardiol 2025, doi:10.1111/anec.70099) — the repo cites this as **prior art, not as this component's novelty**.
- `docs/RESEARCH_CONTRIBUTION.md` L133 heads this as "**What exists elsewhere but not here.**"

**The gap this component claims to fill `[inferred from the three audit scripts' framing]`:**
Prior work reports that a conformal guarantee *holds marginally* on PTB-XL. None of the cited prior work reports what happens to that guarantee when (a) the population is partitioned by sex and age band, (b) the electrodes are physically miswired, or (c) the true disease lies outside the label space. All three audits in `audit/10–13` measure exactly those three failure modes and are framed as "CONTRIBUTION EXPERIMENT 1/2/3" in their module docstrings.

**⚠ Caveat the author must resolve:** the repo's own history records **two retracted novelty claims** (conformal prediction on PTB-XL; attribution→MI-subtype localisation) that turned out to be prior art. `docs/PANEL_ANSWERS.md` L336 preserves one of these retractions. **A systematic literature check that no one has published "conditional validity of conformal ECG guarantees under subgroup / electrode-reversal / out-of-scope stress" is NOT FOUND IN CODEBASE — needs input from author.** This is literature research, not code reading.

---

## 4. Research Question(s) This Component Answers

**FACT — the code does not contain a section literally headed "Research Questions".** The RQs below are `[inferred]` from what each audit script actually measures; the numbered wording is the dossier author's, the measured quantity is not.

| RQ | Question | Measured by |
|---|---|---|
| **RQ1** | Does a conformal miss-rate guarantee fitted marginally on a calibration fold still hold within patient subgroups defined by sex and age band? | `audit/10_conditional_validity.py`, `audit/11_significance.py` |
| **RQ2** | Does group-conditional (Mondrian) calibration restore the bound within those subgroups, and at what cost? | `audit/10_conditional_validity.py`, section B |
| **RQ3** | Does a limb-electrode reversal that passes every signal-quality check change the diagnosis and void the statistical guarantee? | `audit/12_electrode_reversal.py` |
| **RQ4** | When the true disease lies outside the five-class label space, does the system still attach a statistical guarantee — and can a physiological rhythm check, fitted on validation only, withhold it? | `audit/13_out_of_scope.py` |
| **RQ5** | Are the generated Grad-CAM / integrated-gradients explanations faithful to the model, or would a random attribution do as well? | `audit/05_xai_audit.py` (deletion/insertion AUC, model-randomisation Spearman) |
| **RQ6** | Can a recall-first operating point reach ≥ 0.75 recall and ≥ 0.75 accuracy on **every** superclass, chosen on validation only? | `analysis/02_operating_point.py` (`--floor 0.80`, `--report-floor 0.75`) |

**Explicit note:** RQ1–RQ6 are reconstructions. If the panel/paper needs authored RQs in the author's own framing, that is a writing task — the code states hypotheses only implicitly, through what it measures.

---

## 5. Contribution Bullets & Novelty

1. **We measure whether a PAC-conformal ECG miss-rate guarantee survives partition by sex and age, and repair it with Mondrian calibration.**
   → **Adapted.** The mechanism is Mondrian / group-conditional conformal prediction (Vovk et al. 2003; Vovk 2012), cited at `audit/10_conditional_validity.py` L28. What is adapted is its application as an *audit of an existing marginal guarantee* on PTB-XL superclasses, with Holm-corrected significance testing (`audit/11_significance.py`). The method is not new; the measurement on this system is.

2. **We show a limb-electrode reversal passes every quality gate, changes up to 87% of diagnoses, and silently voids the conformal guarantee.**
   → **Novel `[inferred]`.** `src/electrodes.py` implements the three reversals as exact linear maps (`swap_ra_la`, `swap_ra_ll`, `swap_la_ll`, L56-80) and a physiology-rule detector (`detect()`, L119) using aVR polarity and lead-I inversion. Coupling reversal simulation to *conformal-guarantee invalidation* is not attributed to any cited source anywhere in the repo. **Novelty NOT independently verified against literature — needs author's literature check.**

3. **We show 113/114 test-fold atrial-fibrillation recordings receive a statistical guarantee for a disease the label space cannot express, and gate it with a validation-fitted R-R irregularity check.**
   → **Novel `[inferred]`** in the specific combination; the components are standard. R-R variability features (`cv`, `rmssd`, `pnn50`, `irr` in `src/scope.py::rr_features`) are textbook; the contribution claimed is *withholding the conformal claim* rather than adding a class. No citation is attached to this construction in the repo.

4. **We implement a template-grounded report generator with an automated sentence-level verification gate that refuses text asserting anything the model did not compute.**
   → **Engineering.** `src/report.py` (365 L) builds reports from structured findings; `src/verify.py` (265 L) checks them with a hallucinated-term list, directional within-clause negation detection (`_negated()`), and a `SCOPE_DISCLAIMERS` allow-list. Solid implementation, standard rule-based approach, no methodological novelty. Stating it plainly.

5. **We deliver the full pipeline — quality gate, calibration, PAC conformal triage, XAI, report, JSON API, React clinical UI.**
   → **Engineering.** Every element (temperature scaling — Guo et al. 2017; Grad-CAM; integrated gradients; 1-D ResNet with squeeze-excitation) is an existing technique applied conventionally.

**Conservative bottom line:** bullet 1 is *adapted*, bullets 4–5 are *engineering*. Bullets 2 and 3 are the only candidates for *novel*, and neither has been verified against the literature from inside this repo. Given the repo already records two retracted novelty claims (§3), the honest position for a paper is: **the audits and the guarantee-withholding mechanism are the contribution; the underlying methods are not new.**

---

## 6. Contribution → Evidence Traceability Table

| # | Contribution bullet | Implemented where | Evaluated where | Evidence status |
|---|---|---|---|---|
| 1 | Subgroup validity + Mondrian repair | `src/conformal.py` L133-202 (`_pac_order_statistic`, `_conformal_lower`, `_conformal_upper`); `audit/10_conditional_validity.py` | `audit/results/10_conditional_validity.txt` (23 cells, 9 violations, Mondrian 22/23 vs marginal 14/23); `audit/results/11_significance.txt` (2 significant after Holm) | ✅ Complete |
| 2 | Electrode reversal voids the guarantee | `src/electrodes.py` L56-80 (swaps), L119 (`detect`); `src/quality.py::assess` (wiring into gate) | `audit/results/12_electrode_reversal.txt`, `.json` | ⚠️ **Present but sample-size conflict** — see §18 / §25 |
| 3 | Out-of-scope disease receives a guarantee; rhythm gate withholds it | `src/scope.py` L75-145; `src/quality.py::_load_scope_threshold` L36; `checkpoints/scope.json` | `audit/results/13_out_of_scope.txt`, `.json` | ✅ Complete |
| 4 | Grounded report + verification gate | `src/report.py`, `src/verify.py` | `audit/results/03_report_audit.txt`; `audit/results/08_verify_fixes.txt` (26-check regression suite, per `README.md` L204) | ⚠️ Regression pass/fail counts are not summarised as a single line in the results file — see §18 |
| 5 | Full pipeline + API + UI | `src/pipeline.py`, `backend/server.py`, `frontend/src/` (10 files) | `audit/results/04_runtime_audit.txt`; **no automated frontend test exists** | ⚠️ **Bullet 5 has no UI-side evaluation evidence** |

**Paper-writing risks flagged now:**
- Bullet 5's React frontend has **zero test coverage** — no test file, no test runner, no CI (`find` over the tree returned no `test_*.py`, `conftest.py`, `pytest.ini`, `*.yml`, `*.yaml`, `Dockerfile`, `Makefile`, `setup.py`, or `pyproject.toml`).
- Bullet 2's headline numbers in `docs/PANEL_ANSWERS.md` were produced at a different sample size than the results file currently on disk.

---

## 7. Related Work / Prior Approaches Referenced

**FACT — the reference list appears verbatim in two places, `docs/PANEL_ANSWERS.md` L516-525 and `docs/CONTRIBUTION_FINAL.md` L184-193, and is identical in both:**

| # | Citation as written in repo | Where else referenced |
|---|---|---|
| 1 | *Conformal prediction for AMI risk on PTB-XL.* Ann Noninvasive Electrocardiol, 2025. doi:10.1111/anec.70099 | `docs/CONTRIBUTION_FINAL.md` L184 |
| 2 | Strodthoff N. et al. *Explaining deep learning for ECG analysis: building blocks for auditing and knowledge discovery.* Comput Biol Med, 2024 | `docs/PANEL_ANSWERS.md` L336 (explicit non-novelty disclaimer) |
| 3 | Angelopoulos A., Bates S., Fisch A., Lei L., Schuster T. *Conformal Risk Control.* arXiv:2208.02814, 2023 | `src/conformal.py` L54; `docs/RESEARCH_CONTRIBUTION.md` L133, L386 |
| 4 | Vovk V., Lindsay D., Nouretdinov I., Gammerman A. *Mondrian confidence machine.* 2003 | `audit/10_conditional_validity.py` L28 |
| 5 | Vovk V. *Conditional validity of inductive conformal predictors.* ACML, 2012 | `src/conformal.py` L146 |
| 6 | *Pitfalls of Conformal Predictions for Medical Image Classification.* arXiv:2506.18162, 2025 | — |
| 7 | Wagner P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Sci Data 7:154, 2020 | `data/DATASET_LICENSE.txt` (full citation with DOI 10.1038/s41597-020-0495-6) |
| 8 | Strodthoff N. et al. *Deep learning for ECG analysis: benchmarks and insights from PTB-XL.* IEEE JBHI, 2021 | — |
| 9 | Guo C. et al. *On calibration of modern neural networks.* ICML, 2017 | `src/calibration.py` `[inferred]` — method matches, explicit inline citation not confirmed |
| 10 | Adebayo J. et al. *Sanity checks for saliency maps.* NeurIPS, 2018 | `audit/legacy/05_xai_audit.py` L182 |

**Additional citation in the dataset licence** [`data/DATASET_LICENSE.txt`]:
- Goldberger A., Amaral L., Glass L., Hausdorff J., Ivanov P. C., Mark R., … Stanley H. E. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23), e215-e220.

**Comparison table of approaches (as positioned in `docs/CONTRIBUTION_FINAL.md` L16, L69):**

| Approach | Key idea | Limitation as stated in repo |
|---|---|---|
| Marginal split conformal | Threshold = ⌊α(n+1)⌋-th order statistic; miss rate bounded *in expectation over calibration draws* | `src/conformal.py` L141-145: "violated on this single test realisation for CD (0.122 vs alpha 0.10) and HYP (0.174 vs 0.15)" |
| PAC / training-conditional conformal (Vovk 2012) | Coverage of k-th order statistic is Beta(k, n−k+1); pick largest k with P(Beta ≤ α) ≥ 1−δ | `src/conformal.py` L141: "strictly more conservative"; costs calibration positives |
| Mondrian / group-conditional (Vovk et al. 2003) | One threshold per subgroup | `audit/results/10_conditional_validity.txt`: needs enough positives per cell — STTC/age<50 (n=42) yields threshold `-inf` |
| Conformal Risk Control (Angelopoulos 2023) | Bound a general monotone risk, not just miss rate | `docs/CONTRIBUTION_FINAL.md` L16 labels it "Method I use" |

**Incomplete citations that will fail a reference check** (items 1 and 6 have no author list; item 4 has no venue): **needs input from author.**

---

## 8. Domain-Specific Structuring Fit

**Best fit: Motivating-Example → Generalize.**

**Why (FACT):** all three contribution scripts open with a concrete, named failure case before presenting any general mechanism.
- `audit/13_out_of_scope.py` L4-6 opens with a quoted failure statement: *"A five-class ECG model attaches a statistical guarantee to recordings whose actual disease it has no output for. The guarantee certifies the wrong answer, and nothing in the system knows."*
- `src/conformal.py` L143-145 motivates the PAC bound with two named concrete violations (CD 0.122 vs 0.10; HYP 0.174 vs 0.15) before defining the order statistic.
- `src/preprocess.py` L5 opens by naming a specific prior defect: "E-5 the archive resampled ANY length to 5000 samples with no fs check".

**Secondary fit: Threat-model-driven** — the three audits read as an adversarial/stressor model over the guarantee:

| Element | As found in code |
|---|---|
| What the "attacker" (failure source) can do | Present a patient from an under-represented subgroup; miswire two limb electrodes; present a disease outside the label space |
| What it is assumed **not** able to do | Corrupt the signal beyond the quality gate's detection (`src/quality.py::assess` refuses those); alter the calibration fold |
| What the defence guarantees | `checkpoints/conformal_triage.json`: per-class miss-rate bound α at PAC confidence δ = 0.01, **conditional on** correct electrode placement and in-scope rhythm; guarantees are *withheld*, not corrected, when either assumption fails [`README.md` L100-113] |

**Extra content this implies should be captured:** the assumption list above belongs in the paper explicitly — currently it is distributed across `README.md` L100-113, `src/electrodes.py`, and `src/scope.py::LIMITATION` rather than stated once.

**Poor fits:** *Build-up Ablation* — no ablation study exists (§16). *Benchmark-driven* — only one dataset is used. *Theorem-Proof* — the PAC bound is cited, not proved, in this repo.

---

## 9. Method / Design

### 9.1 Module map (data flow, diagram-ready)

`README.md` L148-153 gives the flow as ASCII:

```
quality gate ──► preprocess ──► classify ──► calibrate ──► conformal triage
  │        │                                                      │
  │   electrode + scope checks                                    ▼
  │   (withhold guarantees)              XAI ──► report ──► verify
REFUSED if uninterpretable
```

| Module | LOC | Responsibility (`README.md` L166-176) | Public surface |
|---|---|---|---|
| `src/models.py` | 238 | Architectures (single definition) | `ResidualBlock`, `ECGResNet`, `SEBlock`, `SEResidualBlock`, `MultiKernelStem`, `AttentionPool`, `ECGResNetSE`, `FocalLoss`, `build_model()` |
| `src/paths.py` | 78 | Asset resolution | `find`, `require`, `signals_cache`, `describe` |
| `src/signals.py` | 70 | Reads WFDB or cache | `raw_signals_dir`, `available`, `source_description`, `load` |
| `src/quality.py` | 314 | Flatline / noise / unit / duration / rhythm gate | `QualityReport`, `detect_and_fix_units`, `detect_r_peaks`, `assess` |
| `src/electrodes.py` | 183 | Limb-electrode reversal detection | `swap_ra_la`, `swap_ra_ll`, `swap_la_ll`, `ElectrodeReport`, `detect` |
| `src/scope.py` | 145 | Is this disease inside the label space? | `ScopeReport`, `rr_features`, `load_threshold`, `assess` |
| `src/preprocess.py` | 139 | Band-pass, resample, normalise | `resample_to`, `bandpass`, `center_or_pad`, `normalise`, `load_norm_stats`, `prepare`, `augment` |
| `src/calibration.py` | 129 | Per-class temperature scaling | `TemperatureCalibrator`, `expected_calibration_error`, `calibration_report` |
| `src/conformal.py` | 360 | Risk-controlled triage | `ClassThresholds`, `_pac_order_statistic`, `_conformal_lower`, `_conformal_upper`, `ConformalTriage`, `risk_coverage_curve` |
| `src/xai.py` | 297 | Thread-safe Grad-CAM, signed IG, territory mapping | `Explanation`, `_model_lock`, `grad_cam`, `integrated_gradients`, `lead_attributions`, `localise` |
| `src/report.py` | 365 | Grounded report | `Finding`, `ClinicalReport`, `build_report` (+ 5 private builders) |
| `src/verify.py` | 265 | Safety gate | `VerificationResult`, `verify_report`, `verify_paraphrase`, `safe_paraphrase`, `batch_verify`, `asserted_classes` |
| `src/pipeline.py` | 197 | Single inference entry point | `AnalysisResult`, `ECGPipeline` |
| `src/__init__.py` | 23 | package init | — |
| **Total** | **2,803** | | |

### 9.2 Key algorithms

**(a) PAC (training-conditional) conformal threshold** — `src/conformal.py` L133-158. Original selection logic; the statistical result is cited to Vovk (2012).

```
_pac_order_statistic(n, alpha, delta):
    # coverage of the k-th order statistic ~ Beta(k, n-k+1)
    best = 0
    for k in 1 .. n:
        if BetaCDF(alpha; k, n-k+1) >= 1 - delta:
            best = k
        else:
            break                      # CDF is monotone decreasing in k
    return best
    # fallback if scipy unavailable: floor(alpha * (n+1))  [marginal bound]
```

- `_conformal_lower(scores_pos, alpha, delta)` → threshold = `sort(scores_pos)[m-1]`, i.e. the largest threshold whose miss rate on calibration positives is bounded by α. Returns `(-inf, False, reason)` and declares the class *never rule-out-able* when `m < 1`, reporting the number of calibration positives needed: `ceil(log δ / log(1−α))` under PAC, `ceil(1/α) − 1` under the marginal bound [L176-181].
- `_conformal_upper(scores_neg, beta, delta)` → symmetric, `sort(scores_neg)[::-1][m-1]`.
- Three-zone decision: below `lambda_out` → **RULE_OUT**, above `lambda_in` → **RULE_IN**, between → **REFER** [`src/conformal.py`, surfaced in `frontend/src/components/Interpretation.jsx` L77-83].

**(b) Signal-quality gate** — `src/quality.py::assess`. Order is load-bearing and documented: shape → duration → finite → flat-lead detection (relative criterion, deliberately **before** unit inference) → unit/gain detection → amplitude → noise → rhythm → electrode check → scope check → SQI. Original ordering logic.

**(c) Electrode-reversal detection** — `src/electrodes.py`. The three swaps are exact linear maps of Einthoven/Goldberger relations (original derivation in this repo, standard physiology). `detect()` L119 uses net deflection (`_net_deflection` L102) and dominant polarity (`_dominant_polarity` L112) of aVR and lead I.

**(d) Rhythm-scope check** — `src/scope.py::rr_features` computes `cv`, `rmssd`, `pnn50`, `irr` from R-peak intervals; `assess()` compares `irr` against a threshold loaded from `checkpoints/scope.json` (`DEFAULT_IRR_THRESHOLD = 0.179`). Original composition of standard HRV features.

**(e) Report verification** — `src/verify.py`. `_clause()` L74 and `_negated()` L83 implement *directional within-clause* negation detection so that "MI is ruled out" and "MI is present" are distinguished; `_hallucinated_terms()` L110 rejects any clinical term the model cannot compute, with a `SCOPE_DISCLAIMERS` allow-list so that legitimate limitation sentences (which necessarily name out-of-scope conditions such as atrial fibrillation) are not themselves flagged. Original rule set.

### 9.3 Novel vs. standard/reused — function by function

| Code | Verdict |
|---|---|
| `src/conformal.py::_pac_order_statistic`, `_conformal_lower/_upper`, `ConformalTriage` | **Original implementation of a cited result** (Vovk 2012). Uses `scipy.stats.beta` only. |
| `src/conformal.py::risk_coverage_curve` | Standard risk–coverage analysis, conventional |
| `src/electrodes.py` (all) | **Original** — the linear swap maps and the aVR/lead-I rule are written from physiology, not wrapped from a library |
| `src/scope.py::rr_features` | **Standard HRV features** (cv/rmssd/pnn50), original assembly into an `irr` score |
| `src/quality.py::detect_and_fix_units`, `detect_r_peaks`, `assess` | **Original** heuristics; `detect_r_peaks` is a hand-written detector, not a library call |
| `src/preprocess.py` | **Standard** — wraps `scipy.signal.butter`, `iirnotch`, `filtfilt`, `resample` |
| `src/calibration.py::TemperatureCalibrator` | **Standard** — per-class temperature scaling (Guo et al. 2017), extended with a per-class bias term |
| `src/models.py::ECGResNet`, `SEBlock`, `ECGResNetSE` | **Standard architectures** (1-D ResNet + squeeze-excitation + attention pooling), assembled originally |
| `src/models.py::FocalLoss` | **Standard**; the docstring carries an original constraint — do not combine `alpha` with a balanced sampler |
| `src/xai.py::grad_cam`, `integrated_gradients` | **Standard methods**, hand-implemented (hooks at L86-99); `steps=64`, signed attributions retained |
| `src/xai.py::_model_lock` (WeakKeyDictionary + RLock) | **Original** — thread-safety wrapper so Grad-CAM hooks are safe under Flask's threaded server |
| `src/xai.py::localise`, `TERRITORIES`, `TERRITORY_ARTERY` | **Original heuristic** — lead-group → coronary territory map. `README.md` L275: "a lead-group heuristic, not clinically validated" |
| `src/report.py` (all) | **Original** template/grounding logic |
| `src/verify.py` (all) | **Original** rule-based verification |
| `backend/server.py::render_ecg` L161 | **Original** — clinical ECG paper rendering (25 mm/s, 10 mm/mV, 3×4 + rhythm strip) on matplotlib |
| `frontend/src/**` | **Standard** React/Vite/Tailwind; layout design original |

### 9.4 Notation table

| Symbol | Meaning | Defined in |
|---|---|---|
| α (`alpha`) | Per-class **miss-rate** budget for the rule-out decision — max allowed fraction of true positives ruled out | `src/conformal.py` L163; `checkpoints/conformal_triage.json` |
| β (`beta`) | Per-class **false-alarm** budget for the rule-in decision — max allowed fraction of true negatives ruled in | `src/conformal.py` L189 |
| δ (`delta`) | PAC confidence: the α-bound holds with probability ≥ 1−δ **over the calibration draw** | `src/conformal.py` L133-136 |
| n | Number of calibration examples (positives for λ_out, negatives for λ_in) | `src/conformal.py` L170, L190 |
| k, m | Order-statistic index selected by the PAC / marginal rule | `src/conformal.py` L152-158, L173 |
| λ_out (`lambda_out`) | Lower threshold — below it the class is *ruled out* | `ClassThresholds` L122 |
| λ_in (`lambda_in`) | Upper threshold — above it the class is *ruled in* | `ClassThresholds` L123 |
| T_c | Per-class temperature (5 values) | `checkpoints/calibrator.json` |
| b_c | Per-class bias added after temperature scaling | `checkpoints/calibrator.json` |
| `irr` | R-R irregularity score; > threshold ⇒ rhythm judged out of scope | `src/scope.py::rr_features` |
| SQI | Signal Quality Index, 0–1 | `src/quality.py::QualityReport` |
| Beta(k, n−k+1) | Distribution of the k-th order statistic's coverage | `src/conformal.py` L138 |

---

## 10. Algorithmic Complexity Analysis

**Applicable** — there is original algorithmic logic.

**(a) `_pac_order_statistic(n, α, δ)`** — `src/conformal.py` L152-158
- A single `for k in range(1, n+1)` loop with an early `break` on the first failing k. Each iteration is one `scipy.stats.beta.cdf` evaluation, O(1) amortised.
- **Worst case:** O(n) — when the loop runs to completion (α very loose, every k satisfies the condition).
- **Best/typical case:** O(k\*) where k\* is the returned index, because the loop breaks at k\*+1. Since k\* ≈ α·n for small δ, typical cost is **O(αn)**, i.e. sub-linear in n for the small α values shipped (α ∈ [0.05, 0.20]).
- **Space:** O(1).
- `[inferred]` — the early `break` is valid only if the Beta CDF at fixed α is monotone decreasing in k; the code assumes this without asserting it. Worth stating as an assumption in the paper.

**(b) `_conformal_lower` / `_conformal_upper`** — L161-202
- Dominated by `np.sort` on the calibration scores: **O(n log n)** time, **O(n)** space. The order-statistic search adds O(αn). Total **O(n log n)**.
- Called once per class at fit time, so total fit cost is **O(C · n log n)** for C = 5 classes.
- **Inference cost is O(1) per class** — the fitted λ_out/λ_in are two float comparisons. This is the design property that makes the guarantee free at serve time.

**(c) `ConformalTriage` at inference** — O(C) = O(5), constant.

**(d) `src/quality.py::detect_r_peaks`** — a single pass over the 5,000-sample lead-II vector plus a threshold/refractory sweep: **O(L)** time, O(L) space, L = 5,000.

**(e) `src/scope.py::rr_features`** — O(P) over P R-peaks (typically 8–20 for a 10 s strip). Negligible.

**(f) `src/preprocess.py::bandpass`** — three zero-phase `filtfilt` passes over a 5000×12 array: **O(L·C_leads)** = O(60,000) per record, constant memory in the signal length.

**(g) `src/xai.py::integrated_gradients`** — `steps=64` forward+backward passes: **O(steps · F)** where F is one network forward cost. This is the runtime bottleneck at serve time; `README.md` L277-278 records "~6 s per analysis (plotting and integrated gradients dominate; the classifier itself is ~20 ms)."

**(h) Model forward pass** — `ECGResNetSE`, 1,584,326 parameters, 4 stages. Per-record forward is O(Σ_stages C_in·C_out·k·L_stage); measured at ~20 ms CPU [`README.md` L278].

**(i) `audit/10_conditional_validity.py`** — refits thresholds for G subgroups × C classes: **O(G·C·n log n)**. With G = 5 (male, female, <50, 50-69, ≥70) and C = 5, that is 23 populated cells (2 cells fall below the n≥15 floor).

---

## 11. Experimental Setup

### Hardware

| Item | Value | Source |
|---|---|---|
| Training accelerator | NVIDIA **L4** GPU on Google Colab | `docs/COLAB_GUIDE.md` L1: "Colab L4 — Exact Steps, Zero Wasted Compute Units" |
| Inference | CPU sufficient | `README.md` L44-45: "No GPU needed for inference" |
| Development OS | Windows 11 Home Single Language 10.0.26200 `[inferred from session environment, not from a repo file]` | — |
| Exact CPU / RAM / VRAM figures | **NOT FOUND IN CODEBASE — needs input from author** | no config records them |

### Software

**Python** — `requirements.txt` (verbatim):

| Package | Constraint |
|---|---|
| torch | `>=2.1` |
| numpy | `>=1.24` |
| scipy | `>=1.10` |
| pandas | `>=2.0` |
| scikit-learn | `>=1.3` |
| flask | `>=3.0` |
| flask-cors | `>=4.0` |
| werkzeug | `>=3.0` |
| wfdb | `>=4.1` |
| matplotlib | `>=3.7` |
| tqdm | `>=4.65` |
| psutil | `>=5.9` |

- `requirements.txt` carries an inline note that **flask 2.3.2 is incompatible with werkzeug 3.1.3**, which is why flask is floored at `>=3.0`.
- Commented-out optional dependencies (not installed by default): `transformers>=4.40`, `rouge-score>=0.1.2`, `nltk>=3.8`.
- ⚠️ **All constraints are lower bounds (`>=`), none are pinned.** There is no lockfile, no `pyproject.toml`, no `environment.yml`. Exact resolved versions used for the reported results are **NOT FOUND IN CODEBASE — needs input from author** (this is a reproducibility risk worth recording now via `pip freeze`).

**JavaScript** — `frontend/package.json`:

| Package | Version | Scope |
|---|---|---|
| react | `^19.0.0` | runtime |
| react-dom | `^19.0.0` | runtime |
| vite | `^6.0.7` | dev |
| tailwindcss | `^4.0.0` | dev |
| @tailwindcss/vite | `^4.0.0` | dev |
| @vitejs/plugin-react | `^4.3.4` | dev |

- `README.md` L44-45: "Python 3.10+, Node 18+ (Vite 6 is pinned for Node 20.16 compatibility)."

### Datasets

| Property | Value | Source |
|---|---|---|
| Name | **PTB-XL** v1.0.3 | `data/DATASET_LICENSE.txt` |
| Source URL | https://physionet.org/content/ptb-xl/1.0.3/ | `data/DATASET_LICENSE.txt` |
| Licence | **CC-BY 4.0** — "permits redistribution with attribution" | `data/DATASET_LICENSE.txt` |
| Format | WFDB (`.dat` + `.hea`), 12-lead, 500 Hz, 10 s | `data/DATASET_LICENSE.txt`; `README.md` |
| Included here | `raw_signals/records500/` — 17,221 records at 500 Hz | `data/DATASET_LICENSE.txt` |
| Official size | 21,799 records / 18,869 patients | `analysis/results/01_dataset_deep_audit.txt` |
| Used here | **17,221 records / 15,174 patients** | `analysis/results/01_dataset_deep_audit.txt` |
| Dropped | 4,578 records (**21.0%**) | `analysis/results/01_dataset_deep_audit.txt` |
| Drop criterion | Only SCP codes with `likelihood == 100` retained | `README.md` L271 |

**Split (patient-disjoint, official `strat_fold`):**

| Split | Fold(s) | Records | Unique patients | Source |
|---|---|---|---|---|
| train | 1–8 | **13,801** | 12,109 | `csv/train.csv` (counted) |
| val | 9 | **1,709** | 1,550 | `csv/val.csv` (counted) |
| test | 10 | **1,711** | 1,515 | `csv/test.csv` (counted) |
| **Total** | | **17,221** | | matches the audit figure |

⚠️ **Conflict to flag:** `checkpoints/scope.json` records `n_val: 1696`, while `csv/val.csv` has **1,709** rows. `[inferred]` the 13-record difference is records where `rr_features` returned `None` (fewer than the minimum R-peaks) — `audit/13_out_of_scope.py` L173-177 skips those. Author should confirm.

**Label construction (FACT):**
- `csv/scp_to_superclass_mapping.json` contains **44 entries** mapping PTB-XL SCP-ECG statement codes to the five superclasses: **MI 14 codes, STTC 13, CD 11, HYP 5, NORM 1**.
- The 5 classes are therefore **derived**, not native columns in PTB-XL: PTB-XL ships per-record SCP codes with likelihoods; this component filters to `likelihood == 100` and maps through the JSON above.
- Multi-label: the task is per-class binary (each record may carry more than one superclass) — evidenced by per-class TP/FP/FN/TN tables in `checkpoints/operating_point.json` summing to 1,711 per class independently.

**Preprocessing chain** — `src/preprocess.py`:

| Step | Parameters | Line |
|---|---|---|
| Resample | to `SAMPLING_RATE` = 500 Hz, `scipy.signal.resample` | L36-41 |
| High-pass | Butterworth **order 3**, `HP_HZ = 0.5` Hz, zero-phase `filtfilt` | L30, L55-56 |
| Low-pass | Butterworth **order 4**, `LP_HZ = 40.0` Hz, zero-phase `filtfilt` | L31, L58-59 |
| Notch | IIR notch, `NOTCH_HZ = 50.0` Hz, `NOTCH_Q = 30.0` — "PTB-XL was recorded in Germany -> 50 Hz mains" | L32-33, L61-62 |
| Length | `center_or_pad` to `SIGNAL_LENGTH` (5,000 samples = 10 s @ 500 Hz), symmetric constant pad | L66-80 |
| Normalise | per-lead mean/std from `csv/norm_stats.json` | L83, L97 |
| Augment (train only) | `augment(x, rng)` | L116 |

### Environment

| Item | Value | Source |
|---|---|---|
| Containerisation | **None** — no Dockerfile anywhere in the tree | `find` over full tree |
| CI | **None** — no `.github/workflows`, no `*.yml`/`*.yaml` | `find` over full tree |
| Build/packaging | **None** — no `setup.py`, `pyproject.toml`, `Makefile` | `find` over full tree |
| Env vars (runtime) | `HOST`, `PORT`, `CORS_ORIGINS`, `ECG_CKPT`, `ECG_MODEL`, `ECG_FILTER` — "Defaults are correct for the shipped model" | `README.md` L115-116 |
| Env vars (data) | `$ECG_DATA_DIR` — first entry in the asset-resolution order | `README.md` L241-242 |
| Env var (escape hatch) | `ECG_NO_PACKED=1` — bypasses the preprocessing-mismatch guard in `train/fit_calibration.py` | `train/fit_calibration.py` |
| Env vars (data download) | `PHYSIONET_USER`, `PHYSIONET_PASS` | `.env.example` at repo root |
| `.env` handling | `.gitignore` excludes `.env`, `.env.*`, `*credential*`, `*secret*`, `*.pem`, `*.key`; whitelists `!.env.example` | `.gitignore` |
| Asset resolution order | `$ECG_DATA_DIR` → `csv/` → `data/` → `../_archive/data/` | `README.md` L241-242; `src/paths.py::_candidates` L24 |

### Compute Cost

| Item | Value | Source |
|---|---|---|
| Training wall-clock budget (CLI default) | `--max-minutes` **75.0** | `train/train_gpu.py` L294-295 |
| Training wall-clock budget (actually used for the shipped model) | `max_minutes: 60.0` | `checkpoints/best_model.pt` → `args` |
| Epochs configured | 40 | `checkpoints/best_model.pt` → `args`; `train/train_gpu.py` default |
| Epoch at which the shipped checkpoint was saved | **13** | `checkpoints/best_model.pt` → `epoch` |
| Elapsed-time logging | Implemented: `elapsed = (time.time() - t0) / 60`, printed per epoch and written as `"minutes"` to a history JSON | `train/train_gpu.py` L455-459, L520 |
| Actual training wall-clock for the shipped run | ⚠️ **NOT FOUND IN CODEBASE** — no `checkpoints/history_*.json` exists on disk; `.gitignore` whitelists `!checkpoints/history_*.json` but no such file is present | `ls checkpoints/` returns only `best_model.pt`, `calibrator.json`, `conformal_triage.json`, `operating_point.json`, `scope.json` |
| Colab compute-unit cost | **NOT FOUND IN CODEBASE — needs input from author** | — |
| Inference latency | ~6 s per analysis end-to-end; classifier alone ~20 ms; "plotting and integrated gradients dominate" | `README.md` L277-278 |
| Memory ceiling handling | OOM recovery implemented — batch halving with scheduler rebuild, plus a comment at L408: "Recover instead of dying 38 minutes into a paid session" | `train/train_gpu.py` L408-413 |
| Data-transfer optimisation | Packing 17,221 files into "3 big files, 2.1 GB total" because "Colab reading 17,221 small files *from Drive* is so slow it would dominate training time — you would pay GPU rates to wait on network I/O" | `docs/COLAB_GUIDE.md` L13-17 |
| Disk footprint (**measured**, `du -sh`) | **Total `Component_02/` = 4.1 GB**, of which: `data/raw_signals` **2.0 GB**; packed memmaps `train_X.npy` **1.6 GB** + `val_X.npy` **196 MB** + `test_X.npy` **196 MB** ≈ **2.0 GB**; label/id arrays < 500 KB combined. ⚠️ `README.md` L237-239 claims total ~2.0 GB and `data/` = 1.93 GB — **both understate reality by ~2×** because the packed training arrays are not counted (see §25 C8) |

---

## 12. Parameters / Configuration

### 12.1 Training — `train/train_gpu.py` (argparse)

| Flag | Default | Value in shipped checkpoint |
|---|---|---|
| `--pack` | `False` | `False` |
| `--force-pack` | `False` | `False` |
| `--no-filter` | `False` | `False` (filtering ON) |
| `--model` | `resnet_se` | `resnet_se` |
| `--epochs` | `40` | `40` |
| `--batch` | `128` | `128` |
| `--lr` | `3e-3` | `0.003` |
| `--wd` | `1e-2` | `0.01` |
| `--seed` | `0` | `0` |
| `--workers` | `2` | `2` |
| `--patience` | `0` | `0` |
| `--focal-alpha` | `False` | `False` |
| `--resume` | `False` | `False` |
| `--max-minutes` | **`75.0`** | **`60.0`** ⚠️ conflict |

Other training mechanisms present in `train/train_gpu.py`: OneCycleLR (rebuilt on OOM batch-halving via `make_sched()`), AMP/bfloat16, EMA (`ema: False` in the shipped checkpoint), balanced sampler XOR focal-α (never both — enforced by the `FocalLoss` docstring at `src/models.py` L202), per-worker RNG seeding via `_worker_init()`, NaN guard.

### 12.2 Calibration & conformal — `train/fit_calibration.py` (argparse)

| Flag | Default |
|---|---|
| `--ckpt` | (required/derived) |
| `--model` | `resnet_se` |
| `--filter` | (flag) |
| `--batch` | `64` |
| `--delta` | **`0.05`** |
| `--reuse-logits` / `--from-logits` | (flags) |
| `--seed` | `0` |
| `--preset` | `safety` |

⚠️ **Conflict:** `--delta` default is `0.05` here, `ConformalTriage.__init__` default is `delta=0.05` (`src/conformal.py` L211), but the **shipped `checkpoints/conformal_triage.json` records `delta: 0.01`**, and `audit/11_significance.py` / `audit/12_electrode_reversal.py` both default to `--delta 0.01`. The shipped artifact is the authoritative value; the code defaults were not updated to match.

### 12.3 Shipped conformal configuration — `checkpoints/conformal_triage.json`

`delta = 0.01`

| Class | α (miss budget) | β (false-alarm budget) | λ_out | λ_in | n_pos_cal | n_neg_cal | feasible |
|---|---|---|---|---|---|---|---|
| NORM | 0.20 | 0.20 | 0.1986 | 0.6944 | 699 | 1,010 | ✅ |
| MI | 0.05 | 0.10 | 0.0382 | 0.3386 | 283 | 1,426 | ✅ |
| STTC | 0.10 | 0.15 | 0.1934 | 0.3435 | 455 | 1,254 | ✅ |
| CD | 0.10 | 0.15 | 0.1104 | 0.2793 | 485 | 1,224 | ✅ |
| HYP | 0.15 | 0.15 | 0.0506 | 0.1391 | 134 | 1,575 | ✅ |

Presets available: `safety`, `balanced`, `throughput` (`PRESETS` in `src/conformal.py`); `safety` is the `fit_calibration.py` default.

### 12.4 Shipped calibrator — `checkpoints/calibrator.json`

| Class | Temperature T_c | Bias b_c |
|---|---|---|
| NORM | 0.42794867430858335 | 0.3027973414939085 |
| MI | 0.5938682747415797 | −1.0543560122899927 |
| STTC | 0.5478539203741215 | 0.2044651091719739 |
| CD | 0.5593287249730048 | 0.21812323473808823 |
| HYP | 0.6271006286330354 | −0.950844098546527 |

Provenance block: `fitted_for: {model: resnet_se, ckpt: best_model.pt, filter: true, seed: 0}`. `backend/server.py` refuses to start on a provenance mismatch.

### 12.5 Shipped operating point — `checkpoints/operating_point.json`

| Field | Value |
|---|---|
| `policy` | `recall_first` |
| recall floor | `0.80` |
| report floor | `0.75` |
| `seed` | `0` |
| `fitted_on` | `validation fold 9` |

| Class | Decision threshold |
|---|---|
| NORM | 0.7240 |
| MI | 0.2560 |
| STTC | 0.3755 |
| CD | 0.3395 |
| HYP | 0.0925 |

### 12.6 Shipped scope gate — `checkpoints/scope.json`

| Field | Value |
|---|---|
| `irr_threshold` | 0.17920692444965153 |
| `fpr_budget` | 0.05 |
| `fitted_on` | `val fold 9` |
| `n_val` | 1,696 |

(`src/scope.py::DEFAULT_IRR_THRESHOLD = 0.179` is the hard-coded fallback if the JSON is absent.)

### 12.7 Signal-processing constants — `src/preprocess.py`

`SAMPLING_RATE = 500`, `SIGNAL_LENGTH = 5000`, `HP_HZ = 0.5`, `LP_HZ = 40.0`, `NOTCH_HZ = 50.0`, `NOTCH_Q = 30.0`, Butterworth orders 3 (HP) / 4 (LP).

### 12.8 Analysis & audit script parameters

| Script | Flags & defaults |
|---|---|
| `analysis/01_dataset_deep_audit.py` | `--n-signals 1500` |
| `analysis/02_operating_point.py` | `--floor 0.80`, `--report-floor 0.75`, `--seed 0` |
| `audit/11_significance.py` | `--boot 2000`, `--delta 0.01` |
| `audit/12_electrode_reversal.py` | `--n 600`, `--delta 0.01` |
| `audit/13_out_of_scope.py` | `--n 700`, `--fpr 0.05` |
| `train/preflight.py` | `--batch 128`, `--model resnet_se` |

### 12.9 Model architecture parameters — `src/models.py`

| | `ECGResNet` (baseline) | `ECGResNetSE` (shipped) |
|---|---|---|
| Parameters | **1,018,501** | **1,584,326** |
| Channels | [64, 128, 192, 256] | (64, 128, 256, 320) |
| Kernels | [15, 7, 5, 3] | MultiKernelStem 7 / 15 / 31 |
| Blocks | `ResidualBlock` | `SEResidualBlock` (squeeze-excitation) |
| Pooling | (standard) | `AttentionPool` |

### 12.10 XAI parameters — `src/xai.py`

| Parameter | Value |
|---|---|
| Integrated-gradients steps | `steps = 64` (module default) |
| Attribution sign | **signed** (retained, not absolute) |
| Thread safety | `WeakKeyDictionary` + per-model `RLock` |

⚠️ **Conflict:** `audit/results/05_xai_audit.txt` states "(app.py ships steps=30)" and compares 30 vs 200 steps. The current `src/xai.py` default is `steps=64`, and no `app.py` exists in `Component_02/` (the legacy `app.py` is preserved at `reference/legacy_docs/legacy_app.py`). The XAI audit numbers were produced against the **superseded** system.

### 12.11 Backend parameters — `backend/server.py`

| Parameter | Value | Line |
|---|---|---|
| `MAX_CONTENT_LENGTH` | `MAX_UPLOAD_MB * 1024 * 1024` | L69 |
| Startup validation | `_startup()` | L77 |
| ECG rendering | `render_ecg(signal_mv, cam, r_peaks, dark=False)` — 25 mm/s, 10 mm/mV, 3×4 layout + lead II rhythm strip, calibration pulse | L161 |
| Dark theme | `?theme=dark` query param | `README.md` L62 |

---

## 13. Baseline(s) Compared Against

**FACT — three distinct baselines exist in the repo.**

| Baseline | What it is | Where compared | Result |
|---|---|---|---|
| **B1 — `ECGResNet` (plain 1-D ResNet, 1,018,501 params)** | Architectural baseline for `ECGResNetSE` | `audit/results/02_model_audit.txt` vs `docs/RESEARCH_CONTRIBUTION.md` L277-283 | macro-AUROC 0.9297, macro-AUPRC 0.7864, macro-F1 0.7172 [0.6971, 0.7365] |
| **B2 — marginal split conformal** | The standard conformal threshold, ⌊α(n+1)⌋ | `src/conformal.py` L141-145; `audit/10_conditional_validity.py` section A | Violated on the single test realisation: CD 0.122 vs α 0.10; HYP 0.174 vs α 0.15 |
| **B3 — random attribution** | Null model for XAI faithfulness | `audit/results/05_xai_audit.txt` | Grad-CAM deletion AUC 0.3933 vs random 0.5368; insertion 0.5751 vs 0.5039 |
| **B4 — F1-optimal thresholding** | Alternative operating-point policy | `analysis/results/02_operating_point.txt` sections B vs C | Recall-first trades precision for recall — see §18 |
| **B5 — the superseded ("archive") system** | The previous implementation | `audit/legacy/` (6 scripts); `reference/checkpoints_ecg_only/`, `reference/checkpoints_fusion_leaked/`; `docs/AUDIT_FINDINGS.md` (12 defects) | See §18 leakage results |

**Flags:**
- ⚠️ **No published-benchmark comparison is run in code.** `docs/PANEL_ANSWERS.md` L421 compares against published PTB-XL numbers ("Results above 0.94 use foundation models pretrained on…") but this is prose, not a computed comparison. `README.md` L272-273 explicitly warns: "Labels used only SCP codes with `likelihood == 100`, dropping 21% of PTB-XL. Results are **not directly comparable** to published benchmarks."
- ⚠️ **The B1 comparison is asymmetric.** `docs/RESEARCH_CONTRIBUTION.md` L311 admits this: the 3-seed SE model is compared "against a 1-seed point estimate. A fully symmetric comparison would need seeds for the [baseline]."
- ⚠️ **No comparison against a non-deep baseline** (logistic regression on ECG features, rule-based criteria) exists anywhere.

---

## 14. Evaluation Metrics

| Metric | Computed where | Why it fits | Answers |
|---|---|---|---|
| **Accuracy** | `analysis/02_operating_point.py` → `checkpoints/operating_point.json`, `analysis/results/02_operating_point.txt` | Multi-label binary decisions per class | RQ6 |
| **Recall (sensitivity)** | same | Primary metric — this is a *rule-out* system | RQ6 |
| **Specificity** | same | Complement of the false-alarm burden | RQ6 |
| **NPV** | same | The decision-relevant metric for rule-out: "if I say no, how often am I right?" | RQ6 |
| **Precision (PPV)** | same | Referral burden | RQ6 |
| **F1** | same | Reported for comparability, explicitly *not* the optimisation target | RQ6 |
| **Balanced accuracy (BAcc)** | `analysis/results/02_operating_point.txt` | Class-imbalance-robust summary | RQ6 |
| **TP / FP / FN / TN counts** | `checkpoints/operating_point.json` per class | Raw confusion counts so any derived metric can be recomputed | all |
| **macro-AUROC** | `audit/02_model_audit.py`; training loop (`best_auroc`) | Threshold-free discrimination | RQ6, model selection |
| **macro-AUPRC** | `audit/02_model_audit.py` | Better than AUROC under heavy class imbalance (HYP prevalence 7.7%) | RQ6 |
| **Expected Calibration Error (ECE)** | `src/calibration.py::expected_calibration_error` L99; `calibration_report` L113 | Whether probabilities mean what they say | calibration validity |
| **Brier score** | `src/calibration.py::calibration_report` `[inferred from the function's stated role]` | Proper scoring rule | calibration validity |
| **Empirical subgroup miss rate vs promised α** | `audit/10_conditional_validity.py` | Direct test of conditional validity | **RQ1, RQ2** |
| **Wilson score 95% CI** | `audit/11_significance.py` | Interval for a binomial rate with small n per cell | RQ1 |
| **Exact binomial test p-value** | `audit/11_significance.py` | Is the observed violation beyond chance? | RQ1 |
| **Holm–Bonferroni adjusted p** | `audit/11_significance.py` | 23 simultaneous cells ⇒ multiple-comparison control | RQ1 |
| **Calibration-draw bootstrap** (`--boot 2000`) | `audit/11_significance.py` | P(violate) over resampled calibration draws | RQ1 |
| **Diagnosis-change rate under reversal** | `audit/12_electrode_reversal.py` | Clinical impact of miswiring | **RQ3** |
| **Guarantee violations introduced** | `audit/12_electrode_reversal.py` | Direct measure of guarantee invalidation | **RQ3** |
| **Detector sensitivity / FPR** | `audit/12_electrode_reversal.py`, `audit/13_out_of_scope.py` | Operating characteristic of each gate | RQ3, RQ4 |
| **Rhythm-check AUROC** | `audit/13_out_of_scope.py` (via `sklearn.metrics.roc_auc_score`) | Threshold-free quality of the `irr` score | **RQ4** |
| **Deletion AUC / Insertion AUC** | `src/xai.py::deletion_insertion_auc`; `audit/05_xai_audit.py` | Standard XAI faithfulness measures | **RQ5** |
| **Spearman ρ (trained vs randomised head)** | `audit/05_xai_audit.py` | Adebayo et al. 2018 sanity check | **RQ5** |
| **IG completeness relative error** | `audit/05_xai_audit.py` | Axiomatic check on integrated gradients | RQ5 |
| **QRS alignment** | `src/xai.py::qrs_alignment` | Does attribution land on physiologically meaningful segments? | RQ5 |
| **Signal Quality Index (SQI)** | `src/quality.py` | Gate metric | pre-inference |
| **Bootstrap CI on macro-F1** | `audit/02_model_audit.py` | Uncertainty on the headline metric | RQ6 |
| **Optimism gap** | `audit/02_model_audit.py` | Val-vs-test degradation | overfitting check |
| **Inference latency** | `audit/04_runtime_audit.py`; `README.md` L277 | Systems metric — clinical usability | deployment |

**No metric found for:** energy consumption, throughput under concurrency, or memory ceiling at serve time.

---

## 15. Experimental Repetition & Statistical Robustness

**FACT — repetition is uneven across experiments.**

| Experiment | Repetitions | Seeds | Variance reported |
|---|---|---|---|
| Classifier training | **3 runs** | seeds 0, 1, 2 `[inferred — the docs say "3 seeds"; only seed 0's checkpoint exists on disk]` | macro-AUROC **0.9343 ± 0.0028**, macro-AUPRC **0.8001 ± 0.0029** [`README.md` L134; `docs/PANEL_ANSWERS.md` L28-29] |
| Individual seed AUROCs | 3 values recorded | — | 0.9320 / 0.9374 / 0.9335 [`docs/RESEARCH_CONTRIBUTION.md` L283] |
| Calibration fit | **1** | seed 0 | `checkpoints/calibrator.json` → `fitted_for.seed: 0` |
| Conformal threshold fit | **1** | seed 0 | `checkpoints/conformal_triage.json` |
| Operating-point selection | **1** | seed 0 | `checkpoints/operating_point.json` → `seed: 0` |
| Subgroup validity (RQ1/RQ2) | **1** | seed 0 | but with 2,000-sample calibration-draw bootstrap |
| Electrode reversal (RQ3) | **1** | — | `random_state` fixed |
| Out-of-scope (RQ4) | **1** | `random_state=0` in `test.sample()` [`audit/13_out_of_scope.py` L193] | — |
| XAI faithfulness (RQ5) | **1**, n = 20 records | — | Spearman reported, no CI |

**Formal significance testing found (FACT):**
- **Exact binomial test + Holm–Bonferroni correction** over 23 class×subgroup cells — `audit/11_significance.py`, results in `audit/results/11_significance.txt`.
  - "Cells tested: 23"; "Statistically significant after Holm (<0.05): **2**"
  - `CD / age=<50`: promised ≤ 0.10, observed 0.333, 95% CI [0.232, 0.453], exact p = 2.23e-07, **p_Holm = 5.14e-06**
  - `NORM / age=>=70`: promised ≤ 0.20, observed 0.330, 95% CI [0.247, 0.426], exact p = 1.32e-03, **p_Holm = 2.91e-02**
- **Wilson score 95% confidence intervals** for every subgroup miss rate — same file.
- **Calibration-draw bootstrap**, `--boot 2000`, reporting median miss, 2.5%, 97.5% and `P(violate)` per cell.
- **Bootstrap CI on macro-F1**: 0.7172 [0.6971, 0.7365] — `audit/results/02_model_audit.txt`.
- **Paired t-test on AUPRC across seeds**: "t=8.2, p≈0.015" — `docs/RESEARCH_CONTRIBUTION.md` L280.

**⚠️ Weaknesses to flag explicitly (this is what reviewers call out):**
1. **No results artifact backs the 3-seed numbers.** A repo-wide grep for `0.9343` finds it only in `docs/CONTRIBUTION_FINAL.md` L121, `docs/PANEL_ANSWERS.md` L28, `docs/RESEARCH_CONTRIBUTION.md` L279 and `README.md` L134 — all prose. There is **no `.json`/`.txt` results file** recording the three seed runs, and only one checkpoint (`seed: 0`) exists on disk. The numbers came from Colab console output that was never saved into the repo. **This must be re-run with output captured, or the claim removed.**
2. **Everything downstream of the classifier is single-seed.** Calibration, conformal thresholds, the operating point and all three contribution audits use seed 0 only. `docs/PANEL_ANSWERS.md` L483 acknowledges this: "*retraining* seeds is not [repeated], and is the next run", and L508 lists "Repeat the subgroup analysis across all 3 training seeds" as future work.
3. **The baseline (B1) is a 1-seed point estimate** compared against a 3-seed mean — acknowledged at `docs/RESEARCH_CONTRIBUTION.md` L311.
4. **The AUROC gain over baseline is not significant:** +0.0046 = 1.6σ, labelled in the repo's own table as "**within run-to-run noise**". Only AUPRC (+0.0137, 4.7σ) is labelled "**real**" [`docs/RESEARCH_CONTRIBUTION.md` L279-280]. **Do not claim an AUROC improvement in the paper.**
5. **RQ5 (XAI faithfulness) is n = 20** with no confidence interval.

---

## 16. Ablation Studies

**FACT — one genuine ablation exists; it is a leakage ablation of the superseded system, not of the current one.**

`audit/results/06_leakage_audit.txt` runs a four-arm input ablation:

| Arm | Inputs | macro-AUROC |
|---|---|---|
| A | signal + demographics + **report text** | 0.9567 |
| B | text zeroed | 0.9072 |
| C | signal zeroed | 0.8904 |
| D | signal + demographics zeroed (text only) | 0.8872 |

Interpretation recorded in-file: "Remove the ECG, keep the report text: 0.8904 (93.1% of full)"; "Report text ALONE: 0.8872". This ablation is what identified the label-leaking fusion model, preserved as `reference/checkpoints_fusion_leaked/`.

**Also ablation-like (FACT):**
- **Operating-point policy ablation** — `analysis/results/02_operating_point.txt` compares three policies (A: default, B: F1-optimal, C: recall-first shipped) on the same test fold.
- **Marginal vs PAC vs Mondrian conformal** — `audit/10_conditional_validity.py` sections A and B are effectively an ablation of the calibration scheme.
- **IG step-count ablation** — 30 vs 200 steps, Spearman 0.999, top-1 lead agreement 19/20 [`audit/results/05_xai_audit.txt`] — but produced against the superseded system (§12.10).

**NOT present — ablations that a reviewer will ask for:**

| Missing ablation | How the current code could support it |
|---|---|
| Squeeze-excitation on/off | `src/models.py::build_model(name=...)` already exposes `resnet` vs `resnet_se`; run `train_gpu.py --model resnet` at the same seeds |
| Multi-kernel stem (7/15/31) vs single kernel | `MultiKernelStem` L140 — would need a new `build_model` variant |
| Attention pooling vs global average pooling | `AttentionPool` L155 — same |
| Band-pass filtering on/off | **Already exposed**: `train_gpu.py --no-filter` |
| Focal loss vs balanced sampler | **Already exposed**: `train_gpu.py --focal-alpha` |
| Temperature calibration on/off | `ECGPipeline.from_checkpoint(..., calibrator_path=None)` |
| PAC δ sensitivity (0.01 vs 0.05 vs 0.10) | **Already exposed**: `fit_calibration.py --delta` |
| Conformal preset (safety/balanced/throughput) | **Already exposed**: `fit_calibration.py --preset` |
| Rhythm-gate FPR budget sweep | **Already exposed**: `13_out_of_scope.py --fpr` |

**Flag:** the architecture that carries the headline result (`ECGResNetSE`) has **no component-wise ablation**. The `--model resnet` baseline is the only architectural comparison, and it is 1-seed (§15).

---

## 17. Existing Figures / Visual Assets Inventory

**FACT — there are no static figure assets in this component.**

A recursive `find` for `*.png`, `*.svg`, `*.pdf`, `*.jpg` over the entire tree (excluding `frontend/node_modules/`) returned **zero matches**.

**What exists instead (all generated at runtime, none saved to disk):**

| Asset | Generated by | Notes |
|---|---|---|
| Clinical ECG paper rendering (3×4 lead layout + lead II rhythm strip, 25 mm/s, 10 mm/mV, calibration pulse, Grad-CAM overlay, R-peak markers) | `backend/server.py::render_ecg` L161-268 | Returned as a **base64 PNG string** in the `ecgImage` JSON field [`README.md` L96]; never written to a file |
| Conformal threshold-position bar (per-class position between λ_out and λ_in) | `frontend/src/components/Interpretation.jsx` L47-67 | Live DOM, CSS-only |
| Signed lead-attribution bar chart | `frontend/src/components/Interpretation.jsx` L198-219 | Live DOM |
| ASCII pipeline diagram | `README.md` L148-153 | Text — reusable as a figure source |
| ASCII/Markdown result tables | `analysis/results/*.txt`, `audit/results/*.txt` | Text — reusable as table sources |

**Implication for the paper:** every figure must be created from scratch. The result `.json` files (`analysis/results/02_operating_point.json`, `audit/results/10–13*.json`) contain the underlying numbers in machine-readable form, so plotting scripts can be written against them without re-running any experiment. **Writing those plotting scripts is outstanding work.**

**Notebook outputs:** `train/Component02_Colab.ipynb` exists. Whether its cells contain saved output (the 3-seed console log in particular) is **NOT VERIFIED — worth the author checking**, since §15 flags that the 3-seed numbers have no other artifact.

---

## 18. Results Found in Repo (facts only — no interpretation)

### 18.1 Shipped classifier — test fold 10 (n = 1,711), recall-first operating point

Source: `README.md` L124-131, derived from `checkpoints/operating_point.json` → `test_metrics`.

| Class | Accuracy | Recall | Specificity | NPV | Precision | F1 |
|---|---|---|---|---|---|---|
| NORM | 0.883 | 0.796 | 0.943 | 0.868 | 0.908 | 0.849 |
| MI | 0.884 | 0.836 | 0.893 | 0.967 | 0.591 | 0.692 |
| STTC | 0.868 | 0.803 | 0.892 | 0.926 | 0.729 | 0.764 |
| CD | 0.869 | 0.805 | 0.894 | 0.921 | 0.750 | 0.776 |
| HYP | 0.817 | 0.811 | 0.818 | 0.981 | 0.271 | 0.406 |
| **MACRO** | **0.864** | **0.810** | **0.888** | **0.933** | **0.650** | **0.698** |

Raw confusion counts for NORM at this operating point (`checkpoints/operating_point.json`): tp = 563, fp = 57, fn = 144, tn = 947. For HYP: tp = 107, fp = 288, fn = 25, tn = 1,291.

`analysis/results/02_operating_point.txt`: "**RESULT: ALL CLASSES MEET BOTH TARGETS**" (≥ 0.75 accuracy and ≥ 0.75 recall).

### 18.2 Operating-point policy comparison — same test fold

**Policy A (default thresholds)** — `analysis/results/02_operating_point.txt` §A:

| Class | ACC | REC | SPEC | PREC | NPV | F1 | BAcc | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HYP | 0.939 | 0.311 | 0.992 | 0.759 | 0.945 | 0.441 | 0.651 | 41 | 13 | 91 | 1,566 |
| **MACRO** | **0.901** | **0.668** | **0.944** | **0.800** | **0.922** | **0.714** | **0.806** | | | | |

**Policy B (F1-optimal)** — §B:

| Class | ACC | REC | SPEC | PREC | NPV | F1 | BAcc | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NORM | 0.890 | 0.881 | 0.896 | 0.857 | 0.915 | 0.869 | 0.889 | 623 | 104 | 84 | 900 |
| MI | 0.905 | 0.731 | 0.938 | 0.685 | 0.949 | 0.708 | 0.834 | 196 | 90 | 72 | 1,353 |
| STTC | 0.864 | 0.818 | 0.881 | 0.715 | 0.930 | 0.763 | 0.850 | 373 | 149 | 83 | 1,106 |
| HYP | 0.902 | 0.652 | 0.923 | 0.415 | 0.969 | 0.507 | 0.787 | 86 | 121 | 46 | 1,458 |
| **MACRO** | **0.889** | **0.760** | **0.917** | **0.702** | **0.932** | **0.724** | **0.839** | | | | |

**Cost table (F1-optimal → recall-first)** — §3 of the same file:

| Class | F1 (F1-opt) | F1 (recall-first) | recall gain | precision cost |
|---|---|---|---|---|
| NORM | 0.869 | 0.849 | −0.085 | +0.051 |
| MI | 0.708 | 0.692 | +0.104 | −0.094 |
| STTC | 0.763 | 0.764 | −0.015 | +0.015 |
| CD | 0.775 | 0.776 | +0.087 | −0.091 |
| HYP | 0.507 | 0.406 | +0.159 | −0.145 |

In-file note: "hypertrophy cannot reach F1 0.75 at 7.7% prevalence — that needs AUPRC > 0.8 and the model achieves **0.584** (published norm ~0.54)."

### 18.3 Threshold-free discrimination

| Model | macro-AUROC | macro-AUPRC | macro-F1 | Source |
|---|---|---|---|---|
| Baseline `ECGResNet` (1 seed) | 0.9297 | 0.7864 | 0.7172 [0.6971, 0.7365] | `audit/results/02_model_audit.txt` |
| `ECGResNetSE` (3 seeds) | **0.9343 ± 0.0028** | **0.8001 ± 0.0029** | — | `README.md` L134 ⚠️ prose only, no artifact |
| Individual seeds (AUROC) | 0.9320 / 0.9374 / 0.9335 | — | — | `docs/RESEARCH_CONTRIBUTION.md` L283 ⚠️ prose only |
| Δ vs baseline | +0.0046 (1.6σ) — "within run-to-run noise" | +0.0137 (4.7σ) — "real (t=8.2, p≈0.015)" | — | `docs/RESEARCH_CONTRIBUTION.md` L279-280 |

Also recorded: **optimism gap +0.0106** (`audit/results/02_model_audit.txt`), and `checkpoints/best_model.pt` → `best_auroc = 0.940341655767476` at epoch 13 (**validation fold 9**, not test — do not confuse with the 0.9343 test figure).

### 18.4 Contribution 1 — conditional validity (RQ1/RQ2)

Source: `audit/results/10_conditional_validity.txt`. Configuration: PAC δ = 0.01, α = {NORM 0.20, MI 0.05, STTC 0.10, CD 0.10, HYP 0.15}. Thresholds fitted **marginally** on all of fold 9.

Marginal-threshold miss rates, per class, per subgroup (n in parentheses):

| Class | α | Overall | male | female | age <50 | age 50-69 | age ≥70 |
|---|---|---|---|---|---|---|---|
| NORM | 0.20 | 0.190 | 0.158 (373) | 0.225 (334) | 0.103 (301) | 0.228 (303) | **0.330 (103)** |
| MI | 0.05 | 0.015 | 0.020 (150) | 0.008 (118) | 0.000 (11) | 0.011 (87) | 0.019 (161) |
| STTC | 0.10 | 0.092 | 0.103 (203) | 0.083 (253) | 0.128 (39) | 0.080 (174) | 0.093 (227) |
| CD | 0.10 | 0.099 | 0.085 (260) | 0.117 (223) | **0.333 (66)** | 0.099 (161) | 0.042 (240) |
| HYP | 0.15 | 0.121 | 0.182 (66) | 0.061 (66) | 0.444 (9) | 0.159 (44) | 0.066 (76) |

- "Subgroup violations of the promised bound (n>=15): **9**"
- "Mondrian held in **22/23** class-group cells (**96%**) vs marginal **14/23**"

Mondrian (group-conditional) per-cell thresholds and hold status, excerpt:

| Class | Group | n | threshold | miss | held |
|---|---|---|---|---|---|
| NORM | sex=male | 358 | 0.6762 | 0.153 | ✅ |
| NORM | sex=female | 341 | 0.6605 | 0.207 | ❌ |
| NORM | age=<50 | 319 | 0.8214 | 0.156 | ✅ |
| NORM | age=50-69 | 300 | 0.5922 | 0.185 | ✅ |
| NORM | age=≥70 | 79 | 0.0765 | 0.019 | ✅ |
| MI | sex=male | 158 | 0.0428 | 0.020 | ✅ |
| MI | sex=female | 125 | 0.0080 | 0.000 | ✅ |
| MI | age=50-69 | 114 | 0.0080 | 0.011 | ✅ |
| MI | age=≥70 | 144 | 0.0113 | 0.000 | ✅ |
| STTC | sex=male | 213 | 0.0519 | 0.025 | ✅ |
| STTC | sex=female | 242 | 0.2215 | 0.083 | ✅ |
| STTC | age=<50 | 42 | **−inf** | 0.000 | ✅ |
| STTC | age=50-69 | 194 | 0.0943 | 0.046 | ✅ |
| STTC | age=≥70 | 209 | 0.1377 | 0.075 | ✅ |
| CD | sex=male | 259 | 0.1104 | 0.085 | ✅ |
| CD | sex=female | 226 | 0.0597 | 0.085 | ✅ |
| CD | age=<50 | 57 | 0.0137 | 0.061 | ✅ |
| CD | age=50-69 | 155 | 0.0637 | 0.075 | ✅ |
| CD | age=≥70 | 246 | 0.0784 | 0.037 | ✅ |
| HYP | sex=male | 81 | 0.0278 | 0.121 | ✅ |

(Remaining HYP cells truncated in the extract; the file records 23 cells total, 22 held.)

**Significance** — `audit/results/11_significance.txt`:
- Cells tested: **23**; significant after Holm (< 0.05): **2**
- `CD / age=<50`: promised ≤ 0.10, observed **0.333**, 95% CI [0.232, 0.453], exact p = **2.23e-07**, p_Holm = **5.14e-06**
- `NORM / age=>=70`: promised ≤ 0.20, observed **0.330**, 95% CI [0.247, 0.426], exact p = **1.32e-03**, p_Holm = **2.91e-02**
- In-file conclusion: "2 subgroup violation(s) survive BOTH a Holm-corrected exact test and a Wilson interval that clears the promised bound"

### 18.5 Contribution 2 — electrode reversal (RQ3)

Source: `audit/results/12_electrode_reversal.txt` header: "**200 test records** · model resnet_se · PAC delta 0.01".

**Quality gate passes the miswired signal** (`.json` → `gate`):

| Condition | Passed gate |
|---|---|
| correct placement | 198/200 |
| RA/LA reversal | 198/200 |
| RA/LL reversal | 198/200 |
| LA/LL reversal | 197/200 |

**Diagnostic impact:**

| Reversal | (col 1) 86–87% | (col 2) | (col 3) |
|---|---|---|---|
| RA/LA | 86.0% | 54.5% | 24.5% |
| RA/LL | 87.0% | 57.5% | 52.0% |
| LA/LL | 43.5% | 22.5% | 14.0% |

*(Column headers were not captured in the extract; the file's own summary line reads "Diagnoses changed by a cable swap: **up to 87% of patients**".)*

**Detector operating characteristic:**

| Condition | Value |
|---|---|
| correct placement | **4.5% false-positive rate** |
| RA/LA | **65.5% sensitivity** |
| RA/LL | **60.5% sensitivity** |
| LA/LL | **4.0% sensitivity** |

**"Guarantee violations introduced purely by electrode reversal: 7"**

⚠️ **CONFLICT — three-way, must be resolved before writing:**
1. `docs/PANEL_ANSWERS.md` L142-149 reports the same experiment at **n = 600**: "Measured on 600 test records: Correct placement 589/600; RA/LA reversal **587/600**; RA/LL **589/600**; LA/LL **589/600**."
2. `audit/results/12_electrode_reversal.txt` currently on disk is an **n = 200** run: 198/200, 198/200, 198/200, 197/200.
3. `README.md` L274-276 states detector sensitivity as "**~70%** on RA/LA, **~61%** on RA/LL, 4.5% false positives", while the on-disk results file says **65.5%** and **60.5%**.

The `--n 600` default in `audit/12_electrode_reversal.py` matches the docs; the on-disk results are from a smaller verification run that overwrote the file. **Re-run at `--n 600` and regenerate every quoted number, or restate the docs at n = 200.**

### 18.6 Contribution 3 — out-of-scope disease (RQ4)

Source: `audit/results/13_out_of_scope.txt`.

| Finding | Value |
|---|---|
| Out-of-scope disease in the dataset | **14.3%** of records (`README.md` L266; computed across 7 regex patterns in `audit/13_out_of_scope.py` L62-70) |
| AF / flutter recordings in the test fold | **114** |
| … reported as NORMAL | **2 / 114** |
| … carrying a statistical guarantee | **113 / 114** |
| Rhythm-check threshold (fitted on full fold 9 at 5% FPR) | irr = **0.1792** |
| Test sensitivity | **48.9%** |
| Test false-positive rate | **4.8%** |
| Test AUROC | **0.912** |

Out-of-scope patterns searched: atrial fibrillation, atrial flutter, paced rhythm, SV tachycardia, ventricular tachycardia, atrial ectopy, ventricular ectopy — matched against `report_en` in German and English [`audit/13_out_of_scope.py` L62-70].

**Methodological note recorded in the source** (`audit/13_out_of_scope.py` L167-170): "The threshold shipped in `checkpoints/scope.json` must not depend on `--n`. Fitting it on a subsample made a smaller run silently degrade the deployed system: a `--n 400` run pushed sensitivity from 71% to 28%. Always fit on the FULL validation fold; `--n` only limits the test-side evaluation." — a defect found and fixed during development.

### 18.7 XAI faithfulness (RQ5)

Source: `audit/results/05_xai_audit.txt`.

| Measure | Value | Verdict in file |
|---|---|---|
| Deletion AUC (lower better) | Grad-CAM **0.3933** vs random **0.5368** | → FAITHFUL |
| Insertion AUC (higher better) | Grad-CAM **0.5751** vs random **0.5039** | → FAITHFUL |
| Spearman(trained heatmap, random-head heatmap), n = 20 | **0.163** | (Adebayo sanity check) |
| Spearman(IG 30 steps, 200 steps) | **0.999** | |
| Top-1 lead agreement, 30 vs 200 steps | **19/20** | |
| Spearman(zero baseline, mean baseline) | **0.999** | |
| IG completeness relative error | **35.6%** | in-file note: "A completeness error above ~5% means 30 steps is too coarse" |

⚠️ **Two conflicts:**
- The file states "(app.py ships steps=30)". No `app.py` exists in `Component_02/`; `src/xai.py` ships **`steps=64`**. This audit was run against the **superseded** system (preserved at `reference/legacy_docs/legacy_app.py`).
- The 35.6% completeness figure is known to be an ill-conditioned measurement (it blows up when F(x) ≈ F(baseline)); the development record puts the true median at ≈1.3% at 30 steps, but **the on-disk results file still shows 35.6%**. **Re-run `05_xai_audit.py` against `src/xai.py` before quoting any XAI number.**

### 18.8 Leakage audit of the superseded system

Source: `audit/results/06_leakage_audit.txt` — see §16 table. Headline: "Report text ALONE: **0.8872**" macro-AUROC, i.e. the cardiologist's free-text report alone nearly matched the full fusion model, identifying label leakage.

### 18.9 Regression suite

`README.md` L204: "26 checks that every known defect is closed" — `audit/08_verify_fixes.py`. The results file `audit/results/08_verify_fixes.txt` ends with a full sample report and its LIMITATIONS block rather than a pass/fail tally line. **A machine-readable pass/fail count for the 26 checks is NOT FOUND IN CODEBASE** (the file has no `.json` sibling).

Sample verified output from that file (a real generated report fragment):
> "ST/T change ruled out under a conformal miss-rate bound of 10% (calibrated on 455 positive cases)."
> "LIMITATIONS: This system recognises only 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP). Arrhythmias such as atrial fibrillation, and any condition outside these classes, are NOT detected and their absence here is not evidence of their absence. Intervals (PR, QRS, QT) and QRS axis are not measured. The model was trained on PTB-XL (German cohort, 1989-1996) and has not been validated on this population."
> "AI-generated decision support. NOT a medical device and NOT a diagnosis."

### 18.10 Dataset audit

Source: `analysis/results/01_dataset_deep_audit.txt` — official 21,799 records / 18,869 patients; used 17,221 / 15,174; dropped 4,578 (21.0%); split table by records/patients/folds. Patient-disjointness across folds is verified by the script.

---

## 19. Interpretation Notes

**These are interpretations already written by the author inside the repo — quoted or paraphrased with citation. No new interpretation is added here.**

| Interpretation as written | Source |
|---|---|
| "Recall is bought with precision. For a rule-out system that is the correct trade: a false alarm costs a cardiologist's review, a false negative can cost a life. The referral burden this creates is quantified by the conformal layer, not hidden." | `analysis/results/02_operating_point.txt` §3 |
| "hypertrophy cannot reach F1 0.75 at 7.7% prevalence — that needs AUPRC > 0.8 and the model achieves 0.584 (published norm ~0.54). Its NPV at the shipped operating point is reported above and is the number [that matters]" | `analysis/results/02_operating_point.txt` |
| "this is a *rule-out* system, so it is tuned for sensitivity and NPV, the metrics every cardiology rule-out pathway uses. HYP F1 is 0.41 but its NPV is 0.981" | `README.md` L140-142 |
| "Not one of those guarantees concerns atrial fibrillation, because no guarantee about atrial fibrillation exists. The system certifies what it can measure while being blind to the finding that will cause the stroke." | `audit/13_out_of_scope.py` L137-139 |
| "Note this does NOT reduce diagnostic output. The classifier still reports its five classes. What is withheld is the CLAIM — the sentence promising a bounded miss rate — because that claim was never true for these records." | `audit/13_out_of_scope.py` L227-229 |
| "at a 5% false-positive budget the system stays silent about its guarantee on roughly 1 in 20 normal-rhythm records, in exchange for not making a false promise to [48.9%] of patients whose disease it cannot see." | `audit/13_out_of_scope.py` L231-233 |
| "[The marginal bound] only controls the miss rate *in expectation over repeated calibration draws*. The audit of the marginal version found it violated on this single test realisation for CD (0.122 vs alpha 0.10) and HYP (0.174 vs 0.15), because those classes have few calibration positives." | `src/conformal.py` L141-145 |
| "A suspected electrode reversal does not refuse the record — it is high-quality, just wired wrong — but it withdraws the statistical guarantees, which are calibrated on correctly-placed recordings." | `README.md` L157-160 |
| "no conformal bound covers a class that is not in the label space, so the claim is withheld." | `README.md` L161-162 |
| "macro-AUROC +0.0046 = 1.6σ → **within run-to-run noise**; macro-AUPRC +0.0137 = 4.7σ → **real** (t=8.2, p≈0.015)" | `docs/RESEARCH_CONTRIBUTION.md` L279-280 |
| "A fully symmetric comparison would need seeds for the [baseline]." | `docs/RESEARCH_CONTRIBUTION.md` L311 |
| "Results above 0.94 use foundation models pretrained on [large ECG corpora]" — positioning 0.9343 as inside the published band | `docs/PANEL_ANSWERS.md` L421 |
| "XAI → MI subtype localisation | **Not mine** — Strodthoff et al. 2024, cited" | `docs/PANEL_ANSWERS.md` L336 |
| "Do the slow, boring part on your laptop. Do only the GPU part on Colab." | `docs/COLAB_GUIDE.md` L10 |

**Interpretation that does NOT yet exist and cannot be extracted from code — requires author interpretation, human judgment call:**
- Why `CD / age<50` and `NORM / age≥70` specifically are the two cells that fail. The numbers are there; the clinical/physiological explanation is not written anywhere.
- Whether Mondrian's 22/23 vs 14/23 improvement is *clinically* meaningful or merely statistically visible.
- What the 48.9% rhythm-gate sensitivity means for deployment policy — the repo states the trade-off but does not recommend a budget.
- Why HYP is the hardest class beyond the prevalence argument.
- Whether a 4.5% false-positive rate on electrode-reversal detection is acceptable in a real ED workflow.

---

## 20. Limitations & Threats to Validity

**FACT — a limitations list is already maintained at `README.md` L262-279. Reproduced with sources, plus additions found elsewhere in the code.**

### 20.1 Stated in `README.md`

1. **Single-dataset, single-cohort.** "Trained on PTB-XL only (German cohort, 1989–96). **No external validation.**" (L262)
2. **Label space is incomplete.** "Recognises 5 superclasses. Atrial fibrillation and other arrhythmias are not detected… 14.3% of the dataset carries a documented finding the label space cannot express." (L263-266)
3. **The rhythm gate catches less than half.** "48.9% sensitivity, 4.8% FPR… regular out-of-scope rhythms (paced, monomorphic VT, Brugada, long QT) remain silent failures." (L267-269)
4. **No interval measurement.** "PR/QRS/QT intervals and QRS axis are not measured." (L270) — surfaced in the UI as literal "not measured" rows [`frontend/src/components/Interpretation.jsx` L109-110].
5. **Non-comparable labels.** "Labels used only SCP codes with `likelihood == 100`, dropping 21% of PTB-XL. Results are **not directly comparable** to published benchmarks." (L271-272)
6. **Unvalidated localisation.** "Territory localisation is a lead-group heuristic, not clinically validated." (L273)
7. **Weak electrode detector.** "a physiology rule, not a classifier: ~70% sensitivity on RA/LA, ~61% on RA/LL, 4.5% false positives. LA/LL reversal leaves aVR unchanged and is essentially undetectable this way." (L274-276)
8. **Latency.** "~6 s per analysis." (L277)

### 20.2 Additional threats visible in the code (not in the README)

| Threat | Evidence |
|---|---|
| **Threshold-selection fragility** | `audit/13_out_of_scope.py` L167-170 documents a real incident where fitting on a subsample degraded shipped sensitivity 71% → 28%. Fixed, but demonstrates how sensitive the pipeline is to fit-set size. |
| **Small calibration cells** | `_conformal_lower` returns `-inf` and `guarantee_feasible=False` when positives are too few. STTC/age<50 (n = 42) already hits `-inf` under Mondrian. HYP has only 134 calibration positives. |
| **No dependency pinning** | `requirements.txt` uses only `>=`; no lockfile. Results may not reproduce on a future resolve. |
| **Preprocessing/model coupling** | A guard in `train/fit_calibration.py` exists precisely because filtered packed data fed to an unfiltered model shifted macro-ECE 0.1834 → 0.2094 *silently*. The `ECG_NO_PACKED=1` escape hatch can re-open this. |
| **Calibrator/model provenance** | `checkpoints/calibrator.json` carries `fitted_for` and `backend/server.py` refuses on mismatch — mitigated, but shows the failure mode is real. |
| **Thread safety of XAI** | `src/xai.py::_model_lock` exists because Grad-CAM hooks are not thread-safe under Flask's threaded server. Correct now; a regression here would produce silently wrong attributions. |
| **Assumed monotonicity** | `_pac_order_statistic`'s early `break` assumes the Beta CDF is monotone in k; not asserted. `[inferred]` |
| **Age/sex subgroups only** | The conditional-validity audit covers sex and three age bands. No audit by device, recording site, comorbidity, or signal quality band. |
| **No test coverage** | Zero test files, no CI, no `pytest` configuration. The 26-check `08_verify_fixes.py` is a script, not a test suite, and produces no machine-readable pass/fail. |
| **Single-run downstream** | Everything after training is seed-0 only (§15). |
| **The 3-seed claim has no artifact** | §15 item 1 — a reviewer asking for the raw numbers cannot be shown them. |
| **Two audits are stale** | `05_xai_audit` was run against the superseded system; `12_electrode_reversal` on disk is n = 200 while docs quote n = 600 (§18.5, §18.7). |

### 20.3 Assumptions baked into the code

| Assumption | Where |
|---|---|
| Mains frequency is 50 Hz | `src/preprocess.py` L32 — "PTB-XL was recorded in Germany -> 50 Hz mains". **A 60 Hz recording (US/Japan) would be filtered incorrectly.** |
| Recording is 10 s / 5,000 samples @ 500 Hz | `SIGNAL_LENGTH = 5000`; longer inputs are centre-cropped, shorter are constant-padded |
| Lead order is the standard 12-lead order, lead II at index 1 | `audit/13_out_of_scope.py` L160 uses `pp.bandpass(s)[:, 1]` for R-peak detection |
| Only limb-electrode reversals occur | `src/electrodes.py` handles RA/LA, RA/LL, LA/LL — precordial misplacement is not modelled |
| Reference reports are in German or English | `audit/13_out_of_scope.py` regexes cover both, nothing else |
| Age bands `<50 / 50-69 / ≥70` and binary sex | `audit/10_conditional_validity.py` — no other protected attribute is available in PTB-XL |

---

## 21. Ethical & Societal Considerations

### Data privacy
**Applicable.** PTB-XL is human clinical ECG data with age and sex attributes. It is a **public, de-identified, IRB-cleared research dataset** redistributed under CC-BY 4.0 [`data/DATASET_LICENSE.txt`]. No re-identification attempt exists in the code. Records are referenced by `ecg_id` and `patient_id` — PTB-XL's own pseudonymous identifiers, not names or MRNs. The API accepts uploaded `.dat`/`.hea` pairs; **whether uploaded files are persisted server-side is NOT FOUND IN CODEBASE — needs verification by the author** (`backend/server.py::predict` L413 should be checked for temp-file cleanup before any real-patient use).

⚠️ **Live security item:** PhysioNet download credentials were found hard-coded in plaintext in the superseded system and were moved to `PHYSIONET_USER` / `PHYSIONET_PASS` environment variables, with `.env` and `*credential*` added to `.gitignore`. **The exposed account must be treated as compromised and the password rotated.** This is unresolved as far as the codebase shows.

### Potential misuse
**Applicable, and actively mitigated.** The system produces text that reads like a clinical report. Mitigations present in code:
- Every report carries: "AI-generated decision support. NOT a medical device and NOT a diagnosis. Every report requires review by a qualified clinician before any clinical action is taken." [`audit/results/08_verify_fixes.txt`; `README.md` L10-12]
- `src/verify.py` refuses to emit text asserting findings the model did not compute.
- `refused: true` is returned instead of probabilities when quality control fails, and `README.md` L106-107 instructs integrators: "**Do not treat this as 'normal'.**"
- Guarantees are *withheld* — not silently weakened — on suspected electrode reversal or out-of-scope rhythm [`README.md` L108-113].

**Residual misuse risk:** a downstream integrator who ignores the `refused`, `verification.passed`, `electrode.suspected` and `scope.outOfScope` flags would present an unsafe result as a clean one. The README documents these as "Four fields the caller must respect", but nothing enforces it at the protocol level.

### Fairness / bias
**Applicable — and this is the component's central finding.** The system makes decisions about people, and §18.4 shows the statistical guarantee is **not equally valid across sex and age**: the promised miss-rate bound is violated in 9 of 23 subgroups, with two violations surviving Holm correction (`CD / age<50`: 0.333 vs promised ≤ 0.10; `NORM / age≥70`: 0.330 vs promised ≤ 0.20). Mondrian calibration is implemented as the mitigation and restores 22/23 cells.

**Unmeasured fairness axes:** race/ethnicity, socioeconomic status, and geography are **not present in PTB-XL** and therefore cannot be audited. Since PTB-XL is a single German cohort from 1989–96, generalisation to other populations is untested — an equity concern the README already flags as "No external validation."

### Environmental / compute cost
**Applicable but low.** Training is a single ~60-minute L4 GPU run (§11). The `docs/COLAB_GUIDE.md` is explicitly organised around *minimising* wasted compute (data packing to avoid paying GPU rates for network I/O; time budget; OOM recovery; resume). Total energy is **not logged** — **NOT FOUND IN CODEBASE**.

### Dataset licensing
**Applicable and compliant.** `data/DATASET_LICENSE.txt` states PTB-XL is redistributed "under its Creative Commons Attribution 4.0 licence (CC-BY 4.0), which permits redistribution with attribution", and supplies both required citations (Wagner et al. 2020, Sci Data 7:154, doi:10.1038/s41597-020-0495-6; Goldberger et al. 2000, Circulation 101(23):e215-e220) plus the source URL. `.gitignore` additionally blocks `*.dat`/`*.hea` from version control with the comment "never commit PTB-XL: 20+ GB, restricted redistribution".

---

## 22. Reproducibility / How to Run

### 22.1 Setup

**All commands run from inside the `Component_02/` folder** [`README.md` L20].

```bash
cd Component_02
pip install -r requirements.txt
```

`README.md` L41-42 records a known user error: "Seeing `can't open file '...Component_02\Component_02\backend\server.py'`? You are already inside `Component_02/` — drop the prefix."

### 22.2 Run the system

```bash
# Terminal 1 — API on :5000
python -X utf8 backend/server.py

# Terminal 2 — UI on :5173
cd frontend
npm install          # first time only
npm run dev
```

Open **http://localhost:5173** [`README.md` L27-39].

Requirements: Python 3.10+, Node 18+. No GPU needed for inference.

### 22.3 Reproduce every experiment

From `README.md` L203-224:

```bash
python -X utf8 audit/08_verify_fixes.py             # 26 regression checks
python -X utf8 analysis/01_dataset_deep_audit.py    # dataset integrity
python -X utf8 analysis/02_operating_point.py       # operating point
python -X utf8 audit/10_conditional_validity.py     # contribution 1
python -X utf8 audit/11_significance.py             # significance of contribution 1
python -X utf8 audit/12_electrode_reversal.py       # contribution 2
python -X utf8 audit/13_out_of_scope.py             # contribution 3
```

### 22.4 Retraining

`train/train_gpu.py` (CLI, §12.1), `train/fit_calibration.py` (§12.2), `train/preflight.py` (`--batch 128 --model resnet_se`), and `train/Component02_Colab.ipynb` (the Colab entry point per `docs/COLAB_GUIDE.md` L87). Full procedure in `docs/COLAB_GUIDE.md`.

### 22.5 Entry points and I/O

| Entry point | Input | Output |
|---|---|---|
| `backend/server.py` | HTTP; `multipart/form-data` with `dat_file` + `hea_file` on `POST /api/predict` | JSON (schema at `README.md` L66-97) |
| `src.pipeline.ECGPipeline.from_checkpoint(...)` then `.analyse(signal, fs, with_xai=)` | numpy array (samples × 12 leads) in mV + sampling rate | `AnalysisResult` dataclass |
| `train/train_gpu.py` | packed `.npy` arrays in `data/` or WFDB via `src/signals.py` | `checkpoints/best_model.pt` |
| `analysis/*.py`, `audit/*.py` | reads `csv/`, `checkpoints/`, `data/` | `*/results/*.txt` + `*.json` |

### 22.6 Artifact status

**Yes — this is packaged as a shareable artifact.**
- `README.md` L232-234: "Nothing here reads from outside this folder. Verified by copying it to an empty directory and running the full suite."
- Asset resolution is layered so the folder degrades gracefully: `$ECG_DATA_DIR` → `csv/` → `data/` → `../_archive/data/` [`src/paths.py::_candidates`].
- `README.md` L237-240: total ~2.0 GB; without `data/` ~20 MB, and "upload path still works fully" — `data/` is needed "only to browse the built-in test set".
- Distribution instruction: "Send the whole folder minus `frontend/node_modules/`" [`README.md` L244].
- `.gitignore` is written to keep the reproducibility record in git while excluding weights and data: it whitelists `!checkpoints/calibrator.json`, `!checkpoints/conformal_triage.json`, `!checkpoints/history_*.json`, `!csv/*.csv`, `!csv/norm_stats.json`, `!audit/results/*.json`, `!audit/results/*.txt` under the comment "**KEEP these small artefacts — they ARE the reproducibility record**".

⚠️ **Reproducibility gaps:** no pinned dependency versions; no Docker image; no CI that proves the suite runs on a clean machine; the model weights (`*.pt`) are gitignored with the note "Distribute via Drive/releases" but **no release URL or DOI is recorded anywhere** — **NOT FOUND IN CODEBASE — needs input from author.**

---

## 23. My Individual Role / Contribution Statement

**⚠️ GIT HISTORY IS UNUSABLE FOR ATTRIBUTION. This section cannot be derived from the repository and must be written manually by the author.**

**FACT — what git actually reports:**

| Command | Result |
|---|---|
| `git log --oneline` | `fatal: your current branch 'main' does not have any commits yet` |
| `git log --author=...` | same — **zero commits exist in the repository** |
| `git config user.name` | `VenushanT` |
| `git config user.email` | `aron28416@gmail.com` |
| `git status` | working tree entirely untracked: `.env.example`, `.gitignore`, `Component_02/`, `README.md`, `_archive/` |

**Consequences:**
1. **There is no commit authorship to separate the author's work from teammates'.** Every file in `Component_02/` is untracked. An author-contributions note for the group paper cannot be evidenced from this repo.
2. **There is no version history at all** — no ability to show what was inherited from the previous system versus rebuilt, no rollback point, and a single point of failure for ~2 GB of work.

**FACT — the only in-repo attribution statements:**
- `README.md` L3: "**Venushan T** · part of the Explainable AI System for Cardiovascular Disease Detection and Diagnosis."
- `docs/PANEL_ANSWERS.md` L336 explicitly disclaims one item as *not* the author's: "XAI → MI subtype localisation | **Not mine** — Strodthoff et al. 2024, cited".

**FACT — what is demonstrably inherited vs. rebuilt** (structural evidence, not authorship evidence):
- `reference/legacy_docs/legacy_app.py`, `legacy_index.html`, and four `.txt` overview documents are preserved copies of a **superseded** system.
- `reference/checkpoints_ecg_only/` and `reference/checkpoints_fusion_leaked/` are superseded model checkpoints kept as audit evidence.
- `audit/legacy/` holds 6 scripts described as "audits of the superseded system (need `../_archive`)" [`README.md` L194].
- `docs/AUDIT_FINDINGS.md` documents "The 12 defects found in the previous system" [`README.md` L253].
- `[inferred]` The current `src/` (2,803 lines, 14 modules), `backend/server.py`, `frontend/` (10 source files), `analysis/`, and `audit/08–13` constitute the rebuilt work; the `reference/` and `audit/legacy/` trees constitute the inherited work. **This inference is from folder naming and README labels, not from authorship metadata.**

**ACTION REQUIRED BY AUTHOR:**
1. **`git add` and commit immediately** — currently there is zero version history for the entire project.
2. Write the contribution statement manually: which modules were personally authored, which were adapted from the previous system, which were built jointly.
3. Confirm the git email (`aron28416@gmail.com`) is the one that should appear in the group paper's author-contributions note.

---

## 24. Key Terms / Mini-Glossary

| Term | One-line definition |
|---|---|
| **Superclass** | One of the five broad diagnostic groups this system outputs — NORM (normal), MI (myocardial infarction), STTC (ST/T change), CD (conduction disturbance), HYP (hypertrophy) — obtained by mapping PTB-XL's fine-grained SCP codes through `csv/scp_to_superclass_mapping.json`. |
| **Conformal prediction** | A way of turning any model's score into a decision that comes with a proven bound on how often it will be wrong, calibrated on held-out data. |
| **Marginal validity** | The bound holds *on average over the whole population* — it can still fail badly for a particular subgroup. |
| **Conditional (Mondrian) validity** | The bound is guaranteed *within each subgroup* because a separate threshold is fitted per subgroup. |
| **PAC / training-conditional bound (δ)** | A stronger form of the guarantee: it holds with probability ≥ 1−δ over the random draw of the calibration set, not merely in expectation across many such draws. |
| **α (miss-rate budget)** | The maximum fraction of true cases allowed to be "ruled out" in error, chosen per class before calibration. |
| **Rule-out / Refer / Rule-in** | The three decision zones: below the lower threshold the disease is excluded under a bound; above the upper threshold it is asserted; in between the system declines to decide and refers. |
| **NPV (negative predictive value)** | Of all the patients the system said were negative, the fraction that truly were — the number that matters for a rule-out pathway. |
| **Temperature scaling** | Dividing the model's raw scores by a fitted constant so the resulting probabilities match observed frequencies. |
| **Grad-CAM / Integrated Gradients** | Two methods for showing which parts of the ECG the model actually used — one via gradients at a convolutional layer, one by accumulating gradients along a path from a blank baseline to the real signal. |
| **Deletion / Insertion AUC** | A faithfulness test: progressively remove (or add back) the regions an explanation calls important and watch how fast the prediction collapses (or recovers). |
| **Ablation** | A test where you remove one part of a system to see whether it still works as well. |
| **Out-of-scope** | A condition present in the recording that the five-class label space cannot express at all — atrial fibrillation being the main example. |
| **Electrode reversal** | Two ECG cables swapped at the patient; the recording looks clean but the waveform is a mathematically transformed version of the truth. |
| **SQI (Signal Quality Index)** | A 0–1 score for how usable a recording is; below threshold the system refuses rather than guessing. |
| **Holm–Bonferroni correction** | An adjustment applied when testing many hypotheses at once, so that finding "something significant" by chance becomes unlikely. |

---

## 25. Gaps & Open Questions

### 25.1 Every "NOT FOUND IN CODEBASE — needs input from author"

| # | Gap | Section |
|---|---|---|
| G1 | How this component connects to the other three teammates' components — nothing in `Component_02/` references them | §1 |
| G2 | Literature verification that the electrode-reversal→guarantee-invalidation and out-of-scope→guarantee-withholding contributions are actually novel. **The repo already records two retracted novelty claims — do not repeat that.** | §3, §5 |
| G3 | Exact CPU / RAM / VRAM specifications of the training and development machines | §11 |
| G4 | Exact resolved dependency versions (`pip freeze`) — `requirements.txt` uses only `>=`, no lockfile | §11 |
| G5 | Actual training wall-clock time — logging code exists (`train_gpu.py` L455-459, L520) but no `checkpoints/history_*.json` is on disk | §11 |
| G6 | Colab compute units consumed | §11 |
| G7 | Whether `train/Component02_Colab.ipynb` contains saved output cells with the 3-seed console log | §17 |
| G8 | Machine-readable pass/fail tally for the 26 regression checks — `08_verify_fixes.txt` has no `.json` sibling | §18.9 |
| G9 | Whether `POST /api/predict` persists or cleans up uploaded `.dat`/`.hea` temp files | §21 |
| G10 | Whether the exposed PhysioNet password has been rotated | §21 |
| G11 | Release URL / DOI for the gitignored model weights ("Distribute via Drive/releases" — no location recorded) | §22 |
| G12 | The author's manual contribution statement — **git has zero commits, so this cannot be derived** | §23 |
| G13 | Column headers for the diagnosis-change table in `12_electrode_reversal.txt` | §18.5 |

### 25.2 Conflicting values found — flagged, not silently resolved

| # | Conflict | Locations |
|---|---|---|
| C1 | **Electrode-reversal sample size: 600 vs 200.** Docs report 589/600, 587/600; the on-disk results file is an n=200 run (198/200, 197/200). Script default is `--n 600`. | `docs/PANEL_ANSWERS.md` L142-149 vs `audit/results/12_electrode_reversal.txt` |
| C2 | **Electrode detector sensitivity: ~70%/~61% vs 65.5%/60.5%.** | `README.md` L274-276 vs `audit/results/12_electrode_reversal.txt` |
| C3 | **PAC δ: 0.05 vs 0.01.** Code defaults say 0.05 (`fit_calibration.py --delta`, `ConformalTriage.__init__` L211); the shipped artifact and all audit scripts use 0.01. | `src/conformal.py` L211, `train/fit_calibration.py` vs `checkpoints/conformal_triage.json`, `audit/11_*.py`, `audit/12_*.py` |
| C4 | **`--max-minutes`: 75.0 vs 60.0.** CLI default vs the value actually recorded in the shipped checkpoint's `args`. | `train/train_gpu.py` L294 vs `checkpoints/best_model.pt` |
| C5 | **IG steps: 64 vs 30.** `src/xai.py` ships `steps=64`; the XAI audit says "(app.py ships steps=30)" and no `app.py` exists in this component. **The XAI audit was run against the superseded system.** | `src/xai.py` vs `audit/results/05_xai_audit.txt` |
| C6 | **IG completeness relative error 35.6%** is on disk, but is known to be an ill-conditioned measurement whose true median is ≈1.3% at 30 steps. The results file was never regenerated. | `audit/results/05_xai_audit.txt` |
| C7 | **Validation set size: 1,709 vs 1,696.** `csv/val.csv` has 1,709 rows; `checkpoints/scope.json` records `n_val: 1696`. `[inferred]` 13 records failed `rr_features` (too few R-peaks) — needs confirmation. | `csv/val.csv` vs `checkpoints/scope.json` |
| C8 | **Folder size is understated by ~2×.** `README.md` L191/L237 says `data/` = 1.93 GB and total ≈ 2.0 GB. **Measured (`du -sh`): total = 4.1 GB** — `data/raw_signals` 2.0 GB **plus** ~2.0 GB of packed training memmaps (`train_X.npy` 1.6 GB, `val_X.npy` 196 MB, `test_X.npy` 196 MB) that the README does not count. `.gitignore` is the only file that gets this right ("1.9 GB … **and** packed training memmaps (2 GB)"). Correct the README, and note the ~2.0 GB of memmaps are regenerable from `raw_signals/` via `train_gpu.py --pack`. | `README.md` L191, L237 vs `.gitignore` vs measured |
| C9 | **`README.md` L226 says "All four verified standalone"** but seven commands are listed above it. Stale count. | `README.md` L226 |
| C10 | **`best_auroc = 0.9403` (validation) vs `0.9343` (test, 3 seeds).** Not a contradiction — different splits — but easy to misquote. State the split explicitly wherever either appears. | `checkpoints/best_model.pt` vs `README.md` L134 |

### 25.3 Requires author interpretation — human judgment, not extractable from code

- Why `CD / age<50` and `NORM / age≥70` specifically fail the marginal bound (clinical/physiological explanation).
- Whether Mondrian's 22/23 vs 14/23 improvement is clinically meaningful or only statistically visible.
- What rhythm-gate operating point to recommend for deployment (the repo states the trade-off but makes no recommendation).
- Whether 4.5% false positives on electrode-reversal detection is acceptable in a real workflow.
- Why HYP is the hardest class beyond the prevalence argument.

### 25.4 Structural gaps a reviewer will find

| # | Gap | Fix |
|---|---|---|
| S1 | **The 3-seed headline (0.9343 ± 0.0028 / 0.8001 ± 0.0029) has no results artifact anywhere in the repo** — only prose in four files, and only seed 0's checkpoint exists. | Re-run the 3 seeds capturing stdout to `audit/results/`, or remove the claim. **Highest-priority item.** |
| S2 | **Zero git commits.** No history, no attribution, no backup. | `git add -A && git commit` immediately. |
| S3 | **No tests, no CI, no Dockerfile, no lockfile.** | At minimum record `pip freeze`; a CI job running `08_verify_fixes.py` would be cheap. |
| S4 | **No figures exist** — every paper figure must be drawn from scratch. The `.json` result files make this scriptable without re-running experiments. | Write plotting scripts against `analysis/results/*.json` and `audit/results/*.json`. |
| S5 | **No architectural ablation** of the shipped `ECGResNetSE` (SE block, multi-kernel stem, attention pooling). Several ablations are already exposed by existing CLI flags. | §16 table lists exactly which flags to use. |
| S6 | **Baseline comparison is asymmetric** (3 seeds vs 1 seed) and covers no non-deep baseline. | Run `--model resnet` at seeds 0/1/2. |
| S7 | **Two audits on disk are stale** (`05_xai_audit` against the superseded system; `12_electrode_reversal` at the wrong n). | Re-run both before quoting any of their numbers. |
| S8 | **References 1, 4 and 6 are incomplete** (missing authors and/or venue). | Complete them before the reference list is compiled. |
| S9 | **The 26-check regression suite has no pass/fail summary line**, making "26/26 green" unverifiable from the artifact. | Add a tally line and a `.json` output. |
| S10 | **RQ5 (XAI faithfulness) is n = 20 with no CI.** | Increase n and report a confidence interval. |
| S11 | **No external validation dataset.** Single German cohort, 1989–96. This is the limitation most likely to be raised. | Acknowledge explicitly; consider a second public ECG dataset if time permits. |
| S12 | **Research questions are not stated anywhere in the codebase** — §4 reconstructs them from what the code measures. | Author should write RQ1–RQ6 in their own framing and confirm the reconstruction is faithful. |

---

*Dossier compiled from a full-tree analysis of `Component_02/` (14 `src` modules / 2,803 LOC, 3 training scripts, 2 analysis scripts, 11 audit scripts, 26 result files, 5 checkpoint artifacts, 5 docs, 10 frontend source files). Every numeric value above is traceable to a named file. No number in this document was estimated, rounded from memory, or supplied from outside the repository.*
