# Component Write-Up: UEF-Net — Uncertainty-Aware Ordinal Four-Class Ejection Fraction Severity Grading

> **REVISION NOTE (post-audit remediation).** This dossier was compiled, then the issues it
> identified were fixed in the codebase. Items marked **[FIXED]** below have been resolved;
> items marked **[CORRECTED]** were errors in the *first version of this dossier itself* and
> are retracted. §25 records the final state of every issue.

**Author:** Dilukshan Viyapury (IT22219534) · Project ID R26-IT-083, Component 03
**Repository root:** `c:\Users\dviya\Desktop\Component_03`
**Dossier compiled:** from codebase state at commit `e3b18e6`

---

## 0. Component Abstract

Left-ventricular ejection fraction (EF) is graded manually from echocardiogram video, a process carrying 4–5 EF points of inter-observer variability (`README.md` §1). This component implements **UEF-Net**, an R(2+1)D-18 spatio-temporal network with four prediction heads (EF regression, ordered-cutpoint ordinal, auxiliary softmax class, log-variance) trained on EchoNet-Dynamic with harmonized CAMUS co-training, producing both continuous EF and a four-class severity grade under ~11:1 class imbalance. On the untouched 1,277-study test split a three-seed ensemble achieves **MAE 3.979 EF points, R² 0.818, 73.0 % overall accuracy, 73.7 % balanced accuracy, and 0.723 minimum per-class recall**, with 99.7 % of predictions within one severity class and zero Severe↔Normal confusions (`Dilukshan/training/outputs/ensemble_report.json`). The component additionally quantifies three independent bounds explaining why a ≥0.75-on-all-classes target is not reachable on this data (`README.md` §10).

**Keywords:** Echocardiography; Ejection Fraction; Ordinal Regression; Class Imbalance; Uncertainty Quantification; Deep Learning

---

## 1. Role in the Overall System

**FACT (from `Project_Proposal_IT22219534_Dilukshan_Viyapury.docx` §4.1 and `proposal_figures/fig_system.png`):**
The wider project is described as a four-component clinical decision-support platform for echocardiography. Component 03 is stated to perform "Ejection Fraction Regression and Four-Class Severity Grading". Components 01, 02 and 04 are labelled "(Team Member)" in the system diagram.

**Plain-language paragraph:**
A clinician uploads an echocardiogram video. Component 03 watches the heart beating in that video and answers two things: a number (what percentage of blood the left ventricle pumps out per beat) and a category (Normal, Mild, Moderate, or Severe impairment). It also reports how confident it is, and gives a prediction interval rather than a bare number. It needs only decoded video as input and emits EF value + severity class + confidence + interval, so it can be developed and swapped independently of the other three components.

**⚠️ NOT FOUND IN CODEBASE — needs input from author:** There are **no imports, API calls, shared interfaces, or message contracts** between this repository and any Component 01/02/04 code. The four-component architecture appears only in the proposal document and its figure. The claimed integration is therefore **[inferred]** design intent, not implemented. No integration code exists in `Dilukshan/`.

---

## 2. Problem Statement & Motivation

**The specific problem this component addresses:**

- Predict EF from 2D apical-four-chamber echocardiogram video **and** assign one of four clinical severity classes, where the classes are severely imbalanced (`README.md` §1).
- FACT — class distribution in EchoNet-Dynamic (`README.md` §1, derived from `Dilukshan/preprocessing/artifacts/manifest.csv`):

| Severity | EF range | Studies | Share |
|---|---|---|---|
| Severe | < 30 % | 596 | 5.9 % |
| Moderate | 30–40 % | 718 | 7.2 % |
| Mild | 40–55 % | 1,806 | 18.0 % |
| Normal | ≥ 55 % | 6,910 | 68.9 % |
| **Total** | | **10,030** | 100 % |

**Why it matters (FACT, `README.md` §1):**
- Manual EF estimation takes several minutes per study and requires an experienced operator.
- Reported inter-observer variability is 4–5 EF points, large enough to move a borderline patient across a clinical decision boundary (e.g. a true EF of 41 % may be graded *mild* by one reader and *moderate* by another).
- Under 11:1 imbalance, a degenerate model predicting "Normal" for every study scores 68.9 % overall accuracy while being clinically useless (`README.md` §2).

---

## 3. The Gap

**FACT — three gaps are stated explicitly in `README.md` §2:**

| # | Gap as stated in repo | Evidence cited in repo |
|---|---|---|
| **G1** | EF is modelled as regression or binary reduced-vs-normal, not as the four-class severity grading clinicians use | "Ouyang et al. [1] report MAE and binary AUC only; transformer and graph variants [6], [7] remain regression-only" |
| **G2** | Per-class recall is not reported under ~11:1 imbalance, leaving minority-class safety unverified | The 68.9 % degenerate-baseline argument |
| **G3** | The EF label is treated as exact, despite carrying 4–5 points of measurement noise | "No reviewed echocardiography study encodes annotation noise into the supervision signal" |

**⚠️ Verification required by author:** G1–G3 are assertions written by the author in `README.md`; the repository contains **no literature-survey artifact** (no BibTeX file, no notes file, no PDF collection) substantiating the claim that no prior work does these things. The claim "*to the best of my knowledge*" appears in `README.md` §3. **This requires literature research from the author to defend at review.**

---

## 4. Research Question(s) This Component Answers

**⚠️ FACT: No file in the repository states research questions explicitly.** No `RQ1`/`RQ2` string exists anywhere in the codebase.

The following are **[inferred]** from what the code actually measures, and **require author confirmation/authoring**:

- **RQ1 [inferred]:** Does an ordinal formulation with measurement-uncertainty-derived soft labels achieve EF regression accuracy at or below the published EchoNet-Dynamic benchmark (MAE 4.05) *while additionally* producing four-class severity grades? — *measured by* `run_eval.py` / `run_ensemble.py` reporting MAE, R², and per-class recall together.
- **RQ2 [inferred]:** Does cross-dataset co-training with an intensity-harmonized minority-rich cohort (CAMUS) improve minority-class recall relative to training on EchoNet alone? — *measured by* comparing `outputs/uefnet_drw` (no CAMUS) against `outputs/uefnet_v3` (CAMUS).
- **RQ3 [inferred]:** Can post-hoc calibration that corrects regression-to-the-mean recover extreme-tail (Severe) recall without retraining? — *measured by* `outputs/uefnet_v3/test_report_baseline.json` vs `test_report.json` on identical weights.
- **RQ4 [inferred]:** Does selective prediction (abstention) raise worst-class recall in ordinal grading? — *measured by* `run_selective.py`, `outputs/selective_report.json`. **This RQ is answered negatively.**

---

## 5. Contribution Bullets & Novelty

1. **We derive ordinal supervision targets from the EF label's own measurement noise.** *(13 words)*
   → Formula: `s_k = 1 − Φ((t_k − EF)/σ)`, σ = 4 EF points.
   → **Refinement, not a new method.** The literature check the write-up flagged as missing has now been done, and the closest prior art is **Díaz & Marathe, "Soft Labels for Ordinal Regression", CVPR 2019** — which derives soft ordinal targets rather than hard class membership. Claiming soft ordinal targets as novel would not survive a panel that knows it.

→ What survives as this component's own: the width of the soft target is **not a tuned hyperparameter**. It is fixed at σ = 4 EF points because that is the published inter-observer variability of the clinical measurement itself, so the supervision signal is calibrated to how precisely the label was ever knowable. Díaz & Marathe tune their smoothing; this derives it from the instrument. Implemented in `losses/losses.py L30-36 soft_cumulative_targets()`.

→ **How to say it:** *"Soft ordinal targets are Díaz & Marathe 2019. What is ours is deriving the smoothing width from the label's own measurement noise rather than tuning it — the target is as sharp as the ground truth deserves, and no sharper."*

2. **We enforce ordinal rank consistency structurally via positively-constrained cut-point gaps.**
   → **Adapted.** Adapted from CORAL (Cao et al. 2020, `README.md` ref [12]), which learns independent per-threshold biases that may violate ordering. Modification: single score compared against `anchor + cumsum(softplus(gaps))`, so monotonicity holds by construction (`models/uef_net.py L33-57 OrderedCoralHead`).

3. **We harmonize a minority-rich second cohort's intensity before co-training to prevent a scanner-appearance shortcut.**
   → **Novel (as a diagnosis + fix pairing).** The *use* of auxiliary data is standard; identifying that a class-balanced sampler over-drawing brighter CAMUS studies would let the network learn brightness→severity, and correcting it, is the contribution (`preprocessing/build_camus.py harmonize_to_echonet()`).

4. **We correct regression-to-the-mean shrinkage post-hoc by variance expansion fitted on full validation.**
   → **Adapted.** Variance expansion is a known bias-correction idea; the adaptation is applying it to recover extreme-tail class recall and fitting thresholds on the *full* validation split rather than a 30 % subsample for small-class stability (`engine/calibrate.py L76-110 fit_variance_expansion`, `config.py L117-118`).

**Conservative statement:** Items 2 and 4 are **adaptations of published techniques**, not new methods. Items 1 and 3 are the strongest novelty claims but **both depend on a literature check the repository does not contain**. Seed ensembling, TTA, EMA, DRW, LDS, logit adjustment and conformal prediction are all **Engineering** — standard techniques applied competently, explicitly listed as "Prior work used and cited, not claimed" in `README.md` §3.

---

## 6. Contribution → Evidence Traceability Table

| Contribution bullet | Implemented where | Evaluated where | Evidence status |
|---|---|---|---|
| **1.** Measurement-uncertainty soft ordinal labels | `losses/losses.py L30-36` (`soft_cumulative_targets`), applied `L119-121`; σ from `config.py L78 ef_noise_sigma=4.0` | **⚠️ NO ISOLATED ABLATION.** No run exists with this disabled. It is present in every trained model. | **RISK — unevaluated in isolation** |
| **2.** Ordered-cutpoint ordinal head | `models/uef_net.py L33-57` (`OrderedCoralHead`); gated by `is_v2` L121-122 | **⚠️ NO ISOLATED ABLATION.** v1 (`CoralHead`) vs v2 runs differ in many variables simultaneously. | **RISK — confounded comparison** |
| **3.** Harmonized CAMUS co-training | `preprocessing/build_camus.py` (`harmonize_to_echonet`); merge in `data/dataset.py L~170-195`; enabled by `--extra-manifest` | `outputs/uefnet_drw/test_report.json` (no CAMUS, minrec 0.651) vs `outputs/uefnet_v3/test_report.json` (CAMUS, minrec 0.687) | **PARTIAL — confounded** (v3 also adds v2 architecture + logit adjustment; see §13) |
| **4.** Tail-decompression calibration | `engine/calibrate.py fit_variance_expansion` + `config.py L117-118 calibrate_on_full_val`, `threshold_search_radius` | `outputs/uefnet_v3/test_report_baseline.json` (Severe 0.590) vs `test_report.json` (Severe 0.687) — **identical weights** | **STRONG — fully controlled** |
| *(supporting)* Seed ensembling | `run_ensemble.py` (`average_predictions`) | `outputs/ensemble_report.json` vs single-model `outputs/uefnet_v3/test_report.json` | **STRONG — fully controlled** |
| *(supporting)* Selective prediction | `engine/selective.py`, `run_selective.py` | `outputs/selective_report.json` | **STRONG — negative result** |

**⚠️ Paper-writing risk flagged now:** contributions **1 and 2 have no isolated ablation**. Both are baked into every v2 run. A reviewer asking "how much did the soft labels actually contribute?" cannot currently be answered from this repository.

---

## 7. Related Work / Prior Approaches Referenced

**FACT — 18 references are listed in `README.md` §19.** Those cited as *methods used or compared*:

| Approach | Key idea | Limitation stated in repo | Citation |
|---|---|---|---|
| EchoNet-Dynamic (benchmark) | R(2+1)D regression on echo video; MAE 4.05, R² 0.81 | Regression only; no severity grading; no per-class recall | Ouyang et al., *Nature* 580:252-256, 2020 [1] |
| Ultrasound video transformers | Attention over long frame sequences | "≈ 5.9 MAE"; high data/compute requirement | Reynaud et al., MICCAI 2021, pp. 495-505 [6] |
| Lightweight spatio-temporal graphs | Graph net, lower cost | "≈ 4.2 MAE"; regression only | Thomas et al., MICCAI 2022, pp. 380-390 [7] |
| CORAL | Rank-consistent ordinal regression via shared projection + independent biases | Ordering repaired post hoc, not structural | Cao, Mirjalili, Raschka, *Pattern Recognition Letters* 140:325-331, 2020 [12] |
| Deferred re-weighting (LDAM-DRW) | Train natural first, then class-balanced | — (adopted) | Cao et al., NeurIPS 32, 2019 [8] |
| Effective-number class weighting | `(1−β)/(1−βⁿ)` weights | — (adopted) | Cui et al., CVPR 2019, pp. 9268-9277 [9] |
| Label distribution smoothing | Neighbouring targets carry related information | — (adopted) | Yang et al., ICML 2021, pp. 11842-11851 [10] |
| Logit adjustment | Shift logits by `τ·log P(class)` | — (adopted) | Menon et al., ICLR 2021 [11] |
| Temperature scaling | Post-hoc confidence calibration | — (adopted) | Guo et al., ICML 2017, pp. 1321-1330 [13] |
| Conformal prediction | Finite-sample coverage intervals | — (adopted) | Angelopoulos & Bates, *FnT ML* 16(4):494-591, 2023 [14] |
| VideoMAE | Masked-autoencoder self-supervised video pretraining | "advantage diminishes at this data scale"; listed as future work, **not implemented** | Tong et al., NeurIPS 35, 2022 [15] |
| R(2+1)D | Factorised spatio-temporal convolution | — (backbone used) | Tran et al., CVPR 2018, pp. 6450-6459 [5] |
| Chow's reject option | Classification with abstention | — (adopted for §9 analysis) | Chow, *IEEE T-IT* 16(1):41-46, 1970 [18] |

**⚠️ Flagged in `README.md` §7.5 by the author:** "Figures for [6] and [7] should be verified against the source papers before citation." **These MAE values are unverified.**

---

## 8. Domain-Specific Structuring Fit

**Best fit: Build-up Ablation (ML).**

**Why (FACT):** `README.md` §8.1 presents a staged progression where each stage adds one mechanism and reports the effect:

| Stage | MAE | Min-class recall |
|---|---|---|
| Single model, 30 % validation calibration | 4.166 | 0.590 |
| + tail-decompression calibration | 4.138 | 0.687 |
| + 2-seed ensemble | 3.994 | 0.711 |
| + 3-seed ensemble | 3.979 | 0.723 |

The repository also contains a **secondary Benchmark-driven** aspect (§7.5 compares against published EchoNet-Dynamic results on the same official test split).

**What this structure implies should be captured:**
- ✅ Already captured: the incremental table above; per-stage measured deltas.
- ⚠️ **Missing:** ablations for contributions 1 and 2 (§6). A build-up-ablation paper is expected to isolate *each* added component. Currently only calibration and ensembling are isolated.
- ⚠️ **Missing:** a from-scratch baseline within this codebase — see §13.

---

## 9. Method / Design

### 9.1 Architecture (as words → diagram)

**FACT (`models/uef_net.py`):**

```
Input clip (2 × 32 × 112 × 112)   [grayscale + temporal-difference motion]
        │
   R(2+1)D-18 backbone            torchvision r2plus1d_18, Kinetics-400 weights,
        │                         stem adapted 3→2 channels (uef_net.py L79-104)
        │                         feat_dim 512 (config.py L66); 31.3 M params
        ├──────────┬──────────────┬──────────────┐
        ▼          ▼              ▼              ▼
   reg_head    ord_head       class_head    log_var_head
   Linear→1    OrderedCoral    Linear→4     Linear→1, clamp(-6,4)
   (L120)      Head (L121-122) (L123-124)   (L125-126)
```

**Data flow (module level):**
`preprocessing/run_preprocessing.py` (stages 0–5) → `artifacts/manifest.csv` + `cache/videos/*.npy` → `data/dataset.py EchoClipDataset` → `data/sampler.py build_sampler` → `engine/trainer.py Trainer.fit()` → `outputs/<run>/best.pt` → `engine/evaluate.py run_inference` → `engine/calibrate.py calibrate()` → `outputs/<run>/thresholds.json` → `run_eval.py` / `run_ensemble.py` / `run_selective.py` → `outputs/*.json`.

### 9.2 Key algorithms

**(a) Ordered cut-points** — `models/uef_net.py L50-57`:
```
cutpoints(): if no gaps → anchor
             else → concat[anchor, anchor + cumsum(softplus(raw_gaps) + 1e-4)]
forward(x):  score(x) − cutpoints().unsqueeze(0)
```
Because `softplus(·) > 0`, cut-points are strictly increasing → cumulative probabilities monotone by construction.

**(b) Measurement-uncertainty soft targets** — `losses/losses.py L30-36`:
```
s_k = 1 − Φ((t_k − EF)/σ),  clamped to [1e-4, 1−1e-4]
```

**(c) Six-term objective** — `losses/losses.py L172-181`:
```
total = w_reg·L_reg + w_ord·L_ord + w_consistency·L_con
      + w_class·L_class + w_nll·L_nll + w_rank·L_rank + w_head_consistency·L_heads
```

**(d) Threshold optimisation** — `engine/calibrate.py optimize_thresholds()`: grid search over 3 thresholds within `radius` of clinical boundaries, step 0.5, `min_gap` 2.0; lexicographic score `(min recall, mean recall, macro-F1, −distance from clinical)`.

**(e) Selective prediction** — `engine/selective.py`: 7 uncertainty signals; signal + threshold chosen on VAL by highest coverage meeting target; applied frozen to TEST.

### 9.3 Design decisions with rationale found in code/docs

| Decision | Rationale (quoted/paraphrased) | Source |
|---|---|---|
| `drw_epoch = 15` not 0 | "balancing from epoch 0 gave MAE 5.44 at equal min-recall, versus 4.29 with drw_epoch=15" | `README.md` §4.4; corroborated by `outputs/uefnet_r2p1d` (drw=0, MAE 5.549) vs `outputs/uefnet_drw` (drw=15, MAE 4.502) |
| `eval_use_keyframes = False` | "avoids privileged tracing annotations" — unavailable for a new clinical study | `config.py L145`; `data/dataset.py L214-216` |
| `persistent_workers = False` | "workers are torn down at end of each train epoch so they do NOT coexist with the validation loader's workers (that overlap = 2x processes = out-of-RAM on Windows spawn)" | `engine/trainer.py L118-120` |
| Monotonic upsampling for short clips | "avoids modulo wraparound, which would introduce an artificial last-frame → first-frame motion jump" | `preprocessing/utils/sampling.py L56-60` |
| `calibrate_on_full_val = True` | "the frozen decision rule generalizes far better when fit on ALL of VAL rather than the 30% calibration sub-split (~30 vs ~100 Severe cases)" | `engine/trainer.py L~240-248` |
| Stem adaptation sums RGB kernels | "Summing RGB kernels makes a one-channel grayscale input equivalent to repeating it over the original RGB stem" | `models/uef_net.py L91-93` |

### 9.4 Novel vs. standard/reused — function by function

| Location | Assessment |
|---|---|
| `models/uef_net.py L33-57 OrderedCoralHead` | **Adapted** — modifies CORAL parameterisation |
| `models/uef_net.py L22-30 CoralHead` | **Standard** — CORAL as published; retained for v1 checkpoint compatibility |
| `models/uef_net.py L60-108 _build_backbone` | **Standard wrapper** — calls `torchvision.models.video.r2plus1d_18`; stem-adaptation logic (L79-104) is **custom engineering** |
| `losses/losses.py soft_cumulative_targets` | **Novel** (per §5.1) |
| `losses/losses.py pairwise_rank_loss L72-82` | **Standard** pairwise ranking, custom gap/temperature masking |
| `losses/losses.py` DRW weighting L133-137 | **Standard** — Cao 2019 / Cui 2019 |
| `losses/lds.py` | **Standard** — Yang 2021 |
| `engine/calibrate.py fit_variance_expansion` | **Adapted** (per §5.4) |
| `engine/calibrate.py fit_temperature` | **Standard** — Guo 2017 |
| `engine/calibrate.py fit_conformal / apply_conformal` | **Standard** — split conformal |
| `engine/selective.py` | **Standard** framework (Chow) with **custom** `boundary` signal (distance to nearest clinical threshold ÷ total σ) |
| `models/ema.py` | **Standard** |
| `preprocessing/build_camus.py harmonize_to_echonet` | **Novel** (per §5.3) |
| `preprocessing/utils/sampling.py sample_indices` | **Custom** — cycle-aware window constraint |
| `engine/metrics.py _wilson_interval` | **Standard** statistics, hand-implemented |

### 9.5 Notation table

| Symbol | Meaning | Where |
|---|---|---|
| `EF` | Ejection fraction, % | throughout |
| `t_k` | k-th clinical EF threshold; (30, 40, 55) | `config.py L46 EF_THRESHOLDS` |
| `σ` (`ef_noise_sigma`) | Assumed EF measurement noise SD = 4.0 | `config.py L78` |
| `Φ(·)` | Standard normal CDF | `losses/losses.py L25-27 _ndtr` |
| `s_k` | Soft cumulative target `P(trueEF > t_k \| measuredEF)` | `losses/losses.py L30-36` |
| `y` | Severity class index ∈ {0,1,2,3} | `config.py L47 CLASS_NAMES` |
| `τ` (`logit_adjustment_tau`) | Logit-adjustment strength; 0.5 in final runs | `config.py L95` |
| `β` (`effective_num_beta`) | Effective-number weighting parameter = 0.9999 | `config.py L97` |
| `k` | Variance-expansion factor `std(true)/std(pred)`, clipped [1.0, 1.7] | `engine/calibrate.py fit_variance_expansion` |
| `σ_ale`, `σ_epi` | Aleatoric (log-var head) and epistemic (inter-clip) SD | `engine/selective.py total_uncertainty` |
| `M` | Ensemble member count = 3 | `run_ensemble.py --runs` |
| `N` | TTA clips per study = 10 at eval | `config.py L143`, overridden by `--n-tta` |

---

## 10. Algorithmic Complexity Analysis

**Applicable — for the threshold-optimisation and frontier-search routines only.** The neural network training/inference cost is standard CNN complexity and is not original algorithmic work.

**`engine/calibrate.py optimize_thresholds()`** — exhaustive constrained grid search.

- Grid size per threshold: `g_i = 2·radius_i/step + 1`. With `radius = (8, 8, 6)` (`config.py L118`) and `step = 0.5`: `g = (33, 33, 25)`.
- Triple-nested loop over ordered combinations (L99-119), pruned by `min_gap` — worst case `g₁·g₂·g₃ = 27,225` combinations.
- Each iteration calls `classification_metrics` over N samples → **O(N)**.
- **Time: O(g₁·g₂·g₃·N)** = O(27,225 × 1,288) ≈ 3.5 × 10⁷ elementary operations for VAL. Reasoning: three independent grids, one linear pass per candidate.
- **Space: O(N + K²)** — one prediction vector plus a 4×4 confusion matrix per candidate, not accumulated.
- **Best case** materially better: `min_gap` pruning skips combinations where `t₂ − t₁ < 2` or `t₃ − t₂ < 2`, removing roughly a third of the space in practice.

**`engine/selective.py fit_selective_rule()`** — signal × coverage sweep.

- Signals: `S = 7` (`uncertainty_signals` returns 7 keys).
- Coverage levels: `C = (1.00 − min_coverage)/0.01 + 1` = 41 for `min_coverage = 0.60`.
- Each `evaluate_at_coverage` performs `np.lexsort` → **O(N log N)**, then metrics **O(N)**.
- **Time: O(S·C·N log N)** = O(7 × 41 × 1,288 log 1,288) ≈ 3.7 × 10⁶.
- **Space: O(S·N)** — all signals materialised simultaneously in the returned dict.

**Not applicable** to the remainder of the component: `trainer.py`, `dataset.py`, `evaluate.py` are standard training/inference loops with no original algorithmic content.

---

## 11. Experimental Setup

### Hardware

**FACT (`README.md` §12):** "Windows 11, Python 3.11, CUDA 12.x, NVIDIA RTX 4060 Laptop (8 GB), 32 GB RAM."
**FACT (proposal §13, Table 7.1):** "Laptop, RTX 4060 8 GB, Ryzen 7, 32 GB RAM".

### Software

**FACT — declared (`Dilukshan/training/requirements.txt`):** `torch>=2.1`, `torchvision>=0.16`, `numpy>=1.26`, `pandas>=2.0`, `scipy>=1.11`, `opencv-python>=4.9`, `tqdm>=4.65`, `matplotlib>=3.7`.
**FACT — declared (`Dilukshan/preprocessing/requirements.txt`):** `numpy>=1.26`, `pandas>=2.0`, `opencv-python>=4.9`, `scikit-image>=0.22`, `scipy>=1.11`, `tqdm>=4.65`.

**FACT — actually installed in the environment used:**

| Library | Version |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| opencv-python (cv2) | 4.10.0 |
| matplotlib | 3.10.9 |
| tqdm | 4.67.3 |
| nibabel | 5.4.2 |
| scikit-learn | 1.7.2 |
| scipy | 1.17.0 |

**⚠️ CONFLICT:** `nibabel` is required by `preprocessing/build_camus.py` but appears in **neither** requirements file. A clean install following the documented instructions would fail on CAMUS conversion.

### Datasets

| Dataset | Details | Source |
|---|---|---|
| **EchoNet-Dynamic** | 10,030 apical-4-chamber videos; splits 7,465 / 1,288 / 1,277; EF labels + ED/ES volume tracings | `README.md` §5; `preprocessing/artifacts/manifest.csv` |
| **CAMUS** | 500 patients → 1,000 clips (2CH + 4CH); class distribution Severe 122 / Moderate 194 / Mild 496 / Normal 188 | verified from `preprocessing/artifacts/camus_manifest.csv` |

**License/terms (FACT, `README.md` §18):** both under Research Data Use Agreements prohibiting redistribution; excluded from the repo via `.gitignore`. EchoNet from Stanford Medicine (IRB approved), CAMUS from University Hospital of St Etienne.

**Preprocessing (FACT, `README.md` §4.1; `preprocessing/stage0-5*.py`):** 5 stages — scan/validate → decode to grayscale 112×112 → normalisation statistics by exact full decode → cache `uint8` `.npy` → verify. Results: `artifacts/audit_report.json` reports 10,030 labelled, 0 missing, 0 unreadable, 6 metadata mismatches, elapsed 4.1 s. `artifacts/verification_report.json` reports `overall: PASS_WITH_WARNINGS`, 10,030 valid caches, 0 bad, clip length min 28 / median 171 / max 1002.

**⚠️ CONFLICT (normalisation statistics provenance):**
- `artifacts/norm_stats.json` states `"source": "exact_full_decode_train"`, `n_videos_sampled: 7465`, `n_train_pixels: 16,499,624,960`.
- `artifacts/verification_report.json` states `norm_stats_videos: 512`, `norm_stats_pixels: 51,380,224`.
These disagree by ~15× in video count and ~320× in pixel count. **[inferred]** the verification report recomputed on a 512-video subsample to check drift (it reports `drift_mean 0.001022`, tolerance 0.02) rather than describing the original computation — but **this must be confirmed by the author.**

### Environment

- **FACT:** Windows 11 (`README.md` §12); no Docker/`Dockerfile` present; no `environment.yml`; venv creation documented in `README.md` §12 (`python -m venv .venv`).
- **FACT:** No environment variables are read by the training code (`config.py` uses only `os.cpu_count()`).
- **⚠️ NOT FOUND IN CODEBASE — needs input from author:** no `setup.py`, `pyproject.toml`, or installable package definition.

### Compute Cost

- **FACT (`README.md` §13.3):** "Approximately 20 hours per run on an RTX 4060."
- **FACT (per-epoch timings logged in `outputs/<run>/train_log.csv`, `sec` column):** ~1,730–1,930 s per epoch across runs.
- **FACT (measured, backbone ablation — §16.1):** at the shipped input geometry with AMP, `r2plus1d_18` runs at **1.84 s/step** peaking at **4.22 GB**; `r3d_18` at **0.262 s/step** peaking at **1.86 GB**. End-to-end: **23.5 h against 4.02 h** for the same 45-epoch schedule. This closes the peak-VRAM half of the gap noted below.
- **FACT (proposal §13.2):** stage timings for preprocessing declared as ~6 min (preprocess), ~10 s (split).
- **⚠️ NOT LOGGED:** total wall-clock across all experiments, peak GPU memory, and energy are **not** recorded anywhere. Observed GPU memory during a run was 6.3/8.0 GB (session observation, not a repo artifact). **Recommend logging peak VRAM and cumulative compute now, not at write-up time.**

---

## 12. Parameters / Configuration

**FACT — all from `Dilukshan/training/config.py` (line numbers given).** Values shown are dataclass defaults; the *actually used* values for the final runs are confirmed from `outputs/<run>/config.json`.

| Parameter | Default (line) | Final-run value | Notes |
|---|---|---|---|
| `run_name` | `"uefnet_r2p1d"` (L33) | `uefnet_v3` / `v3b` / `v3c` | |
| `seed` | 1337 (L34) | **1337 / 2024 / 777** | the only variable across ensemble members |
| `clip_len` | 32 (L50) | 32 | |
| `sampling_period` | 2 (L51) | 2 | spans 64 native frames |
| `frame_size` | 112 (L52) | 112 | |
| `in_channels` | 2 (L53) | 2 | grayscale + motion |
| `motion_mode` | `"tempdiff"` (L54) | tempdiff | |
| `aug_pad` | 12 (L57) | 12 | |
| `aug_intensity_jitter` | 0.1 (L58) | 0.1 | |
| `aug_time_jitter` | True (L59) | True | |
| `cycle_aware_probability` | 0.5 (L60) | 0.5 | |
| `backbone` | `"r2plus1d_18"` (L63) | r2plus1d_18 | |
| `pretrained` | True (L64) | True | Kinetics-400 |
| `dropout` | 0.3 (L65) | 0.3 | |
| `feat_dim` | 512 (L66) | 512 | |
| `model_version` | `"uefnet_v1"` (L72) | **`uefnet_v2`** | set by `--v2` |
| `use_class_head` | True (L73) | True | |
| `predict_uncertainty` | True (L74) | True | |
| `aux_channel_init_scale` | 0.05 (L75) | 0.05 | |
| `ef_noise_sigma` | 4.0 (L78) | 4.0 | **C1 parameter** |
| `w_reg` | 1.0 (L79) | 1.0 | |
| `w_ord` | 1.0 (L80) | 1.0 | |
| `w_consistency` | 0.15 (L81) | 0.15 | |
| `huber_delta` | 1.0 (L82) | 1.0 | |
| `lds_kernel_sigma` | 2.0 (L83) | 2.0 | |
| `lds_ks` | 5 (L84) | 5 | |
| `w_class` | 0.5 (L86) | 0.5 | |
| `w_nll` | 0.5 (L87) | 0.5 | |
| `w_rank` | 0.2 (L88) | 0.2 | |
| `w_head_consistency` | 0.1 (L89) | 0.1 | |
| `class_target_sigma` | 4.0 (L90) | 4.0 | |
| `logit_adjustment_tau` | 0.0 (L95) | **0.5** | set by CLI |
| `lds_weight_power` | 0.5 (L96) | 0.5 | |
| `effective_num_beta` | 0.9999 (L97) | 0.9999 | |
| `rank_min_gap` | 3.0 (L98) | 3.0 | EF points |
| `rank_temperature` | 4.0 (L99) | 4.0 | |
| `extra_manifests` | `()` (L105) | `('artifacts/camus_manifest.csv',)` | CAMUS co-training |
| `use_balanced_sampler` | True (L108) | True | |
| `calibrate_on_full_val` | True (L117) | True | **C4 parameter** |
| `threshold_search_radius` | (8.0, 8.0, 6.0) (L118) | (8,8,6) | **C4 parameter** |
| `drw_epoch` | 15 (L124) | 15 | |
| `calibration_fraction` | 0.30 (L125) | 0.30 | |
| `epochs` | 45 (L128) | 45 | |
| `batch_size` | 8 (L129) | 8 | |
| `grad_accum` | 4 (L130) | 4 | effective batch 32 |
| `lr_backbone` | 1e-4 (L131) | 1e-4 | |
| `lr_head` | 1e-3 (L132) | 1e-3 | |
| `weight_decay` | 1e-4 (L133) | 1e-4 | |
| `warmup_epochs` | 2 (L134) | 2 | |
| `grad_clip` | 2.0 (L135) | 2.0 | |
| `amp` | True (L136) | True | |
| `use_ema` | True (L139) | True | |
| `ema_decay` | 0.999 (L140) | 0.999 | |
| `n_tta_clips` | **5** (L143) | **10** (via `--n-tta 10`) | **⚠️ default ≠ used value** |
| `tta_forward_batch` | 8 (L144) | 8 | |
| `eval_use_keyframes` | False (L145) | False | |
| `early_stop_patience` | **12** (L146) | **18** (via `--patience 18`) | **⚠️ default ≠ used value** |
| `num_workers` | `min(4, cpu_count−2)` (L151) | 2–4 | varied across resumes |
| `device` | `"cuda"` (L152) | cuda | |
| `EF_THRESHOLDS` | (30.0, 40.0, 55.0) (L46) | unchanged | |

**⚠️ CONFLICT flagged:** `n_tta_clips` default is 5 but all reported results use 10; `early_stop_patience` default is 12 but runs used 18. A reader running the documented default command would **not** reproduce the reported numbers exactly.

**Preprocessing config:** `preprocessing/config.py` uses a `Config()` dataclass instantiated at L127; individual field lines were not extractable by the same pattern. **NOT FULLY ENUMERATED — author should list preprocessing parameters explicitly.**

---

## 13. Baseline(s) Compared Against

**FACT — internal baselines that exist as runs:**

| Run | Config | Test MAE | R² | Overall | Balanced | Min-recall | Macro-F1 | Source |
|---|---|---|---|---|---|---|---|---|
| `uefnet_r2p1d` | v1, drw_epoch 0, no EMA, no CAMUS | 5.549 | 0.668 | 0.684 | 0.684 | 0.651 | 0.633 | `outputs/uefnet_r2p1d/test_report.json` |
| `uefnet_drw` | v1, drw_epoch 15, EMA, no CAMUS | 4.502 | 0.777 | 0.689 | 0.702 | 0.651 | 0.647 | `outputs/uefnet_drw/test_report.json` |
| `uefnet_v3` | v2 + CAMUS + τ=0.5, single seed | 4.138 | 0.804 | 0.721 | 0.730 | 0.687 | 0.679 | `outputs/uefnet_v3/test_report.json` |
| **3-seed ensemble** | v3 + v3b + v3c | **3.979** | **0.818** | 0.730 | 0.737 | **0.723** | 0.684 | `outputs/ensemble_report.json` |

**FACT — external baseline:** published EchoNet-Dynamic result (Ouyang et al. 2020, MAE 4.05, R² 0.81) quoted in `README.md` §7.5. This is a **literature comparison, not a re-implementation.**

**⚠️ GAPS FLAGGED:**
1. **No re-implemented baseline.** No competing method was trained on this split by this codebase. The comparison to [1], [6], [7] uses *their published numbers*, not controlled re-runs. This is a standard reviewer objection.
2. **The v1 → v3 comparison is confounded.** `uefnet_drw` → `uefnet_v3` changes **four things at once**: architecture (v1→v2), CAMUS co-training, logit adjustment (0→0.5), and calibration method. The isolated effect of any one cannot be read from these runs.
3. **No naive/trivial baseline** (e.g. majority-class predictor, or EF-mean regressor) is computed, though `README.md` §2 cites the 68.9 % degenerate accuracy figure rhetorically.

---

## 14. Evaluation Metrics

**FACT — computed in `engine/metrics.py`:**

| Metric | Function (line) | Type | Why it fits | Answers RQ |
|---|---|---|---|---|
| MAE | `regression_metrics` L91 | Regression | Directly comparable with published EchoNet benchmark | RQ1 |
| RMSE | `regression_metrics` L91 | Regression | Penalises large EF errors | RQ1 |
| R² | `regression_metrics` L91 | Regression | Variance explained | RQ1 |
| Overall accuracy | `classification_metrics` L57 | Classification | Headline figure (caveated for imbalance) | RQ1 |
| Balanced accuracy | `classification_metrics` L57 | Classification | Removes majority-class advantage under 11:1 imbalance | RQ1, RQ2 |
| Per-class recall | `per_class_recall` L34 | Classification | Establishes minority-class safety — the core claim | RQ2, RQ3 |
| Per-class F1 | `per_class_f1` L44 | Classification | Balances precision/recall per class | RQ1 |
| Macro-F1 | `classification_metrics` L57 | Classification | Unweighted class average | RQ1 |
| Minimum class recall | `classification_metrics` L57 | Classification | **Primary model-selection criterion** (`trainer.py` `score = (mn, -mae)`) | RQ2, RQ3, RQ4 |
| Confusion matrix | `confusion_matrix` L27 | Classification | Error-mode structure | RQ1 |
| Wilson interval | `_wilson_interval` L6 | Statistical | Small-count binomial CI (Severe n=83) | all |
| Per-class regression metrics | `regression_metrics_by_class` L110 | Regression | Whether error concentrates in a class | RQ1 |
| Brier / NLL / ECE | `probability_metrics` L120 | Calibration | Probability quality | RQ1 |
| Bootstrap CI on regression | `bootstrap_regression_ci` L139 (n=2000) | Statistical | Interval on MAE | RQ1 |
| Prediction-interval coverage | `prediction_interval_metrics` L170 | Uncertainty | Conformal interval validity | RQ4 |
| Coverage (selective) | `engine/selective.py evaluate_at_coverage` | Selective | Fraction graded vs deferred | RQ4 |

**⚠️ Note:** `probability_metrics`, `bootstrap_regression_ci` and `prediction_interval_metrics` **exist in code** but do **not** appear in `outputs/ensemble_report.json` — they are implemented but not reported in the final result artifact.

---

## 15. Experimental Repetition & Statistical Robustness

**FACT:**
- **Repetition: 3 runs**, seeds 1337 / 2024 / 777 (`outputs/uefnet_v3*/config.json`). Seeds are **fixed and explicitly varied**, not random.
- Seed control: `core/common.py seed_everything`, plus `seed_worker` / `make_torch_generator` for DataLoader determinism; RNG state is captured and restored on resume (`capture_rng_state` / `restore_rng_state`).
- **Per-member variance IS observable** (validation): min-recall 0.693 / 0.701 / 0.697; MAE 3.941 / 3.888 / 4.045 (`README.md` §8.2).
- **Confidence intervals: Wilson intervals implemented** (`metrics.py L6`) and reported in `README.md` §7.2 (e.g. Severe 0.723, 95 % CI [0.618, 0.808]).
- **Bootstrap CI implemented** (`bootstrap_regression_ci`, n_bootstrap = 2000) — **but not present in the final reported artifacts.**
- Reproducibility check: calibration re-run produced byte-identical output (`README.md` §8.2).

**⚠️ WEAKNESSES TO FLAG:**
1. **No formal significance testing anywhere.** No t-test, ANOVA, McNemar, or p-value exists in the codebase (grep for `ttest`, `pvalue`, `scipy.stats` returns nothing in the analysis path). `README.md` §6 states this openly: "No paired significance test is performed between configurations."
2. **n = 3 seeds** is small for variance estimation; no error bars on the headline ensemble result.
3. **The ensemble result itself is a single number** — there is no distribution over ensembles (only one 3-seed combination exists).
4. Test-set results are **single-run per configuration** — the test split is deliberately evaluated once (documented protocol, `README.md` §6), which is methodologically sound but means no test-side variance is available.

---

## 16. Ablation Studies

**FACT — ablations that EXIST as measured comparisons (`README.md` §8):**

| Ablation | Before | After | Controlled? |
|---|---|---|---|
| Tail-decompression calibration | Severe recall 0.590 | 0.687 | ✅ **Fully** — identical weights (`test_report_baseline.json` vs `test_report.json`) |
| Ensembling 1 → 2 seeds | MAE 4.138, min-rec 0.687 | 3.994 / 0.711 | ✅ Fully |
| Ensembling 2 → 3 seeds | MAE 3.994, min-rec 0.711 | 3.979 / 0.723 | ✅ Fully |
| Deferred vs immediate re-weighting | MAE 5.44 (drw=0) | 4.29 (drw=15) | ✅ Fully (corroborated by `uefnet_r2p1d` vs `uefnet_drw` runs) |
| CAMUS co-training | Moderate 0.53 | 0.62 | ⚠️ **Confounded** — README §8 itself labels this "matched epochs across runs, not single-variable" |
| Selective prediction | min-rec 0.723 @100 % | 0.706 @88.4 % | ✅ Fully (negative result) |
| **Backbone: R(2+1)D-18 → R3D-18** (3 seeds each) | acc 0.7298, bal-acc 0.7366 | acc 0.7063, bal-acc 0.7145 | ✅ **Fully** — three matched seeds per architecture, every hyperparameter inherited, verified by a fail-closed parity audit. **R(2+1)D significantly better**: accuracy p = 0.0064, balanced accuracy p = 0.0310, McNemar p = 0.0064. See §16.1 |
| *(superseded)* single-seed backbone run | MAE 4.1417 | MAE 4.1175 | ⚠️ **Confounded and underpowered** — `logit_adjustment_tau` and `n_tta_clips` moved with the backbone, and one seed per side cannot separate architecture from seed variance. Retained as the counter-example that motivated the audit |

**⚠️ ABLATIONS THAT DO NOT EXIST — suggested, given current code structure:**

| Missing ablation | How to run it with existing code |
|---|---|
| **C1 — soft ordinal labels** | `config.py L78 ef_noise_sigma`: set → ~0.01 to approximate hard labels; retrain one seed; compare. Currently no CLI flag exists — would need one. |
| **C2 — ordered-cutpoint head** | Train one seed with `--model-version uefnet_v1` (CoralHead) holding CAMUS + τ constant. **The CLI flag already exists** — this is the cheapest missing ablation. |
| **Logit adjustment τ** | `--logit-adjustment-tau 0.0` vs `0.5`, single variable. Flag exists. |
| **Motion channel** | `config.py L54 motion_mode="none"` + `in_channels=1`. Validation in `config.py L200-201` already enforces the pairing. |
| **Cycle-aware sampling** | `--cycle-aware-probability 0.0` (flag exists in `run_train.py`). |
| **CAMUS, isolated** | Train v2 + τ=0.5 **without** `--extra-manifest`, same seed. Single-variable version of the confounded comparison. |
| **Harmonization, isolated** | `build_camus.py --no-harmonize` then retrain — directly tests the C3 shortcut hypothesis. Flag exists. |

**Each of these costs ~20 h of training** on the documented hardware, except the calibration ones which are ~10 min.

---

### 16.1 Backbone ablation — does the R(2+1)D factorisation actually help? (`outputs/uefnet_r3d/`)

**Motivation.** The R(2+1)D-18 backbone was adopted from the EchoNet-Dynamic benchmark (Ouyang et al. 2020), whose own ablation found it best for EF regression. That is a sound starting point, but it is someone else's result on someone else's formulation. This ablation tests it under *this* component's four-head ordinal design, CAMUS co-training and calibration.

**Design.** The intent was that one variable changes: `--backbone r3d_18` against the shipped `r2plus1d_18`, with seed (1337), epoch schedule (45), clip geometry, heads, losses, co-training manifests and calibration procedure all matched. `r3d_18` is the un-factorised baseline R(2+1)D was designed to beat (Tran et al., CVPR 2018) at near-matched capacity — **33.4 M parameters against 31.5 M** — so the comparison isolates the factorisation rather than confounding it with model size.

> **⚠️ The executed run did not achieve that.** `run_backbone_ablation.py` forwarded `--extra-manifest` from the baseline snapshot but nothing else, so the challenger fell back to `config.py` defaults for two settings the baseline had overridden on the command line:
>
> | Setting | Baseline (`uefnet_v3`) | Challenger (`uefnet_r3d`) | Affects |
> |---|---|---|---|
> | `logit_adjustment_tau` | 0.5 | **0.0** | the training loss |
> | `n_tta_clips` | 5 | **10** | model selection and per-run calibration |
>
> Three variables moved, not one. The conclusion below is unchanged — every metric is a null under both framings, and the two settings push in opposite directions — but **this run does not support the claim that the backbone alone was tested.** It is reported here as a confounded comparison rather than withdrawn, because the null is still informative about the joint change.
>
> The replacement is `run_backbone_ensemble.py`, which inherits every hyperparameter from the baseline snapshot via an explicit config-key-to-flag table and **refuses to report any comparison until a fail-closed parity audit confirms only the backbone and seed differ.** It trains three matched R3D seeds so the comparison is ensemble against ensemble rather than single model against single model.

The comparison is **single model vs single model**. It is deliberately *not* run against the three-seed ensemble headline (MAE 3.979), which would read a variance-reduction effect as an architecture effect.

**PRIMARY RESULT — three seeds per architecture, matched configuration.**

`run_backbone_ensemble.py` trained three R3D seeds (1337, 2024, 777) matching the three shipped R(2+1)D members seed-for-seed, with every hyperparameter inherited from the baseline snapshot and a fail-closed parity audit confirming that only the backbone and the seed differ. Both ensembles use 10 TTA views, and both independently selected `reg_operational_expanded` on VAL.

| Metric | R(2+1)D-18 ×3 | R3D-18 ×3 | Δ | 95 % CI | p |
|---|---|---|---|---|---|
| MAE (EF points) | **3.9794** | 4.0330 | +0.0536 | [−0.041, +0.148] | 0.270 |
| Overall accuracy | **0.7298** | 0.7063 | −0.0235 | [−0.040, −0.008] | **0.0064** |
| Balanced accuracy | **0.7366** | 0.7145 | −0.0221 | [−0.044, −0.002] | **0.0310** |
| Min-class recall | **0.7229** | 0.6975 | −0.0254 | — | — |
| Macro-F1 | **0.6844** | 0.6642 | −0.0202 | — | — |
| R² | **0.8183** | 0.8117 | −0.0066 | — | — |
| Exact McNemar | **72** correct-only | 42 correct-only | 114 discordant | — | **0.0064** |

Paired bootstrap over identical study indices, 10,000 resamples (`outputs/robustness_ensemble_r3d.json`).

**R(2+1)D-18 wins every metric, and three of the four significance tests reject the null at α = 0.05.**

**Interpretation.** The split matters: classification improves significantly while MAE does not (p = 0.270). The factorisation is not making the regression more accurate on average — it is resolving the class boundaries better. That is consistent with what the factorisation does. Separating each 3×3×3 convolution into a (1,3,3) spatial and a (3,1,1) temporal step gives the network an explicit temporal-only stage, and ejection fraction is a temporal quantity: the grade depends on how the ventricle moves between end-diastole and end-systole, not on either frame alone. Meanwhile average EF error is dominated by the ~37 % of studies lying within one MAE of a threshold (§10.2), where a small gain moves studies across a boundary without moving MAE much.

**Baseline reproduction.** Rebuilding the shipped ensemble from its three frozen members reproduced the published regression and classification blocks bit-for-bit (`outputs/ensemble_report_baseline_verify.json` against `outputs/ensemble_report.json`), independently confirming the determinism claimed in `README.md` §8.2.

**SUPERSEDED — the first single-model comparison** (`outputs/uefnet_r3d/`). The original one-seed-per-architecture run found no significant difference anywhere (MAE p = 0.889, accuracy p = 0.217, McNemar p = 0.233) and marginally favoured R3D. It is superseded for two independent reasons, and both were needed to reverse the conclusion:

1. **It was confounded** — `logit_adjustment_tau` (0.5 → 0.0) and `n_tta_clips` (5 → 10) moved with the backbone, as described above.
2. **It was underpowered** — one seed per side cannot separate an architecture effect from seed variance; three per side can.

The superseded run is kept in the repository rather than deleted, because the discrepancy between it and the corrected comparison is itself the evidence that the parity audit was necessary.

**Cost.** Measured on the documented RTX 4060 Laptop at the shipped input geometry (8 × 2 × 32 × 112 × 112, AMP on): R(2+1)D-18 runs at **1.84 s/step** with **4.22 GB** peak activation memory, against R3D-18's **0.262 s/step** and **1.86 GB**. End-to-end the two runs took **23.5 h against 4.02 h** for the same 45-epoch schedule. The wall-clock ratio (≈ 6×) is smaller than the GPU-only ratio (≈ 7×) because at 0.262 s/step R3D outruns the four-worker data pipeline and idles waiting for clips. Mechanically, the factorisation replaces each 3×3×3 convolution with a (1,3,3) spatial and a (3,1,1) temporal convolution, raising the backbone from **20 to 37 `Conv3d`** and **20 to 37 `BatchNorm3d`** layers and producing an intermediate tensor **1.8–2.25× wider than the block's own output**. The workload is memory-bandwidth-bound, so that intermediate traffic — not arithmetic — dominates.

**Reproduce:** `python run_backbone_ablation.py` (train, evaluate, compare), or `--compare-only` if both runs already exist.


---

## 17. Existing Figures / Visual Assets Inventory

**FACT — files present in repository:**

| Path | What it shows |
|---|---|
| `Dilukshan/preprocessing/artifacts/viz/class{0..3}_*_clip.png` | Sample clip frames per severity class (4 files) |
| `Dilukshan/preprocessing/artifacts/viz/class{0..3}_*_motion.png` | Temporal-difference motion channel per class (4 files) |
| `Dilukshan/preprocessing/artifacts/viz/class{0..3}_*_ed_es_contours.png` | ED/ES ground-truth contours per class (4 files) |
| `Dilukshan/training/outputs/<run>/training_curves.png` | Loss / MAE / min-recall per epoch, per run |
| `Dilukshan/training/outputs/<run>/confusion_test.png` | Test confusion matrix per run |
| `Dilukshan/training/outputs/selective_coverage_curve.png` | Per-class recall vs coverage, with operating point marked |
| `proposal_figures/fig_system.png` | 4-component system diagram; Component 03 highlighted |
| `proposal_figures/fig_component.png` | UEF-Net 5-stage internal pipeline |
| `proposal_figures/fig_dataflow.png` | 7-step data flow |
| `proposal_figures/fig_paths.png` | Training vs inference path comparison |
| `proposal_figures/fig_distribution.png` | EF histogram with class boundaries, colour-coded |
| `proposal_figures/fig_gantt.png` | Project schedule with 8 milestones |
| `proposal_figures/fig_ui.png` | Clinician review interface mock-up |

**Directly reusable in the paper:** `fig_distribution.png` (Fig. 1 candidate), `fig_component.png` (architecture), `confusion_test.png` (results), `selective_coverage_curve.png` (§9 analysis), `viz/*` (qualitative examples).

**⚠️ NOT PRESENT:** no calibration/reliability diagram plot, no Bland-Altman plot (though `bland_altman_loa95` exists in `metrics.py`), no per-class error-distribution figure.

---

## 18. Results Found in Repo (facts only)

### 18.1 Primary — 3-seed ensemble (`outputs/ensemble_report.json`)

| Metric | Value |
|---|---|
| Strategy | `reg_operational_expanded` |
| MAE | 3.9794 |
| RMSE | 5.2108 |
| R² | 0.8183 |
| Overall accuracy | 0.7298 |
| Balanced accuracy | 0.7366 |
| Macro-F1 | 0.6844 |
| Min class recall | 0.7229 |
| n | 1,277 |

**Per-class (`ensemble_report.json`):**

| Class | Recall | Precision | F1 |
|---|---|---|---|
| Severe | 0.7229 | 0.9524 | 0.8219 |
| Moderate | 0.7662 | 0.4338 | 0.5540 |
| Mild | 0.7303 | 0.4131 | 0.5277 |
| Normal | 0.7272 | 0.9770 | 0.8338 |

**Confusion matrix (`ensemble_report.json`):** `[[60,23,0,0],[3,59,15,0],[0,50,176,15],[0,4,235,637]]`

**Clinical reference (same file):** overall 0.7956, MAE 3.9825, min-recall 0.4416.

### 18.2 Single-model and baseline runs

| Run | MAE | R² | Overall | Balanced | Min-recall | Macro-F1 | Per-class recall |
|---|---|---|---|---|---|---|---|
| `uefnet_r2p1d` | 5.549 | 0.668 | 0.684 | 0.684 | 0.651 | 0.633 | [0.651, 0.701, 0.701, 0.682] |
| `uefnet_drw` | 4.502 | 0.777 | 0.689 | 0.702 | 0.651 | 0.647 | [0.651, 0.740, 0.743, 0.674] |
| `uefnet_v3` | 4.138 | 0.804 | 0.721 | 0.730 | 0.687 | 0.679 | [0.687, 0.766, 0.755, 0.711] |
| `uefnet_v3` (pre-calibration-fix) | 4.166 | — | 0.742 | 0.708 | 0.590 | — | [0.590, 0.740, 0.747, 0.756] |
| `uefnet_r3d` (backbone ablation, §16.1) | 4.130 | 0.803 | 0.734 | 0.712 | 0.651 | 0.669 | [0.651, 0.753, 0.693, 0.751] |

### 18.3 Selective prediction (`outputs/selective_report.json`)

- Signal selected on VAL: `boundary`; coverage 0.8841 (1,129 graded / 148 deferred)
- Covered: overall 0.7697, balanced 0.7425, min-recall 0.7059; per-class [0.7059, 0.7500, 0.7277, 0.7864]
- **Deferred-subset accuracy 0.4257**
- VAL target 0.75 **not reached** (best min-recall 0.738 at 88 % coverage)

### 18.4 Validation-side per-member (`README.md` §8.2)

| Run | Val min-recall | Val MAE |
|---|---|---|
| `uefnet_v3` | 0.693 | 3.941 |
| `uefnet_v3b` | 0.701 | 3.888 |
| `uefnet_v3c` | 0.697 | 4.045 |

### 18.5 Preprocessing verification

- `artifacts/audit_report.json`: 10,030 labelled / 10,030 on disk / 0 missing / 0 unreadable / 6 metadata mismatch / 0 orphans / 4.1 s elapsed
- `artifacts/verification_report.json`: `PASS_WITH_WARNINGS`, 10,030 valid caches, 0 bad, clip length min 28 / median 171 / max 1002, drift_mean 0.001022 (tolerance 0.02)
- `artifacts/norm_stats.json`: ef_mean 55.777643, ef_std 12.406359, pixel_mean 0.128799, pixel_std 0.195979

### 18.6 CAMUS conversion

- `artifacts/camus_manifest.csv`: **1,000 rows**; class distribution Severe 122 / Moderate 194 / Mild 496 / Normal 188 (verified by direct count)

---

## 19. Interpretation Notes

**Interpretations already written by the author, quoted with source:**

- **On the ceiling** (`README.md` §10.1): "Balanced accuracy is 0.737… reaching 0.75 requires balanced accuracy ≈ 0.764 — a property of the model, not of any decision rule."
- **On boundary ambiguity** (`README.md` §10.2): "Over a third of the cohort lies within one MAE of a threshold, and the dominant error mode (Normal → Mild, 235 cases) is exactly the EF = 55 cluster."
- **On the label-noise floor** (`README.md` §10.3): "At MAE 3.979 the model already operates at the level of human reader disagreement; further reduction would require it to be more self-consistent than the annotations it learns from."
- **On the selective negative result** (`README.md` §9): "Moderate occupies a 10-point interior band (30–40), so every Moderate study is near a decision boundary by construction, whereas Normal is open-ended (≥55)… aggregate accuracy rises and the floor falls."
- **On uncertainty validity** (`README.md` §9): "accuracy on deferred studies is 0.426 versus 0.770 on graded studies, confirming that the learned log-variance head plus test-time disagreement genuinely identify hard cases."
- **On precision** (`README.md` §7.3): "At 0.36 % (UA) and 0.46 % (STEMI) prevalence…" — **⚠️ this sentence in `README.md` §7.3 refers to UA/STEMI, which are Component 04's classes, not Component 03's.** Appears to be copied text. **Requires author correction.**
- **On the calibration fix** (`README.md` §15): "Calibration fitted on 30 % of validation (~30 Severe studies) → Severe threshold failed to generalise to test."

**Requires author interpretation — human judgment call, not extractable from code:**
- Whether the ceiling analysis constitutes a *contribution* or a *limitation* in the paper's framing.
- Clinical significance of 99.7 % within-one-class and zero catastrophic errors (needs clinician input).
- Whether the selective-prediction negative result is publishable as a finding or should be relegated to an appendix.

---

## 20. Limitations & Threats to Validity

**FACT — stated by author (`README.md` §16), 7 items:** target not met (0.723–0.766 vs 0.75); small Severe test set (n=83, ±9.5 % CI); partially controlled cross-dataset ablation; single-cohort evaluation; single-institution geographic origin (US and France); low interior-class precision (Moderate 0.434, Mild 0.413); not a certified medical device.

**ADDITIONAL threats visible in code, not listed by the author:**

| Threat | Evidence |
|---|---|
| **Hardcoded EF noise σ = 4.0** | `config.py L78` — the C1 novelty depends on a literature-derived constant that is never tuned or sensitivity-tested |
| **Slope/expansion clipping** | `calibrate.py fit_affine_ef` bounds slope to [0.5, 1.5]; `fit_variance_expansion` clips k to [1.0, 1.7]. If the unclipped value hits a bound, the correction is truncated — `slope_unclipped` / `k_unclipped` are recorded but not checked |
| **Threshold search is bounded** | `threshold_search_radius = (8,8,6)` — the optimum could lie outside the searched region and would be silently missed |
| **VAL used twice** | Model selection *and* calibration both consume validation data. `calibration_fraction = 0.30` exists to separate them, but `calibrate_on_full_val = True` overrides this for the final calibration — an accepted trade-off documented in `trainer.py` but a mild optimistic bias |
| **Windows-specific worker handling** | `trainer.py L118-120` sets `persistent_workers=False` for Windows spawn RAM reasons; performance characteristics differ on Linux |
| **`num_workers` varied across resumes** | 2 vs 4 across interrupted/resumed runs — changes the augmentation RNG stream, so runs are not bit-reproducible end-to-end despite seed control |
| **Interrupted training** | `uefnet_v3c` best epoch 29 of 45 planned; training was stopped early rather than completing the schedule (session logs, `train_log.csv`) |
| **`nibabel` undeclared** | See §11 — clean-install reproduction of CAMUS conversion fails |
| **No tests for training code** | Tests exist only under `preprocessing/tests/` (4 files); `training/` has **no test suite** |

---

## 21. Ethical & Societal Considerations

- **Data privacy:** **Applicable.** Both datasets are patient echocardiograms. **FACT (`README.md` §18):** both released after institutional ethical review and fully anonymised at source; "No attempt is made to re-identify any individual." No consent-handling code exists in the repository (none required — de-identified secondary use).
- **Data redistribution:** **Applicable and handled.** `.gitignore` excludes `Dataset/`, `*.avi`, `*.nii*`, `cache/`, `*.npy`, `*.pt`. Verified: the pushed GitHub repository contains **zero** medical files or checkpoints.
- **Potential misuse:** **Applicable.** `README.md` §18: "Intended use is decision support, not autonomous diagnosis… Deployment would require prospective validation on the target population and appropriate regulatory clearance." Misuse risk = deployment as an autonomous grader without clinician review.
- **Fairness/bias:** **Applicable but NOT ANALYSED.** The model makes clinical severity decisions about people. **⚠️ NO subgroup analysis exists** — no breakdown by age, sex, ethnicity, or image quality anywhere in the codebase, despite EchoNet containing demographic fields. `README.md` §16 notes single-institution origin but performs no fairness evaluation. **This is a genuine gap for a clinical-AI paper.**
- **Environmental/compute cost:** **Applicable.** ~20 h per run × 3 seeds ≈ 60 h GPU on an RTX 4060 laptop, plus baseline runs. Not measured in kWh or CO₂e. **NOT LOGGED.**
- **Dataset licensing:** **Applicable and handled.** Both under Research Data Use Agreements; obtained under those agreements; not redistributed; CAMUS carries a mandatory-citation requirement, satisfied in `README.md` §19 [4].

---

## 22. Reproducibility / How to Run

**FACT — from `README.md` §12–13:**

**Setup:**
```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas opencv-python matplotlib tqdm nibabel
```

**Data:** not included; obtain EchoNet-Dynamic from `https://echonet.github.io/dynamic/` and CAMUS from `https://www.creatis.insa-lyon.fr/Challenge/camus/`. Place EchoNet at `Dilukshan/Dataset/`.

**Pipeline:**
```bash
cd Dilukshan/preprocessing
python run_all.py                    # ⚠️ CONFLICT: actual file is run_preprocessing.py
python build_camus.py

cd ../training
python run_train.py --v2 --extra-manifest artifacts/camus_manifest.csv \
    --logit-adjustment-tau 0.5 --seed 1337 --run-name uefnet_v3  --epochs 45 --patience 18
# repeat with --seed 2024 --run-name uefnet_v3b, --seed 777 --run-name uefnet_v3c

python run_train.py --calibrate-only --run-name uefnet_v3 --n-tta 10
python run_eval.py --run-name uefnet_v3 --n-tta 10
python run_ensemble.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10
python run_selective.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10 --plot
```

**Resume:** `python run_train.py --resume --run-name uefnet_v3b`

**Entry points:** `run_train.py`, `run_eval.py`, `run_ensemble.py`, `run_selective.py`, `preprocessing/run_preprocessing.py`, `preprocessing/build_camus.py`.

**Outputs:** `outputs/<run>/{best.pt, last.pt, config.json, norm_stats.json, thresholds.json, train_log.csv, training_curves.png, validation_partition.json, test_report.json, confusion_test.png}`, plus `outputs/ensemble_report.json`, `outputs/selective_report.json`.

**Artifact status:** Code is public at `https://github.com/Dilukshan285/Research_Component_03`. **It is a code artifact, not a data or model artifact** — datasets and checkpoints are excluded by licence and size. A third party with dataset access could reproduce the pipeline; they could not verify the reported numbers without ~60 h of GPU time.

**⚠️ CONFLICTS FOUND:**
1. `README.md` §13.2 documents `python run_all.py` but the file in `preprocessing/` is **`run_preprocessing.py`**. The documented command **will fail**.
2. `nibabel` is in the pip line of §12 but absent from both `requirements.txt` files.
3. Default `n_tta_clips=5` and `early_stop_patience=12` differ from the values used (10, 18) — see §12.

---

## 23. My Individual Role / Contribution Statement

**FACT — git history:**
- `git config user.name` = **Dilukshan Viyapury**, `user.email` = `Dviyapury@gmail.com`
- `git rev-list --count HEAD` = **4 commits**
- `git shortlog -sne` returned **empty output** — no per-author breakdown was produced
- Commits: `e3b18e6` "Implement code changes to enhance functionality and improve performance", `0a6780e` "Update ensemble_report.json with revised metrics and additional run configuration", `e77fbff` "Refactor code structure for improved readability and maintainability", `553a25d` "UEF-Net: uncertainty-aware ordinal EF severity grading"

**⚠️ Git history is INSUFFICIENT for a contribution statement, for three reasons:**
1. Only 4 commits exist for a project of this size (~60 source files), all made after the work was substantially complete. The history does not track incremental authorship.
2. `git shortlog -sne` produced no output, so no author attribution table can be generated.
3. **No teammate commits exist in this repository at all** — Components 01/02/04 are in separate locations (Component 04 was found at `c:\Users\dviya\Desktop\Component_4\Component_04`, a *different repository*).

**[inferred] from repository structure:** all code under `Dilukshan/` appears to be single-author work by Dilukshan Viyapury, since the directory is named after the author and no other contributor appears anywhere in the history.

**⚠️ NOT FOUND IN CODEBASE — needs input from author:** the author-contributions note for the group paper must be written manually. Specifically needed: which parts (if any) were built jointly, whether any code was adapted from teammates, and what the division of labour was for shared infrastructure.

---

## 24. Key Terms / Mini-Glossary

- **Ejection fraction (EF)** — the percentage of blood the left ventricle pumps out per heartbeat; the primary measure of cardiac pumping function.
- **Ordinal regression** — classification where the classes have a natural order, so confusing neighbours is a smaller error than confusing distant classes.
- **CORAL** — a technique that makes an ordinal model's cumulative probabilities decrease monotonically, so its predictions stay internally consistent.
- **Class imbalance** — when some classes have far more examples than others; here roughly 11 Normal studies for every Severe one.
- **Deferred re-weighting (DRW)** — train on the natural class distribution first, then switch to class-balanced weighting partway through.
- **Test-time augmentation (TTA)** — run the model on several different clips from the same video and average the answers to reduce variance.
- **Calibration (post-hoc)** — adjusting decision thresholds after training, using validation data, without changing the model's weights.
- **Aleatoric vs epistemic uncertainty** — noise inherent in the data (aleatoric, from the log-variance head) versus the model's own disagreement across views (epistemic).
- **Selective prediction** — allowing the model to say "I don't know" and defer to a human, reported alongside *coverage* (the fraction it did answer).
- **Balanced accuracy** — the average of the per-class recalls; unlike overall accuracy it is not inflated by the majority class.
- **Ablation** — a test where you remove one part of a system to see whether it still works as well.
- **Conformal prediction** — a method producing prediction intervals with a guaranteed coverage rate under mild assumptions.

---

## 25. Gaps & Open Questions

**Status key:** ✅ **[FIXED]** resolved in the codebase · ⚠️ **[OPEN]** still requires author action · ❌ **[CORRECTED]** this dossier was wrong; claim retracted

### A. Requires literature research from the author

| # | Item | Status |
|---|---|---|
| 1 | **Novelty claims C1/C3 unverified.** No literature-survey artifact exists in the repo to substantiate "no prior echo work encodes label measurement noise into ordinal supervision". | ⚠️ **[OPEN]** — cannot be fixed by code |
| 2 | **MAE figures for refs [6] Reynaud (≈5.9) and [7] Thomas (≈4.2) unverified**; flagged in `README.md` §7.5 by the author. | ⚠️ **[OPEN]** — verify against source papers |
| 3 | **Whether any published method reports MAE < 3.979** on this split, which determines the strength of the defensible claim. | ⚠️ **[OPEN]** |

### B. Requires author decision / authoring

| # | Item | Status |
|---|---|---|
| 4 | **No research questions stated anywhere.** RQ1–RQ4 in §4 are inferred. | ⚠️ **[OPEN]** — confirm or rewrite |
| 5 | **Framing of the ceiling analysis** (contribution vs limitation). | ⚠️ **[OPEN]** — judgment call |
| 6 | **Clinical significance** of 99.7 % within-one-class / zero catastrophic errors. | ⚠️ **[OPEN]** — needs clinician input |
| 7 | **Author-contributions statement** must be written manually (git history inadequate — 4 commits, `git shortlog -sne` empty). | ⚠️ **[OPEN]** |

### C. Missing evidence / experiments

| # | Item | Status |
|---|---|---|
| 8 | **C1 (soft labels) and C2 (ordered-cutpoint head) have no isolated ablation.** | ⚠️ **[OPEN]** — ~20 h GPU each. C2 is cheapest: `--model-version uefnet_v1` flag already exists |
| 9 | **No re-implemented baseline** trained on this split. | ⚠️ **[PARTIALLY MITIGATED]** — `uefnet_r2p1d` (v1, drw_epoch 0, no EMA, no CAMUS) is an internal near-benchmark configuration at MAE 5.549; it is *not* a faithful re-implementation of [1] |
| 10 | **v1→v3 comparison changes four variables at once.** | ⚠️ **[OPEN]** — single-variable runs would each cost ~20 h |
| 11 | **No significance testing.** | ✅ **[FIXED]** — `engine/robustness.py` implements paired bootstrap (10,000 resamples, identical indices) + exact binomial McNemar; entry point `run_robustness.py --compare-with`. Verified against known answers (10-vs-0 discordant → p = 0.001953; symmetric → p = 1.0; null difference → p ≈ 1.0) |
| 12 | **No fairness/subgroup analysis** — original claim said "despite the dataset containing demographic fields". | ❌ **[CORRECTED]** + ✅ **[FIXED]** — **EchoNet-Dynamic contains NO demographic fields.** `FileList.csv` columns are `FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split`. Demographic fairness is **not assessable** on this cohort. **Acquisition-subgroup robustness** (frame rate, recording length, EDV, ESV) is now implemented in `engine/robustness.py subgroup_report()` and documented in `README.md` §16 item 7 |
| 13 | **Compute cost not logged** (no total wall-clock, peak VRAM, or energy). | ⚠️ **[OPEN]** — per-epoch `sec` is logged in `train_log.csv`; totals and VRAM are not |
| 14 | **Implemented-but-unreported metrics:** `probability_metrics`, `bootstrap_regression_ci`, `prediction_interval_metrics`, `bland_altman_loa95`. | ⚠️ **[OPEN]** — exist in `metrics.py`, absent from result artifacts |
| 15 | **`uefnet_v3c` stopped at epoch 29 of 45.** | ✅ **[FIXED]** — justification now stated in `README.md` §16 item 8: val min-recall peaked at epoch 29 (0.604) with no improvement for 8 epochs while val MAE drifted 4.130 → 4.201; `best.pt` held the peak; calibrated val performance (0.697 / 4.045) is in line with the two full-schedule members |

### D. Conflicts / inconsistencies

| # | Item | Status |
|---|---|---|
| 16 | **Normalisation-statistics conflict** — `norm_stats.json` 7,465 videos / 16.5 × 10⁹ px vs `verification_report.json` 512 videos / 51.4 × 10⁶ px. | ✅ **[RESOLVED — not a contradiction]** They measure different things: stage 3 estimates from 16 frames/video, stage 4 refines over all cached TRAIN pixels, stage 5 recomputes on a `--stats-sample` (default 512) **drift check** (drift_mean 0.001022 vs 0.02 tolerance). Additionally the artifact's stale `source` string (`exact_full_decode_train`, from an older code version) was corrected to `exact_cached_train_pixels` to match `stage4_cache_clips.py L230`, the contradictory `n_frames_per_video: 16` field removed, and a `provenance_note` added. Documented in `README.md` §4.1 |
| 17 | **Documented command broken** — `README.md` said `python run_all.py`; actual file is `run_preprocessing.py`. | ✅ **[FIXED]** — both occurrences corrected (command + repository-layout tree) |
| 18 | **`nibabel` undeclared** in both requirements files but required by `build_camus.py`. | ✅ **[FIXED]** — added to `preprocessing/requirements.txt` with a cross-reference note in `training/requirements.txt` |
| 19 | **Config defaults ≠ values used** — `n_tta_clips` 5 (used 10), `early_stop_patience` 12 (used 18). | ✅ **[FIXED]** — defaults changed to 10 and 18 in `config.py` so the documented command reproduces the published numbers. Existing runs are unaffected: each `outputs/<run>/config.json` is restored by `restore_for_evaluation()` |
| 20 | **`README.md` §7.3 allegedly contained Component 04 text** about "UA" and "STEMI" prevalence. | ❌ **[CORRECTED]** — **this dossier was wrong.** The match was a false positive: the substring `stemi` occurs inside *epi**stemi**c*. No Component 04 text exists in `README.md`. Claim retracted |
| 21 | **`thresholds_baseline.json` / `test_report_baseline.json` provenance undocumented** — manually-saved copies used for the C5 ablation. | ⚠️ **[OPEN]** — no script records how they were produced; author should document or regenerate reproducibly |
| 22 | **`preprocessing/config.py` parameters not enumerable** by the pattern used for training config (uses `Config()` at L127). | ⚠️ **[OPEN]** — author should list preprocessing parameters explicitly for §12 completeness |
| 23 | **No integration code** with Components 01/02/04. | ✅ **[FIXED — documented]** — `README.md` §16 item 9 now states the four-component platform is design intent, not implemented integration; Component 03 is self-contained and consumes only decoded video |

### Summary

- ✅ **Fixed in codebase:** 8 items (11, 12, 15, 16, 17, 18, 19, 23)
- ❌ **Retracted dossier errors:** 2 (12's premise, 20)
- ⚠️ **Still open, requiring author action:** 12 items — of which **8, 10 and 13 are the ones a reviewer is most likely to press**, and **8 (the C2 ablation) is the cheapest to close** since the `--model-version uefnet_v1` flag already exists
