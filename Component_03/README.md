# UEF-Net

**Uncertainty-aware Ordinal Deep Learning for Four-Class Ejection Fraction Severity Grading from Echocardiography**

Four-class left-ventricular ejection fraction (EF) severity classification on EchoNet-Dynamic under ~11:1 class imbalance, with harmonized cross-dataset co-training, learned predictive uncertainty, and a strictly leak-free evaluation protocol.

> **Headline result (untouched test split, n = 1,277):**
> **MAE 3.979 EF points · R² 0.818 · 73.0 % overall accuracy · 73.7 % balanced accuracy · all four classes 0.723–0.766 recall · 99.7 % within one severity class · zero catastrophic misclassifications.**

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Research gap](#2-research-gap)
3. [Contributions](#3-contributions)
4. [Method](#4-method)
5. [Datasets](#5-datasets)
6. [Evaluation protocol](#6-evaluation-protocol)
7. [Results](#7-results)
8. [Ablations and measured effects](#8-ablations-and-measured-effects)
9. [Selective prediction — a negative result](#9-selective-prediction--a-negative-result)
10. [Analysis of the performance ceiling](#10-analysis-of-the-performance-ceiling)
11. [Repository layout](#11-repository-layout)
12. [Installation](#12-installation)
13. [Reproducing the results](#13-reproducing-the-results)
14. [Configuration reference](#14-configuration-reference)
15. [Implementation notes and bugs fixed](#15-implementation-notes-and-bugs-fixed)
16. [Limitations](#16-limitations)
17. [Future work](#17-future-work)
18. [Ethics and data use](#18-ethics-and-data-use)
19. [References](#19-references)

---

## 1. Motivation

Left-ventricular ejection fraction is the primary index of cardiac pumping function and the measurement on which heart-failure treatment pathways are decided. It is estimated manually from echocardiogram video by tracing the ventricular border at end-diastole and end-systole — a process that takes several minutes, requires an experienced operator, and carries reported inter-observer variability of **4–5 EF points** [2].

That variability matters because clinical categories are separated by fixed thresholds:

| Severity | EF range | Studies in EchoNet-Dynamic | Share |
|---|---|---|---|
| Severe | < 30 % | 596 | 5.9 % |
| Moderate | 30 – 40 % | 718 | 7.2 % |
| Mild | 40 – 55 % | 1,806 | 18.0 % |
| Normal | ≥ 55 % | 6,910 | 68.9 % |
| **Total** | | **10,030** | **100 %** |

A patient at a true EF of 41 % may be graded *mild* by one reader and *moderate* by another, and those two labels lead to different treatment decisions. An automated grader must therefore be judged not only on average regression error but on **how reliably it assigns each severity category** — including the rare ones.

---

## 2. Research gap

Three gaps in the published literature motivate this work.

| # | Gap | Evidence |
|---|---|---|
| **G1** | EF is modelled as **regression** or **binary reduced-vs-normal**, not as the four-class severity grading clinicians actually use | Ouyang et al. [1] report MAE and binary AUC only; transformer and graph variants [6], [7] remain regression-only |
| **G2** | **Per-class recall is not reported** under ~11:1 imbalance, leaving minority-class safety unverified | A degenerate model predicting *Normal* for every study scores 68.9 % overall accuracy while being clinically useless |
| **G3** | The EF label is treated as **exact**, despite carrying 4–5 points of measurement noise | No reviewed echocardiography study encodes annotation noise into the supervision signal |

---

## 3. Contributions

| # | Contribution | Type | Measured effect |
|---|---|---|---|
| **C1** | **Measurement-uncertainty soft ordinal labels.** EF is treated as a noisy measurement; cumulative ordinal targets are derived analytically as `s_k = 1 − Φ((t_k − EF)/σ)` with σ = 4 EF points, so borderline studies contribute proportionate rather than absolute supervision. | Novel formulation | Core ordinal supervision; addresses **G3** |
| **C2** | **Ordered-cutpoint ordinal head.** Rank consistency `P(y>t₀) ≥ P(y>t₁) ≥ …` is guaranteed *by construction* via positively-constrained cut-point gaps, rather than repaired post hoc as in CORAL [12]. | Architectural modification | Eliminates rank violations structurally |
| **C3** | **Cycle-aware clip sampling with a motion channel.** Clips are constrained to contain the ED→ES contraction that physically defines EF; a temporal-difference channel supplies explicit wall motion without extra labels. | Domain-informed design | Physically grounded features |
| **C4** | **Harmonized cross-dataset co-training.** CAMUS (mean EF 44) is merged into training only, with affine intensity harmonization to prevent the network from exploiting scanner brightness as a shortcut for the minority class. | Novel application + diagnosis | Moderate recall 0.53 → 0.62 (VAL, matched epochs) |
| **C5** | **Tail-decompression calibration.** Variance expansion plus full-validation threshold fitting corrects the regression-to-the-mean shrinkage that pushes true-Severe studies into the adjacent class. | Novel post-hoc procedure | **Severe recall 0.590 → 0.687** (identical weights) |
| **C6** | **Characterisation of the performance ceiling.** Quantified boundary ambiguity, the `min-recall ≤ balanced-accuracy` bound, and the label-noise floor `MAE ≥ 0.8σ`. | Original analysis | Explains the residual gap |
| **C7** | **Selective-prediction analysis (negative result).** Deferring uncertain studies raises aggregate accuracy but *not* worst-class recall, because minority classes occupy the boundary regions that abstention removes. | Novel negative finding | Overall 0.730 → 0.770 at 88.4 % coverage; min-recall 0.723 → 0.706 |

**Prior work used and cited, not claimed:** R(2+1)D backbone [5], CORAL base formulation [12], deferred re-weighting [8], effective-number class weighting [9], label distribution smoothing [10], logit adjustment [11], temperature scaling [13], conformal prediction [14].

---

## 4. Method

### 4.1 Preprocessing

A five-stage pipeline converts raw AVI studies into a verified frame cache and a single manifest. Decoding is done once and reused, which removes video decoding from the training loop — necessary to make repeated experiments feasible on one GPU.

| Stage | Operation | Output |
|---|---|---|
| 1 | Scan the cohort, join `FileList.csv` with `VolumeTracings.csv`, validate EF ranges and split labels | candidate index |
| 2 | Decode every study to grayscale, resize to 112 × 112, verify frame counts against metadata | decoded frames |
| 3 | EF statistics over the TRAIN split; pixel statistics first estimated from 16 uniformly-sampled frames per video, then **refined by stage 4 over all cached TRAIN pixels** (16,499,624,960) | `pixel_mean` 0.1288, `pixel_std` 0.1960, `ef_mean` 55.78, `ef_std` 12.41 |
| 4 | Write `uint8` arrays to disk and record cache paths | `cache/videos/*.npy` |
| 5 | Verify every cached study reloads, matches its recorded length, and carries a valid label | audit + verification reports |

**Normalisation-statistics provenance.** Three numbers describe different computations and should not be confused: `norm_stats.json` records `n_videos_sampled = 7465` (all TRAIN studies) and `n_train_pixels = 16,499,624,960` (every cached TRAIN pixel, the stage-4 refinement). Separately, `stage5_verify.py` recomputes pixel statistics on a `--stats-sample` subsample (default 512 videos, 51,380,224 pixels) purely as a **drift check**, and reports `drift_mean = 0.001022` against a 0.02 tolerance. The subsample figure is a verification artefact, not the statistic used for training.

**Manifest.** One row per study with `FileName`, `Split`, `EF`, `ef_class`, `FPS`, `NumberOfFrames`, `ed_frame`, `es_frame`, `class_weight`, `sample_weight`, `ef_density_weight`, `cache_path`, `cached_ok`. All 10,030 studies passed verification with zero decode failures.

**Clip construction (C3).** Ejection fraction is defined by the volume change between end-diastole and end-systole, so a clip that misses the contraction carries little signal. During training, clips are constrained to contain the annotated ED→ES transition with probability `cycle_aware_probability` = 0.5, and sampled label-free otherwise — deliberately mixing the two so the network does not acquire a train/deployment mismatch, since tracings are unavailable at inference. Studies shorter than the requested span are sampled monotonically by linear interpolation across the available frames rather than by modulo wraparound, which would introduce an artificial last-to-first motion discontinuity.

**Motion channel (C3).** A second input channel is the signed temporal difference between consecutive frames, `m_t = g_t − g_{t−1}` (with `m_0 = m_1`), giving the network an explicit representation of wall motion without additional labels.

**Augmentation (training only).** Reflect-pad by 12 px and random-crop back to 112 × 112; multiplicative and additive intensity jitter of ±0.1. Evaluation is deterministic.

The sampling and motion routines are imported directly by the training dataset from the preprocessing package, so there is a single implementation and no train/preprocess skew.

### 4.2 Architecture

```
Input clip  (2 × 32 × 112 × 112)          grayscale + temporal-difference motion
      │
   R(2+1)D-18 backbone                    Kinetics-400 pretrained, 2-channel stem
      │                                   31.3 M parameters
      ├──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
 Regression     Ordinal head    Class head    Log-variance
 head           (ordered        (softmax,     head
 → EF (z)        cut-points)     auxiliary)   → predictive σ²
                → P(y > t_k)    → p(class)
```

**Backbone.** R(2+1)D-18 [5] factorises 3-D convolution into spatial then temporal operations, increasing non-linearity while reducing parameters. The input stem is adapted from three RGB channels to two (grayscale + motion); RGB kernels are summed into channel 0 so a grayscale input is equivalent to the pretrained response, and auxiliary channels start as a small learned residual.

**Ordered-cutpoint head (C2).** A single severity score `s(x)` is compared against learned cut-points

```
τ₀ = a,        τ_k = a + Σ_{j<k} softplus(g_j) + ε
logits_k = s(x) − τ_k
```

Because `softplus(·) > 0`, the cut-points are strictly increasing, so the cumulative probabilities are monotone by construction.

### 4.3 Training objective

```
L = w_reg·L_reg + w_ord·L_ord + w_class·L_class
  + w_nll·L_nll + w_rank·L_rank + w_con·L_con
```

| Term | Definition | Purpose |
|---|---|---|
| `L_reg` | LDS-weighted Huber on standardised EF | Regression accuracy under imbalance [10] |
| `L_ord` | BCE against soft cumulative targets `s_k = 1 − Φ((t_k − EF)/σ)` | **C1** — ordinal supervision with label noise |
| `L_class` | Soft cross-entropy on the auxiliary head, with logit adjustment `+ τ·log P(class)` [11] | Minority-class margin |
| `L_nll` | Gaussian negative log-likelihood `½(e^{−log σ²}(ŷ−y)² + log σ²)` | Trains the uncertainty head |
| `L_rank` | Smooth pairwise ordering penalty for pairs separated by ≥ 3 EF points | Enforces global ordering |
| `L_con` | MSE between regression-implied and ordinal-head cumulative probabilities | Couples the two heads |

### 4.4 Imbalance handling

Three mechanisms, activated on a deferred schedule (`drw_epoch = 15`):

1. **Sampling** — natural distribution for epochs 0–14, class-balanced `WeightedRandomSampler` thereafter [8]
2. **Loss weighting** — effective-number class weights `(1−β)/(1−β^{n_c})`, β = 0.9999 [9]
3. **Logit adjustment** — classification logits shifted by `τ·log P(class)`, τ = 0.5 [11]

The deferred schedule was selected empirically: balancing from epoch 0 gave MAE 5.44 at equal min-recall, versus **4.29** with `drw_epoch = 15`.

### 4.5 Test-time augmentation and seed ensembling

Two variance-reduction stages are applied at inference. Both operate on frozen weights — no additional training is involved beyond producing the ensemble members.

**Multi-clip test-time augmentation.** A single 32-frame clip samples only part of a recording, so the EF estimate inherits the variance of *which* cardiac cycle was captured. Each study is therefore evaluated with **N = 10 deterministic clips** whose start positions are spread evenly across the valid window. Sampling is **label-free** — the annotated ED/ES frames are not consulted, because they would not exist for a new clinical study. The per-clip outputs are averaged at study level:

```
EF(study)       = mean_v  EF(clip_v)
P_ord(study)    = mean_v  P_ord(clip_v)          cumulative ordinal probabilities
P_class(study)  = mean_v  P_class(clip_v)        auxiliary softmax distribution
σ_epistemic     = std_v   EF(clip_v)             inter-clip disagreement
σ_aleatoric²    = mean_v  exp(log σ²(clip_v))    learned predictive variance
```

The two dispersion terms are retained and combined by the law of total variance, `σ_total = √(σ_aleatoric² + σ_epistemic²)`, and are consumed by the conformal interval and by the selective-prediction analysis (§9).

**Seed ensembling.** *M* = 3 models are trained with **identical configuration and different random seeds** (1337, 2024, 777). The seed governs weight initialisation of the heads, data ordering, the balanced sampler draws and augmentation, producing genuinely decorrelated members while holding architecture, data and hyperparameters fixed — so the ensemble is a controlled variance-reduction step rather than a confound.

Member predictions are averaged **after** clip aggregation:

```
for each member m:      clip-level outputs  →  study-level outputs   (TTA, above)
across members:         EF, P_ord, P_class, σ  →  arithmetic mean
```

Averaging in this order keeps every member on the same 1,277 studies in the same order; `run_ensemble.py` asserts this and raises rather than silently averaging mismatched splits.

Each member is loaded with **its own saved configuration and frozen normalisation statistics**, so a member trained under different preprocessing could never be silently mixed in.

**Ensemble-level calibration.** The decision rule is fitted on the **ensemble's** validation predictions, not per member and not averaged from member rules. The sequence is: average members on VAL → select and freeze one strategy → average members on TEST → apply the frozen strategy once. Members therefore contribute only predictions; a single decision rule governs the reported result.

Empirically the ensemble improves every criterion the design targets — MAE 4.138 → 3.994 → 3.979 and min-class recall 0.687 → 0.711 → 0.723 for M = 1, 2, 3 — with diminishing returns consistent with variance reduction of order 1/M (§8).

### 4.6 Post-hoc calibration

Fitted on validation data only, then frozen. Candidate rules compared:

| Rule | Description |
|---|---|
| `reg_clinical_raw` | Published boundaries on raw regression output (no tuning) |
| `reg_clinical_affine` | Least-squares affine EF correction, slope bounded to [0.5, 1.5] |
| `reg_operational_thresholds` | Learned thresholds, searched within ±(8, 8, 6) of clinical boundaries |
| **`reg_operational_expanded`** | **C5** — variance expansion `EF′ = μ_t + k(EF − μ_p)`, then learned thresholds |
| `ordinal_rank` / `ordinal_argmax` / `class_argmax` | Head-specific decision rules |
| `probability_blend` | Convex blend of regression / ordinal / class distributions with temperature scaling [13] |

Selection is lexicographic on (min-class recall, balanced accuracy, macro-F1, proximity to clinical boundaries). **`reg_operational_expanded` won on all three ensemble configurations**, independently validating C5.

Split-conformal prediction [14] supplies per-study intervals with finite-sample coverage.

---

## 5. Datasets

| Dataset | Studies | Views | Use | Mean EF |
|---|---|---|---|---|
| **EchoNet-Dynamic** [1] | 10,030 | Apical 4-chamber | Train / val / test | 55.8 |
| **CAMUS** [4] | 500 patients → 1,000 clips | Apical 2-ch + 4-ch | **Train only** | 44.2 |

**Official splits:** 7,465 train · 1,288 validation · 1,277 test.

### 5.1 Cross-dataset co-training (C4)

CAMUS is markedly richer in impaired function — approximately half its patients have EF < 45 %:

| Class | CAMUS clips | Share | EchoNet train | **Merged train** | Increase |
|---|---|---|---|---|---|
| Severe | 122 | 12.2 % | 460 | **582** | **+27 %** |
| Moderate | 194 | 19.4 % | 488 | **682** | **+40 %** |
| Mild | 496 | 49.6 % | 1,333 | **1,829** | **+37 %** |
| Normal | 188 | 18.8 % | 5,184 | **5,372** | +4 % |
| **Total** | **1,000** | | 7,465 | **8,465** | |

**Intensity harmonization.** Raw CAMUS pixels are systematically brighter (mean 49.2, σ 56.6) than EchoNet (mean 32.8, σ 50.0). Because the class-balanced sampler over-draws the minority-rich CAMUS studies, an uncorrected offset would let the network associate *brightness* with *severity* — a shortcut that cannot transfer to the EchoNet test set. Every CAMUS clip is therefore affinely remapped:

```
p′ = (p − μ_CAMUS)/σ_CAMUS · σ_EchoNet + μ_EchoNet
```

After harmonization the per-clip grayscale statistics match (CAMUS 0.916 vs EchoNet 0.940 normalised σ). The operation is idempotent, guarded by a marker file.

**No leakage.** CAMUS enters the TRAIN split only; validation (1,288) and test (1,277) are verified to contain zero CAMUS studies.

---

## 6. Evaluation protocol

The protocol is designed so results survive independent replication.

1. **Official splits respected.** No re-partitioning.
2. **Validation is partitioned.** Model selection uses one part; post-hoc calibration is fitted on the other (or on full validation for tiny-class threshold stability).
3. **Test is evaluated once**, with every decision parameter already frozen. No thresholds, blends, temperatures or scalings are tuned on test.
4. **Two results always reported:**
   - **Clinical reference** — published boundaries (30/40/55) on the raw regression output, no post-hoc rule
   - **Operational** — the validation-selected frozen decision rule
5. **Label-free inference.** Expert ED/ES tracings are used for training-time sampling only, never at evaluation, since they are unavailable for a new clinical study.
6. **Uncertainty reported** — Wilson intervals on every per-class recall; split-conformal intervals on EF.

**Statistical treatment.** Per-class recalls are reported with 95 % Wilson score intervals, which are appropriate for binomial proportions at small counts (Severe has 83 test studies). Where an interval is wide — notably Severe at ±9.5 % — that width is stated rather than suppressed.

Because every configuration is evaluated on the *same* test studies, comparisons between them use **paired** tests rather than two independent intervals: a paired bootstrap over studies (10,000 resamples, identical indices for both systems) for MAE, accuracy and balanced accuracy, and an **exact binomial McNemar test** on discordant classifications. These are implemented in `engine/robustness.py` and run via `run_robustness.py --compare-with` (§13.5). Reported ablation deltas should be quoted with the paired interval, not as bare differences.

---

## 7. Results

### 7.1 Primary result — three-seed ensemble, test split (n = 1,277)

| Metric | Value | 95 % CI |
|---|---|---|
| **MAE** | **3.979 EF points** | — |
| **R²** | **0.818** | — |
| RMSE | 5.211 | — |
| **Overall accuracy** | **72.98 %** (932/1,277) | [0.705, 0.753] |
| **Balanced accuracy** | **73.66 %** | — |
| Macro-F1 | 0.684 | — |
| **Minimum class recall** | **0.723** | — |
| **Within-one-class** | **99.69 %** (1,273/1,277) | — |
| **Catastrophic errors** | **0** | — |

### 7.2 Per-class performance

| Class | Correct / Total | Recall | 95 % Wilson CI | Precision | F1 |
|---|---|---|---|---|---|
| Severe (< 30) | 60 / 83 | **0.723** | [0.618, 0.808] | 0.952 | 0.822 |
| Moderate (30–40) | 59 / 77 | **0.766** | [0.660, 0.847] | 0.434 | 0.554 |
| Mild (40–55) | 176 / 241 | **0.730** | [0.671, 0.782] | 0.413 | 0.528 |
| Normal (≥ 55) | 637 / 876 | **0.727** | [0.697, 0.756] | 0.977 | 0.834 |

All four recalls fall within a **0.043 band** — an unusually tight spread for an 11:1 imbalanced problem.

### 7.3 Confusion matrix

| True ↓ / Predicted → | Severe | Moderate | Mild | Normal |
|---|---|---|---|---|
| **Severe** | **60** | 23 | 0 | 0 |
| **Moderate** | 3 | **59** | 15 | 0 |
| **Mild** | 0 | 50 | **176** | 15 |
| **Normal** | 0 | 4 | 235 | **637** |

**Clinical safety.** Only **4 of 1,277** predictions are off by more than one severity class. **No severely-impaired study was graded Normal, and no Normal study was graded Severe** — the two confusions with the greatest clinical consequence never occur.

### 7.4 Two operating points

| Rule | MAE | Overall | Balanced | Min-recall |
|---|---|---|---|---|
| **Operational** (validation-frozen) | 3.979 | 0.730 | 0.737 | **0.723** |
| **Clinical reference** (raw EF @ 30/40/55) | 3.983 | **0.796** | 0.653 | 0.442 |

The contrast is itself a finding: applying the published boundaries directly yields higher aggregate accuracy while abandoning the minority classes. Reporting only the clinical reference would conceal a min-recall of 0.44.

### 7.5 Comparison with published work

Only MAE is directly comparable — these studies share the EchoNet-Dynamic test split, and almost none report four-class metrics.

| Method | Year | Task | MAE | R² | 4-class? |
|---|---|---|---|---|---|
| Ouyang et al. [1] (*Nature*, benchmark) | 2020 | Regression | 4.05 | 0.81 | ✗ |
| Reynaud et al. [6] (transformer) | 2021 | Regression | ≈ 5.9 | — | ✗ |
| Thomas et al. [7] (lightweight graph) | 2022 | Regression | ≈ 4.2 | — | ✗ |
| **UEF-Net (this work)** | 2026 | **Regression + 4-class** | **3.979** | **0.818** | ✓ |

> **Claim discipline.** This work *matches and slightly improves on* the reference benchmark while additionally performing four-class severity grading. It does **not** claim state of the art — more recent optimised regression pipelines report lower MAE. Figures for [6] and [7] should be verified against the source papers before citation.

### 7.6 Error analysis

Of 1,277 studies, **345 are misclassified**. Their distribution is highly structured rather than diffuse:

| True → Predicted | Count | Share of all errors | Class distance |
|---|---|---|---|
| **Normal → Mild** | **235** | **68.1 %** | 1 |
| Mild → Moderate | 50 | 14.5 % | 1 |
| Severe → Moderate | 23 | 6.7 % | 1 |
| Moderate → Mild | 15 | 4.3 % | 1 |
| Mild → Normal | 15 | 4.3 % | 1 |
| Normal → Moderate | 4 | 1.2 % | 2 |
| Moderate → Severe | 3 | 0.9 % | 1 |

**Three observations.**

1. **98.8 % of errors are single-class displacements** (341 of 345). Only four studies are misplaced by two categories, and none by three.
2. **A single confusion dominates.** Normal → Mild accounts for 68 % of all errors and occurs at the EF = 55 boundary — precisely where 343 test studies lie within ±4 EF (§10.2). This is not a distributed weakness but one crowded threshold.
3. **Errors are asymmetric toward severity.** Normal → Mild (235) exceeds Mild → Normal (15) by more than fifteen to one, and Severe → Moderate (23) exceeds Moderate → Severe (3). The frozen decision rule is biased toward assigning the *more impaired* label, which is the clinically safer direction for a screening aid.

**The four distant errors** are all Normal → Moderate, i.e. studies with true EF ≥ 55 whose calibrated prediction falls into the 30–40 band. No study is displaced by three categories in either direction. Whether these four are individually recoverable — for example whether they carry elevated predictive uncertainty and would be caught by the selective rule of §9 — has not been examined and is left as future work.

---

## 8. Ablations and measured effects

Every design decision was measured on data never used to fit it.

| Change | Metric before | Metric after | Δ | Controlled? |
|---|---|---|---|---|
| **C5 — tail-decompression calibration** | Severe recall **0.590** | **0.687** | **+0.097** | ✅ **Fully** — identical weights, only the calibration changed |
| **Ensembling 1 → 2 seeds** | MAE 4.138, min-rec 0.687 | MAE 3.994, min-rec 0.711 | −0.144 / +0.024 | ✅ Fully — same config, different seeds |
| **Ensembling 2 → 3 seeds** | MAE 3.994, min-rec 0.711 | MAE 3.979, min-rec 0.723 | −0.015 / +0.012 | ✅ Fully |
| **C4 — CAMUS co-training** | Moderate 0.53 | Moderate 0.62 | +0.09 | ⚠️ Matched epochs across runs, not single-variable |
| **Deferred vs immediate re-weighting** | MAE 5.44 | MAE 4.29 | −1.15 | ✅ Fully |

### 8.1 Minimum-recall progression

| Stage | MAE | Min-class recall |
|---|---|---|
| Single model, 30 % validation calibration | 4.166 | 0.590 |
| **+ C5 tail-decompression calibration** | 4.138 | **0.687** |
| **+ 2-seed ensemble** | 3.994 | **0.711** |
| **+ 3-seed ensemble (final)** | **3.979** | **0.723** |

### 8.2 Ensemble member consistency

| Run | Seed | Validation min-recall | Validation MAE | Validation per-class |
|---|---|---|---|---|
| `uefnet_v3` | 1337 | 0.693 | 3.941 | 0.70 / 0.69 / 0.70 / 0.72 |
| `uefnet_v3b` | 2024 | 0.701 | 3.888 | 0.72 / 0.71 / 0.70 / 0.71 |
| `uefnet_v3c` | 777 | 0.697 | 4.045 | 0.72 / 0.71 / 0.70 / 0.70 |

Members agree closely, and `reg_operational_expanded` was independently selected as the winning strategy in all three — evidence that C5 is not a fitting artefact. Repeating calibration reproduced results bit-for-bit.

---

## 9. Selective prediction — a negative result

**Hypothesis.** Since errors concentrate near decision boundaries, allowing the model to abstain on its least-confident studies (Chow's rule [18]) should raise recall on the graded subset above 0.75 for all classes.

**Implementation.** Seven model-derived uncertainty signals — learned aleatoric σ from the log-variance head, inter-clip epistemic disagreement, total variance, boundary proximity normalised by σ, probability margin, entropy, and a combined score. Signal and threshold selected on validation, applied once to test.

**Result at the validation-selected operating point (88.4 % coverage):**

| Metric | Full coverage | Selective (88.4 %) |
|---|---|---|
| Overall accuracy | 0.730 | **0.770** ↑ |
| Balanced accuracy | 0.737 | 0.743 ↑ |
| **Minimum class recall** | **0.723** | **0.706** ↓ |

**Coverage sweep (test):**

| Coverage | Overall | Severe | Moderate | Mild | Normal |
|---|---|---|---|---|---|
| 100 % | 0.730 | 0.72 | 0.77 | 0.73 | 0.73 |
| 90 % | 0.762 | 0.71 | 0.75 | 0.73 | 0.77 |
| 80 % | 0.801 | 0.76 | **0.66** | 0.74 | 0.82 |
| 70 % | 0.847 | 0.79 | **0.61** | 0.73 | **0.88** |
| 50 % | 0.929 | 0.82 | **0.58** | 0.67 | **0.97** |

**Finding.** Abstention raises aggregate accuracy monotonically (0.730 → 0.929) but *degrades* worst-class recall. The mechanism is structural:

> **Moderate occupies a 10-point interior band (30–40), so every Moderate study is near a decision boundary by construction, whereas Normal is open-ended (≥ 55) and most of its studies lie far from any boundary. A boundary-proximity abstention rule therefore preferentially defers minority-class studies while retaining easy majority studies — aggregate accuracy rises and the floor falls.**

**Selective prediction cannot repair worst-class recall in ordinal grading when the minority classes *are* the boundary region.** To our knowledge this has not been reported for EF severity grading.

**The uncertainty estimates are nonetheless validated:** accuracy on deferred studies is **0.426** versus **0.770** on graded studies, confirming that the learned log-variance head plus test-time disagreement genuinely identify hard cases.

---

## 10. Analysis of the performance ceiling

Three independent bounds explain why the ≥ 0.75-on-all-classes target is not met.

### 10.1 The arithmetic bound

```
min-class recall ≤ balanced accuracy       (a minimum cannot exceed a mean)
```

Balanced accuracy is **0.737**. Post-hoc calibration *redistributes* recall between classes; it cannot raise the mean. Given the observed 0.043 spread, min-recall ≈ balanced accuracy − 0.014, so reaching 0.75 requires balanced accuracy ≈ **0.764** — a property of the model, not of any decision rule.

### 10.2 Boundary ambiguity

Distance from each test study's true EF to the nearest decision boundary:

| Within | Studies | Share |
|---|---|---|
| ± 2 EF | 228 | 17.9 % |
| ± 3 EF | 346 | 27.1 % |
| **± 4 EF (≈ model MAE)** | **469** | **36.7 %** |
| ± 5 EF | 606 | 47.5 % |

Concentration by boundary: **EF = 30 → 63 studies · EF = 40 → 63 · EF = 55 → 343.**

Over a third of the cohort lies within one MAE of a threshold, and the dominant error mode (Normal → Mild, 235 cases) is exactly the EF = 55 cluster.

### 10.3 The label-noise floor

For a perfect predictor evaluated against labels carrying Gaussian noise of standard deviation σ:

```
E|error| = σ·√(2/π) ≈ 0.80 σ
```

| Inter-observer σ | Minimum attainable MAE |
|---|---|
| 4.0 | **3.19** |
| 4.5 | **3.59** |
| 5.0 | **3.99** |

With reported inter-observer variability of 4–5 EF points [2], the floor is **3.2–4.0**. At **MAE 3.979 the model already operates at the level of human reader disagreement**; further reduction would require it to be more self-consistent than the annotations it learns from.

---

## 11. Repository layout

```
Dilukshan/
├── preprocessing/
│   ├── run_preprocessing.py       pipeline driver (stages 0-5)
│   ├── stage1_scan.py … stage5_verify.py
│   ├── build_camus.py             CAMUS → EchoNet format + intensity harmonization
│   ├── utils/sampling.py          cycle-aware sampling + motion channel (shared with training)
│   ├── utils/io_utils.py
│   └── artifacts/                 manifest, normalisation statistics, audit reports, visualisations
└── training/
    ├── config.py                  single source of truth; serialised with every run
    ├── core/
    │   ├── common.py              seeding, checkpointing, RNG capture/restore, CSV logging
    │   └── plots.py               training curves, confusion matrices
    ├── data/
    │   ├── dataset.py             manifest loading, extra-manifest merge, augmentation
    │   └── sampler.py             DRW-aware class-balanced sampler
    ├── models/
    │   ├── uef_net.py             backbone + four heads (CoralHead / OrderedCoralHead)
    │   └── ema.py                 exponential moving average of weights
    ├── losses/
    │   ├── losses.py              six-term multi-task objective
    │   └── lds.py                 label distribution smoothing
    ├── engine/
    │   ├── trainer.py             training loop, EMA, resume, final calibration
    │   ├── evaluate.py            multi-clip TTA inference
    │   ├── calibrate.py           strategy fitting, thresholds, conformal, temperature
    │   ├── metrics.py             per-class metrics, Wilson intervals, bootstrap CIs
    │   ├── selective.py           selective prediction / abstention
    │   └── robustness.py          acquisition subgroups, paired bootstrap, exact McNemar
    ├── run_train.py               training entry point
    ├── run_eval.py                frozen-strategy test evaluation
    ├── run_ensemble.py            multi-seed ensemble evaluation
    ├── run_selective.py           coverage–recall analysis
    └── run_robustness.py          subgroup robustness + paired significance tests
```

---

## 12. Installation

**Tested environment:** Windows 11, Python 3.11, CUDA 12.x, NVIDIA RTX 4060 Laptop (8 GB), 32 GB RAM.

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas opencv-python matplotlib tqdm nibabel
```

`nibabel` is required only for CAMUS conversion. The design fits an 8 GB GPU through mixed precision, gradient accumulation and bounded TTA batching; no cloud compute is required.

---

## 13. Reproducing the results

### 13.1 Obtain the data

**Not included in this repository** — both datasets are distributed under Research Data Use Agreements that prohibit redistribution.

- **EchoNet-Dynamic** — <https://echonet.github.io/dynamic/>
- **CAMUS** — <https://www.creatis.insa-lyon.fr/Challenge/camus/>

Place EchoNet at `Dilukshan/Dataset/` (`FileList.csv`, `VolumeTracings.csv`, `Videos/`).

### 13.2 Preprocess

```bash
cd Dilukshan/preprocessing
python run_preprocessing.py          # stages 0-5: audit, labels, keyframes, stats, cache, verify
python build_camus.py                # convert + harmonise CAMUS (optional but recommended)
```

### 13.3 Train the three ensemble members

```bash
cd ../training
python run_train.py --v2 --extra-manifest artifacts/camus_manifest.csv \
    --logit-adjustment-tau 0.5 --seed 1337 --run-name uefnet_v3  --epochs 45 --patience 18
python run_train.py --v2 --extra-manifest artifacts/camus_manifest.csv \
    --logit-adjustment-tau 0.5 --seed 2024 --run-name uefnet_v3b --epochs 45 --patience 18
python run_train.py --v2 --extra-manifest artifacts/camus_manifest.csv \
    --logit-adjustment-tau 0.5 --seed 777  --run-name uefnet_v3c --epochs 45 --patience 18
```

Approximately 20 hours per run on an RTX 4060. Training is fully resumable:

```bash
python run_train.py --resume --run-name uefnet_v3b
```

### 13.4 Calibrate and evaluate

```bash
python run_train.py --calibrate-only --run-name uefnet_v3 --n-tta 10   # per member
python run_eval.py     --run-name uefnet_v3 --n-tta 10                 # single model
python run_ensemble.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10     --save-predictions                                                     # primary result
python run_selective.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10 --plot
```

### 13.5 Robustness and significance analysis

`--save-predictions` writes per-study arrays so the analyses below need no re-inference:

```bash
# acquisition-subgroup robustness for the ensemble
python run_robustness.py     --predictions outputs/predictions_test_uefnet_v3_uefnet_v3b_uefnet_v3c.npz

# paired significance test: ensemble vs single model
python run_ensemble.py --runs uefnet_v3 --n-tta 10 --save-predictions     --out outputs/single_model.json
python run_robustness.py     --predictions outputs/predictions_test_uefnet_v3_uefnet_v3b_uefnet_v3c.npz     --compare-with outputs/predictions_test_uefnet_v3.npz
```

Reports paired-bootstrap confidence intervals and p-values on MAE, accuracy and balanced accuracy, plus an exact McNemar test on discordant classifications. Because both systems are evaluated on the same studies, the paired form is the correct instrument; two independent intervals would understate the evidence.

### 13.6 Outputs

```
outputs/<run>/best.pt, last.pt, config.json, norm_stats.json,
              thresholds.json, train_log.csv, training_curves.png,
              validation_partition.json, test_report.json, confusion_test.png
outputs/ensemble_report.json
outputs/selective_report.json, selective_coverage_curve.png
```

---

## 14. Configuration reference

All hyperparameters live in `training/config.py` and are serialised with each run.

| Group | Parameter | Value | Note |
|---|---|---|---|
| Input | `clip_len` / `sampling_period` | 32 / 2 | spans 64 native frames (~1.4 cycles) |
| | `frame_size`, `in_channels` | 112, 2 | grayscale + motion |
| | `motion_mode` | `tempdiff` | temporal difference |
| | `cycle_aware_probability` | 0.5 | mix of guided and label-free sampling |
| Model | `backbone`, `model_version` | `r2plus1d_18`, `uefnet_v2` | |
| | `dropout` | 0.3 | |
| Loss | `ef_noise_sigma` | 4.0 | **C1** measurement noise |
| | `w_reg` / `w_ord` / `w_class` | 1.0 / 1.0 / 0.5 | |
| | `w_nll` / `w_rank` / `w_consistency` | 0.5 / 0.2 / 0.15 | |
| | `logit_adjustment_tau` | 0.5 | **C7** |
| | `lds_kernel_sigma`, `lds_ks` | 2.0, 5 | |
| Imbalance | `drw_epoch` | 15 | deferred re-weighting |
| | `effective_num_beta` | 0.9999 | |
| Optimisation | `epochs`, `batch_size`, `grad_accum` | 45, 8, 4 | effective batch 32 |
| | `lr_backbone` / `lr_head` | 1e-4 / 1e-3 | warmup 2 epochs + cosine |
| | `weight_decay`, `grad_clip`, `amp` | 1e-4, 2.0, True | |
| | `use_ema`, `ema_decay` | True, 0.999 | |
| Evaluation | `n_tta_clips` | 10 | label-free views |
| | `calibrate_on_full_val` | True | **C5** tiny-class stability |
| | `threshold_search_radius` | (8, 8, 6) | **C5** widened Severe search |
| | `eval_use_keyframes` | False | no privileged tracings |

---

## 15. Implementation notes and bugs fixed

| Issue | Impact | Resolution |
|---|---|---|
| `np.random.randint(0, 2³²−1)` overflowed the 32-bit C long on Windows | **Crashed all training** on first batch | Explicit `dtype=np.int64` |
| CAMUS brightness offset (+16 mean) | Would let the network use scanner appearance as a minority-class shortcut | Affine intensity harmonization (§5.1) |
| Harmonization not idempotent — black-background clipping prevents exact stat matching | Repeated runs would corrupt cached clips | Marker file `_harmonized.json` guards re-application |
| Calibration fitted on 30 % of validation (~30 Severe studies) | Severe threshold failed to generalise to test | `calibrate_on_full_val` (~100 Severe studies) |
| Log-variance head computed but discarded at inference | Learned uncertainty unavailable downstream | `ef_aleatoric_std` exposed by `run_inference` |
| DataLoader workers persisted across train/val phases | 16 concurrent processes exhausted 32 GB RAM | `persistent_workers=False`, workers reduced to 4 |
| RNG state restored to GPU after `map_location=cuda` | Resume failed | Coerce to CPU uint8 before restore |

**Reproducibility.** Seeds, RNG state, optimiser state, EMA weights and the full configuration are checkpointed each epoch; runs resume with at most one epoch of lost work. Repeated calibration of a fixed checkpoint reproduces byte-identical results.

---

## 16. Limitations

1. **The ≥ 0.75-on-all-classes target is not met.** Recalls span 0.723–0.766. §10 establishes that this is bounded by balanced accuracy (0.737), boundary ambiguity (36.7 % within ±4 EF), and label noise, rather than by insufficient tuning.
2. **Small extreme-tail test set.** Severe contains 83 studies, giving a 95 % Wilson interval of roughly ±9.5 % on its recall. Any single-figure comparison at that sample size is statistically fragile.
3. **Cross-dataset ablation is only partially controlled.** The CAMUS effect is measured across runs at matched epochs, not as a single-variable ablation. The calibration and ensembling effects *are* fully controlled.
4. **Single-cohort evaluation.** CAMUS is used for training only; no external cohort is held out for testing.
5. **Geographic representativeness.** Both datasets originate from single institutions in the United States and France. Performance on other populations is not established.
6. **Precision on the interior classes is low** (Moderate 0.434, Mild 0.413) — a direct consequence of optimising worst-class recall under heavy imbalance. Applications prioritising precision would select a different operating point.
7. **No demographic fairness analysis is possible on this cohort.** EchoNet-Dynamic's `FileList.csv` carries only `FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split` — there are **no age, sex or ethnicity fields**. Subgroup robustness is therefore assessed over *acquisition* characteristics (frame rate, recording length, ventricular volume) via `run_robustness.py`; demographic fairness is **not claimed and not assessable** here, and would require a cohort that carries those attributes.
8. **`uefnet_v3c` was stopped at epoch 29 of a scheduled 45.** Its validation minimum-recall peaked at epoch 29 (0.604) and had not improved for 8 subsequent epochs while validation MAE drifted upward (4.130 → 4.201), so `best.pt` already held the peak and the remaining epochs were not run. Its calibrated validation performance (min-recall 0.697, MAE 4.045) is in line with the two members that completed the full schedule, and all three were selected by the same frozen criterion. This is stated rather than presented as a completed run.
9. **The four-component platform (§4.1 of the proposal) is design intent, not implemented integration.** This repository contains no interface, adapter or contract with Components 01/02/04; those exist in separate repositories. Component 03 is self-contained and consumes only decoded video.
10. **Not a certified medical device.** Decision support only.

---

## 17. Future work

| Direction | Rationale | Expected effect |
|---|---|---|
| **Self-supervised pretraining (VideoMAE [15])** on pooled unlabelled echocardiography | Kinetics-400 transfer is from natural video; echo-specific features may transfer better | +0.01–0.03 balanced accuracy (uncertain at this data scale) |
| **Additional low-EF cohorts** | Severe remains the scarcest class; more minority data is the only lever that raises balanced accuracy substantially | +0.02–0.05 balanced accuracy |
| **External validation** — hold CAMUS out as a test cohort | Establishes cross-scanner generalisation | Credibility, not accuracy |
| **Prospective local validation** | Required before any deployment | — |
| **Larger ensembles (5–10 seeds)** | Diminishing returns; asymptotes near MAE 3.85 | +0.005–0.010 balanced accuracy |
| **Multi-view fusion (2CH + 4CH)** | Simpson's biplane uses both views; EchoNet provides only A4C | Potentially significant if paired views become available |

---

## 18. Ethics and data use

Both datasets were released by their originating institutions after ethical review and full anonymisation. EchoNet-Dynamic was collected at Stanford Medicine under institutional review board approval; CAMUS was collected at the University Hospital of St Etienne within the regulations of its local ethics committee.

- No new patient data is collected and no human participants are involved.
- No attempt is made to re-identify any individual.
- **Neither dataset is redistributed.** This repository excludes all video, derived frame caches, and any artefact permitting reconstruction of the source data.
- **Intended use is decision support, not autonomous diagnosis.** Outputs are presented for clinician review with explicit uncertainty intervals; the interface requires confirmation rather than passive acceptance. Deployment would require prospective validation on the target population and appropriate regulatory clearance.

---

## 19. References

[1] D. Ouyang, B. He, A. Ghorbani, N. Yuan, J. Ebinger, C. P. Langlotz, P. A. Heidenreich, R. A. Harrington, D. H. Liang, E. A. Ashley, and J. Y. Zou, "Video-based AI for beat-to-beat assessment of cardiac function," *Nature*, vol. 580, no. 7802, pp. 252–256, 2020.

[2] R. M. Lang *et al.*, "Recommendations for cardiac chamber quantification by echocardiography in adults," *J. Am. Soc. Echocardiogr.*, vol. 28, no. 1, pp. 1–39, 2015.

[3] G. Savarese and L. H. Lund, "Global public health burden of heart failure," *Cardiac Failure Review*, vol. 3, no. 1, pp. 7–11, 2017.

[4] S. Leclerc *et al.*, "Deep learning for segmentation using an open large-scale dataset in 2D echocardiography," *IEEE Trans. Med. Imaging*, vol. 38, no. 9, pp. 2198–2210, 2019.

[5] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for action recognition," in *Proc. CVPR*, 2018, pp. 6450–6459.

[6] H. Reynaud, A. Vlontzos, B. Hou, A. Beqiri, P. Leeson, and B. Kainz, "Ultrasound video transformers for cardiac ejection fraction estimation," in *Proc. MICCAI*, 2021, pp. 495–505.

[7] S. M. Thomas, A. Lefebvre, and P.-M. Jodoin, "Light-weight spatio-temporal graphs for segmentation and ejection fraction prediction in cardiac ultrasound," in *Proc. MICCAI*, 2022, pp. 380–390.

[8] K. Cao, C. Wei, A. Gaidon, N. Aréchiga, and T. Ma, "Learning imbalanced datasets with label-distribution-aware margin loss," in *Proc. NeurIPS*, vol. 32, 2019, pp. 1567–1578.

[9] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. CVPR*, 2019, pp. 9268–9277.

[10] Y. Yang, K. Zha, Y. Chen, H. Wang, and D. Katabi, "Delving into deep imbalanced regression," in *Proc. ICML*, 2021, pp. 11842–11851.

[11] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in *Proc. ICLR*, 2021.

[12] W. Cao, V. Mirjalili, and S. Raschka, "Rank consistent ordinal regression for neural networks with application to age estimation," *Pattern Recognition Letters*, vol. 140, pp. 325–331, 2020.

[13] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321–1330.

[14] A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591, 2023.

[15] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training," in *Proc. NeurIPS*, vol. 35, 2022, pp. 10078–10093.

[16] P. Ponikowski *et al.*, "2016 ESC guidelines for the diagnosis and treatment of acute and chronic heart failure," *European Heart Journal*, vol. 37, no. 27, pp. 2129–2200, 2016.

[17] World Health Organization, "Cardiovascular diseases (CVDs) fact sheet," WHO, Geneva, 2021.

[18] C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Trans. Information Theory*, vol. 16, no. 1, pp. 41–46, 1970.

---

## Citation

If this work is useful, please cite the underlying datasets:

```bibtex
@article{ouyang2020video,
  title   = {Video-based AI for beat-to-beat assessment of cardiac function},
  author  = {Ouyang, David and He, Bryan and Ghorbani, Amirata and Yuan, Neal and
             Ebinger, Joseph and Langlotz, Curtis P. and Heidenreich, Paul A. and
             Harrington, Robert A. and Liang, David H. and Ashley, Euan A. and Zou, James Y.},
  journal = {Nature},
  volume  = {580}, number = {7802}, pages = {252--256}, year = {2020},
  doi     = {10.1038/s41586-020-2145-8}
}

@article{leclerc2019deep,
  title   = {Deep Learning for Segmentation using an Open Large-Scale Dataset
             in 2D Echocardiography},
  author  = {Leclerc, Sarah and Smistad, Erik and Pedrosa, Joao and Ostvik, Andreas and others},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {38}, number = {9}, pages = {2198--2210}, year = {2019},
  doi     = {10.1109/TMI.2019.2900516}
}
```

---

## Author

**Dilukshan Viyapury** — IT22219534
BSc (Hons) in Information Technology, specialising in Information Technology
Faculty of Computing, Sri Lanka Institute of Information Technology

Project **R26-IT-083**, Component 03.

---

## Licence

Source code is released for academic use. The EchoNet-Dynamic and CAMUS datasets remain under the terms of their respective Research Data Use Agreements and are **not** redistributed here.
