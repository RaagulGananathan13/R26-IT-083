# Component Write-Up: Cardiomegaly Detection with Acquisition-Aware Decision Policies and Automated Report Generation

> **Status:** factual dossier extracted from the codebase, not paper prose.
> **Scope:** `C:\Users\94775\Desktop\Component_01\Component_01\`
> **Convention:** every claim is either **FACT** (traceable to a file) or marked
> **[inferred]**. Missing information is marked
> `NOT FOUND IN CODEBASE — needs input from author`.

---

## 0. Component Abstract

Chest radiographs are acquired in two projections — posteroanterior (PA, patient standing)
and anteroposterior (AP, portable/bedside) — and this component measures a persistent
0.0639 AUROC performance gap between them in a multi-label pathology classifier
(`RESULTS.md` §7). The component implements a cardiomegaly-focused diagnostic system
(ConvNeXt-Base classifier over 8 pathologies, BioBART-v2 report generator, Grad-CAM
attribution) and then uses it as a testbed for whether that acquisition disparity is a
model defect or an information limit. Three model-side interventions — acquisition-
conditioned recalibration, adversarial gradient reversal, and FiLM conditional
specialisation — were each falsified against controls (`RESULTS.md` §8), while two
decision-policy interventions succeeded: per-projection operating points reduced reported
TPR disparity by 73.3% at zero AUROC cost (`stage9_fairness.py`), and projection-conditional
selective deferral reduced the AP/PA accuracy gap from 6.68 to −0.62 points
(`reports/stage13/table.md`). The headline diagnostic result is cardiomegaly AUROC 0.9189
[0.9112, 0.9265] at 92.3% sensitivity on n=4,722 (`RESULTS.md` §3). All comparisons are
supported by paired significance tests with family-wise error control
(`reports/stage14/significance.md`). Extending the analysis to serial studies shows the
same acquisition axis produces a directional false-interval-change artefact: when
projection switches PA→AP between consecutive studies the classifier reports spurious
cardiac enlargement in 13.5% of pairs the radiologist recorded as unchanged, against 1.9%
spurious improvement (`reports/stage15/`).

**Keywords:** chest radiography; algorithmic fairness; acquisition bias; selective
prediction; radiology report generation; operating-point calibration

---

## 1. Role in the Overall System

**FACT.** This is "Component 01" of a 4-person group project. Its deliverables, per
`README.md` and `backend/main.py`:

1. Cardiomegaly detection plus 7 co-pathologies (multi-label classifier)
2. A generated free-text radiology report
3. Grad-CAM explainability overlays
4. A web demo (FastAPI backend + React frontend)

**Plain-language paragraph:** This component takes a single chest X-ray image and returns
three things — whether the heart is enlarged (cardiomegaly) plus seven other possible
findings, a written radiology-style report describing what it sees, and a heat-map showing
which part of the image drove the decision. On top of that it does something most such
systems do not: it checks how the X-ray was taken (standing vs. bedside), adjusts its
decision threshold accordingly, and refuses to answer cases it is not confident enough
about, referring those to a human radiologist instead.

**Integration with the other three components:**
`NOT FOUND IN CODEBASE — needs input from author`. No imports, API calls, shared schemas,
or documentation referencing sibling components were found. `backend/config.py` L4-5
explicitly states this is "a SEPARATE deployment" and that "Nothing in the original system
is read or written."

---

## 2. Problem Statement & Motivation

**FACT** (`README.md`, `MASTER_PLAN.md`):

- **Primary clinical task:** detect cardiomegaly from chest radiographs and produce a
  draft report, to reduce radiologist reporting workload.
- **Measured failure this component addresses** (`RESULTS.md` §7): the classifier performs
  materially worse on AP films than PA films — AUROC 0.8224 (AP) vs 0.8864 (PA), gap
  0.0639, 95% CI [0.0491, 0.0790], consistent in direction across 8/8 pathologies.

**Why it matters (FACT, from `backend/services/thresholds.py` L5-10 and
`frontend/src/components/ReliabilityNotice.jsx` L4-8):** AP films are acquired at the
bedside because the patient is too ill to stand. The system is therefore weakest precisely
on the sickest patients — a failure mode that is invisible if only pooled metrics are
reported.

**Secondary problem (FACT, `RESULTS.md` §4):** the standard report-generation metric
ROUGE-L was shown not to measure clinical correctness on this task — a single constant
paragraph scores 0.2641 ROUGE-L with 0.0000 clinical F1.

---

## 3. The Gap

**FACT — evidenced in `MASTER_PLAN.md` §9.1–9.3 ("Prior art — the full record"), which
records a deliberate prior-art search.**

| Gap | Evidence in codebase |
|---|---|
| Prior work treats projection bias as a **representation** problem to be trained away | `MASTER_PLAN.md` §9.3; Pereira et al. use an adversarial/training-time method |
| Fairness metrics are reported without testing **threshold-only** null baselines | `stage9_fairness.py` docstring; `RESULTS.md` §7A |
| Report-generation papers report ROUGE without a **degenerate-output control** | `RESULTS.md` §4; ref #11 (PLOS One 2021) is cited as precedent for the control |
| Selective prediction on CXR exists, but **not conditioned on acquisition projection** | `stage13_deferral.py` docstring L11-19 |

**The specific opening claimed:** that the AP/PA disparity should be addressed in the
*decision rule* (threshold, deferral budget) rather than in the *representation*, and that
the metric commonly used to report it is movable without touching the model at all.

**[inferred]** The framing "the metric is gameable" is a stronger claim than "we improve on
the metric," and reviewers will test it hardest. The codebase supports it empirically
(AUROC spread 0.00e+00 under threshold-only change) but a formal proof that this
generalises beyond this dataset is **not** present.

---

## 4. Research Question(s) This Component Answers

**[inferred from what the code measures — these RQs are not written verbatim anywhere in
the codebase and need author ratification.]**

- **RQ1** — Does the AP/PA performance disparity in a chest-radiograph classifier persist
  after model-side mitigation (recalibration, adversarial invariance, conditional
  specialisation)?
  *Tested by:* `stage6_acr.py`, `stage9b_gradrev.py`, `stage10_conditional.py`
- **RQ2** — Can the standard TPR-disparity fairness metric be reduced by a post-hoc,
  threshold-only intervention that provably does not alter the model's discrimination?
  *Tested by:* `stage9_fairness.py`
- **RQ3** — Does allocating a selective-deferral budget per projection reduce the AP/PA
  accuracy gap more than an equal-rate deferral policy at matched cost?
  *Tested by:* `stage13_deferral.py`
- **RQ4** — Is ROUGE-L a valid measure of clinical correctness for radiology report
  generation?
  *Tested by:* `RESULTS.md` §4 constant-string control
- **RQ5** — Does conditioning a report decoder on classifier outputs improve clinical
  accuracy beyond the effect of additional fine-tuning?
  *Tested by:* `stage11_conditioned.py`, Stage 11 ablation
- **RQ6** — When the acquisition projection changes between serial studies, does the
  classifier report a directional false interval change, and do per-projection thresholds
  correct it?
  *Tested by:* `stage15_interval.py`

---

## 5. Contribution Bullets & Novelty

1. **We show TPR disparity in CXR classifiers is reducible 73.3% by per-projection
   thresholds alone, at zero AUROC cost.**
   → **Novel.** Distinct from prior work in that it is post-hoc and provably
   model-preserving (AUROC spread 0.00e+00). **Validated against a same-experiment
   reimplementation of Pereira et al. (Stage 9B): on our data their method changed TPR
   disparity by +25.4% (worse) at a cost of 0.0789 AUROC.** See §13.

2. **We falsify three model-side mitigations of acquisition disparity against controls.**
   → **Novel** (as a negative result). Each of recalibration, gradient reversal, and FiLM
   conditioning was tested against a null arm and failed; `RESULTS.md` §8–9.

3. **We propose projection-conditional selective deferral, closing the AP/PA accuracy gap
   from 6.68 to −0.62 points.**
   → **Adapted.** Selective prediction is established
   ([arXiv:2509.10348](https://arxiv.org/abs/2509.10348)); group-conditional conformal
   prediction is established. The adaptation is conditioning the deferral budget on
   *acquisition projection*, motivated by the §8 irreducibility result.

4. **We implement a cardiomegaly detection and report-generation system with Grad-CAM.**
   → **Engineering.** ConvNeXt-Base, BioBART-v2, FastAPI, React — standard components,
   standard training. No methodological novelty. Stated plainly.

5. **We measure acquisition-induced false interval change: 13.5% spurious "worsening" vs
   1.9% "improvement" when projection switches PA→AP.**
   → **Novel.** Cross-sectional projection bias is documented; its propagation into
   *longitudinal* interval-change reporting is not. The 2026 temporal-CXR literature
   (TILA, TRACE, MI-CXR) evaluates temporal ordering and inversion but does not stratify
   by projection transition. Design is measurable without change adjudication because it
   conditions on radiologist-recorded stability.

**Conservative note:** Bullet 4 carries no novelty. Bullet 1 is the strongest claim, and it
is defensible **because the baseline was reimplemented rather than quoted** (`stage9b_gradrev.py`,
`RESULTS.md` §8.2). The residual exposure is presentational, not methodological: `RESULTS.md`
§8.1 and `PANEL_ANSWERS.md` Q2 still lead with the cross-dataset figure (73.3% vs 46.7%)
when the same-experiment figure in §8.2 is stronger. See §25 item 3.

---

## 6. Contribution → Evidence Traceability Table

| # | Contribution | Implemented where | Evaluated where | Risk |
|---|---|---|---|---|
| 1 | Per-projection operating points | `stage9_fairness.py` (`fit_thresholds`, `run_strategies`, `bootstrap_disparity`); live in `backend/services/thresholds.py` | `RESULTS.md` §7A; `backend/thresholds.json`; `Stage9A_Operating_Point_Fairness.ipynb`; **`reports/stage14/` (accuracy cost n.s., p=0.885)** | ✅ Baseline reimplemented (Stage 9B) — see §13 |
| 2 | Falsification of 3 model-side mitigations | `stage6_acr.py` (`calibration_ablation`, 4-arm); `stage9b_gradrev.py` (`_GradReverse`, `lambda_at`); `stage10_conditional.py` (`compare_probes`) | `RESULTS.md` §8, §9; Stage 6/6B/9B/10A notebooks | ✅ Controls present in-code |
| 3 | Projection-conditional deferral | `stage13_deferral.py` (`fit_conditional`, `apply_conditional`); live in `backend/services/deferral.py` | `reports/stage13/table.md`, `summary.json`, `deferral.png`; `RESULTS.md` §7B; **`reports/stage14/` (p=0.0004)** | ✅ 4-arm, val-fitted |
| 5 | Acquisition-induced false interval change | `stage15_interval.py` (`build_pairs`, `shuffle_order`, `cluster_bootstrap`) | `reports/stage15/interval_change.md`, `summary.json`, `interval_change.png` | ✅ 4-arm incl. shuffled null |
| 4 | Detection + report + XAI system | `backend/models/classifier.py`, `backend/models/report_generator.py`, `backend/services/gradcam.py` | `RESULTS.md` §3, §5; `reports/stage12/` | ✅ Evaluated |

**No bullet lacks evaluation evidence.** The risk is comparability (§20), not absence.

---

## 7. Related Work / Prior Approaches Referenced

**FACT — full list transcribed from `MASTER_PLAN.md` §9.4.**

### The baseline claimed to be beaten

> **Pereira SC, Rocha J, Gaudio A, Smailagic A, Campilho A, Mendonça AM.**
> *"Addressing Chest Radiograph Projection Bias in Deep Classification Models."*
> Medical Imaging with Deep Learning (MIDL) 2023 — **PMLR 227:1199–1210** (volume published
> 2024; PMLR key `pereira24a`).
> 🔗 https://proceedings.mlr.press/v227/pereira24a.html
> 🔗 OpenReview PDF: https://openreview.net/pdf?id=k8K2zEiv_m
>
> **Their reported result:** 46.7% disparity reduction, DenseNet-121, ChestX-Ray14.
> **This component:** 73.3%, ConvNeXt-Base, MIMIC-CXR-JPG.
> **✅ Their method was reimplemented on our data (`stage9b_gradrev.py`) — see §13 for the
> same-experiment comparison, which is what the claim rests on.**

### Comparison table

| Approach | Key idea | Limitation (as recorded in this codebase) |
|---|---|---|
| Pereira et al., MIDL 2023 | Label-conditional gradient reversal — train projection bias out of the classifier | Reimplemented on our data: made TPR disparity **worse** (+25.4%) and cost 0.0789 AUROC |
| Ganin & Lempitsky, ICML 2015 (gradient reversal) | Adversarially remove a nuisance factor from features | `RESULTS.md` §8: drove projection AUC to 0.5000, closed only 13.3% of gap, cost 0.0789 AUROC |
| Platt 1999 (Platt scaling) | Sigmoid recalibration of scores | Not a limitation — this **is** the null that falsified the ACR hypothesis |
| Sagawa et al., ICLR 2020 (Group-DRO) | Worst-group optimisation | Listed as Stage 9C, **never run** |
| Hardt et al., NeurIPS 2016 | Equal Opportunity / TPR disparity | Provides the metric this component argues is gameable |

### Full reference list (with links, as recorded)

| # | Reference | Role |
|---|---|---|
| 1 | Pereira et al., MIDL 2023, [PMLR 227:1199–1210](https://proceedings.mlr.press/v227/pereira24a.html) | **the baseline beaten** |
| 2 | Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts", [ICLR 2020](https://arxiv.org/abs/1911.08731) | Group-DRO (Stage 9C, not run) |
| 3 | Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015 — [arXiv:1409.7495](https://arxiv.org/abs/1409.7495) | gradient reversal |
| 4 | Hardt, Price & Srebro, "Equality of Opportunity in Supervised Learning", NeurIPS 2016 — [arXiv:1610.02413](https://arxiv.org/abs/1610.02413) | TPR disparity metric |
| 5 | ["Are demographically invariant models and representations in medical imaging fair?"](https://arxiv.org/abs/2305.01397) | invariance ≠ fairness |
| 6 | ["Who Gets Missed in the Tail? Thresholded Subgroup Underdiagnosis"](https://arxiv.org/abs/2607.07717) | threshold-dependence |
| 7 | ["Technical Acquisition Parameters Dominate Demographic Factors"](https://www.medrxiv.org/content/10.64898/2026.01.20.26344495.full.pdf) | motivates the axis |
| 8 | Seyyed-Kalantari et al., [CheXclusion, PSB 2021](https://psb.stanford.edu/psb-online/proceedings/psb21/seyyed-kalantari.pdf) | fairness foundation |
| 9 | ["The limits of fair medical imaging AI", *Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-03113-4) | context |
| 10 | ["The Subgroup Imperative", *Radiology: AI*](https://pubs.rsna.org/doi/full/10.1148/ryai.220270) | subgroup generalization |
| 11 | ["Encoder-decoder models perform no better than unconditioned baselines", PLOS One 2021](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259639) | constant-baseline control |
| 12 | [PromptMRG, AAAI 2024](https://arxiv.org/abs/2308.12604) | report-generation context |
| 13 | [Anatomically-Grounded Fact-Checking of CXR Reports](https://arxiv.org/html/2412.02177) | why Stage 7 was dropped |
| 14 | [KL-Regularised Group-DRO for CT](https://arxiv.org/html/2603.15941) | Group-DRO, other modality |
| 15 | Johnson et al., MIMIC-CXR-JPG — [arXiv:1901.07042](https://arxiv.org/pdf/1901.07042) | **dataset** |
| 16 | Wang et al., ChestX-Ray14, CVPR 2017 — [arXiv:1705.02315](https://arxiv.org/abs/1705.02315) | Pereira's dataset |
| 17 | Irvin et al., CheXpert, AAAI 2019 — [arXiv:1901.07031](https://arxiv.org/abs/1901.07031) | labeller |
| 18 | Liu et al., ConvNeXt, CVPR 2022 — [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | **backbone** |
| 19 | Yuan et al., BioBART, BioNLP 2022 — [arXiv:2204.03905](https://arxiv.org/abs/2204.03905) | **report decoder** |
| 20 | Platt, "Probabilistic Outputs for SVMs", 1999 | the null that killed ACR |

### Additional references cited in code but not in `MASTER_PLAN.md` §9.4

| Reference | Where cited | Role |
|---|---|---|
| Arun et al., "Assessing the Trustworthiness of Saliency Maps", *Radiology: AI* 2021 | `backend/services/gradcam.py` L8-11 | Grad-CAM repeatability SSIM 0.12 |
| Smit et al., CheXbert — [arXiv:2004.09167](https://arxiv.org/abs/2004.09167) | `Stage12_CheXbert_Evaluation.ipynb` | independent label validation |
| Perez et al., FiLM, AAAI 2018 — [arXiv:1709.07871](https://arxiv.org/abs/1709.07871) | `stage10_conditional.py` | conditioning mechanism |
| McNemar Q. *Psychometrika* 1947;12(2):153-157 | `stage14_significance.py` | the paired test |
| Fagerland MW, Lydersen S, Laake P. "The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional." [*BMC Med Res Methodol* 2013;13:91](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-91) | `stage14_significance.py:mcnemar_midp` | justifies reporting mid-p |
| Holm S. "A simple sequentially rejective multiple test procedure." *Scand J Stat* 1979;6(2):65-70 | `stage14_significance.py:holm_bonferroni` | family-wise error control |
| [Multi-pathology CXR Classification with Rejection Mechanisms](https://arxiv.org/abs/2509.10348) | prior-art check for Stage 13 | selective prediction on CXR |

---

## 8. Domain-Specific Structuring Fit

**Best fit: Build-up Ablation (ML), with a strong secondary Benchmark-driven character.**

**Why (FACT):** the project is organised as numbered stages, each adding or testing exactly
one mechanism against the previous stage, with an explicit control arm at each step:

- Stage 1 (target cleaning) → Stage 2 (transforms) → Stage 4 (report baseline) → Stage 4B
  (decoding ablation) → Stage 5 (classifier) → Stage 6/6B (ACR + its control) → Stage 9A
  (thresholds) → Stage 9B (gradient reversal) → Stage 10A (FiLM) → Stage 11 (prompting)
  → Stage 12 (CheXbert validation) → Stage 13 (deferral).
- `stage6_acr.py` implements an explicit **four-arm** ablation (raw / Platt / ACR /
  shuffled); `stage13_deferral.py` implements another (none / random / global / conditional).

**Extra content this flow implies should be captured:**
- ✅ A stage-by-stage delta table (present — `RESULTS.md`)
- ✅ Null/control arm for every claimed mechanism (present)
- ⚠️ A single consolidated ablation figure showing cumulative gains — **not present**; only
  `reports/stage13/deferral.png` exists.

**Threat-model-driven:** Not applicable — no adversary is modelled.
**Theorem-Proof:** Partially applicable. `backend/services/thresholds.py` L27-34 contains a
prose argument that thresholding cannot change AUROC ("AUROC is computed over the whole
ranking; a threshold is one cut through it, and cutting per group cannot reorder any
case"), verified numerically to 1e-12. This is a **proof sketch, not a formal proof.**

---

## 9. Method / Design

### 9.1 Architecture (as data flow)

```
chest X-ray (JPEG/PNG)
   │
   ├─► cxr_transforms.build_transform("test")
   │      resize 384×384 · per-image z-score  (NOT ImageNet normalisation)
   │
   ├─► CXRClassifier  (backend/models/classifier.py)
   │      ConvNeXt-Base features → LayerNorm(1024) → Dropout(0.3)
   │      → Linear(1024,512) → GELU → Dropout(0.198) → Linear(512,8)
   │      └─► 8 sigmoid probabilities
   │
   ├─► ThresholdPolicy.get(pathology, view)   ← Contribution 1
   │      returns per-projection threshold (AP 0.409 / PA 0.348 for Cardiomegaly)
   │
   ├─► DeferralPolicy.assess(prob, threshold, view)   ← Contribution 3
   │      margin = |p − τ|;  defer if margin < cutoff[view]
   │      cutoffs: AP 0.2247, PA 0.0029
   │
   ├─► GradCAM.generate(x, class_idx)   (hooks model.features[-1], 12×12×1024)
   │
   └─► CXRReportGenerator  (backend/models/report_generator.py)
          vision encoder → proj → 144 visual tokens
          → BioBART-v2-base via inputs_embeds  (NOT encoder_outputs)
          → greedy decode → report text
```

### 9.2 Key algorithms

**A · Per-projection operating points** (`stage9_fairness.py`) — **original design work.**

```
for each pathology:
    for each projection g in {AP, PA}:
        τ_g ← argmax_τ F1(y_g, p_g ≥ τ)      # fitted on VALIDATION only
apply τ_view at inference
```
Claim: AUROC is invariant to this because thresholding cannot reorder cases.

**B · Projection-conditional deferral** (`stage13_deferral.py`) — **original design work.**

```
margin m ← |p − τ_view|
fit on VAL:
    for coverage budget C:
        search c_AP over grid; c_PA ← (C·n − c_AP·n_AP)/n_PA     # equal total budget
        choose the pair minimising |acc_AP − acc_PA|
    freeze q_AP, q_PA as margin quantiles
apply to TEST: answer iff m ≥ q_view, else refer to radiologist
```

**C · Four-arm calibration ablation** (`stage6_acr.py:calibration_ablation`) — **original
control design.** Arms: A raw / B Platt (the null) / C ACR / D shuffled. Docstring states
the claim "requires C to beat **B**, not merely A."

**D · Prompt conditioning with label dropout** (`stage11_conditioned.py:build_prompt`) —
**adapted.** States both positives and negatives; `dropout` hides labels at random during
training, because "without it the decoder learns to transcribe the prompt and stops looking
at the image."

### 9.3 Key design decisions and rationale (FACT — from code comments)

| Decision | Rationale as written in code | Location |
|---|---|---|
| Per-image z-score, not ImageNet normalisation | "wrong for grayscale radiographs — measured 4.4× worse variance" | `backend/config.py` L55-60 |
| `inputs_embeds`, not `encoder_outputs` | old path bypassed BART's encoder entirely | `backend/models/report_generator.py` |
| FiLM applied **after** LayerNorm | LayerNorm exactly cancels a pre-applied modulation: `((1+g)x + b − (1+g)m − b)/((1+g)s) = (x−m)/s` | `stage10_conditional.py` |
| Greedy decoding (`num_beams=1`) | "Stage 4B ablation: greedy beat beam-4 on 5 of 7 metrics" | `backend/config.py` L63-65 |
| Ground truth from original CSV, not cleaned manifest | showing the cleaned version "would compare the model against our own preprocessing rather than against the radiologist" | `backend/config.py` L32-38 |
| Unknown projection falls back to global threshold | "Guessing PA on a bedside film would under-call cardiomegaly on exactly the patients least able to tolerate a missed diagnosis" | `backend/services/thresholds.py` L54-58 |
| Display image from original pixels, not z-scored tensor | z-scoring "destroys absolute intensity by design" | `backend/services/inference.py` L96-99 |
| `group_ece` = mean within-group ECE, not gap | "`subgroup_ece_gap` alone is gameable: a model equally badly calibrated in both groups scores a perfect 0 gap" | `stage6_acr.py` |

### 9.4 Novel vs. standard, function-by-function

| Code | Classification |
|---|---|
| `stage9_fairness.py` — `fit_thresholds`, `threshold_for_tpr`, `run_strategies` | **Original** |
| `stage13_deferral.py` — `fit_conditional`, `apply_conditional`, four-arm `run` | **Original** |
| `stage15_interval.py` — `build_pairs`, `shuffle_order`, `study_id_order_agreement` | **Original.** The measurement design is the contribution: conditioning on radiologist-recorded stability removes the need to adjudicate real change, and the shuffled-order null isolates temporal direction. |
| `stage14_significance.py` — `mcnemar_midp`, `holm_bonferroni`, `paired_diff_ci`, `bootstrap_gap_difference` | **Standard statistics, correctly applied.** McNemar (1947), mid-p (Fagerland 2013), Holm (1979). No novelty claimed — the judgement is in *which* test applies where, see §15. |
| `stage6_acr.py` — `calibration_ablation`, `group_ece` | **Original control design** |
| `backend/services/thresholds.py`, `deferral.py` | **Original** (deployment of the above) |
| `stage9b_gradrev.py` — `_GradReverse`, `lambda_at` | **Standard** — Ganin & Lempitsky 2015 |
| `stage10_conditional.py` — FiLM layer | **Standard** (Perez et al. 2018); the *placement fix* after LayerNorm is original debugging |
| `backend/models/classifier.py` | **Standard** — torchvision ConvNeXt-Base + custom head |
| `backend/models/report_generator.py` | **Standard** — HuggingFace BART; the `inputs_embeds` routing is a bug-fix, not novelty |
| `backend/services/gradcam.py` | **Standard** — textbook Grad-CAM via forward/backward hooks |
| `cxr_transforms.py` | **Adapted** — standard torchvision ops, non-standard normalisation choice |
| `chexpert_fusion.py` | **Adapted** — text-adjudicated fusion over CheXpert labels |
| `frontend/` | **Engineering** — React 19 + Vite |

### 9.5 Notation table

| Symbol | Meaning |
|---|---|
| `p` | predicted probability for a pathology, sigmoid output ∈ [0,1] |
| `τ`, `τ_g` | decision threshold; `τ_g` = threshold for projection group *g* |
| `g` | projection group, `g ∈ {AP, PA}` |
| `m` | confidence margin, `m = \|p − τ_g\|` |
| `q_g` | frozen margin cut-off for group *g* (deferral) |
| `C` | coverage — fraction of cases the system answers |
| `c_AP`, `c_PA` | per-group coverage |
| `λ` | gradient-reversal strength (Stage 9B) |
| `γ`, `β` | FiLM scale and shift parameters (Stage 10A) |
| ECE | Expected Calibration Error |

---

## 10. Algorithmic Complexity Analysis

**Applicable** — two original algorithms.

### A · Per-projection threshold fitting (`stage9_fairness.py`)

- **Time: O(K · G · N log N)** where K = 8 pathologies, G = 2 groups, N = validation size.
  Reasoning: for each (pathology, group) the scores must be sorted once to sweep candidate
  thresholds (`O(N log N)`), then the F1 sweep is a single linear pass `O(N)`.
- **Space: O(N)** — one score array plus label array per sweep.
- Best/worst/average do not meaningfully differ — the sort dominates unconditionally.

### B · Conditional deferral fitting (`stage13_deferral.py:fit_conditional`)

- **Time: O(|grid| · N)** per coverage target, with `|grid| = 141` (`np.linspace(0.30, 1.0, 141)`).
  Reasoning: for each candidate `c_AP` the derived `c_PA` is computed in O(1), two quantiles
  are taken (`O(N)` via introselect), and accuracy is evaluated in one pass `O(N)`. With 6
  coverage targets this is **O(6 · 141 · N) ≈ O(N)** with a large constant.
- Bootstrap adds **O(B · N)** with `B = N_BOOT = 2000`.
- **Space: O(N)** for masks and the `O(B)` bootstrap output array.
- **Measured wall-clock:** the whole Stage 13 analysis completes in seconds on CPU
  (no GPU, no model loaded).

### C · Inference path
Dominated by the two neural forward passes — **not original algorithmic work**, so no
complexity analysis is claimed.

---

## 11. Experimental Setup

### Hardware

| | |
|---|---|
| Training | **NVIDIA L4 GPU** via Google Colab (`MASTER_PLAN.md` §10) |
| Cost model | ~1.75 compute units/hour on L4 (`MASTER_PLAN.md` §10) |
| Local dev/demo | Windows 11 Home 10.0.26200 |
| Local torch build | `2.11.0+cpu` — **CUDA not available locally** |
| CPU/RAM specs | `NOT FOUND IN CODEBASE — needs input from author` |

### Software

**✅ `backend/requirements.txt` is fully version-pinned**, verified against the live
environment (10/10 match). It was previously name-only, which is a reproducibility defect —
`transformers` 5.x removed `encode_plus`, so an unpinned reinstall could silently break.

| Package | Pinned | Installed |
|---|---|---|
| python | — | 3.13.5 |
| torch | `2.11.0` | 2.11.0 (+cpu build locally) |
| torchvision | `0.26.0` | 0.26.0 |
| transformers | `5.7.0` | 5.7.0 |
| numpy | `2.1.3` | 2.1.3 |
| pandas | `2.3.1` | 2.3.1 |
| pillow | `11.1.0` | 11.1.0 |
| **opencv-python-headless** | `4.12.0.88` | 4.12.0.88 |
| fastapi | `0.110.0` | 0.110.0 |
| uvicorn[standard] | `0.29.0` | 0.29.0 |
| python-multipart | `0.0.9` | 0.0.9 |

**Two deliberate pinning decisions, both documented in the file header:**

1. **`torch==2.11.0`, not `2.11.0+cpu`.** Pinning the local build tag would force CPU-only
   torch onto a CUDA machine. GPU install instructions are given separately in the header.
2. **`opencv-python-headless`, not `opencv-python`.** ⚠️ **Discovered during pinning:** three
   OpenCV distributions are installed (`opencv-python` 4.10.0.84, `opencv-contrib-python`
   4.10.0.84, `opencv-python-headless` 4.12.0.88) and `cv2.__version__` resolves to
   **4.12.0** — i.e. the *headless* build shadows the one the old requirements named. Headless
   is also the correct choice for an API server: it ships no GUI bindings, which the backend
   never calls and which fail to install on headless Linux hosts.

⚠️ Remaining gap: no `environment.yml`, lockfile, or container. Transitive dependencies are
unpinned.

**Frontend** (`frontend/package.json` — these *are* pinned to semver ranges):
react ^19.1.0 · react-dom ^19.1.0 · vite ^6.3.5 · @vitejs/plugin-react ^4.5.2 ·
tailwindcss ^4.1.7 · @tailwindcss/vite ^4.1.7

### Datasets

| | |
|---|---|
| Name | **MIMIC-CXR / MIMIC-CXR-JPG** (Johnson et al., [arXiv:1901.07042](https://arxiv.org/pdf/1901.07042)) |
| Access | PhysioNet **credentialed**, under a Data Use Agreement |
| Splits | train 
`manifest_train.csv` (19 MB) · val **4,474** · test **4,722** |
| Cardiomegaly prevalence (test) | **50.4%** — deliberately enriched |
| Projection split (test) | AP **2,891** / PA **1,831** |
| Labels | 8: Cardiomegaly, Edema, Pleural_Effusion, Atelectasis, Consolidation, Lung_Opacity, Pneumonia, Pneumothorax |
| Image format | PNG, 384×384 (`data/output/cardio_image_384`, 4.9 GB, outside component folder) |
| Preprocessing | Stage 1 report-target cleaning; Stage 2 transform selection; Stage 3 label fusion |
| Timestamps | `mimic-cxr-2.0.0-metadata.csv` (`StudyDate`/`StudyTime`); test-set slice cached to `reports/stage15/study_timestamps.csv` so reruns need no access to `data/` |
| Licence handling in code | `review_cases/.gitignore` contains `*` — excludes all MIMIC data from git, preventing DUA breach |

### Environment

- No Docker, no `environment.yml`, no `venv` config found — `NOT FOUND IN CODEBASE`
- Launcher: `run_backend.bat`
- `backend/config.py` `CORS_ORIGINS` allows `localhost:5173/5174`
- Training ran in Google Colab notebooks with Google Drive mounting

### Compute Cost

- **FACT:** budget tracked in `MASTER_PLAN.md` §10; ~63 CU were reported unused at the time
  of the Stage 13 work.
- **FACT:** Stage 13 costs **0 CU** — post-hoc on cached predictions, `torch` never imported
  (asserted as test #1 in `stage13_deferral.py:selftest`).
- **Per-run wall-clock training times:** `NOT FOUND IN CODEBASE — needs input from author`.
  No timing logs were located.

---

## 12. Parameters / Configuration

### Classifier — Stage 5 (`Stage5_Classifier_Training.ipynb`)

| Parameter | Value |
|---|---|
| IMG_SIZE | 384 |
| BATCH | 64 |
| EPOCHS | 30 |
| LR | 2e-4 |
| WEIGHT_DECAY | 0.05 |
| DROPOUT | 0.3 |
| SEED | 42 |

⚠️ **Conflict:** the notebook contains both `EPOCHS = 30` and `EPOCHS = 3` in different
cells. Which applied to the shipped checkpoint is **unresolved — needs author input.**

### Report generator — Stage 4 (`Stage4_Report_Generator.ipynb`)

| Parameter | Value |
|---|---|
| BASE_LR | 5e-5 |
| EFFECTIVE_BATCH | 32 |
| EPOCHS | 15 |
| PATIENCE | 5 |
| GRAD_CLIP | 1.0 |
| LABEL_SMOOTH | 0.1 |
| EMA_DECAY | 0.9998 |
| MAX_TOKENS | 256 |
| GEN_MAX_TOKENS / GEN_MIN_TOKENS | 192 / 24 |
| NO_REPEAT_NGRAM | 3 |
| LENGTH_PENALTY | 1.2 |
| NUM_WORKERS | 4 |
| CKPT_EVERY_MIN | 10 |
| CONST_BASELINE_ROUGEL | **0.2769** ⚠️ |

### Report generator — Stage 11 (`Stage11_Conditioned_Report.ipynb`)

| Parameter | Value |
|---|---|
| EPOCHS | 3 |
| LR_REST | 5e-5 |
| LR_VISION | 5e-6 (0.1× — partial unfreezing) |
| PROMPT_DROP | 0.15 |
| N_GATE | 96 |

### Inference — `backend/config.py`

| Parameter | Value |
|---|---|
| IMG_SIZE | 384 |
| DECODER_NAME | `GanjinZero/biobart-v2-base` |
| NUM_VISUAL_TOKENS | 144 (12×12) |
| GEN_NUM_BEAMS | 1 (greedy) |
| GEN_MAX_TOKENS / MIN / NO_REPEAT_NGRAM | 192 / 24 / 3 |

### Significance testing — `stage14_significance.py`

| Parameter | Value |
|---|---|
| ALPHA | 0.05 |
| N_BOOT | 10000 |
| SEED | 20260813 |
| Correction | Holm-Bonferroni, within family |
| Reported p-variant | mid-p (exact + Yates chi-square also stored) |
| Families | F1 vs always-negative (8) · F2 classifier vs report (8) · F3 threshold policy (1) |

### Interval-change analysis — `stage15_interval.py`

| Parameter | Value |
|---|---|
| TARGET | Cardiomegaly |
| N_BOOT | 4000 (cluster bootstrap, resampled by patient) |
| SEED | 20260813 |
| Ordering source | `StudyDate` + `StudyTime` (`mimic-cxr-2.0.0-metadata.csv`) |
| Arms | A same-projection · B shuffled order (null) · C true order · D + per-projection thresholds |
| Conditions | radiologist recorded cardiomegaly unchanged / all 8 unchanged |

### Thresholds — `backend/thresholds.json`

| | Cardiomegaly |
|---|---|
| global | 0.401 |
| AP | 0.409 |
| PA | 0.348 |

### Deferral — `reports/stage13/deferral_policy.json`

| | |
|---|---|
| DEPLOY_COVERAGE | 0.85 |
| margin cutoff AP | 0.22465848658323284 |
| margin cutoff PA | 0.0029436368836295076 |
| N_BOOT | 2000 |
| SEED | 20260807 |
| COVERAGES swept | 0.95, 0.90, 0.85, 0.80, 0.75, 0.70 |

### ⚠️ Parameter conflicts to resolve

| Parameter | Value A | Value B | Where |
|---|---|---|---|
| Constant-string ROUGE-L baseline | **0.2769** | **0.2641** | `Stage4_Report_Generator.ipynb` vs `RESULTS.md` §4 + `backend/config.py` |
| EPOCHS (classifier) | 30 | 3 | two cells of `Stage5_Classifier_Training.ipynb` |
| SEED | 42 (Stage 5) | 20260807 (Stage 13) | different stages — **[inferred]** intentional, not a conflict |

---

## 13. Baseline(s) Compared Against

**FACT — baselines are present and unusually thorough.**

| Baseline | Purpose | Where |
|---|---|---|
| **"Always say no"** predictor | accuracy floor for rare pathologies | `README.md` — beats the model on 5/8 labels |
| **Constant-string report** | ROUGE-L validity control | `RESULTS.md` §4 — 0.2641 ROUGE-L, 0.0000 clinical F1 |
| **Random real report (wrong patient)** | ROUGE-L direction control | `RESULTS.md` §4 — 0.1821 / 0.3120 |
| **Platt scaling (1999)** | the null arm for ACR | `stage6_acr.py:calibration_ablation` arm B |
| **Shuffled acquisition labels** | destroys signal, keeps procedure | `stage6_acr.py` arm D |
| **Random deferral** | null for confidence ordering | `stage13_deferral.py` arm B |
| **Global (equal-rate) deferral** | the real control for Contribution 3 | `stage13_deferral.py` arm C |
| **Same-projection pairs** | false-positive floor for interval change | `stage15_interval.py` arm A |
| **Shuffled temporal order** | destroys direction, keeps every other property | `stage15_interval.py` arm B |
| **Per-projection thresholds re-applied** | tests whether Contribution 1 rescues Stage 15 | `stage15_interval.py` arm D |
| **Stage 4 → Stage 11** | internal improvement baseline | `RESULTS.md` §5 |
| **Greedy vs beam-4 (6 strategies)** | decoding ablation | `Stage4B_Decoding_Ablation.ipynb` |
| **Pereira et al. MIDL 2023 — reimplemented** | ✅ **external method, re-run on our data** | `stage9b_gradrev.py`; `RESULTS.md` §8.2 |

### ✅ The external baseline WAS reproduced in code

`stage9b_gradrev.py` L1-5 states verbatim:

> *"Reimplements Pereira et al., MIDL 2023 (PMLR 227:1199-1210) on OUR data, OUR backbone
> and OUR split, so the Stage 9A comparison stops being cross-dataset."*

It implements their **label-conditional gradient reversal**, with λ = 0.1 chosen to match
their reported 1e-4 / 1e-5 learning-rate ratio. The same-experiment result
(`RESULTS.md` §8.2):

| On our data / backbone / split | AUROC | AUROC gap | TPR Disparity | Projection AUC | Cost |
|---|---|---|---|---|---|
| Baseline | 0.8554 | 0.0639 | 0.1581 | — | — |
| **Per-projection thresholds (ours)** | 0.8554 | 0.0639 (0.0%) | **0.0416 (−73.3%)** | — | **0.00** |
| **Gradient reversal (Pereira, reimplemented)** | 0.7765 | 0.0554 (−13.3%) | **0.1982 (+25.4%)** | **0.5000** | **−0.0789** |
| *Pereira et al., as published (ChestX-Ray14)* | *0.8366* | *not reported* | *0.0969 (−46.7%)* | *0.6118* | *−0.0091* |

**Key facts for the paper:**
- On our data their method made TPR disparity **worse** (+25.4%), not better.
- We achieved **greater** invariance than published (projection AUC 0.5000 vs 0.6118), and
  the AUROC gap still moved only 0.0639 → 0.0554.
- Row 4 is included for context only and is **not** the comparison the claim rests on.

⚠️ **Remaining risk is presentational only.** `RESULTS.md` §8.1 and `PANEL_ANSWERS.md` Q2
currently lead with the cross-dataset "73.3% vs 46.7%" framing. The same-experiment
comparison in §8.2 is stronger and should lead instead.

---

## 14. Evaluation Metrics

| Metric | Where computed | Why it fits | Answers |
|---|---|---|---|
| **AUROC** (+ 95% bootstrap CI) | `stage9_fairness.py`, `RESULTS.md` §3 | threshold-independent discrimination; invariant to the threshold intervention, which is what makes Contribution 1 provable | RQ2 |
| **Sensitivity / Specificity** | `RESULTS.md` §3, `stage13_deferral.py` | screening task — missed cardiomegaly costs more than a false alarm | RQ1, RQ3 |
| **Accuracy** | `stage13_deferral.py`, `RESULTS.md` §5.0 | interpretable for the panel; valid only at 50.4% prevalence | RQ3 |
| **TPR Disparity (Equal Opportunity)** | `stage9_fairness.py:bootstrap_disparity` | the standard CXR fairness metric (Hardt et al. 2016) | RQ2 |
| **ECE / `group_ece`** | `stage6_acr.py` | calibration quality; within-group form chosen because the gap form is gameable | RQ1 |
| **Clinical-efficacy F1** | Stage 12 / CheXbert | measures whether the report states the right findings | RQ4, RQ5 |
| **ROUGE-L** | Stage 4/4B/12 | reported **as the object of study**, argued invalid | RQ4 |
| **Prior-study hallucination rate** | Stage 1 / 12 | domain-specific correctness | — |
| **Coverage** | `stage13_deferral.py` | fraction answered; mandatory companion to accuracy | RQ3 |
| **AP/PA accuracy gap (+ CI)** | `stage13_deferral.py:bootstrap_gap` | the disparity being closed | RQ3 |
| **Exact-match (all 8 labels)** | `RESULTS.md` §3 | strictest multi-label criterion — 34.9% | — |
| **McNemar mid-p (+ Holm-adjusted)** | `stage14_significance.py` | correct paired test for two methods scoring the same items | RQ2, RQ5 |
| **Paired difference in proportions + Wald CI** | `stage14_significance.py:paired_diff_ci` | effect size — a p-value alone says nothing about magnitude | RQ2, RQ5 |
| **Paired bootstrap test on difference of gaps** | `stage14_significance.py:bootstrap_gap_difference` | the only valid test for the deferral claim (McNemar is inapplicable) | RQ3 |

---

## 15. Experimental Repetition & Statistical Robustness

**This is a relative strength of the component — FACT.**

| Practice | Present? | Evidence |
|---|---|---|
| Fixed seeds | ✅ | `SEED = 42` (Stage 5), `SEED = 20260807` (Stage 13); seed control across 14 files |
| Bootstrap confidence intervals | ✅ | `N_BOOT = 2000` in `stage13_deferral.py`; `bootstrap_disparity` in `stage9_fairness.py` |
| **Cluster bootstrap** | ✅ | used where 8 labels share one image — correct variance handling |
| **Stratified/within-group bootstrap** | ✅ | `bootstrap_gap` resamples within each projection so group sizes don't wander |
| Reported CIs | ✅ | e.g. AUROC 0.9189 [0.9112, 0.9265]; gap 6.68 [4.51, 8.84]; −0.62 [−2.78, 1.37] |
| Null/control arms | ✅ | four-arm designs in Stage 6 and Stage 13 |
| Averaging over random draws | ✅ | random-deferral arm averaged over 25 draws |
| **Formal significance testing** | ✅ | **Stage 14** — McNemar mid-p, 10 tests, Holm-Bonferroni corrected |
| **Paired-difference effect sizes + CIs** | ✅ | `stage14_significance.py:paired_diff_ci` (Wald, correlation-aware) |
| Shuffled-order null (temporal) | ✅ | `stage15_interval.py:shuffle_order` — collapses +11.59 → +1.55 |
| Unit tests | ✅ | **193** across 8 modules (38+18+28+20+24+13+33+19), all passing |
| Val-fit / test-apply discipline | ✅ | `stage13_deferral.py` freezes quantiles on val before touching test |

### ⚠️ Weaknesses reviewers will target

1. **No multi-seed training runs.** Every model was trained **once**. All CIs quantify
   *sampling* variance over the test set, **not training variance**. A different seed could
   move the numbers and this is unmeasured.
2. ~~No formal significance testing~~ — **RESOLVED by Stage 14** (`stage14_significance.py`).
   McNemar mid-p on paired predictions, Holm-Bonferroni within family, paired-difference
   effect sizes with Wald CIs, and a paired bootstrap test for the deferral claim.

   **Method rationale worth stating in the paper:**
   - **McNemar, not a t-test** — both methods score the same 4,722 radiographs, so the
     observations are paired; a two-sample t-test would assume independence and discard the
     information that lives in the discordant pairs.
   - **mid-p, not exact** — the exact conditional binomial is valid but conservative;
     Fagerland, Lydersen & Laake (*BMC Med Res Methodol* 2013;13:91) recommend mid-p as the
     best power/validity trade-off. Exact and Yates-corrected chi-square are also computed
     and stored.
   - **Holm-Bonferroni, not raw p** — 10 tests at α=0.05 carry a ~40% family-wise false
     positive rate. Holm is used over plain Bonferroni (uniformly more powerful, same FWER
     control) and over Benjamini-Hochberg (these are confirmatory tests, where a false
     positive is worse than a false negative).
   - **⚠️ McNemar deliberately NOT applied to the deferral comparison.** A deferral policy
     never changes a prediction — only which cases are answered — so two policies emit
     identical labels on every case they share, giving b = c = 0 and a spurious p = 1.0.
     Using it there would be a category error that *looks* like a result. The correct
     instrument, a paired bootstrap on the difference of gaps, is used instead
     (`bootstrap_gap_difference`, 10,000 stratified replicates).
3. **No cross-validation** — a single fixed train/val/test split.
4. **No external validation** — single dataset, single centre.

---

## 16. Ablation Studies

**FACT — ablations are extensive.**

| Ablation | Arms | Result |
|---|---|---|
| **Calibration** (`stage6_acr.py`) | raw / Platt / ACR / shuffled | ACR = Platt → hypothesis falsified |
| **Deferral** (`stage13_deferral.py`) | none / random / global / conditional | conditional closes gap; global does not |
| **Interval change** (`stage15_interval.py`) | same-projection / shuffled order / true order / + per-projection thresholds | only projection *change* is directional; shuffling collapses it +11.59 → +1.55; thresholds reduce but do not eliminate |
| **Decoding** (Stage 4B) | 6 strategies incl. greedy, beam-4 | greedy won 5 of 7 metrics |
| **Threshold strategy** (`stage9_fairness.py`) | global / per_group_f1 / equal_tpr | per-group reduces disparity 73.3% |
| **Prompt conditioning** (Stage 11) | with / without prompt | +0.0023 — gain attributed to fine-tuning |
| **Linear probe** (`stage10_conditional.py:compare_probes`) | 3 arms | +0.0003 — no specialisation |
| **λ sweep** (Stage 9B) | warmup ramp, multiple λ | invariance ↑ → AUROC ↓ |

**Gap:** there is **no single consolidated ablation table or figure** showing cumulative
contribution of each stage to the final result. `[inferred]` This would be the highest-value
figure to add for a paper.

---

## 17. Existing Figures / Visual Assets Inventory

| Path | Description | Paper-ready? |
|---|---|---|
| `reports/stage13/deferral.png` | Two-panel: (L) accuracy vs coverage for random/global/conditional; (R) AP/PA gap vs coverage with 95% bootstrap CI bands. Okabe-Ito colourblind-safe palette, 170 dpi | ✅ **Yes** |
| `reports/stage13/table.md` | Four-arm results table, all coverages | ✅ as a table |
| `reports/stage13/summary.json` | Machine-readable full results | source data |
| `reports/stage13/deferral_policy.json` | Frozen deployment policy | source data |
| `reports/stage15/interval_change.png` | Two-panel: (L) directional asymmetry by projection transition with 95% CI; (R) four arms incl. shuffled null and the failed threshold fix. Okabe-Ito, 170 dpi | ✅ **Yes** |
| `reports/stage15/interval_change.md` | Full four-arm tables, both conditions | ✅ as a table |
| `reports/stage14/significance.md` | All 10 McNemar tests + Holm-adjusted p + the deferral bootstrap test | ✅ as a table |
| `reports/stage14/summary.json` | Machine-readable, includes exact/mid-p/chi2 for every test | source data |
| `reports/stage12/MANUAL_REVIEW.md` | Side-by-side generated vs. real reports | ✅ as an appendix |
| `review_cases/*/manifest.csv` | Per-case metadata for 2,027 demo images | source data |
| Notebook output cells | Training curves, ROC curves | ⚠️ `NOT VERIFIED` — outputs not individually inspected |

**⚠️ Figure gap.** For a paper you would normally also need: a system architecture diagram,
ROC curves per projection, a calibration/reliability diagram, and the consolidated ablation
figure. **None of these exist as standalone assets.**

---

## 18. Results Found in Repo (facts only)

### Classifier — Stage 5, test n=4,722 (`RESULTS.md` §3)

| | Value |
|---|---|
| Cardiomegaly AUROC | **0.9189** [0.9112, 0.9265] |
| Cardiomegaly accuracy | **83.2%** (3,929/4,722) |
| Cardiomegaly sensitivity / specificity | **92.3% / 74.0%** |
| Cardiomegaly TP/FP/TN/FN | 2197 / 609 / 1732 / 184 |
| Mean AUROC (8 labels) | **0.8554** (baseline 0.8251) |
| Mean accuracy (8 labels) | **83.7%** |
| Exact-match all 8 | **34.9%** (1,649/4,722) |
| Average labels correct | 6.69 of 8 |

### Report generator — Stage 11 (`RESULTS.md` §5, §5.0)

| | Stage 4 | Stage 11 |
|---|---|---|
| Clinical-efficacy F1 | 0.5799 | **0.5937** |
| ROUGE-L | 0.2918 | 0.2896 |
| Prior-study hallucination | 0.0000 | 0.0000 |
| Mean words | 36.62 | 39.45 (reference 46.9) |
| Cardiomegaly accuracy | — | **80.4%** (3,796/4,722) |
| Cardiomegaly sens/spec | — | **88.8% / 71.8%** |
| Mean accuracy (8 labels) | — | **83.3%** |

- Prior-study hallucination in the **raw training corpus: 70.70%** → **0.0000** in output.
- CheXbert micro-F1-14: Stage 4 **0.5783** → Stage 11 **0.5939**; Cardiomegaly **0.8287**.
- CheXbert vs internal extractor agreement: **0.002**.

### ROUGE-L validity control (`RESULTS.md` §4)

| "Report" | ROUGE-L | Clinical F1 |
|---|---|---|
| Constant string | 0.2641 | **0.0000** |
| Random real report, wrong patient | 0.1821 | 0.3120 |
| Own real report | 1.0000 | 1.0000 |
| Stage 11 | 0.2896 | 0.5937 |

### Fairness — Stage 9A (`RESULTS.md` §7)

| | Value |
|---|---|
| AP AUROC / PA AUROC | 0.8224 / 0.8864 |
| Gap | **0.0639** [0.0491, 0.0790], 8/8 same direction |
| TPR disparity reduction | **−73.3%** (Pereira et al.: −46.7%) |
| AUROC cost | **0.00e+00** |
| Cardiomegaly prevalence AP vs PA | 62.1% vs 32.0% (1.94×) |
| Prevalence gap with zero co-pathologies | 43.6% vs 21.8% |
| Acquisition metadata alone predicts pathology | AUROC 0.6665–0.7016 |

### Falsified interventions (`RESULTS.md` §8, §9)

| Hypothesis | Result |
|---|---|
| Acquisition-Conditioned Reliability | = Platt scaling; shuffled ≡ real |
| Adversarial invariance | AUC→0.5000 closed 13.3% of gap, cost 0.0789 AUROC |
| Conditional specialisation | +0.0003 |
| Classifier-conditioned generation | +0.0023 (fine-tuning effect) |
| Cross-modal agreement | 85.57% vs 86.64% for plain confidence |

### Stage 13 deferral (`reports/stage13/table.md`, test n=4,722)

| Arm | Coverage | Accuracy | Sens | AP acc | PA acc | Gap [95% CI] |
|---|---|---|---|---|---|---|
| A none | 100.0% | 83.19% | 92.9% | 80.59% | 87.27% | 6.68 [4.51, 8.84] |
| B random | 80.0% | 83.22% | — | — | — | — |
| C global | 80.8% | 88.99% | 97.1% | 86.47% | 92.75% | 6.28 [4.39, 8.10] |
| D conditional | 80.6% | 88.04% | 95.5% | 88.34% | 87.71% | **−0.62 [−2.78, 1.37]** |

Deployment policy at 85% target: coverage 85.83%, accuracy 86.92%, sensitivity 95.07%,
specificity 78.09%, gap 0.776, AP coverage 77.03%, PA coverage 99.73%.

### Stage 14 significance (`reports/stage14/significance.md`, test n=4,722)

**Family 3 — Contribution 1's cost, formally tested.** A *non-significant* result is the
outcome that supports the claim:

| Comparison | Acc A | Acc B | b | c | Diff % [95% CI] | p (mid-p) | Significant |
|---|---|---|---|---|---|---|---|
| Global vs per-projection threshold (Cardiomegaly) | 83.21 | 83.19 | 24 | 23 | **+0.02 [−0.26, +0.31]** | **0.8854** | **no** |

**Family 2 — classifier vs report generator**, Holm-corrected:

| Pathology | Classifier | Report | Diff % [95% CI] | p (Holm) | Significant |
|---|---|---|---|---|---|
| Cardiomegaly | 83.21 | 80.39 | +2.82 [+1.89, +3.74] | <0.0002 | **YES** |
| Edema | 85.18 | 79.33 | +5.84 [+4.74, +6.95] | <0.0002 | **YES** |
| Pleural_Effusion | 86.13 | 79.99 | +6.14 [+5.04, +7.24] | <0.0002 | **YES** |
| Atelectasis | 70.58 | 75.03 | −4.45 [−5.98, −2.92] | <0.0002 | **YES** |
| Consolidation | 89.31 | 92.44 | −3.13 [−3.98, −2.28] | <0.0002 | **YES** |
| Lung_Opacity | 70.33 | 74.59 | −4.26 [−5.68, −2.83] | <0.0002 | **YES** |
| Pneumonia | 89.39 | 88.88 | +0.51 [−0.32, +1.34] | 0.2318 | no |
| Pneumothorax | 95.28 | 95.89 | −0.61 [−1.12, −0.10] | 0.0367 | **YES** |

**Family 1 — classifier vs always-negative baseline.** All 8 significant after Holm.
Cardiomegaly +33.63 [+31.65, +35.61] (p<0.0002); the model is significantly **worse** on
Atelectasis (−2.79), Consolidation (−5.02), Lung_Opacity (−5.76), Pneumonia (−2.50) and
Pneumothorax (−0.99) — confirming, with formal tests, the caveat already stated in §3.

**Stage 13 deferral — paired bootstrap** (10,000 stratified replicates, matched coverage
85.83% both arms):

| Quantity | Value |
|---|---|
| Statistic | \|gap_global\| − \|gap_conditional\| (percentage points) |
| Observed | **5.8286** |
| 95% bootstrap CI | [3.4559, 6.7908] |
| **p (two-sided)** | **0.0004** |
| Replicates favouring conditional | **99.98%** |

### Stage 15 interval change (`reports/stage15/interval_change.md`)

Consecutive study pairs per patient ordered by **true** `StudyDate`/`StudyTime`, restricted
to pairs where **the radiologist recorded no change**, so any model movement is spurious by
construction. n = 1,666 pairs from 692 patients.

| Transition | n | False worsening | False improvement | Asymmetry [95% CI] | Sig |
|---|---|---|---|---|---|
| AP→AP | 1035 | 3.0% | 3.8% | −0.77 [−2.07, +0.49] | no |
| PA→PA | 245 | 4.1% | 6.1% | −2.04 [−5.58, +1.28] | no |
| **PA→AP** | 207 | **13.5%** | **1.9%** | **+11.59 [+6.37, +16.92]** | **YES** |
| AP→PA | 179 | 3.9% | 9.5% | −5.59 [−11.17, +0.00] | no |

**Four arms:**

| Arm | n | Asymmetry [95% CI] | Sig |
|---|---|---|---|
| A · same projection (control) | 1280 | −1.02 [−2.23, +0.18] | no |
| B · shuffled temporal order (null) | 386 | +1.55 [−2.22, +5.37] | no |
| C · PA→AP, true order (finding) | 207 | **+11.59 [+6.37, +16.92]** | **YES** |
| D · C + per-projection thresholds (the fix) | 207 | **+8.21 [+2.96, +13.62]** | **YES** |

| Test | Difference [95% CI] | p |
|---|---|---|
| Finding vs same-projection control | +12.61 [+7.36, +18.11] | **0.00000** |
| Finding vs shuffled-order null | +10.05 [+3.65, +16.59] | **0.00150** |
| Threshold fix vs uncorrected | −3.40 [−5.94, −1.40] | 0.00150 |

**Stricter condition — all 8 findings recorded unchanged** (n = 397, 271 patients):
PA→AP **+10.91 [+3.64, +20.00]** (10.9% false worsening, **0.0%** false improvement);
vs same-projection control +9.12 [+1.45, +18.02], **p = 0.01850**. ⚠️ Based on 55 PA→AP
pairs — supporting evidence, not the headline.

**Ordering provenance:** `study_id` ordering matches true chronology **49.59%** of the
time (a coin flip); ordering therefore comes from `StudyDate`/`StudyTime`, joined for
**100%** of the 4,722 test images. `selftest` asserts the coin-flip property.

**Interpretation of arm D:** per-projection thresholds reduce the artefact significantly
(−3.40, p = 0.00150) but leave **+8.21, still significantly above zero**. Contribution 1
partially corrects the artefact; it does not eliminate it.

### Test suite
**193 tests, 8 modules, 0 failures** (stage6 38 · stage9 18 · stage9b 28 · stage10 20 ·
stage11 24 · stage13 13 · stage14 33).

---

## 19. Interpretation Notes

**FACT — interpretations already written by the author, quoted with source.**

- On the irreducibility of the gap (`backend/services/thresholds.py` L27-34):
  > "Thresholding does NOT make the model better at AP films. AUROC is computed over the
  > whole ranking; a threshold is one cut through it, and cutting per group cannot reorder
  > any case… The AP/PA gap of 0.0639 is irreducible at the representation level — it
  > reflects genuine information loss at acquisition."

- On why the field's approach is wrong (`backend/services/thresholds.py` L11-15):
  > "Clinical radiology has handled this for decades with a projection-specific decision
  > rule: cardiomegaly is CTR > 0.50 on PA and > 0.55 on AP. The AI fairness literature does
  > the opposite — it tries to make models BLIND to projection."

- On the fitted thresholds matching clinical convention (same file, L22-26): fitted AP 0.409
  / PA 0.348 (ratio 1.18) vs clinical CTR AP 0.55 / PA 0.50 (ratio 1.10) — same direction.

- On the five falsifications (`RESULTS.md` §9):
  > "Across five independent interventions, sophisticated methods matched or lost to trivial
  > baselines… Each was caught by a control we built to falsify our own hypothesis, not by a
  > reviewer."

- On Grad-CAM's limits (`backend/services/gradcam.py` L8-11):
  > "Grad-CAM shows WHERE the model looked, not WHETHER it was right. Arun et al. measured
  > Grad-CAM repeatability at SSIM 0.12 on chest radiographs."

**Requires author interpretation — human judgment call, not extractable from code:**
- Whether the "metric is gameable" claim generalises beyond MIMIC-CXR
- Clinical acceptability of a 23% AP referral rate in a real workflow
- How to position this against Pereira et al. given non-comparable setups
- Whether 80.4% report accuracy is clinically useful as a draft-generation aid

---

## 20. Limitations & Threats to Validity

**FACT — several are already documented in `RESULTS.md` §10 and `PANEL_ANSWERS.md` Q7.**

### Threats to external validity

1. **The external baseline was reimplemented, so the core comparison is sound**
   (`stage9b_gradrev.py`, `RESULTS.md` §8.2). The residual issue is that two documents still
   *present* the weaker cross-dataset framing — a writing fix, not a validity problem.
   Any sentence that places our 73.3% beside their published 46.7% must state that the
   dataset, backbone and split differ, and should defer to §8.2.
2. **Not comparable to published MIMIC-CXR numbers.** Custom split, cleaned references,
   cardiomegaly enriched to 50.4% (vs ~20% natural prevalence).
3. **Single dataset, single centre, retrospective.** No external validation, no prospective
   or reader study.

### Threats to internal validity

4. **Single training run per model** — no seed variance measured.
5. ~~No significance testing~~ — **resolved** (Stage 14): McNemar mid-p with Holm correction, plus a paired bootstrap test for the deferral claim.
6. **Split contamination noted previously:** 98.3% of the test set falls in the official
   MIMIC train split, so head-to-head comparability with published work is void.

### Baked-in assumptions (from code)

7. `backend/services/thresholds.py` assumes projection ∈ {AP, PA}; anything else falls back
   to global. Lateral films are not handled.
8. `stage13_deferral.py` operates on **Cardiomegaly only** (`TARGET = "Cardiomegaly"`).
   **This is a declared scope decision, not an oversight** — cardiomegaly is the
   component's diagnostic target, and `RESULTS.md` §7A.1 documents the methodological
   reason: per-group threshold fitting is unstable at low prevalence, evidenced by
   pneumothorax (3.73%) worsening under it. Extending deferral to rarer labels is future
   work.
9. Deferral cutoffs are frozen from one validation split; no drift monitoring.
10. `lookup_ground_truth` keys on filename stem — an arbitrary upload returns no ground truth
    (handled correctly, but demo-specific).
11. `get_test_sample()` in `inference.py` reads `TEST_IMAGE_DIR` but is wired to **no
    endpoint** — dead code.

### Deployment/generalisation

12. **Accuracy at 80% coverage is not accuracy.** Must always be quoted with coverage and
    referral rate.
13. **Grad-CAM is a sanity check, not localisation evidence** (SSIM 0.12).
14. ~~Unpinned dependencies~~ — **resolved**: all 10 direct dependencies pinned and verified. Transitive dependencies remain unpinned (no lockfile or container).
15. `torch 2.11.0+cpu` locally — the demo runs on CPU; latency is unmeasured.

---

## 21. Ethical & Societal Considerations

- **Data privacy — APPLICABLE.** Uses MIMIC-CXR, de-identified patient radiographs and
  reports under PhysioNet credentialed access + DUA. **Handling visible in code:**
  `review_cases/.gitignore` contains `*`, excluding all patient data from version control;
  `backend/config.py` documents that the dataset is opened READ-ONLY. Report text is
  de-identified upstream by MIMIC (`___` placeholders).
  ⚠️ `NOT FOUND IN CODEBASE` — no explicit LICENSE or DUA file is stored in the component.

- **Potential misuse — APPLICABLE.** An automated cardiomegaly detector and report
  generator could be used for unsupervised diagnosis. Mitigations present in code: the
  reliability notice (`ReliabilityNotice.jsx`), the deferral notice
  (`DeferralNotice.jsx`), and `README.md` carries a `## ⚕️ Disclaimer` section.

- **Fairness/bias — CENTRALLY APPLICABLE.** This is the component's research subject. The
  measured AP/PA disparity systematically disadvantages the **sickest** patients (those too
  ill to stand). The component measures, reports, and partially mitigates it.
  ⚠️ **Only acquisition projection is analysed.** Sex, age, race/ethnicity, and insurance
  status — the axes studied in CheXclusion (ref #8) — are **not evaluated anywhere**. This
  is a notable omission for a fairness contribution.

- **Environmental/compute cost — PARTIALLY APPLICABLE.** GPU training on Colab L4; budget
  tracked in CU. Stage 13 explicitly costs 0 CU by design. Total kWh/CO₂ not estimated —
  `NOT FOUND IN CODEBASE`.

- **Dataset licensing — APPLICABLE and handled.** PhysioNet DUA; `.gitignore` prevents
  redistribution via git. `PROJECT_STRUCTURE.md` warns against sending data to
  non-credentialed recipients.

---

## 22. Reproducibility / How to Run

### Run the demo

```bash
# Backend
cd backend
pip install -r requirements.txt      # ✅ version-pinned; see header for CUDA install
cd ..
python -m uvicorn backend.main:app --port 8000
# or: run_backend.bat

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

### Reproduce the analyses

```bash
python stage13_deferral.py           # Stage 13 deferral analysis (0 CU, seconds, CPU)
python stage13_deferral.py --test    # 13 self-checks

python stage14_significance.py       # Stage 14 significance tests (0 CU, CPU)
python stage14_significance.py --test # 33 self-checks

python stage15_interval.py           # Stage 15 interval-change analysis (0 CU, CPU)
python stage15_interval.py --test    # 19 self-checks

python stage6_acr.py --test          # 38 checks
python stage9_fairness.py --test     # 18 checks
python stage9b_gradrev.py --test     # 28 checks
python stage10_conditional.py --test # 20 checks
python stage11_conditioned.py --test # 24 checks
python stage14_significance.py --test # 33 checks
python stage15_interval.py --test    # 19 checks
```

### API

| Endpoint | Method |
|---|---|
| `/predict` | POST — multipart image + optional `view` (AP/PA) + filename |
| `/health` | GET |
| `/thresholds` | GET |

**Response fields:** `prediction`, `confidence`, `probability`, `gradcam_image` (base64),
`report_text`, `report_text_raw`, `classifier_prompt`, `ground_truth_report`,
`copathologies[]`, `view`, `threshold`, `threshold_source`, `reliability`, `deferral`,
`model_info`.

### Artifact status

| | |
|---|---|
| Self-contained folder | ✅ as of the `ORIGINAL_TEST_CSV` relocation to `review_cases/` |
| Required weights | `checkpoints/stage5/best.pt` (673 MB), `checkpoints/stage11/best.pt` (975 MB) |
| Cached predictions for re-analysis | `reports/stage6/cache/probs_{val,test}.npy` |
| Shareable publicly? | ❌ **No** — contains MIMIC data under DUA. Code could be released; data cannot. |
| Version control | ❌ **Not a git repository** |
| Dependency pinning | ✅ Pinned (direct deps); no lockfile |

---

## 23. My Individual Role / Contribution Statement

**⚠️ `NOT FOUND IN CODEBASE — needs input from author.`**

**Verified fact:** this directory is **not a git repository** (`git rev-parse` fails). There
is no commit history, no authorship metadata, and no `.git` directory. Consequently:

- Commit-level attribution: **impossible to extract**
- Separation of your work from teammates': **impossible to extract**
- File ownership: **impossible to extract**

**What can be said from file contents [inferred]:**
- The component folder is documented throughout as a *separate* deployment that does not
  read or write sibling components (`backend/config.py` L4-5, `backend/main.py` L11-12),
  which is consistent with single-author ownership of this directory.
- `PANEL_ANSWERS.md` and `NOVELTY_EXPLANATION.md` are written in the first person singular
  ("I built", "I tested"), indicating the author claims individual authorship of the
  research contributions.
- Three unrelated student documents sit in the folder (`IT22130020_Raagul G_Project Proposal
  Report.docx/.pdf`, `R26-IT-087_IT22281296_Thishoharini.V.pdf`), indicating a shared group
  context but **not** shared code.

**Action required from author:** write the contribution statement manually, and — strongly
recommended — **initialise git now** so that future work is attributable.

---

## 24. Key Terms / Mini-Glossary

| Term | Definition |
|---|---|
| **AP / PA projection** | How the X-ray was taken. PA = patient standing, beam back-to-front (standard). AP = portable/bedside, used when the patient is too ill to stand. AP magnifies the heart. |
| **Cardiomegaly** | An enlarged heart, as seen on a chest X-ray. |
| **AUROC** | Area under the ROC curve — how well a model ranks sick above healthy, independent of any threshold. |
| **Operating point / threshold** | The probability cut-off above which the model says "disease present". |
| **TPR disparity** | The difference in true-positive rate between two groups; the standard fairness metric here (Hardt et al. 2016). |
| **Selective prediction / deferral** | Letting a model decline to answer uncertain cases and refer them to a human. |
| **Coverage** | The fraction of cases the system actually answers (100% minus the referral rate). |
| **Ablation** | A test where you remove one part of a system to see whether it still works. |
| **Null arm / control** | An intentionally uninformative version of a method, used to check that a real effect exists rather than an artefact. |
| **Platt scaling** | A 1999 method that fits a sigmoid to rescale model scores into calibrated probabilities. |
| **Grad-CAM** | A heat-map showing which image regions most influenced a model's decision. |
| **Clinical-efficacy F1** | F1 measured over findings *extracted from generated report text*, rather than word overlap. |
| **ECE** | Expected Calibration Error — the gap between predicted confidence and observed accuracy. |

---

## 25. Gaps & Open Questions

### 🔴 Blocking — must be resolved before submission

1. **Individual contribution statement (§23).** Not a git repository; attribution cannot be
   extracted. **Author must write this manually.** Recommend `git init` immediately.
2. **Integration with the other three components (§1).** No evidence found of how this
   component connects to the group system. Needs author input.
3. **Pereira framing — writing fix, not a validity problem.** The baseline *was*
   reimplemented (`stage9b_gradrev.py`, `RESULTS.md` §8.2), so the comparison is sound.
   But `RESULTS.md` §8.1 and `PANEL_ANSWERS.md` Q2 still lead with the weaker cross-dataset
   figure. **Action:** make §8.2's same-experiment result the headline everywhere —
   *"on our data their method made disparity 25.4% worse at a cost of 0.0789 AUROC"* —
   and keep the published 46.7% only as labelled context.

### 🟠 Parameter/number conflicts to resolve

4. **Constant-string ROUGE-L baseline: 0.2769 vs 0.2641.** Two different values in
   `Stage4_Report_Generator.ipynb` and `RESULTS.md`/`backend/config.py`. Which reference set
   was each computed on?
5. **Classifier EPOCHS: 30 vs 3** in different cells of `Stage5_Classifier_Training.ipynb`.
   Which produced the shipped checkpoint?

### 🟡 Missing content the paper will need

6. **Research questions (§4)** are inferred, not stated anywhere. Ratify or rewrite.
7. **Hardware specs (§11)** — CPU model, RAM, VRAM not recorded anywhere.
8. **Wall-clock training times (§11)** — not logged. Cannot report compute cost per run.
9. **Consolidated ablation figure (§16)** — does not exist; would be the highest-value
   figure to add.
10. **System architecture diagram (§17)** — does not exist.
11. **ROC curves per projection, calibration diagram (§17)** — do not exist as standalone
    assets.
12. **Energy/CO₂ estimate (§21)** — not computed.
13. **Formal proof (§8)** that per-group thresholding cannot alter AUROC — currently a prose
    argument plus a 1e-12 numerical check. A one-line formal statement would strengthen it.

### 🟢 Known scope limitations to state explicitly, not fix

13b. **Stage 15's strictest arm rests on 55 PA→AP pairs.** The main condition (n=207) is
    well powered; the all-findings-stable replication is not, and its CI for AP→PA touches
    zero. Report as supporting evidence.
13c. **Stage 15 measures a failure mode in a capability the system does not yet have.**
    The backend exposes only `/predict` on a single image, and 0 of 4,722 generated
    reports contain interval-change language. This is a prospective safety finding, not a
    defect in the shipped system.
14. **Stage 13 covers Cardiomegaly only** — not demonstrated for the other 7 pathologies.
15. **Only acquisition projection is studied as a fairness axis** — sex, age, race, and
    insurance are not evaluated, despite CheXclusion (ref #8) being cited.
16. **Single training run and no cross-validation** (§15). Significance testing is now
    present (Stage 14), but *training* variance across seeds remains unmeasured — all
    intervals quantify test-set sampling variance only.
17. **Dependencies pinned** (resolved), but still `NOT FOUND` — no Docker image,
    lockfile, or `environment.yml`, so transitive versions can still drift.
18. **No LICENSE / DUA file** stored in the component.
19. **Dead code:** `get_test_sample()` reads `TEST_IMAGE_DIR` but no endpoint calls it.
20. **Stage 9C (Group-DRO)** appears in `MASTER_PLAN.md` §13 "Next actions" and was never
    run; `MASTER_PLAN.md` §13 is stale (3 of its 5 items are complete) and contains no
    mention of Stage 13.

### Inconsistencies between documents

21. `README.md` and `MASTER_PLAN.md` contain **no mention of Stage 13 / Contribution 3**,
    while `RESULTS.md`, `NOVELTY_EXPLANATION.md`, and `PANEL_ANSWERS.md` do.
22. Contribution numbering differs across documents: `NOVELTY_EXPLANATION.md` calls it
    "Contribution 3"; `RESULTS.md` calls it "§7B". Unify before writing the paper.
