# Component_01 — FINAL ANALYSIS
### MIMIC-CXR Report Generation: root causes, verdicts, and execution plan

*Analysis date: 2026-07-28. All figures marked "measured" were computed directly against the
project's own CSVs, PNGs, model outputs and source files.*

---

## ⚠️ THE HEADLINE NUMBER

A **single fixed sentence emitted for every test image** — ignoring the X-ray completely —
was scored against the project's test set (n=4,786):

| System | ROUGE-L |
|---|---|
| Trained ConvNeXt + BART model | **0.2739** |
| Constant string, image never looked at | **0.2481** |
| One-bit oracle (only knows cardiomegaly yes/no) | **0.2506** |
| **Value added by the entire vision pipeline** | **+0.0258** |
| **Value added over one single bit of information** | **+0.0233** |

The whole vision stack — ConvNeXt-Base, 384×384 images, 46k training pairs, the projection
layer, BART — is worth **2.6 ROUGE-L points over printing the same sentence every time.**

The constant string scoring 0.2481:

> *"No acute cardiopulmonary process. PA and lateral chest radiographs. The lungs are clear.
> There is no pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal."*

Supporting constants measured:

| Constant | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Most frequent training report | 0.3250 | 0.1014 | 0.2300 |
| Model's most-emitted output (32/100) | 0.3073 | 0.1065 | 0.2232 |
| Model's 2nd most-emitted output | 0.2639 | 0.0622 | 0.1761 |
| **Val-optimised constant** | **0.3286** | **0.1300** | **0.2481** |
| One-bit oracle | 0.3383 | 0.1347 | 0.2506 |

### Two consequences

1. **The "ROUGE-L 30%" target is not a meaningful goal.** 24.8% requires no model at all.
   Chasing 30% could be satisfied by a system that never looks at an image. Clinical-efficacy
   F1 must become the primary metric, with ROUGE-L reported alongside.
2. **The architecture rebuild is mandatory, not optional.** The image signal is genuinely not
   reaching the decoder.

---

## ✅ CORRECTION — label noise is NOT the binding constraint

An earlier draft of this analysis claimed label noise caused the weak co-pathology F1.
**That claim is refuted by measurement.**

Method: compute the *ceiling F1* — what a **perfect** model, predicting official-CheXpert
truth exactly, would score against the custom labels the project actually trained and tested
on. If label noise were binding, that ceiling would sit near the model's F1. It does not.

| Pathology | Prev% | **Ceiling F1** | Model F1 | Gap | Real constraint |
|---|---|---|---|---|---|
| Cardiomegaly | 50.0 | 0.974 | 0.856 | 0.117 | model has room |
| Edema | 22.7 | 0.953 | 0.670 | 0.283 | model has room |
| Pleural_Effusion | 31.3 | 0.965 | 0.772 | 0.193 | model has room |
| Atelectasis | 23.5 | 0.974 | 0.541 | **0.432** | model has room |
| Lung_Opacity | 23.2 | 0.941 | 0.471 | **0.470** | model has room |
| Pneumonia | 8.0 | 0.816 | 0.298 | **0.518** | model has room |
| Pneumothorax | 3.4 | 0.847 | 0.351 | **0.496** | prevalence |
| Consolidation | 5.9 | 0.905 | 0.231 | **0.674** | prevalence |

Label noise caps performance at 0.82–0.97. The model sits at 0.23–0.86.

Additionally confirmed: these F1 values were computed **at optimal thresholds**
(`training/god_tier_model_audit.py:309`, thresholds 0.10–0.37), not at 0.5. Threshold tuning
is already spent — no easy win remains there.

### Real causes of weak co-pathology detection

1. **No `pos_weight` in `BCEWithLogitsLoss`.** `LABEL_IMPORTANCE = [3.0,1.5,1.5,1,1,1,1,1]`
   scales per-*label* loss; it does nothing about positive/negative imbalance *within* a
   label. At 3.7% prevalence the model is rewarded for always predicting "no".
2. **Too few positives.** Train counts: Pneumothorax 1,363 · Consolidation 2,077 ·
   Pneumonia 2,711.
3. **The 1:1 cardiomegaly balancing discarded the corpus.** 46,274 of 227,827 available
   studies were used (**20%**) to force cardiomegaly to 50%. Every co-pathology paid for it.

**Revised recommendation:** switching to official CheXpert labels is still worth doing — it is
essentially free, it is the field standard, and it makes results comparable to published work.
But it is **not** the co-pathology fix. The fix is **more data + `pos_weight`**.

*Nuance:* 79–98.5% of custom *negatives* are "not mentioned" in official CheXpert rather than
explicitly denied. That is the standard blank-as-negative convention and is defensible, but it
means the negative class is mostly silence, not denial.

---

## 🔴 ROOT CAUSE #1 — The hallucination (data, not model)

Measured across all 46,274 training targets:

| Un-groundable language | % of reports |
|---|---|
| "unchanged / no change / stable / constant" | 47.5% |
| `___` de-identification placeholder | 27.6% |
| "prior / previous / earlier study" | 21.3% |
| "as compared to / in comparison with" | 19.6% |
| "again seen / persistent / remains" | 19.3% |
| "interval change / new since" | 14.9% |
| **ANY prior-reference language** | **69.7%** |
| **Sentences impossible to ground in one image** | **24.7%** |

The model is not malfunctioning. It was taught that 7 out of 10 correct reports mention a
previous study. It has never seen one, so it invents one — and it **amplifies**: 63% of
generations contain prior-language versus 51% of references, because that phrasing is the
safest high-frequency filler.

**No architecture change fixes this. Only cleaning the targets fixes this.**

---

## 🔴 ROOT CAUSE #2 — Mode collapse (architecture)

Measured from `models/report_generator/sample_reports_100.txt`:

| Measure | Value |
|---|---|
| Unique generated reports | 55 / 100 |
| **Unique first sentences** | **14 / 100** |
| Single output emitted verbatim | **32 / 100** |
| Vocabulary | 248 words vs **1,046** in references (ratio **0.24**) |
| Length | 32 words vs 55 |
| Unique *reference* reports | 98 / 100 — the data is fine |

### Ranked defects

**① BART's encoder is bypassed — CRITICAL.**
`training/train_report_generator.py:172` passes `encoder_outputs=BaseModelOutput(...)`.
BART's pretrained encoder is instantiated, never used, never trained. The decoder's
cross-attention was pretrained expecting encoder outputs with a specific scale, LayerNorm
geometry and positional structure; it receives raw projected ConvNeXt features instead and
learns to ignore them. **This single line explains most of the collapse — and is why the
constant baseline is only 0.026 behind.**

**② No positional encoding on the 144 visual tokens.**
The 12×12 grid is flattened with zero position information. The decoder cannot distinguish
apex from base, left from right — on a task that is entirely about *where*.

**③ One `Linear(1024→768)` carries the whole image→language burden** (~800K params) on a
frozen trunk.

**④ The frozen trunk was supervised through global average pooling** — its 12×12 map was
never rewarded for spatial discrimination, only for its average.

**⑤ Beam search amplifies collapse.** `num_beams=4, early_stopping=True,
length_penalty=1.0` on a weakly-conditioned model converges to the corpus-modal string.
Generated length 32 vs reference 55 is the signature.

**⑥ Checkpoint selection is noise** — best model chosen on ROUGE-L over **50 greedy
samples** (SE ≈ ±0.02–0.03, larger than epoch-to-epoch differences).

**⑦ `MAX_LEN=512, padding="max_length"`** but reports are median 50 words → ~90% of every
label tensor is padding.

---

## 📋 DIRECT ANSWERS

### "Were the positive and negative labels correct?"

**For cardiomegaly — yes.** 95.6% agreement with official MIMIC-CXR CheXpert. The primary
task's labels are sound.

Agreement vs official CheXpert (definite 0/1 rows only):

| Pathology | Agreement | False-neg |
|---|---|---|
| Cardiomegaly | 95.6% | 729 |
| Pleural_Effusion | 94.8% | 451 |
| Pneumothorax | 94.8% | 425 |
| Edema | 93.3% | 639 |
| Consolidation | 85.9% | 190 |
| Lung_Opacity | 84.7% | 842 |
| Atelectasis | 80.2% | 265 |
| Pneumonia | 79.3% | 686 |
| No_Finding | 71.8% | 3,825 |

**Real bug found** — `scripts/multi_label/layer1_keywords.py:102-110`.
`_check_explicit_negation()` scans the 50 characters before the *first* keyword hit and flags
negation if `"no "` appears anywhere in that window. In radiology prose "no" is everywhere, so
negation bleeds across sentence boundaries:

> *"No pleural effusion. Cardiomegaly is present."* → **Cardiomegaly marked ABSENT**

Confirmed instances exist in the data (*"Enlargement of the cardiac silhouette without
pulmonary edema"* → labeled 0 at `conf=1.00`). Both `_check_uncertainty` and
`_check_explicit_negation` also use `text.find()` — **only the first occurrence** — so a
finding negated in the impression and asserted in the findings is only half-seen.

Per the ceiling analysis: **this bug is real and worth fixing, and it is not what caps F1.**

### "Do I need to do preprocessing again?"

#### Images: NO. Do not re-download.

The anisotropic `resize((384,384))` at
`scripts/stage6_image_linking/download_cardio_384.py:129` does **not** corrupt cardiomegaly.
CTR is *cardiac width ÷ thoracic width* — both horizontal, so horizontal scaling cancels in
the ratio. The primary target is geometrically safe. Not worth 46,000 re-downloads.

Fix with transform changes only — zero re-download:

- **Per-image intensity normalization or CLAHE.** Measured per-image mean drifts
  **97.8 → 126.2**, std 74.5 → 84.1, entirely uncorrected. Highest-value free change.
- **Delete `ColorJitter` and `RandomAutocontrast`.** For CXR, intensity *is* the signal for
  edema and opacity — these augment away the needed feature.
- **Rotation 10° → 5°.** 10° perturbs the cardiothoracic geometry being measured.
- **Fix train/inference skew.** Training has no `Resize`;
  `backend/services/inference.py:43` does. Uploaded images take a different path than
  training images.

#### Report text: YES — mandatory, full rebuild.

| Stage | Action |
|---|---|
| 1 | Drop sentences matching prior-comparison patterns |
| 2 | Drop sentences containing `___` |
| 3 | **Rewrite** temporal→static rather than delete: *"Moderate cardiomegaly persists"* → *"Moderate cardiomegaly"*. Deleting loses real findings. |
| 4 | Strip recommendations/communication (*"Dr. ___ was paged"*, *"recommend CT"*) |
| 5 | Rebuild as `FINDINGS: … IMPRESSION: …` — **correct order** |
| 6 | Drop rows under 8 words after cleaning |
| 7 | **QA gate:** re-run detection regex. Target <2% residual. |

Current target construction is `impression + " " + findings`
(`scripts/build_cardiomegaly_dataset.py:207`) — **inverted** versus real radiology order —
and only 57.9% of rows have both sections (31.5% impression-only, 10.7% findings-only), so
the target format is inconsistent.

Apply identically to **train, val and test**. A cleaned model scored against uncleaned
references is meaningless.

Also fix: **5.31% of test reports appear verbatim in train.**

---

## 🛠 EXECUTION PLAN (Google Colab)

| # | Stage | Cost | Impact | Status |
|---|---|---|---|---|
| **1** | Clean report targets (7-stage spec) | ~2 h CPU | **Kills the hallucination** | **MANDATORY** |
| **2** | Fix transforms (per-image norm, drop ColorJitter, 5°, skew) | ~30 min | Free AUROC | **MANDATORY** |
| **3** | Swap to official CheXpert labels | ~1 h CPU | Standard, comparable | Recommended |
| **4** | **Rebuild report model** | 6–10 h GPU | **The ROUGE fix** | **MANDATORY** |
| **5** | Drop 1:1 balancing, retrain classifier on full MIMIC + `pos_weight` | 8–12 h GPU | **The co-pathology fix** | High value |
| **6** | Classifier-conditioned generation | +2 h | Co-pathology grounding | High value |

### Stage 4 spec

- Stop replacing `encoder_outputs`; prepend projected visual tokens as *inputs* so BART's
  pretrained encoder processes them.
- Add learned 2D positional embeddings to the 12×12 grid.
- Replace the single Linear with a small cross-attention resampler (~32 learned queries).
- Unfreeze the last ConvNeXt stage at low LR.
- Swap `facebook/bart-base` → **BioBART-base**.
- Decoding: drop `early_stopping`, `length_penalty ≈ 1.2–1.5`, `min_length ≈ 40`,
  `max_length = 200`.
- Validate on **≥500 samples**, not 50.

### Stage 6

Highest impact-per-hour item and the direct route to co-pathology explanations: feed the
classifier's predicted labels as a text prefix —
*"Cardiomegaly: present. Edema: mild. Pleural effusion: absent. →"* — so generation is
grounded in something the model gets right at AUROC 0.92.

### If only 15 GPU-hours are available

Stages 1–3 are CPU-only — do them regardless. Then spend the entire GPU budget on stage 4.
Stages 1–2 alone remove the hallucination even on the current architecture.

### Logistics

- Do not put 46k loose PNGs on Google Drive — small-file I/O will dominate runtime. Tar into
  shards, extract to Colab local disk per session.
- Report-gen checkpoints are 1.69 GB because they redundantly store frozen vision weights.
  Save trainable params only.

---

## 📊 EXPECTED RESULTS — honest

| Metric | Now | After 1–3 | After 4–6 |
|---|---|---|---|
| Prior-study hallucination | 63% of outputs | **<3%** | <3% |
| Unique first sentences /100 | 14 | ~25 | 50–70 |
| ROUGE-L | 0.274 | 0.24–0.28 ¹ | 0.29–0.33 ² |
| **Margin over constant baseline** | **+0.026** | — | **+0.05–0.09** |
| Consolidation / Pneumonia F1 | 0.23 / 0.30 | — | 0.40–0.55 |

¹ **Expect ROUGE-L to move sideways or dip after cleaning — that is success.** Text the model
was scoring "for free" is being deleted. Judge stage 1 by hallucination rate, not ROUGE.
² Estimate, not measured.

**Track margin-over-constant-baseline (0.2481) as the real metric.** Absolute ROUGE-L hides
everything that is wrong here.

---

## Evaluation protocol

- Primary metric: **clinical-efficacy F1** (CheXbert/CheXpert labels applied to generated vs
  reference reports, micro and macro).
- Secondary: ROUGE-L, BLEU-4, **margin over the 0.2481 constant baseline**.
- Hallucination metric: prior-reference rate over generated text, using the same regex family
  as the cleaning QA gate. Target <3%.
- Always report the constant-baseline control alongside any ROUGE figure.
- Validate report generation on ≥500 samples with bootstrapped CIs; 50 is noise.
- Classifier: select checkpoints on mean AUROC, not Cardiomegaly-only. Report at true
  prevalence, not the 50/50 balanced set.

---

## Confidence

**Measured on project data — trust fully:** the constant baseline (0.2481), the ceiling-F1
table, the 69.7% / 24.7% hallucination rates, all label-agreement figures, the mode-collapse
statistics, image intensity drift, and every code defect cited by line number.

**Engineering judgment, not independently verified:** the published SOTA range, the post-fix
projection table, Colab time estimates, and the specific BioBART/resampler choices — standard
and sound, but not benchmarked on this setup.

**Corrected during analysis:** label noise is *not* the binding constraint on co-pathology F1
(ceiling 0.82–0.97 vs model 0.23–0.86). The cause is imbalance and data volume. Priority moved
from "fix labels" to "more data + `pos_weight`".

---

## Verified environment facts

- 46,274 images: train 36,938 / val 4,550 / test 4,786. Exactly 1 image per study.
- Frontal only: AP 28,230, PA 18,044.
- **Zero patient and study leakage across splits** (verified).
- Report lengths: mean 56 words, median 50, p99 149, max 329 — versus `MAX_LEN = 512`.
- Official `data/raw/mimic-cxr-2.0.0-chexpert.csv` joins **100%** on `study_id` and is already
  on disk.
- Classifier test (optimal thresholds): Cardiomegaly AUROC 0.9235 · Edema 0.8921 ·
  Pleural_Effusion 0.9153 · Pneumothorax 0.8696 · Atelectasis 0.7800 · Consolidation 0.7498 ·
  Pneumonia 0.7458 · Lung_Opacity 0.7247 · MEAN AUROC 0.8251.
