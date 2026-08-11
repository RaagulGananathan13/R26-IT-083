# Explainable AI System for Cardiovascular Disease Detection and Diagnosis

### Component: Cardiomegaly Detection with XAI and Automatic Report Generation

**Name:** Raagul Gananathan
**IT Number:** IT22130020

---

## Project Description

This component takes a chest X-ray and does three things with it: predicts whether the patient has cardiomegaly (an enlarged heart) along with seven other chest pathologies, shows you *where* on the image the model was looking when it made that call, and writes a draft radiology report in plain clinical language.

The reason for all three is that a prediction on its own isn't much use to a radiologist. A number like `Cardiomegaly: 0.94` tells you nothing about whether the model looked at the heart or at a piece of tubing in the corner of the film. So every prediction comes with a Grad-CAM heatmap and a written report, and — this is the part I ended up spending most of the project on — the system is honest about when it shouldn't be trusted.

I should say up front what this is and isn't. It's a decision-support prototype. It is not a diagnostic tool, it has not been clinically validated, and every output needs a radiologist to check it. Throughout this README I've tried to report what I actually measured rather than what I hoped for.

---

## Objectives

1. **Detect cardiomegaly** from a single frontal chest X-ray, with a confidence score rather than a bare yes/no.
2. **Detect seven co-occurring pathologies** in the same pass — edema, pleural effusion, atelectasis, consolidation, lung opacity, pneumonia, pneumothorax — because these rarely appear alone and a cardiomegaly-only model gives a misleading picture.
3. **Make the prediction explainable** with Grad-CAM, so a clinician can see whether the model attended to the cardiac silhouette or to something irrelevant.
4. **Generate a readable draft report** that reflects the actual image rather than reciting a generic template.
5. **Audit the system for fairness** across how the X-ray was taken. This started as a sanity check and turned into the main research contribution.

---

## Technologies Used

### AI and Deep Learning
| | |
|---|---|
| Vision backbone | ConvNeXt-Base (ImageNet-pretrained), 384×384 input |
| Text decoder | BioBART-v2-base (`GanjinZero/biobart-v2-base`) |
| Explainability | Grad-CAM on the final convolutional stage |
| Framework | PyTorch 2.x, torchvision, Hugging Face Transformers |
| Metrics | ROUGE, scikit-learn, custom bootstrap implementations |

### Backend
| | |
|---|---|
| API | FastAPI (Python) |
| Frontend | React |
| Image handling | Pillow, OpenCV |

### Training
| | |
|---|---|
| Hardware | Google Colab, NVIDIA L4 (24 GB) |
| Precision | bfloat16 mixed precision, `channels_last` memory format |
| Optimiser | AdamW, cosine schedule with warmup |
| Stability | EMA weight averaging, gradient clipping, non-finite-loss abort |
| Reliability | Resumable Drive checkpointing, automatic batch-size finder |

### Dataset
| | |
|---|---|
| Source | MIMIC-CXR / MIMIC-CXR-JPG (PhysioNet, credentialed) |
| Split | Patient-disjoint — 36,362 train / 4,474 val / **4,722 test** |
| Views | Frontal only (AP and PA) |
| Labels | 8 pathologies, text-adjudicated fusion of CheXpert labels + report text |

> ⚠️ MIMIC-CXR is credentialed data under a PhysioNet Data Use Agreement. No images, reports, or derived CSVs are in this repository — only code that regenerates them.

---

## How It Works

Two trained models share a single vision backbone.

```mermaid
flowchart TD
    A["Chest X-ray upload"] --> B["Preprocessing<br/>384x384 · grayscale · per-image z-score"]
    B --> C["ConvNeXt-Base backbone"]
    C --> D["Model 1<br/>Multi-label classifier"]
    C --> E["Model 2<br/>BioBART report generator"]
    D --> F["8 pathology probabilities"]
    D --> G["Grad-CAM heatmap"]
    E --> H["Draft radiology report"]
    F --> I["Per-projection thresholds<br/>AP vs PA"]
    I --> J["Final output + reliability flag"]
    G --> J
    H --> J
```

### Model 1 — Image Classifier

A 384×384 chest X-ray goes through ConvNeXt-Base and comes out as eight independent probabilities. The head is a small MLP:

```
pooled features (1024)
   -> LayerNorm
   -> Dropout(0.3)
   -> Linear(1024 -> 512)
   -> GELU
   -> Dropout(0.198)
   -> Linear(512 -> 8)
```

Two things mattered more than I expected.

**Per-image z-score normalisation instead of ImageNet statistics.** ImageNet constants are written for natural RGB photographs. Chest X-rays are grayscale with a completely different intensity distribution, and using ImageNet normalisation made the variance across images **4.4× worse** than using the raw pixels. Normalising each image by its own mean and standard deviation fixed it.

**Class weighting.** Pneumothorax appears in about 4% of the training set. Without `pos_weight` the model is rewarded for simply never predicting it — and that is exactly what the first version learned to do.

**Grad-CAM** is computed on `features[7]`, the last convolutional stage. The gradient of the cardiomegaly logit is backpropagated to that layer, producing a 12×12 map of which regions drove the prediction. That map is upsampled and overlaid on the original X-ray.

### Model 2 — Report Generator

```mermaid
flowchart LR
    A["X-ray"] --> B["ConvNeXt<br/>shared weights"]
    B --> C["12x12x1024<br/>feature map"]
    C --> D["Flatten to<br/>144 tokens"]
    D --> E["Projection MLP<br/>1024 -> 768"]
    E --> F["BART ENCODER<br/>adds positions"]
    F --> G["BART decoder<br/>greedy"]
    G --> H["FINDINGS: ...<br/>IMPRESSION: ..."]
```

The 12×12 spatial map is flattened into **144 visual tokens** and projected from 1024 to 768 dimensions through a two-layer MLP:

```
LayerNorm(1024) -> Linear(1024->768) -> GELU -> Dropout -> Linear(768->768) -> LayerNorm
```

Those tokens go into BART as **`inputs_embeds`**, and that detail matters more than anything else in this model.

An earlier version passed them as `encoder_outputs` instead, which skips BART's pretrained encoder entirely. The decoder's cross-attention was pretrained to read encoder *outputs* with a specific scale and structure — hand it raw projected convolutional features and it simply can't use them, so it falls back on its language prior. The model was writing fluent, plausible reports from memory without meaningfully looking at the X-ray. Switching to `inputs_embeds` means BART's encoder actually runs and adds its own learned positional embeddings, so the decoder can tell the apex from the base.

The last ConvNeXt stage is **unfrozen at 0.1× the base learning rate**; earlier stages stay frozen. Freezing the whole trunk left too little capacity to adapt, and unfreezing all of it destroyed the features the classifier depends on.

Decoding is **greedy** (`num_beams=1`), `min_length=24`, `max_length=192`. I originally used beam search with 4 beams because that's the conventional choice. An ablation across six strategies showed greedy beat beam-4 on five of seven metrics, so I switched.

### Cleaning the reports — at the source, not the output

MIMIC-CXR reports are dictated in a workflow where the radiologist can see the patient's previous scans. So the text is full of *"compared to the prior study,"* *"unchanged from the previous exam,"* *"findings discussed with Dr. ___ at 3:45 PM."*

A model trained on that learns to say those things — about patients it has never seen, referencing scans that don't exist. **70.70%** of the raw training targets contained this language.

My first instinct was to strip it out of the generated text with regex. That works, but it papers over the problem: the model still spends capacity learning to produce text that gets deleted afterwards. Instead I clean the **training targets**, so the pattern is never learned. That's Stage 1 — 67 unit tests, 98.45% of the corpus retained, and fabricated prior-study references in generated reports now sit at exactly **0.0000** across all 4,722 test images.

### The shared backbone

Model 2's vision encoder is loaded from Model 1's trained checkpoint rather than trained from scratch. Both models therefore see identical visual features. If the classifier says cardiomegaly is present, the report generator is reasoning from the same evidence — they can't disagree because of a representation mismatch.

---

## Features

- ✅ **Cardiomegaly detection** — AUROC **0.9189**
- ✅ **7 co-pathologies** in the same forward pass
- ✅ **Grad-CAM heatmaps** per pathology
- ✅ **Report generation** — clinical F1 **0.5937**, zero fabricated prior-study references
- ✅ **Acquisition fairness audit** — performance reported separately for AP and PA films
- ✅ **Per-projection operating points** — 73.3% less subgroup disparity at no accuracy cost
- ✅ **104 unit tests** across four modules, including controls built to falsify my own methods
- ✅ **Reproducible** — every stage is a self-contained notebook with a smoke-test mode

---

## Project Structure

```
Component_01/
│
├── cxr_transforms.py                       # preprocessing, single source of truth
├── chexpert_fusion.py                      # text-adjudicated label fusion
│
├── Stage1_Report_Target_Cleaning.ipynb     # strip prior-study language
├── Stage2_Image_Transforms.ipynb           # normalisation experiments
├── Stage4_Report_Generator.ipynb           # BioBART training
├── Stage4B_Decoding_Ablation.ipynb         # greedy vs beam search
├── Stage5_Classifier_Training.ipynb        # ConvNeXt training
│
├── stage6_acr.py                           # acquisition reliability  (38 tests)
├── Stage6_Acquisition_Conditioned_Reliability.ipynb
├── Stage6B_Validation.ipynb                # four-arm ablation
│
├── stage9_fairness.py                      # operating-point analysis (18 tests)
├── Stage9A_Operating_Point_Fairness.ipynb
│
├── stage9b_gradrev.py                      # gradient reversal        (28 tests)
├── Stage9B_Gradient_Reversal.ipynb
├── Stage9B_Lambda_Calibration.ipynb
│
├── stage10_conditional.py                  # conditional heads        (20 tests)
├── Stage10A_Feature_Probe.ipynb
│
├── MASTER_PLAN.md                          # full research record
├── RESULTS.md                              # results section
└── README.md
```

Data folders (`stage1_clean/`, `stage3_labels/`, `training_manifest/`) and `checkpoints/` are gitignored — they hold MIMIC-CXR derivatives and model weights.

---

## Model Performance

### Classifier — test set, n = 4,722

Thresholds fitted on validation, applied to test.

| Pathology | Prev. | **AUROC** | Acc. | **Sens.** | Spec. | PPV |
|---|---|---|---|---|---|---|
| **Cardiomegaly** | 50.4% | **0.9189** | 83.2% | **92.3%** | 74.0% | 78.3% |
| Pleural Effusion | 31.1% | 0.9289 | 86.1% | 81.2% | 88.4% | 75.9% |
| Pneumothorax | 3.7% | 0.9141 | 95.3% | 54.0% | 96.9% | 40.1% |
| Edema | 22.6% | 0.9132 | 85.2% | 75.9% | 87.8% | 64.6% |
| Consolidation | 5.7% | 0.8167 | 89.3% | 38.4% | 92.4% | 23.3% |
| Atelectasis | 26.6% | 0.8096 | 70.5% | 79.3% | 67.3% | 46.8% |
| Pneumonia | 8.1% | 0.7959 | 89.4% | 31.1% | 94.5% | 33.4% |
| Lung Opacity | 23.9% | 0.7462 | 70.3% | 61.6% | 73.1% | 41.8% |
| **Mean** | | **0.8554** | 83.7% | 64.2% | 84.3% | 50.5% |

Up from **0.8251** on the previous version.

#### Cardiomegaly in plain numbers

```
correct       : 3,929 of 4,722   (83.2%)
missed cases  :   184
false alarms  :   609

of 2,381 patients WITH cardiomegaly    -> caught  2,197  (92.3%)
of 2,341 patients WITHOUT cardiomegaly -> cleared 1,732  (74.0%)

AUROC 0.9189   95% CI [0.9112, 0.9265]
```

92.3% sensitivity means it misses roughly 1 case in 13. For a triage tool that's the
right trade — it over-calls rather than missing disease.

#### ⚠️ Why I don't lead with accuracy

Accuracy of a model that simply always says "no disease":

| Pathology | Model | Always "no" | Gain |
|---|---|---|---|
| **Cardiomegaly** | 83.2% | 49.6% | **+33.6** |
| Pleural Effusion | 86.1% | 68.9% | **+17.2** |
| Edema | 85.2% | 77.4% | **+7.8** |
| Pneumothorax | 95.3% | 96.3% | −1.0 |
| Pneumonia | 89.4% | 91.9% | −2.5 |
| Atelectasis | 70.5% | 73.4% | −2.9 |
| Consolidation | 89.3% | 94.3% | −5.0 |
| Lung Opacity | 70.3% | 76.1% | −5.8 |

**Five of eight pathologies lose to doing nothing, on accuracy.** That's an artefact of
F1-optimal thresholds on rare diseases — the model deliberately over-calls to catch
cases. AUROC shows discrimination is genuine. So I report **AUROC and sensitivity**, and
quote accuracy only next to its baseline.

```
all 8 labels correct on one X-ray : 34.9%  (1,649 of 4,722)
average labels correct per X-ray  : 6.69 of 8
```

---

### The report generator, expressed as accuracy

The metrics further down (ROUGE-L, clinical F1) are the standard ones, but they're hard to
read without background. So here's the plain question: **of 4,722 test X-rays, how often did
the generated report reach the same conclusion as the radiologist?**

| | **Classifier** | **Report generator** |
|---|---|---|
| **Cardiomegaly accuracy** | **83.2%** (3,929/4,722) | **80.4%** (3,796/4,722) |
| Sensitivity | 92.3% | 88.8% |
| Specificity | 74.0% | 71.8% |
| **Mean accuracy, 8 pathologies** | **83.7%** | **83.3%** |

The report generator is **2.8 points** behind the classifier on cardiomegaly, and within
**0.4 points** averaged over all eight — despite a much harder job. The classifier emits a
number; the report generator has to write English that happens to contain the right
clinical conclusion. Labels are extracted from the generated text with my own extractor,
which I validated against CheXbert to within **0.002** micro-F1.

Both models favour sensitivity over specificity (92.3%/88.8% vs 74.0%/71.8%). That's
deliberate — the thresholds were fitted for F1 on a screening task, where a missed
cardiomegaly costs more than a false alarm that gets reviewed.

> ⚠️ Same caveat as above: this test set is enriched to 50.4% cardiomegaly, so accuracy is
> meaningful **for cardiomegaly** and misleading for the rarer findings. Read it next to the
> "always say no" table, never on its own.

---

### ⭐ ROUGE-L doesn't measure what people think it measures

Before the report numbers, here's a result worth its own section. I scored three
degenerate "reports" against all 4,722 references using the identical pipeline that
scores my model:

| "Report" | **ROUGE-L** | **Clinical F1** |
|---|---|---|
| **A constant string, identical for every patient** | **0.2641** | **0.0000** |
| A random real report from a *different* patient | 0.1821 | 0.3120 |
| The patient's own real report | 1.0000 | 1.0000 |
| my model (Stage 11) | 0.2896 | 0.5937 |

Two things fall out of this:

**A clinically worthless report scores 91% of my model's ROUGE-L.** One fixed paragraph,
the same for all 4,722 patients, reaches 0.2641 — while identifying **exactly zero**
findings correctly.

**ROUGE-L and clinical F1 rank in opposite directions.** The constant string beats a real
report from the wrong patient on ROUGE-L (0.2641 vs 0.1821) but loses catastrophically on
clinical F1 (0.0000 vs 0.3120). ROUGE-L prefers generic radiology English over a genuine
report; clinical F1 correctly prefers the opposite.

Radiology reports are so templated — *"the lungs are clear"*, *"no pleural effusion"*,
*"cardiomediastinal silhouette"* — that you score well just by writing plausible
radiology English. This matches
[PLOS One 2021](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259639),
which found encoder-decoder report generators do no better than unconditioned baselines.

**So clinical-efficacy F1 is my primary report metric. ROUGE-L is only ever reported
next to its constant-string control.**

---

### Report Generator — test set, n = 4,722

| Metric | Stage 4 | **Stage 11 (shipped)** | Change |
|---|---|---|---|
| **CheXbert micro-F1 (14)** | 0.5783 | **0.5939** | **+0.0157** |
| CheXbert micro-F1 (5) | 0.6580 | **0.6700** | +0.0120 |
| Clinical F1 *(internal extractor)* | 0.5799 | 0.5937 | +0.0138 |
| ROUGE-L | 0.2918 | 0.2896 | −0.0022 |
| Constant-string control | 0.2641 | 0.2641 | — |
| **Fabricated prior-study references** | 0.0000 | **0.0000** | ✅ held |
| Mean words | 36.6 | 39.4 | +2.8 *(reference: 46.9)* |

**Scored with CheXbert**, the labeller the field actually uses — so the clinical number is
measured the standard way. It also validated my own extractor: the two agree to within
**0.002** on both models, which means every clinical figure elsewhere in this project
holds up.

**Cardiomegaly is the strongest of all 14 CheXbert findings — F1 0.8287** (precision
0.7886, recall 0.8732). Fracture, Pleural Other and Lung Lesion score near zero because
my classifier covers only 8 pathologies and those aren't among them.

The gain mechanism is visible in the split: recall **+0.0441**, precision −0.0204. Stage
11 writes longer reports, mentions more findings, catches more and over-calls slightly
more.

I ship Stage 11: it trades 0.0022 of a metric a worthless constant string nearly matches,
for 0.0138 of the metric that measures whether the report states the right findings.
*(The ROUGE-L difference is about one standard error at n=4,722 and isn't established as
significant.)*

#### What actually caused the gain — the ablation

Stage 11 conditions the decoder on the classifier's predictions via a text prompt. I
tested whether that was really responsible, by evaluating the same weights twice:

| Configuration | Clinical F1 |
|---|---|
| Stage 4 (no fine-tuning) | 0.5799 |
| Stage 11, **prompt removed** | 0.5913 |
| Stage 11, **prompt present** | 0.5937 |

```
gain from extra fine-tuning : +0.0114   (83%)
gain from the PROMPT        : +0.0023   (17%, within noise)
```

Inverting the prompt changed only **18 of 96** reports. **Classifier conditioning did not
work.** The improvement is real, but it comes from additional fine-tuning — and I report
it that way.

---

### ⚠️ Comparability

These numbers are **not** directly comparable to published MIMIC-CXR results, for three
reasons I checked rather than assumed:

1. **The split is mine, not the official one.** Patient-disjoint — zero subject, study or
   image overlap, verified — but 98.3% of my test images are officially *training* data.
   Re-running on the official split isn't possible: only 155 official-test films are both
   available to me and free of my training patients.
2. **My reference text is cleaned.** Stage 1 removes prior-study language, shifting the
   reference distribution. Cleaning alone moved the constant baseline from 0.2481 to 0.2769.
3. **Different filtering** — frontal-only, cardiomegaly-enriched.

**CheXbert doesn't fix this.** It removed the *labeller* mismatch, not the *data*
mismatch. Published MIMIC-CXR work reports micro-F1-14 around 0.47; I get 0.5939 — but
that is **not** a win. My test set is cardiomegaly-enriched at 50.4% prevalence, and
cardiomegaly is my best finding (F1 0.8287). Over-representing your strongest class
inflates a micro-average. I report the number; I don't claim the comparison.

---

## Research Contribution — Acquisition Fairness

While auditing the classifier I found something I wasn't looking for.

Chest X-rays come in two projections. **PA** is the standard: the patient stands, the beam travels back-to-front. **AP** is used when the patient is too ill to stand, taken portably at the bedside. AP films magnify the heart, the scapulae overlie the lung fields, and image quality is lower.

The classifier is significantly worse on AP films:

| | AUROC |
|---|---|
| PA (standing) | 0.8864 |
| AP (bedside) | 0.8224 |
| **Gap** | **0.0639**, 95% CI [0.0491, 0.0790] |

Significant for 7 of 8 pathologies, same direction for all 8. **And AP films come from the sickest patients** — so the model is weakest exactly where it matters most. Pooled evaluation cannot see this at all.

I then tested three ways of fixing it.

```mermaid
flowchart TD
    A["AP/PA gap = 0.0639"] --> B["Adjust thresholds<br/>per projection"]
    A --> C["Adversarial invariance<br/>remove projection info"]
    A --> D["Conditional heads<br/>exploit projection info"]
    B --> E["Gap unchanged<br/>proven to 1e-12"]
    C --> F["Gap -13.3%<br/>cost -0.0789 AUROC"]
    D --> G["Gap +0.0003<br/>no benefit"]
    E --> H["The gap is irreducible<br/>at the model level"]
    F --> H
    G --> H
```

**Finding 1 — the standard fairness metric can be gamed.** TPR Disparity, the metric the literature uses for this problem, is defined at a single decision threshold. AUROC is computed over the whole *ranking* of scores, and a threshold is just one cut through that ranking — so changing it per group **cannot reorder any case**. I used that to cut TPR Disparity by **73.3% at exactly zero accuracy cost**, while the real discrimination gap stayed identical to **1e-12**.

For comparison, [Pereira et al. (MIDL 2023)](https://proceedings.mlr.press/v227/pereira24a.html) reduced the same metric by 46.7% using adversarial training that required full retraining and cost 0.91 AUC points.

**Finding 2 — removing projection information doesn't help.** I reimplemented their gradient-reversal method on my own data and drove projection AUC to **0.5000** — complete invariance, better than the 0.61 they report. The gap moved only 0.0639 → 0.0554, at a cost of 0.0789 AUROC.

**Finding 3 — adding projection information doesn't help either.** The opposite strategy, giving the model projection-specific capacity, gained **+0.0003**. Nothing.

**Conclusion:** the AP/PA gap is irreducible at the representation level. It isn't a shortcut the model learned, and it isn't a metric artifact — AP images genuinely carry less usable information. It can't be engineered away downstream. It has to be addressed at acquisition, or by flagging low-reliability reads for human review instead of pretending parity exists.

---

## What Didn't Work

Including this because the failures took as long as the successes and shaped the final design.

| Idea | Outcome |
|---|---|
| **Acquisition-Conditioned Reliability** | Looked like a 90% fairness improvement. A proper four-arm control showed it was plain Platt scaling from 1999. Dropped. |
| **Sentence-Level Evidence Gating** | Four papers already do this, and better. Dropped after a prior-art search. |
| **Grad-CAM stability score** | Already published in 2021 — and they found Grad-CAM repeatability is poor (SSIM 0.12). Dropped. |
| **Conditional specialisation** | A cheap linear-probe gate returned +0.0003 before I committed the GPU budget. Cancelled. |
| **Classifier-conditioned reports** | Measured +0.0736 of headroom, built it, and the ablation showed the prompt contributed +0.0023. The gain was fine-tuning. |
| **Label noise hypothesis** | I assumed weak co-pathology performance was caused by bad labels. Measuring the ceiling F1 (0.82–0.97 vs model 0.23–0.86) proved it wasn't. |

The pattern in all of these is the same: I assumed a number instead of measuring it. The controls that killed these ideas are in the repository alongside the ones that worked.

---

## References

1. Pereira S.C. et al. **Addressing Chest Radiograph Projection Bias in Deep Classification Models.** MIDL 2023, [PMLR 227:1199–1210](https://proceedings.mlr.press/v227/pereira24a.html)
2. Johnson A.E.W. et al. **MIMIC-CXR-JPG: A large publicly available database of labeled chest radiographs.** [arXiv:1901.07042](https://arxiv.org/pdf/1901.07042)
3. Irvin J. et al. **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels.** AAAI 2019, [arXiv:1901.07031](https://arxiv.org/abs/1901.07031)
4. Liu Z. et al. **A ConvNet for the 2020s (ConvNeXt).** CVPR 2022, [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
5. Yuan H. et al. **BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model.** BioNLP 2022, [arXiv:2204.03905](https://arxiv.org/abs/2204.03905)
6. Selvaraju R.R. et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** ICCV 2017, [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
7. Ganin Y., Lempitsky V. **Unsupervised Domain Adaptation by Backpropagation.** ICML 2015, [arXiv:1409.7495](https://arxiv.org/abs/1409.7495)
8. Hardt M. et al. **Equality of Opportunity in Supervised Learning.** NeurIPS 2016, [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)
9. Seyyed-Kalantari L. et al. **CheXclusion: Fairness gaps in deep chest X-ray classifiers.** [PSB 2021](https://psb.stanford.edu/psb-online/proceedings/psb21/seyyed-kalantari.pdf)
10. **Encoder-decoder models for chest X-ray report generation perform no better than unconditioned baselines.** [PLOS One 2021](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259639)
11. Arun N. et al. **Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging.** *Radiology: AI* 2021
12. **The limits of fair medical imaging AI in real-world generalization.** [*Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-03113-4)

Full prior-art analysis is in [`MASTER_PLAN.md`](MASTER_PLAN.md); detailed results are in [`RESULTS.md`](RESULTS.md).

---

## License

The **code** in this repository is released under the **MIT License**.

The **data is not**. MIMIC-CXR is credentialed and governed by the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimic-cxr/view-dua/2.0.0/). No images, reports, or derived data files are distributed here. To reproduce this work you need your own PhysioNet credentialed access and completed CITI training.

Model weights are not distributed either, since they are derived from credentialed data.

---

## ⚕️ Disclaimer

This is a research prototype built for an undergraduate final-year project. It is **not** a medical device, has **not** undergone clinical validation, and must **not** be used for diagnosis or treatment decisions. Every output requires review by a qualified radiologist.

The system is measurably less accurate on AP (bedside) films — the ones taken of the sickest patients. That limitation is reported here rather than hidden.
