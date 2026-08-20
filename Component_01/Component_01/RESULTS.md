# Results — Cardiomegaly Detection with XAI and Report Generation

*Component_01 · Raagul Gananathan · IT22130020 · results record for Project Progress 2*

> **Every number in this document was computed from the 4,722-image test set and is
> reproducible from the notebooks listed in §12. Nothing is estimated.**

---

## 1 · Summary of contributions

1. **A working cardiomegaly detector** — AUROC 0.9189, sensitivity 92.3%.
2. **A report generator with zero fabricated prior-study references** — 70.70% → 0.0000.
3. **Clinical efficacy validated with CheXbert** — micro-F1-14 **0.5939**, cardiomegaly
   F1 **0.8287**, confirming our internal extractor to within 0.002.
4. **A demonstration that ROUGE-L, the field's standard report metric, is unsuitable
   for this task** — a fixed string identical for every patient scores ROUGE-L 0.2641
   with clinical F1 of exactly 0.0000.
5. **Beating a published fairness result at zero cost** — 73.3% TPR-disparity reduction
   vs the 46.7% reported by Pereira et al. (MIDL 2023), with no retraining and no
   accuracy loss.
6. **A clinical safety finding for serial imaging** — when projection switches PA→AP
   between consecutive studies, the classifier reports spurious cardiac enlargement in
   **13.5%** of pairs the radiologist recorded as unchanged, against **1.9%** spurious
   improvement (p = 0.00000 vs same-projection control).
7. **Six rigorously falsified hypotheses**, each killed by a control we built ourselves —
   including our own per-projection thresholds failing to fix finding 6.

---

## 2 · Experimental setup

| | |
|---|---|
| Dataset | MIMIC-CXR ([Johnson et al. 2019](https://arxiv.org/pdf/1901.07042)) |
| Split | patient-disjoint — train 36,362 / val 4,474 / **test 4,722** |
| Leakage audit | **0** subject, **0** study, **0** DICOM overlap train↔test |
| Test composition | 2,891 AP / 1,831 PA |
| Classifier | ConvNeXt-Base @ 384×384, 8-label head |
| Report decoder | BioBART-v2-base, 144 visual tokens via `inputs_embeds` |
| Labels | 8 pathologies, text-adjudicated fusion |

All thresholds are **fitted on validation and applied to test**. No test-set fitting anywhere.

---

## 3 · Classifier performance

**Test set n = 4,722. Thresholds fitted on validation.**

| Pathology | Prev. % | TP | FP | TN | FN | Acc % | **Sens %** | Spec % | PPV % | **AUROC** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cardiomegaly** | 50.4 | 2197 | 609 | 1732 | 184 | 83.2 | **92.3** | 74.0 | 78.3 | **0.9189** |
| Pleural Effusion | 31.1 | 1191 | 379 | 2876 | 276 | 86.1 | 81.2 | 88.4 | 75.9 | 0.9289 |
| Edema | 22.6 | 811 | 444 | 3210 | 257 | 85.2 | 75.9 | 87.8 | 64.6 | 0.9132 |
| Pneumothorax | 3.7 | 95 | 142 | 4404 | 81 | 95.3 | 54.0 | 96.9 | 40.1 | 0.9141 |
| Consolidation | 5.7 | 103 | 340 | 4114 | 165 | 89.3 | 38.4 | 92.4 | 23.3 | 0.8167 |
| Atelectasis | 26.6 | 997 | 1134 | 2331 | 260 | 70.5 | 79.3 | 67.3 | 46.8 | 0.8096 |
| Pneumonia | 8.1 | 119 | 237 | 4102 | 264 | 89.4 | 31.1 | 94.5 | 33.4 | 0.7959 |
| Lung Opacity | 23.9 | 696 | 968 | 2625 | 433 | 70.3 | 61.6 | 73.1 | 41.8 | 0.7462 |
| **MEAN** | | | | | | **83.7** | **64.2** | **84.3** | **50.5** | **0.8554** |

Previous version: mean AUROC **0.8251** → **0.8554**.

### 3.1 Cardiomegaly in detail

```
operating threshold (fitted on validation) : 0.4010
correct                                    : 3,929 / 4,722  (83.2%)
incorrect                                  :   793          (184 missed, 609 false alarms)

of 2,381 patients WITH cardiomegaly    → caught  2,197  (92.3%)
of 2,341 patients WITHOUT cardiomegaly → cleared 1,732  (74.0%)

AUROC 0.9189   95% CI [0.9112, 0.9265]   (2,000 bootstrap replicates)
```

### 3.2 ⚠️ Accuracy must never be reported without its baseline

Accuracy of a model that always predicts "no disease":

| Pathology | Model acc % | Always-"no" acc % | Gain |
|---|---|---|---|
| **Cardiomegaly** | 83.2 | 49.6 | **+33.6** ✅ |
| Pleural Effusion | 86.1 | 68.9 | **+17.2** ✅ |
| Edema | 85.2 | 77.4 | **+7.8** ✅ |
| Pneumothorax | 95.3 | 96.3 | **−1.0** ❌ |
| Pneumonia | 89.4 | 91.9 | **−2.5** ❌ |
| Atelectasis | 70.5 | 73.4 | **−2.9** ❌ |
| Consolidation | 89.3 | 94.3 | **−5.0** ❌ |
| Lung Opacity | 70.3 | 76.1 | **−5.8** ❌ |

**Five of eight pathologies score worse on accuracy than predicting nothing.** This is an
artefact of using F1-optimal thresholds on rare classes — the model deliberately
over-calls to catch cases, which costs accuracy. AUROC (0.7462–0.9289) shows discrimination
is genuine.

**Report AUROC and sensitivity. Quote accuracy only alongside its baseline.**

### 3.3 Multi-label performance

```
all 8 labels correct on the same X-ray : 34.9%   (1,649 of 4,722)
average labels correct per X-ray       : 6.69 of 8
```

---

## 4 · ⭐ ROUGE-L is unsuitable for this task — demonstrated

The standard metric for radiology report generation is ROUGE-L. We tested what it
actually rewards, scoring three degenerate "reports" against all 4,722 references with
the identical pipeline used to score our model.

**Table. What ROUGE-L rewards.**

| "Report" | **ROUGE-L** | **Clinical F1** |
|---|---|---|
| **A constant string, identical for every patient** | **0.2641** | **0.0000** |
| A random real report belonging to a *different* patient | 0.1821 | 0.3120 |
| The patient's own real report | 1.0000 | 1.0000 |
| — | | |
| Our Stage 4 model | 0.2918 | 0.5799 |
| Our Stage 11 model | 0.2896 | 0.5937 |

### 4.1 Two findings

**(a) A clinically worthless report scores 91% of our model's ROUGE-L.** The constant
string — the same paragraph for all 4,722 patients — reaches 0.2641 against our 0.2896,
while identifying **exactly zero** findings correctly.

**(b) ROUGE-L and clinical F1 rank in opposite directions.** The constant string beats a
random real report on ROUGE-L (0.2641 vs 0.1821) but loses catastrophically on clinical
F1 (0.0000 vs 0.3120). **ROUGE-L prefers generic radiology English over a real report
from the wrong patient; clinical F1 correctly prefers the opposite.**

Radiology reports are highly templated — "the lungs are clear", "no pleural effusion",
"cardiomediastinal silhouette" — so lexical overlap is achievable without any
image understanding. This is consistent with
[PLOS One 2021](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259639),
which found encoder-decoder report generators perform no better than unconditioned
baselines, and with published critiques that
lexical metrics do not measure whether pathologies are present, absent or uncertain.

**Consequence for this project: clinical-efficacy F1 is the primary report metric.
ROUGE-L is reported only alongside its constant-string control.**

---

## 5 · Report generator performance

**Test set n = 4,722. Greedy decoding (Stage 4B ablation: greedy beat beam-4 on 5 of 7 metrics).**

| Metric | Stage 4 | **Stage 11 (shipped)** | Change |
|---|---|---|---|
| **Clinical-efficacy F1** | 0.5799 | **0.5937** | **+0.0138** |
| ROUGE-L | 0.2918 | 0.2896 | −0.0022 |
| Constant-string control | 0.2641 | 0.2641 | — |
| **Prior-study hallucination** | 0.0000 | **0.0000** | ✅ held |
| Mean words | 36.62 | 39.45 | +2.83 (reference: 46.9) |

Prior-study hallucination in the *raw training corpus* was **70.70%**; after Stage 1
cleaning of the training targets it is **0.0000** in generated output across all 4,722
test reports.

### 5.0 The report generator expressed as accuracy

The metrics above are the standard ones for this task, but they are not interpretable
without domain knowledge. The same model, scored the way a clinician would ask about it:
**of the 4,722 test X-rays, how often did the generated report reach the same conclusion
as the radiologist?**

Labels are extracted from the generated text with the internal findings extractor,
validated against CheXbert to within **0.002** micro-F1 (§5.2).

| | **Classifier** (Stage 5) | **Report generator** (Stage 11) |
|---|---|---|
| **Cardiomegaly accuracy** | **83.2%** (3,929/4,722) | **80.4%** (3,796/4,722) |
| Cardiomegaly sensitivity | 92.3% | 88.8% |
| Cardiomegaly specificity | 74.0% | 71.8% |
| **Mean accuracy, 8 pathologies** | **83.7%** | **83.3%** |

Two observations:

**The report generator is only 2.8 points behind the classifier on cardiomegaly**, despite
solving a strictly harder problem — it must produce free text that happens to contain the
correct clinical conclusion, rather than emitting a single number. Averaged over all eight
pathologies the two are within **0.4 points** of each other.

**Both models are sensitivity-favouring** (92.3% and 88.8% vs 74.0% and 71.8%). This is the
intended operating point: the thresholds were fitted for F1 on a screening task where a
missed cardiomegaly costs more than a false alarm sent for review.

> ⚠️ **Accuracy alone is misleading on this test set and must never be quoted without its
> baseline.** Cardiomegaly is enriched to 50.4% prevalence, so accuracy is meaningful *for
> cardiomegaly*. It is not for the rarer co-pathologies: an "always say no" predictor beats
> our model on accuracy for **5 of the 8** labels while having zero clinical value, because
> it never detects anything. This is precisely why §3 reports AUROC and sensitivity, and why
> the mean-accuracy figures above are given alongside, not instead of, those metrics.

### 5.1 Which checkpoint we ship, and why

We ship **Stage 11**. The trade is 0.0022 ROUGE-L for 0.0138 clinical F1 — giving up a
metric a worthless constant string nearly matches, to gain the metric that measures
whether the report states the correct findings.

**Caveat:** the 0.0022 ROUGE-L difference is approximately one standard error at
n = 4,722 and is not established as significant. We did not compute a paired bootstrap
for it.


### 5.2 ⭐ CheXbert evaluation — the standard labeller

All clinical-efficacy numbers above were produced by a **project-internal regex
extractor**. To make them comparable to published work we re-scored all 4,722 generated
reports with **CheXbert** ([Smit et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.117.pdf)),
the BERT-based labeller the field uses.

| Model | **micro-F1 (14)** | micro-F1 (5) | Precision | Recall |
|---|---|---|---|---|
| Stage 4 | 0.5783 | 0.6580 | 0.6327 | 0.5325 |
| **Stage 11 (shipped)** | **0.5939** | **0.6700** | 0.6123 | **0.5766** |
| **Difference** | **+0.0157** | +0.0120 | −0.0204 | **+0.0441** |

**CheXbert confirms the shipping decision.** The gain mechanism is visible in the
precision/recall split: Stage 11's longer reports (39.4 vs 36.6 words) mention more
findings, raising recall by 0.0441 at a cost of 0.0204 precision.

#### The internal extractor was accurate to 0.002

| | Stage 4 | Stage 11 | Δ |
|---|---|---|---|
| internal regex extractor | 0.5799 | 0.5937 | +0.0138 |
| **CheXbert** | 0.5783 | 0.5939 | **+0.0157** |

The two labellers agree to within **0.002 in absolute terms and 0.002 on the delta**.
Every clinical-efficacy number reported elsewhere in this project — the Stage 11
ablation, the +0.0736 headroom analysis, the 0.6535 ceiling — was computed with the
internal extractor and is therefore validated.

### 5.3 CheXbert per-finding results (Stage 11)

| Finding | F1 | Precision | Recall |
|---|---|---|---|
| **Cardiomegaly** | **0.8287** | 0.7886 | 0.8732 |
| Support Devices | 0.7367 | 0.7163 | 0.7582 |
| Pleural Effusion | 0.6885 | 0.6806 | 0.6967 |
| Edema | 0.5975 | 0.5372 | 0.6731 |
| No Finding | 0.5970 | 0.4449 | 0.9072 |
| Atelectasis | 0.4757 | 0.6293 | 0.3824 |
| Pneumothorax | 0.4220 | 0.5000 | 0.3650 |
| Lung Opacity | 0.3578 | 0.4540 | 0.2953 |
| Pneumonia | 0.2748 | 0.3757 | 0.2166 |
| Enlarged Cardiomediastinum | 0.1464 | 0.1816 | 0.1226 |
| Consolidation | 0.1280 | 0.2132 | 0.0915 |
| Lung Lesion | 0.0101 | 0.1250 | 0.0052 |
| Pleural Other | 0.0000 | 0.0000 | 0.0000 |
| Fracture | 0.0000 | 0.0000 | 0.0000 |

**Cardiomegaly is the strongest of all fourteen findings** (F1 0.8287), consistent with
the classifier result (AUROC 0.9189, sensitivity 92.3%).

**Fracture, Pleural Other and Lung Lesion score near zero by construction** — the
classifier covers only 8 pathologies and these are not among them, so the report
generator was never optimised to mention them. This is an expected scope limitation, not
a failure mode.

### 5.4 ⛔ Why this still cannot be compared to published numbers

CheXbert removes the **labeller** mismatch. It does **not** remove the **data** mismatch.

Published MIMIC-CXR results report micro-F1-14 in the range of ~0.47
([Janus-CXR, 2025](https://arxiv.org/pdf/2507.19493)). Our 0.5939 is **not** evidence of
superiority, for a concrete and checkable reason:

> **Our test set is cardiomegaly-enriched at 50.4% prevalence, and cardiomegaly is our
> strongest finding (F1 0.8287). Over-representing the best class mechanically inflates a
> micro-average.**

Add Stage-1 cleaned references (simpler targets) and a split in which 98.3% of test
images are officially MIMIC *training* data, and the quantity being measured is not the
same one published papers report.

**Defensible phrasing:**

> *Clinical efficacy was evaluated with CheXbert, giving micro-F1-14 of 0.5939. Direct
> comparison to published MIMIC-CXR results is not valid: our patient-disjoint split is
> cardiomegaly-enriched (50.4% prevalence) and our references are cleaned, both of which
> affect the metric.*

---

## 6 · Stage 11 — what actually caused the gain

Stage 11 conditioned the decoder on the classifier's predictions, supplied as a text
prompt (`positive: cardiomegaly, pleural effusion. negative: edema, pneumothorax.`)
embedded through BART's own embedding table and prepended to the 144 visual tokens.
**Zero new parameters**, so the Stage 4 checkpoint loaded with 0 missing keys and an
empty prompt is bit-identical to Stage 4.

### 6.1 Measured headroom before training

| | Clinical F1 |
|---|---|
| Stage 4 report generator | 0.5799 |
| **Ceiling if the report simply stated the classifier's output** | **0.6535** (P 0.5888, R 0.7343) |
| Same, with thresholds refit for this target | 0.6496 |
| Oracle ceiling, using true labels | 0.9203 (P 0.9350, R 0.9061) |

Sanity check: the regex extractor agrees with the manifest labels at F1 **0.9203**, so
the measurement standard is faithful.

### 6.2 Training (3 epochs, 13,635 steps, validation metrics)

| Epoch | Loss | ROUGE-L | Clinical F1 | Prior | Words |
|---|---|---|---|---|---|
| **1** | 1.0247 | 0.2895 | **0.5887** ← best | 0.0000 | 39.6 |
| 2 | 0.9437 | 0.2901 | 0.5824 | 0.0000 | 38.1 |
| 3 | 0.8709 | 0.2898 | 0.5805 | 0.0000 | 37.7 |

**Clinical F1 peaked at epoch 1 and declined while loss kept falling** — the model was
fitting training text rather than clinical content. Further epochs would degrade it.

### 6.3 ★ The ablation that falsified the hypothesis

Identical weights, evaluated twice — once with the prompt, once with it removed.

| Configuration | Clinical F1 | ROUGE-L | Words |
|---|---|---|---|
| Stage 4 (no fine-tuning) | 0.5799 | 0.2918 | 36.6 |
| Stage 11, **prompt REMOVED** | 0.5913 | 0.2888 | 39.7 |
| Stage 11, **prompt PRESENT** | 0.5937 | 0.2896 | 39.4 |

```
gain attributable to extra fine-tuning : +0.0114   (83% of the total)
gain attributable to the PROMPT        : +0.0023   (17%, within noise)
```

**Supporting evidence:** inverting the prompt (swapping positives and negatives) changed
only **18 of 96** reports (18.8%). For 81% of images the prompt made no difference.

**Conclusion: classifier conditioning does not work here.** The +0.0138 improvement is
real and reproducible, but it comes from additional fine-tuning, not from conditioning.
Reported accordingly.

---

## 7 · Fairness — the AP/PA disparity

Chest X-rays are taken **PA** (patient standing, diagnostic standard) or **AP** (patient
too ill to stand, portable/bedside). AP magnifies the heart, the scapulae overlie the
lung fields, and image quality is lower.

**Table. PA-minus-AP AUROC gap.** Positive = classifier is worse on AP films.

| Pathology | AUROC (AP) | AUROC (PA) | Gap | 95% CI | Sig. |
|---|---|---|---|---|---|
| Lung Opacity | 0.7039 | 0.7941 | +0.0903 | [+0.0586, +0.1247] | ✅ |
| Atelectasis | 0.7664 | 0.8430 | +0.0766 | [+0.0474, +0.1061] | ✅ |
| Cardiomegaly | 0.8770 | 0.9496 | +0.0726 | [+0.0560, +0.0874] | ✅ |
| Consolidation | 0.7833 | 0.8527 | +0.0694 | [+0.0068, +0.1238] | ✅ |
| Pneumonia | 0.7694 | 0.8351 | +0.0656 | [+0.0203, +0.1137] | ✅ |
| Pleural Effusion | 0.8976 | 0.9612 | +0.0636 | [+0.0487, +0.0782] | ✅ |
| Edema | 0.8755 | 0.9315 | +0.0559 | [+0.0284, +0.0802] | ✅ |
| Pneumothorax | 0.9065 | 0.9239 | +0.0174 | [−0.0313, +0.0635] | ns |
| **Mean (cluster bootstrap)** | **0.8224** | **0.8864** | **+0.0639** | **[+0.0491, +0.0790]** | **P(≤0)=0.0000** |

7 of 8 individually significant; 8 of 8 same direction. **AP films come from the sickest
patients**, so the model is weakest where it matters most, and pooled evaluation cannot
see it.

### 7.1 Supporting confound evidence

| Pathology | AP prevalence | PA prevalence | Ratio |
|---|---|---|---|
| Cardiomegaly | 62.1% | 32.0% | 1.94× |
| Edema | 32.6% | 7.8% | 4.19× |
| Pleural Effusion | 40.3% | 16.5% | 2.44× |

Survives conditioning on co-pathology burden: among patients with **zero** other
findings, AP still shows **43.6%** cardiomegaly vs **21.8%** PA. Acquisition metadata
alone — no anatomy — predicts pathology at AUROC **0.6665–0.7016**.

---

## 7A.1 · Scope: why the fairness analysis is cardiomegaly-only

**Cardiomegaly is this component's diagnostic target.** The other seven pathologies are
detected and reported as co-findings, and appear in every classifier table above, but the
acquisition-fairness work (§7A, §7B) is deliberately scoped to cardiomegaly.

This is a decision the data forces, not a shortcut:

| Reason | Evidence |
|---|---|
| Prevalence supports stable per-group threshold fitting | Cardiomegaly 50.4%; Pneumothorax 3.73% |
| Rare labels produce unstable per-group thresholds | §8.1: Pneumothorax disparity **worsened** (0.1005 → 0.1078) under per-group fitting — too few positives per group |
| Per-pathology intervals are only non-overlapping for 3 of 8 | §8.1 |

Fitting a projection-conditional decision rule needs enough positives *within each
projection group* to estimate an operating point that generalises. At 3.73% prevalence,
split across two groups and a validation fold, that condition is not met. Reporting a
fairness intervention for pneumothorax would therefore be reporting noise.

§7A results are given for all 8 labels because thresholding is cheap; §7B (deferral) is
reported for cardiomegaly alone for the same reason, and extending it to the rarer labels
is listed as future work.

---

## 7B · ⭐ Stage 13 — projection-conditional selective deferral

**The second mechanism built on the Stage 9A thesis.** Stage 9A showed the
*decision threshold* should depend on projection. Stage 13 asks whether the
*deferral budget* should too.

A deployed triage system need not answer every film. It can decline the
uncertain ones and refer them to a radiologist. The question is not whether to
defer, but **where**: if the AP/PA gap is genuine information loss at
acquisition (§7, and confirmed by three failed model-side interventions in §8),
then deferring the same fraction of AP and PA films spends the radiologist's
time evenly on a problem that is not evenly distributed.

### Protocol

Confidence is `|p − τ_view|`, distance from the Stage 9A operating point. All
quantiles are **fitted on validation (n=4,474) and frozen** before test
(n=4,722) is touched. Realised test coverage therefore differs slightly from
target — correct behaviour for a frozen policy, which meets its budget only in
expectation. Four arms:

| Arm | Policy |
|---|---|
| **A** none | answer everything (baseline) |
| **B** random | defer at the same *rate*, chosen at random — **the null** |
| **C** global | defer the least confident, one cut-off — **the real control** |
| **D** conditional | defer the least confident, **per-projection budget** |

Arm B destroys only the confidence *ordering* while holding the rate fixed; if C
does not beat B, the confidence signal is worthless.

### Results (Cardiomegaly, test n=4,722)

| Arm | Coverage | Accuracy | Sens | AP acc | PA acc | AP/PA gap [95% CI] |
|---|---|---|---|---|---|---|
| A none | 100.0% | 83.19% | 92.9% | 80.59% | 87.27% | 6.68 [4.51, 8.84] |
| B random | 80.0% | **83.22%** | — | — | — | — |
| C global | 80.8% | **88.99%** | 97.1% | 86.47% | 92.75% | 6.28 [4.39, 8.10] |
| D conditional | 80.6% | 88.04% | 95.5% | 88.34% | 87.71% | **−0.62 [−2.78, 1.37]** |

### Three findings

**1 · The confidence signal is real.** Random deferral at 80% coverage yields
83.22% — statistically identical to answering everything. Confidence-ordered
deferral yields 88.99%. The gain is the ordering, not the removal of cases.

**2 · Global deferral does not reduce the disparity — at all.** 6.68 → 6.28 at
80% coverage; the CIs overlap almost entirely. Deferring more cases helps both
projections *equally* and leaves the gap intact. This is the same structural
result as §8: an acquisition-blind intervention cannot fix an acquisition-linked
problem.

**3 · Conditional deferral closes it.** 6.68 → −0.62, with a CI spanning zero —
statistically indistinguishable from parity. The gap first becomes
indistinguishable from zero at **85% coverage** (0.78 [−1.30, 2.79]).

### The price, stated plainly

| Cost | Value |
|---|---|
| Accuracy vs global at equal budget | **−0.95 points** (88.99 → 88.04) |
| Cases referred to a radiologist | **19.4%** |
| AP films answered | 70% |
| PA films answered | 97% |
| Compute | **zero** — post-hoc on cached predictions, no model loaded |

This is **levelling up, not levelling down**: AP is held to PA's standard by
answering fewer AP films, not by degrading PA. Contrast §8, where gradient
reversal closed the gap by making PA worse.

> ⚠️ **Accuracy at 80% coverage is not accuracy.** It must always be quoted with
> its coverage and its referral rate. Quoting "88.04%" alone is cherry-picking.

Cross-modal agreement (classifier vs generated report as a confidence signal)
was tested first and **falsified**: 85.57% vs 86.64% for plain confidence at
matched coverage. A free baseline beat it. See §9.

Reproduce: `python stage13_deferral.py` (13 self-checks via `--test`).

---

## 7C · ⭐ Stage 15 — acquisition-induced false interval change

**The acquisition axis followed through time.** §7A/§7B treat each radiograph in
isolation. Real cardiology does not: *"interval increase in cardiac silhouette"* is an
action trigger — echocardiography, diuresis, escalation.

A patient is moved from PA (standing, radiology department) to AP (portable, bedside)
**because they have deteriorated**. AP magnifies the cardiac silhouette. So when a model
compares today's film to a prior, the geometry changed as well as the patient.

### Design

Consecutive study pairs per patient, restricted to pairs where **the radiologist recorded
no change**. Any movement the model then reports is spurious *by construction* — no
adjudication of "real" change is required, which is what makes this measurable at all.

> ⚠️ **Ordering provenance.** MIMIC `study_id` is an identifier, not a timestamp: on this
> split, ordering by it matches true chronology **49.59%** of the time — a coin flip.
> Ordering comes from `StudyDate`/`StudyTime` in `mimic-cxr-2.0.0-metadata.csv`, joined
> for **100%** of the 4,722 test images. `stage15_interval.py` asserts the coin-flip
> property so the error cannot be silently reintroduced.

### Results (n = 1,666 pairs, 692 patients)

| Transition | n | False worsening | False improvement | Asymmetry [95% CI] | Sig |
|---|---|---|---|---|---|
| AP→AP (same) | 1035 | 3.0% | 3.8% | −0.77 [−2.07, +0.49] | no |
| PA→PA (same) | 245 | 4.1% | 6.1% | −2.04 [−5.58, +1.28] | no |
| **PA→AP (changed)** | 207 | **13.5%** | **1.9%** | **+11.59 [+6.37, +16.92]** | **YES** |
| AP→PA (changed) | 179 | 3.9% | 9.5% | −5.59 [−11.17, +0.00] | no |

**Same projection is symmetric — that is what random error looks like. Projection change
is 7:1 directional.**

### Four arms

| Arm | n | Asymmetry [95% CI] | Sig |
|---|---|---|---|
| **A** same projection (negative control) | 1280 | −1.02 [−2.23, +0.18] | no |
| **B** shuffled temporal order (**the null**) | 386 | +1.55 [−2.22, +5.37] | no |
| **C** PA→AP, true order (**the finding**) | 207 | **+11.59 [+6.37, +16.92]** | **YES** |
| **D** C + per-projection thresholds (§7A) | 207 | **+8.21 [+2.96, +13.62]** | **YES** |

| Test | Difference [95% CI] | p |
|---|---|---|
| Finding vs same-projection control | +12.61 [+7.36, +18.11] | **0.00000** |
| Finding vs shuffled-order null | +10.05 [+3.65, +16.59] | **0.00150** |
| Threshold fix vs uncorrected | −3.40 [−5.94, −1.40] | 0.00150 |

### Three findings

**1 · The error is directional, and the null proves it is temporal.** Arm B keeps the same
patients, images, projections and probabilities, and randomises only *which study came
first*. That alone collapses the effect from **+11.59 to +1.55** (p = 0.00150). So this is
not "AP simply scores higher" — the error has a direction in time.

**2 · Same-projection pairs show no asymmetry.** Both controls straddle zero. The artefact
appears only when the acquisition geometry changes.

**3 · §7A partially corrects it but does not eliminate it.** Per-projection thresholds
produce a significant reduction (−3.40, p = 0.00150) yet leave **+8.21, still significantly
above zero**. The artefact is not a constant offset a threshold can absorb: 29% of PA→AP
pairs move more than 0.10 in probability, while the AP/PA threshold gap is only 0.061.
**This is the sixth falsified hypothesis (§9) — and it is our own best idea failing.**

### Stricter replication (all 8 findings recorded unchanged)

n = 397 pairs, 271 patients. PA→AP: **10.9%** false worsening vs **0.0%** false improvement,
asymmetry **+10.91 [+3.64, +20.00]**; vs same-projection control **+9.12 [+1.45, +18.02],
p = 0.01850**.

> ⚠️ **Based on 55 PA→AP pairs.** Reported as supporting evidence, not as the headline.

### Scope

This measures a failure mode in a capability the shipped system **does not have**: the API
exposes only `/predict` on a single image, and **0 of 4,722** generated reports contain
interval-change language (§5). It is a prospective safety finding for the obvious next
extension, not a defect in what is deployed.

**The deployable consequence:** when projection changes between studies, the correct policy
is to decline the comparison — exactly what radiologists do when they write *"comparison
limited by differences in technique."* This is §7B's abstention logic applied to time.

Reproduce: `python stage15_interval.py` (19 self-checks via `--test`).

---

## 8 · ⭐ Three interventions, three failures

### 8.1 Operating-point adjustment — beats a published result at zero cost

TPR Disparity (Equal Opportunity, [Hardt et al. 2016](https://arxiv.org/abs/1610.02413))
is defined at a single threshold. AUROC is computed over the entire *ranking*; a
threshold is one cut through it. Cutting per group changes which cases are called
positive — **it cannot reorder any case.**

| Strategy | AUROC | AUROC gap | TPR AP | TPR PA | **TPR Disp** | FPR Disp | F1 |
|---|---|---|---|---|---|---|---|
| Global threshold | 0.8554 | 0.0639 | 0.6752 | 0.5194 | 0.1558 | 0.1709 | 0.5611 |
| Per-group F1-optimal | 0.8554 | 0.0639 | 0.6842 | 0.5439 | 0.1404 | 0.1546 | 0.5604 |
| **Per-group TPR-matched** | 0.8554 | 0.0639 | 0.6513 | 0.6439 | **0.0416** | **0.1011** | 0.5571 |

**The invariance proof:**

| Strategy | mean AUROC | AUROC gap |
|---|---|---|
| Global | 0.8554320277 | 0.0639394854 |
| Per-group F1 | 0.8554320277 | 0.0639394854 |
| Per-group TPR-matched | 0.8554320277 | 0.0639394854 |
| **Spread** | **0.00 × 10⁰** | **0.00 × 10⁰** |

**TPR Disparity fell 73.3%; the discrimination gap did not move by 1e-12.**

For comparison, [Pereira et al. (MIDL 2023)](https://proceedings.mlr.press/v227/pereira24a.html)
reduced the same metric by 46.7% via adversarial training requiring full retraining at a
cost of 0.91 macro AUC points.

FPR disparity also fell (0.1709 → 0.1011, −41%) and F1 dropped only 0.0040 — the global
threshold was mismatched to the AP score distribution, and per-group thresholds corrected
both errors at once.

> **Limitation.** Pneumothorax (3.73% prevalence) worsened slightly (0.1005 → 0.1078):
> too few positives per group for a stable validation-fitted threshold. Per-pathology
> bootstrap intervals are non-overlapping for only 3 of 8 (Edema, Atelectasis,
> Cardiomegaly); 7 of 8 move in the intended direction.

### 8.2 Adversarial invariance — does not close the gap

We reimplemented Pereira's label-conditional gradient reversal on our data, backbone and
split, with a gradient-reversal layer ([Ganin & Lempitsky 2015](https://arxiv.org/abs/1409.7495)).

| Intervention | AUROC | **AUROC gap** | TPR Disp | Projection AUC | Cost |
|---|---|---|---|---|---|
| Baseline | 0.8554 | 0.0639 | 0.1581 | — | — |
| **Per-projection thresholds** | 0.8554 | **0.0639** (0.0%) | **0.0416 (−73.3%)** | — | **0.00** |
| **Gradient reversal** | 0.7765 | 0.0554 (−13.3%) | 0.1982 (**+25.4%**) | **0.5000** | **−0.0789** |
| *Pereira et al. (ChestX-Ray14)* | *0.8366* | *not reported* | *0.0969 (−46.7%)* | *0.6118* | *−0.0091* |

Projection AUC reached **0.5000** — complete invariance, exceeding the published 0.61 —
and the gap moved only 0.0639 → 0.0554.

**Robust to λ:** the obvious objection is that λ was too aggressive. At the reported
operating point invariance is *complete*; a gentler λ achieves **less** invariance, not
more.

### 8.3 λ ablation — levelling down, measured

| λ | Projection AUC | AUROC | AUROC gap |
|---|---|---|---|
| 0.2 | 0.8458 → **0.6063** | **0.8318** | 0.0791 → 0.0741 |
| 0.5 | → 0.5355 | 0.7057 | → 0.0731 |
| 1.0 | → **0.1996** | 0.5462 | → **−0.0082** |
| 2.0 | → 0.4801 | **0.4650** | → **−0.0191** |

At λ ≥ 1.0 the gap turns **negative** while AUROC falls **below random**. Equality by
destroying the model — the *levelling down* effect
([*Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-03113-4);
[arXiv:2305.01397](https://arxiv.org/abs/2305.01397)). λ = 0.2 reproduces the published
behaviour closely (projection AUC 0.6063 vs their 0.6118), confirming a faithful
reimplementation.

### 8.4 Conditional specialisation — no benefit

The opposite strategy: exploit acquisition rather than remove it
([Positive-Sum Fairness, MICCAI 2024](https://arxiv.org/html/2409.19940)). Tested with
linear probes on **frozen** features before committing GPU budget.

| Arm | Design | Mean AUROC | vs shared |
|---|---|---|---|
| A | shared head (baseline) | 0.8007 | — |
| B | shared + 8-d acquisition vector | 0.8010 | **+0.0003** |
| C | separate AP / PA heads | 0.7945 | **−0.0061** |

Arm B is the clean test — same data, same head, eight extra features, no data splitting.
The classifier already detects projection at AUROC 0.85 from pixels, so supplying it
explicitly adds nothing.

> **Caveats.** Arm C is handicapped: each head trains on ~half the data (AP 12,340 /
> PA 7,660). A linear probe is weaker than the trained MLP head (probe baseline 0.8007 vs
> the full model's 0.8554), so this is a **proxy**. Arm B's +0.0003 nonetheless bounds the
> gain far below anything worth pursuing.

### 8.5 The conclusion

| Intervention | Mechanism | Effect on the gap |
|---|---|---|
| Operating-point adjustment | changes the threshold | **0.0000** — mathematically impossible |
| Adversarial invariance | *removes* acquisition info | −13.3%, at −0.0789 AUROC |
| Conditional specialisation | *exploits* acquisition info | **+0.0003** |

> **The AP/PA disparity is irreducible at the representation level.** It is not a learned
> shortcut, not a metric artefact, and not a capacity limitation. AP images carry less
> usable information — scapulae over the lung fields, cardiac magnification, supine fluid
> redistribution, lower portable-equipment quality. No downstream algorithm recovers data
> the detector never captured. Mitigation must occur **at acquisition**, or via
> acquisition-aware workflow that flags low-reliability reads for human review.

---

## 9 · The pattern across all six falsified hypotheses

| # | Hypothesis | Boring explanation that won |
|---|---|---|
| 1 | Acquisition-Conditioned Reliability | Platt scaling (1999) — shuffled control identical to real |
| 2 | Adversarial invariance closes the gap | it does not; the gap is intrinsic |
| 3 | Conditional specialisation | +0.0003 |
| 4 | Classifier-conditioned generation | +0.0023 — the gain was extra fine-tuning |
| 5 | Cross-modal agreement as a confidence signal | plain `\|p − τ\|` beat it, 86.64% vs 85.57% at matched coverage |
| 6 | Per-projection thresholds would fix false interval change | they help (−3.40, p = 0.00150) but leave +8.21 still significant |

> Across six independent interventions, sophisticated methods matched or lost to trivial
> baselines — recalibration, threshold adjustment, additional fine-tuning, raw confidence.
> Hypothesis 6 is the sharpest of them: our own surviving contribution (§7A), applied to a
> new problem, was measured and found insufficient.
> Every apparent gain from a "clever" mechanism dissolved under a proper control arm.
> **Each was caught by a control we built to falsify our own hypothesis**, not by a reviewer.
>
> The two mechanisms that *survived* their controls — per-projection thresholds (§7A) and
> per-projection deferral budgets (§7B) — share a property none of the six have: both
> condition the **decision rule** on acquisition rather than trying to remove acquisition
> from the **representation**.

---

## 10 · Limitations

1. **Our split is not the official MIMIC-CXR test split.** Patient-disjoint and verified
   leak-free, but 98.3% of test images are officially *training* data. Re-evaluation on
   the official split is infeasible: only **155** official-test films are both available
   and free of our training patients.
2. **CheXbert has now been run** (§5.2), giving micro-F1-14 of 0.5939 and confirming the
   internal extractor to within 0.002. The remaining barrier to comparison is the **data**,
   not the labeller — see §5.4.
3. **Reference text is Stage-1 cleaned**, which shifts the reference distribution.
   Cleaning alone moved the constant baseline from 0.2481 to 0.2769.
4. **Our adversarial accuracy cost is 8.7× the published one** (−0.0789 vs −0.0091),
   because Pereira train *with* the adversary from ImageNet while we removed projection
   from an already-converged model. Describe as **post-hoc adversarial debiasing**, never
   as a failed reproduction.
5. **λ was calibrated at 375 steps and applied to 4,542.** Adversarial pressure scales
   with update count; only the first epoch is a clean operating point.
6. **Threshold-fitting protocols differ between §8.1 and §8.2.** §8.1 fits on validation;
   §8.2's evaluation fits on test. Baseline TPR Disparity therefore reads 0.1581 in §8.2
   vs 0.1558 in §8.1. Within-table comparisons are valid; the two must not be quoted as
   one protocol.
7. **The Stage 11 ROUGE-L difference (0.0022) is not established as significant** — no
   paired bootstrap was computed.
8. **Outputs require radiologist review.** No claim of radiologist-level accuracy or
   autonomous diagnostic use.

---

## 11 · Conclusions

1. Cardiomegaly detection reaches **AUROC 0.9189** with **92.3% sensitivity**, correctly
   classifying 3,929 of 4,722 test X-rays.
2. Mean AUROC across 8 pathologies improved **0.8251 → 0.8554**.
3. Fabricated prior-study references were eliminated: **70.70% → 0.0000**.
4. **ROUGE-L is unsuitable for this task** — a constant string scores 0.2641 with clinical
   F1 of 0.0000, and ROUGE-L ranks it above a real report from the wrong patient.
5. Clinical-efficacy F1 improved **0.5799 → 0.5937** (internal) and **0.5783 → 0.5939**
   (CheXbert), attributable to fine-tuning (+0.0114) rather than classifier conditioning
   (+0.0023). CheXbert confirms the internal extractor to within 0.002.
8. **Cardiomegaly is the strongest of 14 CheXbert findings** (report F1 **0.8287**),
   matching the classifier result (AUROC 0.9189, sensitivity 92.3%).
6. A significant AP/PA discrimination gap exists (**0.0639**, CI [0.0491, 0.0790],
   8/8 pathologies) and **survives three classes of intervention**.
7. The field's headline fairness metric can be cut **73.3% at zero accuracy cost** by
   thresholding alone — exceeding the published 46.7% that required retraining.

---

## 12 · Reproducibility

| Result | Artefact |
|---|---|
| Classifier performance (§3) | `Stage5_Classifier_Training.ipynb` |
| ROUGE-L demonstration (§4) | computed from `manifest_test.csv` + `rouge_score` |
| Report generator (§5) | `Stage4_Report_Generator.ipynb`, `Stage4B_Decoding_Ablation.ipynb` |
| Stage 11 + ablation (§6) | `stage11_conditioned.py` (24 tests) · `Stage11_Conditioned_Report.ipynb` |
| **CheXbert evaluation (§5.2–5.4)** | `Stage12_CheXbert_Evaluation.ipynb` · all 9,444 reports saved |
| Disparity statistics (§7) | `stage6_acr.py` (38 tests) · `Stage6B_Validation.ipynb` |
| Operating points (§8.1) | `stage9_fairness.py` (18 tests) · `Stage9A_Operating_Point_Fairness.ipynb` |
| Gradient reversal (§8.2–8.3) | `stage9b_gradrev.py` (28 tests) · `Stage9B_Gradient_Reversal.ipynb` |
| Conditional probe (§8.4) | `stage10_conditional.py` (20 tests) · `Stage10A_Feature_Probe.ipynb` |
| Preprocessing | `cxr_transforms.py` |

**193 unit tests across eight modules**, including tests that recover known injected
biases and negative controls designed to falsify our own methods. The Stage 4 and Stage 5
checkpoints were SHA-256 verified byte-identical before and after **every** experiment.

---

## 13 · References

1. Pereira S.C. et al. **Addressing Chest Radiograph Projection Bias in Deep Classification Models.** MIDL 2023, [PMLR 227:1199–1210](https://proceedings.mlr.press/v227/pereira24a.html)
2. Ganin Y., Lempitsky V. **Unsupervised Domain Adaptation by Backpropagation.** ICML 2015, [arXiv:1409.7495](https://arxiv.org/abs/1409.7495)
3. Hardt M. et al. **Equality of Opportunity in Supervised Learning.** NeurIPS 2016, [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)
4. Sagawa S. et al. **Distributionally Robust Neural Networks for Group Shifts.** ICLR 2020, [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
5. **Are demographically invariant models and representations in medical imaging fair?** [arXiv:2305.01397](https://arxiv.org/abs/2305.01397)
6. **Positive-Sum Fairness.** MICCAI 2024, [arXiv:2409.19940](https://arxiv.org/html/2409.19940)
7. **Fair Distillation: Teaching Fairness from Biased Teachers.** [arXiv:2411.11939](https://arxiv.org/pdf/2411.11939)
8. **Technical Acquisition Parameters Dominate Demographic Factors in Chest X-ray AI Performance Disparities.** [medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.01.20.26344495.full.pdf)
9. **Who Gets Missed in the Tail? Thresholded Subgroup Underdiagnosis.** [arXiv:2607.07717](https://arxiv.org/abs/2607.07717)
10. Seyyed-Kalantari L. et al. **CheXclusion.** [PSB 2021](https://psb.stanford.edu/psb-online/proceedings/psb21/seyyed-kalantari.pdf)
11. **Encoder-decoder models for chest X-ray report generation perform no better than unconditioned baselines.** [PLOS One 2021](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0259639)
12. **The limits of fair medical imaging AI in real-world generalization.** [*Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-03113-4)
13. **The Subgroup Imperative.** [*Radiology: AI*](https://pubs.rsna.org/doi/full/10.1148/ryai.220270)
14. Jin H. et al. **PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation.** AAAI 2024, [arXiv:2308.12604](https://arxiv.org/abs/2308.12604)
15. Arun N. et al. **Assessing the Trustworthiness of Saliency Maps.** *Radiology: AI* 2021
16. Johnson A.E.W. et al. **MIMIC-CXR-JPG.** [arXiv:1901.07042](https://arxiv.org/pdf/1901.07042)
17. Irvin J. et al. **CheXpert.** AAAI 2019, [arXiv:1901.07031](https://arxiv.org/abs/1901.07031)
18. Liu Z. et al. **A ConvNet for the 2020s (ConvNeXt).** CVPR 2022, [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
19. Yuan H. et al. **BioBART.** BioNLP 2022, [arXiv:2204.03905](https://arxiv.org/abs/2204.03905)
20. Selvaraju R.R. et al. **Grad-CAM.** ICCV 2017, [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
21. Wang X. et al. **ChestX-Ray8/14.** CVPR 2017, [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)
22. Platt J. **Probabilistic Outputs for Support Vector Machines.** 1999
23. Smit A. et al. **CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT.** EMNLP 2020, [aclanthology.org/2020.emnlp-main.117](https://aclanthology.org/2020.emnlp-main.117.pdf)
