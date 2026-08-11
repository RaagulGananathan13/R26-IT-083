# Panel Answers — Component 01

Prepared answers to the questions the panel is most likely to ask, with the
measured number for each. **Every figure here is reproducible from this repo.**

The feedback from Progress 1 was:

> *"Live demo was there. Other than the existing coding, there is no evidence of
> any proper independent contribution to the component. The component produces
> acceptable outputs."*

This document exists to answer exactly that.

---

## The 30-second answer

> I have **three contributions** and **five hypotheses I disproved myself**.
>
> The system detects cardiomegaly at **83.2% accuracy, 92.3% sensitivity, AUROC
> 0.9189**, and writes a report that reaches the radiologist's conclusion
> **80.4%** of the time.
>
> But the part that's mine is this: I found that the standard fairness metric in
> this field can be moved by 73% **without changing the model at all** — and then
> I ran three experiments trying to prove myself wrong, and published all three
> when they succeeded in proving me wrong.

---

## Q1 · "What is your accuracy?"

**Say this:**

| | Classifier | Report generator |
|---|---|---|
| **Cardiomegaly accuracy** | **83.2%** (3,929 / 4,722) | **80.4%** (3,796 / 4,722) |
| Sensitivity (catches disease) | **92.3%** | 88.8% |
| Specificity | 74.0% | 71.8% |
| Mean accuracy, 8 pathologies | **83.7%** | **83.3%** |
| Cardiomegaly AUROC | **0.9189** [0.9112, 0.9265] | — |

**Then immediately add the honest qualifier** — do not wait to be asked:

> "Accuracy is the wrong headline number for the rare pathologies. On this test
> set a model that always says 'no disease' beats mine on accuracy for 5 of the 8
> labels — while detecting nothing at all. That's why I fit thresholds for
> sensitivity and report AUROC alongside. For cardiomegaly specifically, accuracy
> is meaningful, because prevalence is 50.4%."

**Why volunteer the weakness?** Because if they find it and you didn't mention it,
every other number you gave becomes suspect.

---

## Q2 · "What is your independent contribution?" *(the one that matters)*

**Three contributions. Lead with the second one — it's the strongest.**

### Contribution 1 — The fairness metric this field uses is gameable

TPR disparity is the standard fairness metric for chest X-ray AI. I showed it can
be reduced **73.3%** by changing only the decision threshold per projection —
**zero retraining, zero architecture change, AUROC spread 0.00e+00** (identical to
12 decimal places, i.e. the model is provably untouched).

A published MIDL 2023 method achieved **46.7%** on the same metric using a
trained adversarial approach.

> **The implication:** a metric that a free post-processing step beats a trained
> method on is not measuring what people think it measures.

### Contribution 2 — The disparity cannot be fixed by a better model

I tested all three families of intervention:

| Approach | Result |
|---|---|
| Recalibration (acquisition-conditioned) | = Platt scaling from 1999 |
| Adversarial invariance (gradient reversal) | gap unchanged; pushed harder, model collapsed |
| Conditional specialisation | +0.0003 — noise |

Driving projection-predictability to **AUC 0.5000** — complete invariance, better
than the published method's 0.61 — closed only **13.3%** of the gap and cost
**0.0789 AUROC**.

> **The finding:** the AP/PA gap is information genuinely absent from the image,
> not a defect in the representation. AP films are taken at the bedside on the
> sickest patients: scapulae overlie the lungs, the heart is magnified, the
> patient is supine. **You cannot recover information the camera never captured.**

### Contribution 3 — So I built the thing that does work

If the model can't be fixed, the *decision policy* can. The system now declines to
answer cases too close to its threshold and refers them to a radiologist — and it
refers **23% of bedside films but only 0.3% of standing films.**

| Policy | Accuracy | AP/PA gap |
|---|---|---|
| Answer everything | 83.2% | 6.68 |
| Defer evenly (control) | 89.0% | **6.28** ← barely moves |
| **Defer per projection** | 88.0% | **−0.62** ← eliminated |

> **Deferring evenly does not reduce the disparity at all.** Only conditional
> deferral closes it. That's Contribution 1's thesis confirmed a second time by an
> independent mechanism.

---

## Q3 · "How do we know you didn't just get lucky?"

**This is the answer to give.** It is the strongest thing in the project.

> "Because I built the experiments designed to destroy my own ideas, and five of
> them worked."

| # | My idea | What beat it |
|---|---|---|
| 1 | Acquisition-conditioned reliability | Platt scaling (1999) |
| 2 | Adversarial invariance | nothing — gap unchanged |
| 3 | Conditional specialisation | +0.0003, i.e. noise |
| 4 | Classifier-conditioned generation | just training longer |
| 5 | Cross-modal agreement as confidence | plain confidence: 86.64% vs 85.57% |

Every winner in the right-hand column is trivial. **Five times, the sophisticated
method lost to the simple one.**

> "Each of those was caught by a control I built to falsify my own hypothesis —
> not by a reviewer. The two ideas that survived went through the same controls.
> That's why I'd ask you to believe those two."

---

## Q4 · "Is ROUGE-L not the standard metric? Why is yours low?"

**Because ROUGE-L is invalid for this task, and I demonstrated it rather than
asserting it.**

| "Report" | ROUGE-L | Clinical F1 |
|---|---|---|
| **One constant paragraph, same for every patient** | **0.2641** | **0.0000** |
| A real report from the *wrong* patient | 0.1821 | 0.3120 |
| My model | **0.2896** | **0.5937** |

> "A single fixed paragraph, identical for all 4,722 patients, scores 91% of my
> model's ROUGE-L while identifying **zero** findings correctly. And ROUGE-L ranks
> the constant string *above* a genuine report from the wrong patient, while
> clinical F1 ranks them the opposite way. The two metrics disagree on direction.
> That's why I optimise clinical F1."

---

## Q5 · "Why should we trust your own label extractor?"

> "I validated it against **CheXbert**, the standard published labeller. They agree
> to within **0.002** micro-F1. I didn't ask you to take my extractor on faith — I
> checked it against the field's."

---

## Q6 · "Did you improve on the baseline system?"

| | Before | After |
|---|---|---|
| Classifier mean AUROC | 0.8251 | **0.8554** |
| Report clinical F1 | 0.5799 | **0.5937** |
| Prior-study hallucination | 70.70% (corpus) | **0.0000** |
| Report vs constant-string control | **below it** | **above it** |

The most important row is the last one. The original report model **scored below a
fixed string** — because a bug routed image features around BART's encoder
entirely, so the decoder never saw the image. Fixing that (`inputs_embeds` instead
of `encoder_outputs`) is what made it a radiology model rather than a language
model with a decoration attached.

Hallucination was removed by cleaning the training **targets**, not by regex on the
**output** — so the model never learned the artefact in the first place.

---

## Q7 · "What are the limitations?" *(have this ready — it builds credibility)*

1. **Not comparable to published MIMIC-CXR numbers.** Custom split, cleaned
   references, cardiomegaly enriched to 50.4%. I will not claim SOTA and any
   head-to-head against published figures would be dishonest.
2. **The AP/PA gap is not fixed** — it is *measured, proven irreducible at the
   model level, and surfaced to the user.* The deferral policy manages it; it does
   not remove it.
3. **Deferral costs coverage.** 88.0% accuracy applies to the 81% of cases the
   system answers. It is never quoted alone.
4. **Grad-CAM is a sanity check, not localisation evidence.** Published
   repeatability on chest X-rays is SSIM 0.12 (Arun et al., *Radiology: AI* 2021).
5. **Single dataset, single centre.** No external validation.
6. **Retrospective.** No prospective or reader study.

---

## Q8 · "What did you *newly use*?" *(supervisor's framing)*

**Data:** per-image z-score replacing ImageNet normalisation (measured 4.4× better
for grayscale) · target-level cleaning · text-adjudicated label fusion (beats
official CheXpert 8/8) · uncertainty downweighting · no h-flip (laterality is
diagnostic)

**Training:** `pos_weight` clamped at 8 · EMA weight averaging · bf16 +
`channels_last` · automatic batch-size probing · cosine schedule with warmup ·
gradient clipping · non-finite-loss abort · partial unfreezing at 0.1× LR · label
dropout · atomic checkpointing with auto-resume

**Architecture:** `inputs_embeds` encoder fix · BioBART over BART · shared vision
encoder · zero-parameter prompt conditioning

**Inference:** per-class F1-optimal thresholds fitted on validation ·
**per-projection operating points** · **projection-conditional deferral**

**Evaluation:** constant-string control · always-say-no baseline · cluster +
stratified bootstrap · four-arm ablation with a Platt null · greedy-vs-beam
ablation · CheXbert validation · **141 unit tests** · seed control across 14 files

> ⚠️ **I do not use ensembling.** If asked, say so plainly — don't claim it.

---

## The closing line

> "I didn't set out to build a better model — I set out to find out whether the
> thing everyone measures is real. It isn't. I have five failed experiments and two
> surviving ones to show for it, and I'd rather hand you the five than pretend they
> never happened."

---

## Numbers to have memorised

| | |
|---|---|
| Test set | **4,722** X-rays |
| Cardiomegaly accuracy / sensitivity | **83.2% / 92.3%** |
| Cardiomegaly AUROC | **0.9189** |
| Report accuracy (cardiomegaly) | **80.4%** |
| Disparity reduction | **73.3%** (published method: 46.7%) |
| Cost of that reduction | **zero** — AUROC spread 0.00e+00 |
| Hallucination | **70.70% → 0.0000** |
| Hypotheses I disproved | **5** |
| Unit tests | **141** |
