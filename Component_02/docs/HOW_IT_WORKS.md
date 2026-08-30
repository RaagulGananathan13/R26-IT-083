# How It Works — The Whole System, From Scratch

**Venushan T** · Component 02 · XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting

*This document assumes no background in machine learning or cardiology. It
explains what the data is, what the models are, how a diagnosis is produced, and
why the safety layers exist. §6 is the deep explanation of conformal prediction —
the idea the whole contribution rests on.*

For the research framing and novelty claims see [CONTRIBUTION_FINAL.md](CONTRIBUTION_FINAL.md).
For the defect audit that motivated the rebuild see [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md).

---

## Contents

1. [The data — what an ECG is to a computer](#1-the-data)
2. [What the system predicts](#2-what-the-system-predicts)
3. [What a "model" actually is](#3-what-a-model-actually-is)
4. [The two models](#4-the-two-models)
5. [The eight steps of a diagnosis](#5-the-eight-steps)
6. [**Conformal prediction, explained properly**](#6-conformal-prediction-explained-properly)
7. [Why two models](#7-why-two-models)
8. [What is standard and what is mine](#8-what-is-standard-and-what-is-mine)
9. [Glossary](#9-glossary)
10. [Where everything lives](#10-where-everything-lives)

---

## 1. The data

The heart produces small electrical signals each time it beats. An ECG machine
places electrodes on the body and records them.

**12-lead** means twelve simultaneous recordings from twelve different angles —
like filming one object with twelve cameras placed around it. The leads are
named I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. A problem on the front
wall of the heart appears clearly in some leads and barely in others. That is
exactly why twelve exist: **the lead that lights up tells you *where* the problem
is.**

Each recording is **10 seconds long, sampled 500 times per second**:

```
500 samples/second × 10 seconds = 5,000 numbers per lead
5,000 numbers × 12 leads        = 60,000 numbers per patient
```

To the computer one patient's ECG is a grid of 60,000 numbers, shape `(12, 5000)`.
That grid is the input. Everything after this is arithmetic on it.

**The dataset — PTB-XL.** 21,799 real ECGs from a German hospital, recorded
1989–1996, each labelled by a cardiologist. This project uses 17,221 of them,
split by the dataset's own official scheme:

| Split | Folds | Records | Purpose |
|---|---|---|---|
| Train | 1–8 | 13,801 | the model learns from these |
| Validation | 9 | 1,709 | **calibration** — see §6 |
| Test | 10 | 1,711 | final evaluation, touched last |

No patient appears in more than one split. That matters: if the same person's
ECG were in both training and test, the model could score well by memorising the
patient rather than learning the disease.

---

## 2. What the system predicts

Five categories, called **superclasses**:

| Code | Plain English | What it means clinically |
|---|---|---|
| **NORM** | Normal ECG | nothing abnormal detected |
| **MI** | Myocardial infarction | heart attack — current or past damage |
| **STTC** | ST/T change | strain or reduced blood supply to the muscle |
| **CD** | Conduction disturbance | the heart's electrical wiring is not carrying signals properly |
| **HYP** | Ventricular hypertrophy | heart muscle wall has thickened |

**This is multi-label, not multi-class.** The system does not pick one winner. It
answers **five independent yes/no questions**, because a real patient can have
several conditions at once:

```
NORM: 0.12    MI: 0.81    STTC: 0.44    CD: 0.09    HYP: 0.03
```

Five numbers between 0 and 1. That is the model's raw output — and on its own it
is not yet safe to show anyone. §5 and §6 explain why.

---

## 3. What a "model" actually is

A neural network is **a mathematical function with adjustable knobs**. It takes
60,000 numbers in and produces 5 numbers out. It begins random and useless.

**Training** means: show it thousands of ECGs whose answers are already known,
measure how wrong it is, and nudge the knobs slightly in the direction that
reduces the error. Repeat millions of times. This model has roughly 1.5 million
knobs (called *parameters* or *weights*).

### Why a CNN

For signals, the standard tool is a **1D convolutional neural network**.

A **convolution** is a small pattern-detector — say 15 samples wide — that slides
along the signal from start to finish, asking at every position: *"does the shape
here match the thing I have learned to look for?"* One detector might learn to
fire on the sharp QRS spike of a heartbeat. Another on a drooping ST segment.

The network learns what those detectors should look for. Nobody programs them.

Stack them in layers and they compose:

```
layer 1  →  tiny shapes: edges, spikes, slopes
layer 2  →  parts of a heartbeat: the QRS complex, the T wave
layer 3  →  relationships between parts, and across leads
layer 4  →  "this combination across these leads indicates infarction"
```

Each layer also **halves the length** (`stride=2` in the code), so 5,000
timepoints compress step by step into a short summary. A final small layer turns
that summary into the five probabilities.

### What "ResNet" means

Each layer has a **shortcut** that lets the signal skip past it. Without
shortcuts, deep networks train badly — the correction signal fades before it
reaches the early layers. It is a fix for a training problem, not a clinical
feature. Every modern deep network has something like it.

---

## 4. The two models

Both are served, and they are genuinely different networks.

### `resnet` — the baseline · 1,018,501 parameters

The Progress-1 model. Four residual blocks, detector widths 15 → 7 → 5 → 3, then
**average pooling** — it averages the whole 10 seconds into one summary.

### `resnet_se` — the improved model · 1,584,326 parameters

Same skeleton plus three additions:

| Addition | What it does, plainly | Why it should help an ECG |
|---|---|---|
| **Multi-kernel stem** | looks at three detector widths at once — 7, 15 and 31 samples | the sharp QRS spike and the slow T wave live at different time-scales; one width cannot see both well |
| **Squeeze-Excitation (SE)** | after each layer, asks *"which of my detectors matter for THIS patient?"* and turns useful ones up, irrelevant ones down | a volume knob per feature, set per patient rather than fixed |
| **Attention pooling** | learns which moments to weight instead of averaging everything equally | a 1-second abnormality gets diluted by averaging across 10 seconds; attention keeps it |

### Do the additions actually help?

Honestly: **the accuracy difference is small and mostly within noise.** macro-AUROC
0.9297 → 0.9343. Only AUPRC clears run-to-run variation. PTB-XL has sat at
0.92–0.94 for years and this project does not claim to beat it.

The registry in [models.py](../src/models.py) now also defines four **ablation
variants** — `resnet_se_no_se`, `resnet_se_no_stem`, `resnet_se_no_attn`,
`resnet_se_plain` — which switch each addition off individually so the question
*"does the SE block actually earn its place?"* can be answered with evidence
rather than assertion. Parameter counts stay within 3% of each other, so any
difference is attributable to the component and not to model size.

**Why keep the old model rather than delete it?** Because two models that
disagree tell you something one model cannot. See §7.

---

## 5. The eight steps

The pipeline in [pipeline.py](../src/pipeline.py). The order is fixed and nothing
skips ahead.

```
quality gate → preprocess → classify → calibrate → triage → explain → report → verify
```

### Step 1 — Quality gate · [quality.py](../src/quality.py)

**Before the model sees anything**, check the signal is interpretable at all.

- Is a lead flat? (an electrode fell off)
- Are the units wrong? (microvolts recorded as millivolts — a factor of 1000)
- Are the limb electrodes swapped? (left arm / right arm reversed)
- Is it pure noise?

If it fails, the system **refuses** and produces no diagnosis.

**Why this exists:** the audit found the old system took a completely flat signal
— a disconnected patient — and reported **"myocardial infarction, probability
0.691"**. A neural network always outputs *something*. It has no concept of "this
is not an ECG." The gate is what gives it one.

### Step 2 — Preprocess · [preprocess.py](../src/preprocess.py)

Band-pass filter (remove slow baseline drift and high-frequency muscle noise),
resample to 500 Hz, normalise to the scale the model was trained on.

> `resnet` was trained **without** filtering; `resnet_se` **with** it. Serving a
> model the wrong preprocessing degrades it silently — it moved calibration error
> from 0.183 to 0.209 during development. Each model therefore carries its own
> preprocessing setting, read from its own artefacts.

### Step 3 — Classify

60,000 numbers in, 5 raw scores out, squashed into the range 0–1.

### Step 4 — Calibrate · [calibration.py](../src/calibration.py)

**A model's "80%" usually does not mean 80%.**

The audit found the old model predicted hypertrophy at **4.14× its true
frequency**. Every confidence number a clinician would have seen was wrong. The
cause was mundane: class imbalance was corrected twice during training.

**Temperature scaling** fixes it. Divide each class's scores by a learned number
until, measured across hundreds of patients, "70%" comes true about 70% of the
time. Calibration error dropped **0.183 → 0.018**.

It only *rescales* — it never reorders patients — so it cannot break the
guarantees established next.

### Step 5 — Conformal triage · [conformal.py](../src/conformal.py)

Turns probabilities into one of three decisions per class, with a mathematical
bound on how often the decision is wrong. **This is §6 — the whole next section.**

### Step 6 — Explain · [xai.py](../src/xai.py)

Two techniques answering two different questions:

- **Grad-CAM → *when*.** Which moments in the 10 seconds drove the decision.
  *"Strongest at 3.7 seconds."*
- **Integrated Gradients → *which lead*.** Ranks all 12 leads by contribution.
  Because leads correspond to physical regions, `V2` and `V3` dominating means
  **the front wall — LAD artery territory**.

The explanation is not decoration beside the diagnosis; its output becomes text
in the report.

### Step 7 — Write the report · [report.py](../src/report.py)

**No free-text AI writes this.** The old system used a language model that
invented "atrial fibrillation" in 42 patients — a condition this model cannot
even detect.

Instead the numbers fill **fixed templates**. If the classifier did not find
something, no sentence exists that can express it. Hallucination becomes
structurally impossible rather than merely unlikely.

### Step 8 — Verify · [verify.py](../src/verify.py)

A final checker reads the finished report and compares every claim against the
actual findings. Anything added, dropped or invented → the report is withheld and
manual interpretation is demanded.

Result: **200/200 reports pass, 0 self-contradictory.** The old system
contradicted itself — asserting NORM *and* an abnormality — in 5.8% of reports.

### The whole journey

```
  raw ECG file (.dat / .hea)
        │
        ▼
  [ QUALITY GATE ] ──── flat / noisy / wrong units ──▶ REFUSED, no diagnosis
        │ acceptable
        ▼
  [ PREPROCESS ]     filter, resample, normalise
        │
        ▼
  [ CNN MODEL ]      60,000 numbers ──▶ 5 raw scores
        │
        ▼
  [ CALIBRATE ]      make "70%" actually mean 70%
        │
        ▼
  [ CONFORMAL ]      each class ──▶ rule out │ refer │ rule in
        │
        ▼
  [ EXPLAIN ]        when (Grad-CAM) + which lead (Integrated Gradients)
        │
        ▼
  [ REPORT ]         fill templates — never free text
        │
        ▼
  [ VERIFY ]         block anything unsupported
        │
        ▼
  clinical report + triage level + disclaimer
```

---

## 6. Conformal prediction, explained properly

This is the idea the contribution rests on. It is worth reading twice.

### 6.1 The problem with an ordinary threshold

The model outputs a probability. To make a decision you need a cutoff — say
"above 0.5 means the disease is present."

But **0.5 is arbitrary.** So is 0.4, or the value that maximises F1 score. None of
them tells you what the choice *costs*.

The audit measured that cost on the old system. At its chosen cutoff:

> **29.5% of real heart attacks in the test set were reported as absent** — with
> no indication that anything was uncertain.

Nearly one in three. And no published PTB-XL paper puts a bound on that number,
because F1 — the metric everyone reports — weighs a missed infarction and an
unnecessary referral **equally**. Clinically they are not remotely equal.

### 6.2 The flip

Conformal prediction reverses the question.

| | Ordinary threshold | Conformal |
|---|---|---|
| You choose | the cutoff | **the error rate you will accept** |
| You discover afterwards | the error rate | the cutoff |

You say: *"I will accept missing at most 5% of infarctions."* The data then tells
you where the cutoff must sit to honour that.

### 6.3 How the cutoff is found — worked example

Use **fold 9**, the calibration split. The model was never trained on it, so its
scores there behave like scores on genuinely new patients.

1. Take every patient in fold 9 who **truly has MI**. There are **283** of them.
2. Ask the model to score each one. It gets some right (high scores) and some
   wrong (low scores).
3. **Sort those 283 scores from lowest to highest.**

```
 lowest scores  ← the infarctions the model is worst at spotting
 0.01  0.02  0.04  0.05  0.07  0.09  ...  0.88  0.94  0.97
                            ▲
                     cut here → λ_out
```

4. You want to miss at most 5%. 5% of 283 is about 14. So cut at the **14th
   lowest score**. Call it `λ_out`.

Now the rule is: *any new patient scoring below `λ_out` is ruled out.*

**Why does that work?** Because among the calibration positives, only 14 of 283 —
about 5% — fell below that line. A new MI patient is, statistically, just another
draw from the same population. So the chance they land below the line is also
about 5%.

That is the entire mechanism. Counting and sorting.

### 6.4 Why it must be a separate fold

If you found `λ_out` using the training data, the model would already have seen
those patients and would score them unrealistically well. The cutoff would be far
too optimistic, and the promise would be a lie.

Fold 9 exists solely so the promise is measured on data the model has never met.
Fold 10 then confirms the promise held on data *neither* the model nor the
threshold has met.

```
folds 1–8   train the model
fold 9      find the thresholds      ← model never trained here
fold 10     check the promise held   ← nothing ever touched this
```

### 6.5 What "distribution-free" means

This is the strong part. The argument in §6.3 used **no assumption**:

- not that the model is accurate
- not that the probabilities are meaningful
- not that the data follows a bell curve or any other shape

Only that **new patients resemble calibration patients** (formally:
*exchangeability*).

So the guarantee holds for *any* model. A weak model does not break it — a weak
model simply produces a very low `λ_out` and ends up ruling almost nothing out.
**The guarantee degrades into caution, never into false confidence.** That is
exactly the property you want in a clinical system.

### 6.6 The three zones

One cutoff gives two answers. This system uses **two** cutoffs and gives three:

```
   score
     0 ─────────── λ_out ─────────── λ_in ─────────── 1
        🟢 RULE OUT    🟡 REFER        🔴 RULE IN
```

| Zone | Meaning | The promise |
|---|---|---|
| 🟢 **RULE OUT** | safe to say the disease is absent | misses at most **α** of true cases |
| 🟡 **REFER** | **"I don't know"** — a cardiologist must look | none needed; a human decides |
| 🔴 **RULE IN** | confident the disease is present | false alarms bounded by **β** |

**The yellow zone is the new part.** The old system had only yes and no — it was
*forced* to answer every case, including the ones it had no business answering.

Letting the machine say *"I am not sure"* is what produced the headline result:

> **Missed infarctions that never reach a clinician: 29.5% → 1.5%.**

`λ_out` and `λ_in` are found separately per class, because the acceptable miss
rate differs by disease. Missing a heart attack is far worse than missing
hypertrophy, so MI gets α = 0.05 while HYP gets α = 0.15.

### 6.7 The cost, stated honestly

The safety comes from referring the uncertain cases to a human. **About half of
patients (50.9%) land in the yellow zone.**

That is a real workload cost and the project reports it rather than hiding it.
Other operating points exist — the `balanced` preset refers 28.5% at a higher
miss rate. The deliverable is not a single number; it is **the whole trade-off
curve**, so a hospital can choose its own point.

### 6.8 One refinement: δ (delta)

The §6.3 argument gives a promise that is true **on average over many possible
calibration sets**. But you only have one calibration set, and you might have
been unlucky with it.

Setting **δ = 0.01** buys a stronger form: *"with 99% confidence, the miss rate is
below α — for this specific calibration set."* It costs a slightly more
conservative threshold. In this system, δ = 0.05 left 2 of 5 guarantees violated;
δ = 0.01 held all five.

### 6.9 What conformal does **not** promise — and the contribution

This is the point the research contribution turns on, so read it carefully.

The guarantee is **marginal**. It holds *on average across the whole test
population*.

**A cardiologist never treats the average.** They treat a 42-year-old with
palpitations, or a 78-year-old woman with atypical chest pain. And nothing in the
marginal guarantee prevents the misses from piling up in one of those groups.

Testing this on PTB-XL produced the finding:

| Class | Promised | **Overall** | Under 50 | 70+ |
|---|---|---|---|---|
| CD | ≤ 0.10 | 0.099 ✓ | **0.333** ✗ | 0.042 |
| NORM | ≤ 0.20 | 0.190 ✓ | 0.103 | **0.330** ✗ |

> **Conduction disturbance, patients under 50: promised at most 10% missed,
> delivered 33.3%. Three times the advertised bound — and the overall figure of
> 9.9% gives no hint of it.**

Conduction disease in a young adult raises Brugada syndrome, ARVC and inherited
conduction disease — causes of sudden cardiac death in the young. It is precisely
the group where a miss is least acceptable, and precisely where the system fails
worst.

**The fix** is group-conditional (*Mondrian*) calibration — fit a separate
threshold per subgroup instead of one for everybody. It raises the cells that
satisfy the bound from 14/23 to 22/23.

**The fix is not free**, which is the second finding: each group needs enough
calibration positives of its own. STTC in under-50s had only 42 and returned
λ = −∞ — meaning that group **can never be certified at all**. Conditional
validity costs calibration data, and the groups that need it most tend to have
the least.

---

## 7. Why two models

Each model carries **its own** calibrator and **its own** conformal thresholds.
They cannot borrow each other's guarantees — a calibrator is only valid for the
scores it was fitted on. [zoo.py](../src/zoo.py) enforces this.

When both run on the same patient, the rule is:

> **A disease is ruled out only if EVERY model rules it out. Any disagreement
> becomes REFER.**

### Why that is safe, not just cautious

This is not a guess-and-hope heuristic. A true case is missed by the combined
rule only if **all** models miss it. So:

```
P(combined rules out | patient has the disease)
      ≤  min over models of  P(that model rules out | patient has it)
      ≤  min over models of  α
```

The combined miss rate is bounded by the **tightest** single-model guarantee.

### Measured on all 1,711 test records

| Disease | Missed by `resnet` | Missed by `resnet_se` | **Combined** |
|---|---|---|---|
| NORM | 4.4% | 3.3% | **1.8%** |
| MI | 0.7% | 1.5% | **0.0%** |
| STTC | 7.7% | 9.2% | **5.5%** |
| CD | 10.6% | 9.9% | **7.0%** |
| HYP | 9.8% | 12.1% | **5.3%** |

The combined rule misses fewer true cases than *either* model on every class,
because the two models do not make the same mistakes. The price is more
referrals (for example CD: 11.8% → 17.8%).

### The number worth reporting

> **The two models disagree on at least one class in 58.9% of records, and reach
> opposite conclusions — one ruling a disease in while the other rules it out —
> in 10.5% (180 of 1,711).**

A single-model deployment shows the clinician one of those two answers, with a
mathematical guarantee attached, and no indication the other exists.

Reproduce with `python -X utf8 audit/14_multi_model.py`.

---

## 8. What is standard and what is mine

| Piece | Status |
|---|---|
| 1D ResNet on PTB-XL | **Standard** — published many times |
| ~0.93 macro-AUROC | **Standard** — the field has sat here for years; not a claim |
| Conformal prediction as a method | **Existing** (Vovk; Angelopoulos et al.) |
| Conformal applied to ECG | **Existing** — 2025 prior art, cited |
| Grad-CAM, Integrated Gradients | **Existing methods** |
| Quality gate, calibration, report verifier | **My engineering** — closing 12 audited defects |
| **Subgroup validity of ECG conformal guarantees** | **Mine** — no prior work |
| **Marginal validity hiding a 3× subgroup violation** | **Mine** — measured, n reported |
| **Mondrian as the remedy, with its data cost quantified** | **Mine** (method is Vovk 2003; the ECG application and cost analysis are mine) |
| **Two-model agreement gate + the 10.5% opposite-conclusion rate** | **Mine** |

**The one-line version:**

> *"I didn't try to beat a benchmark that has been stuck for years. I asked a
> different question — what has to be true before you would let this near a real
> patient? It has to know when it is unsure, and it has to be able to prove how
> often it is wrong."*

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **Lead** | one of the 12 viewing angles of the heart |
| **Parameter / weight** | one adjustable knob inside the network (~1.5 million here) |
| **Convolution** | a small pattern-detector slid along the signal |
| **Stride 2** | the layer halves the signal length as it processes it |
| **Residual / skip connection** | a shortcut past a layer, so deep networks train properly |
| **Multi-label** | several answers can be true at once (not "pick one") |
| **AUROC** | a 0–1 score for how well the model ranks sick above healthy; 0.5 = coin flip |
| **AUPRC** | like AUROC but honest about rare diseases — the better metric here |
| **Calibration** | making "70%" actually mean 70% |
| **ECE** | a number measuring how badly calibration is off (lower is better) |
| **Conformal prediction** | choosing the error rate first and letting data find the cutoff |
| **α (alpha)** | the miss rate you agree to accept for a class |
| **δ (delta)** | confidence that the promise holds for *your* calibration set |
| **λ_out / λ_in** | the two cutoffs defining rule-out and rule-in |
| **Marginal guarantee** | true on average across everyone |
| **Conditional guarantee** | true *within* each subgroup — the harder, safer version |
| **Mondrian conformal** | a separate threshold per subgroup |
| **Exchangeability** | new patients resemble calibration patients; the one assumption |
| **Grad-CAM** | highlights *when* in the recording the decision came from |
| **Integrated Gradients** | ranks *which leads* drove the decision |
| **Fold** | one slice of the dataset; PTB-XL ships 10 official ones |

---

## 10. Where everything lives

| File | Responsibility |
|---|---|
| [src/models.py](../src/models.py) | every architecture, plus the registry and ablation variants |
| [src/quality.py](../src/quality.py) | the refuse-if-uninterpretable gate |
| [src/preprocess.py](../src/preprocess.py) | filter, resample, normalise |
| [src/calibration.py](../src/calibration.py) | temperature scaling |
| [src/conformal.py](../src/conformal.py) | the three-zone risk-controlled triage |
| [src/xai.py](../src/xai.py) | Grad-CAM, Integrated Gradients, lead → territory mapping |
| [src/report.py](../src/report.py) | template-based report generation |
| [src/verify.py](../src/verify.py) | blocks any unsupported claim |
| [src/pipeline.py](../src/pipeline.py) | the eight steps, in order, for one model |
| [src/zoo.py](../src/zoo.py) | serving several models at once, each with its own safety layer |

**Run it:**

```bash
python -X utf8 backend/server.py          # API on :5000
cd frontend && npm run dev                # UI on :5173
```

**Check it:**

```bash
python -X utf8 audit/08_verify_fixes.py        # 26 checks, every defect closed
python -X utf8 audit/10_conditional_validity.py # the subgroup finding
python -X utf8 audit/14_multi_model.py          # two-model serving + evaluation
```

---

> **AI-generated decision support. NOT a medical device and NOT a diagnosis.**
> Every report requires review by a qualified clinician.
