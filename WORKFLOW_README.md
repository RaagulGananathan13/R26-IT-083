# How the system works

**R26-IT-083 — Explainable AI for Cardiovascular Disease Detection and Diagnosis**

A plain-language companion to [`CLINICAL_WORKFLOW.md`](CLINICAL_WORKFLOW.md), which
carries the guideline citations and the exact decision rules. This file explains
the same thing in the order a patient experiences it.

> ⚕️ **Research prototype. Not a medical device.** Nothing here is clinically
> validated and nothing here is a diagnosis. Every output requires review by a
> qualified clinician.

---

## The short version

A patient arrives at the emergency department with chest pain. The system checks
them in **six steps, in order** — and sometimes stops early, because a result can
make the next test pointless.

```
 1. T + 0 min      Notes and vitals      Is this a possible heart problem?
 2. T + 10 min     Heart tracing (ECG)   Is there an infarct pattern right now?
 3. T + 15–60 min  Chest X-ray           What else could be killing them?
 4. T + 1–6 h      Blood test            Does the biomarker confirm it?
 5. T + 6–24 h     Heart scan (echo)     How well is the pump working?
 6. T + 24 h       Everything known      Which kind of heart attack was it?
```

**Three results stop the workup immediately:**

| Stop | Why |
|---|---|
| Heart attack on the ECG | Reperfusion. Door-to-balloon is a 90-minute target; mortality is 8 % under it against 20 % over. Nothing may delay it. |
| Collapsed lung on the X-ray | Needs decompression now. The answer was never the heart. |
| Lowest risk band at triage | The chest-pain fast track is not entered at all. |

---

## Step by step, with what each model can name

### 1 · Triage assessment — Component 04
**T + 0 min · nothing has been ordered yet**

Vitals, chief complaint, history. Four outcomes:

| Class | What it means |
|---|---|
| `No_ACS` | Not a coronary problem. |
| `UA` | **Unstable angina** — the heart is starving for blood, but not yet damaged. |
| `NSTEMI` | A smaller heart attack. Some muscle damage. |
| `STEMI` | A large heart attack. An artery is fully blocked. |

At this point the laboratory channel carries **exactly 0.0 % attribution** — no
troponin exists ten seconds after someone walks through the door, and the
component proves it does not use one.

**→ Lowest band exits. Everything else goes to the ECG.**

---

### 2 · 12-lead ECG — Component 02
**T + 10 min · this is a guideline deadline, not a target**

Five patterns:

| Class | What it means |
|---|---|
| `NORM` | Normal tracing. |
| `MI` | **Myocardial infarction** — heart attack pattern. |
| `STTC` | ST/T change — the trace is the wrong shape; the heart is stressed. |
| `CD` | Conduction disturbance — the electrical wiring is slow or blocked. |
| `HYP` | Hypertrophy — the heart muscle has thickened. |

The **quality gate runs before the classifier**, so a trace that fails quality
control produces no probability at all. That is deliberate: an uninterpretable
ECG must never be able to return a reassuring number.

**→ `MI` stops everything. Anything else continues.**

---

### 3 · Chest radiograph — Component 01
**T + 15–60 min · usually a portable film at the bedside**

Eight findings from one forward pass. They do two different jobs:

| Class | Job |
|---|---|
| `Cardiomegaly` | **The heart is enlarged** → go straight to the echo |
| `Edema` | Fluid in the lungs — the heart is congesting them → echo |
| `Pneumothorax` | **Collapsed lung** → stop, this is the emergency |
| `Pleural Effusion` | Fluid around the lung |
| `Pneumonia` | Lung infection |
| `Consolidation` | A solid patch of lung |
| `Atelectasis` | A collapsed segment |
| `Lung Opacity` | A non-specific cloudy area |

The last five are **mimics** — chest pain that is not the heart at all. This
step exists to find them.

**→ Collapsed lung stops. Enlarged heart or fluid jumps to the echo. Otherwise
continue to the blood test.**

---

### 4 · Troponin — Component 04 again
**T + 1–6 h · the same record, re-scored**

A damaged heart leaks troponin into the blood. Three arms:

- **Rule out** → discharge pathway
- **Rule in** → confirmed; admit
- **Observe zone** → neither. About **40 % of patients land here**, and that is
  exactly who the echo is for.

---

### 5 · Echocardiogram — Component 03
**T + 6–24 h · needs a sonographer, so it is scheduled**

Measures **ejection fraction** — the share of blood the left ventricle pushes
out each beat. A healthy heart pushes out more than half.

| Grade | Ejection fraction | What it means |
|---|---|---|
| `Severe` | under 30 % | Pump is very weak |
| `Moderate` | 30–40 % | Pump is weak |
| `Mild` | 40–55 % | Pump is mildly impaired |
| `Normal` | 55 % and above | Pump is fine |

**40 % is the clinically decisive line.** Below it, a parallel heart-failure
pathway opens alongside the coronary one.

---

### 6 · Final subtype — Component 04, third and last time
**T + 24 h · everything is now knowable**

The same four classes as step 1 — but now with the whole workup behind them.

---

## Why Component 04 runs three times

This is the single most important idea in the pathway, and it is measured rather
than asserted.

**Unstable angina is *defined* by a normal troponin.** You cannot separate it
from NSTEMI until the biomarker comes back. So the same model, on the same
patient, gets better purely because more information exists:

| When it runs | Unstable angina correctly identified |
|---|---|
| Step 1 — at the door | **37.3 %** |
| Step 4 — after the biomarker | **58.2 %** |
| Step 6 — workup complete | **80.0 %** |

That climb is not the model improving. **It is the blood test arriving.** The
horizon curve recovers a clinical fact from the data rather than being told it.

---

## What ties the four together

The four components answer different questions from different data and share no
findings — a cardiomegaly probability and an ejection fraction have nothing in
common. What they share is that **each was built around a mechanism that
declines to commit when its own evidence is weak.**

| Component | How it says "I am not sure" |
|---|---|
| 01 · Radiograph | Per-projection operating points; borderline bedside films are deferred |
| 02 · ECG | Quality gate before any probability; guarantees withdrawn on electrode reversal or an out-of-scope rhythm |
| 03 · Echo | Calibrated prediction interval, learned measurement noise, disagreement between clips |
| 04 · Triage | Declared information horizon; clinician referral below a frozen confidence |

The backend normalises all four into **one verdict** — `actionable`, `caution`,
`deferred`, `withheld`, `unavailable` — so one rule applies everywhere:

> **Do not act on a result that is not `actionable`.**

Findings marked `withheld` are **not rendered at all**, because showing a
suppressed probability beside a warning invites it to be used anyway.

---

## What each component shows for its answer

Every component now returns a per-case explanation, not just a number. They are
different kinds of object because the four data types are different, but each
one answers the same question: *why this patient, this study, this reading?*

| Component | What you see | What it is |
|---|---|---|
| **01 · Radiograph** | Heat map over the film, with an original/overlay/heatmap toggle | Grad-CAM on the final convolutional stage, for the named finding |
| **02 · ECG** | A curve over the ten-second strip, with the attention peaks marked | Temporal Grad-CAM. Its **shape** is the point: one sharp spike means the call rests on a single complex, a broad plateau means it does not |
| **03 · Echo** | Heat map on the strongest frames, plus importance across the clip | Grad-CAM on the last spatiotemporal convolution, taken against the **continuous** ejection fraction rather than the grade |
| **04 · Triage** | Signed bars per feature, and a breakdown by evidence channel | SHAP for this one patient, over the two gradient-boosted models behind P(ACS) |

Each carries the limit of what it can support, because a saliency map is the
easiest thing in the system to over-read:

- **Radiograph** — Grad-CAM repeatability on chest films was measured at SSIM
  0.12. It is a check on *where the model looked*, never localisation evidence.
- **ECG** — the peak times and the curve are computed by different code paths.
  That they agree is a cross-check; it is stated so the agreement is checkable.
- **Echo** — the map is computed at **4 × 7 × 7** and interpolated up to
  32 × 112 × 112 for display. The smooth edges are interpolation, not evidence,
  and the payload says so. It also names **which clip** it came from: the
  reported EF is a mean over ten clips and both ensemble members, and a
  single-clip map does not explain that mean.
- **Triage** — averaging SHAP across the two models attributes the mean margin,
  not the calibrated probability. It is a direction of evidence, not an additive
  decomposition.

### The one worth showing live

Component 04's channel breakdown is computed per patient, and at stage 1 the
**laboratory channel reads exactly 0.0 %**:

| Horizon | Laboratory attribution, this patient | Published cohort figure |
|---|---|---|
| H = 0 · at the door | **0.000 %** | 0.0 % |
| H = 6 · after the first troponin | 2.500 % | 4.6 % |
| H = 24 · workup complete | 18.980 % | 29.6 % |

The information horizon is not asserted from a table here — it is enforced, per
patient, and visible while the panel watches. A pipeline with a temporal leak
cannot produce a zero in the top row.

---

## Models per component

| Component | Architectures trained | Notes |
|---|---|---|
| **01 · Radiograph** | **2** — ConvNeXt classifier, BioBART report generator | Two models, two tasks: one classifies, one writes the report, conditioned on the first |
| **02 · ECG** | **3** — `resnet_se`, `resnet`, and the `no_se` ablation, three seeds each | Component-wise ablation: the stem and attention pooling change nothing (p = 0.74), while squeeze-excitation **loses** 0.0042 AUROC (p = 0.0040), almost all of it on hypertrophy |
| **03 · Echo** | **2** — R(2+1)D-18 and R3D-18, three seeds each | Both fully trained and scored on the same test split: R(2+1)D MAE 3.979 / R² 0.818 against R3D 4.033 / 0.812 |
| **04 · Triage** | **2** — LightGBM and XGBoost, ensembled | Both at stage 1 and stage 2, with isotonic calibration, at three horizons |

**Component 02 now has the sharpest architecture result.** Three architectures at
three seeds each, compared by paired bootstrap on the untouched test fold, showed
that its own three additions do not earn their 566k parameters — and isolated
squeeze-excitation as the single component responsible, on the one class whose
evidence is amplitude. See `Component_02/audit/architecture_comparison/FINDINGS.md`.

---

## The research posture the four components share

The components share no findings and no code, and their individual contributions are
graded differently. What they do share is a habit, and it is worth stating as a claim in
its own right:

**Every component measured its own failure and reported it.**

| Component | What it found against itself |
|---|---|
| **01 · Radiograph** | Three interventions were tried to close the AP/PA accuracy gap. All three failed, and the negative result is published rather than buried. The gap is closed by deferring more bedside films to a human — not by claiming the model improved. |
| **02 · ECG** | The conformal guarantee holds marginally and **breaks for identifiable subgroups**. Measured, significance-tested, and fixed with group-conditional calibration — with the coverage cost of that fix also reported. |
| **03 · Echo** | The backbone was inherited from a benchmark. It was tested against the un-factorised baseline at three matched seeds, and the first attempt at that test was found to be confounded and re-run. |
| **04 · Triage** | A previously published AUROC of 0.9889 was traced to temporal leakage, reproduced under controlled conditions, and **retracted**. The honest figure is 0.9560. |

A four-person project where each member independently found and reported a limitation in
their own work is a stronger claim than any single accuracy number in it.

---

## ⚠️ Prior art — verify before presenting

Each component's novelty write-up now names the closest prior art rather than leaving it
for a reviewer to find. **These citations were added from recall and have not been
verified against the published record.** Check each one before it goes on a slide:

| Component | Must cite | Why |
|---|---|---|
| 01 | **Hardt, Price & Srebro, NeurIPS 2016** | Per-group thresholds *are* their post-processing method. Claiming the technique would be fatal. |
| 01 | **Jones et al., ICLR 2021** | Selective classification can *magnify* group disparities — the closest adjacent result, and complementary to this one. |
| 02 | **Vovk, ACML 2012**; **Barber et al., 2021** | Conditional validity, and its impossibility in general, are established theory. The measurement on ECG is what is new. |
| 03 | **Díaz & Marathe, CVPR 2019** | Soft ordinal targets are theirs. Deriving the width from label measurement noise is the refinement. |

---

## What this pathway does not establish

- **No component diagnoses anything.** Four decision-support outputs, each
  requiring clinician review.
- **The four components share no patients.** MIMIC-CXR, PTB-XL,
  EchoNet-Dynamic/CAMUS and MIMIC-IV-ED are four separate cohorts. This is how
  the components *would* compose clinically, justified against published
  guidelines — not a validated end-to-end study on one population.
- **No joint model was trained** across the modalities, and no combined accuracy
  is claimed. Every figure belongs to exactly one component.
- **Component 02 recognises five superclasses.** Its `MI` class does not separate
  STEMI from NSTEMI, and atrial fibrillation and other arrhythmias are outside
  the label space entirely.
- **Component 04's UA/NSTEMI boundary rests on ICD coding**, not adjudicated
  labels.

---

## Running it

Two terminals, from the repository root:

```bash
cd backend  && python run.py --warm      # http://127.0.0.1:8000
cd frontend && npm run dev               # http://localhost:3001
```

Open **Clinical pathway** in the console. Pick a sample record, press
**Begin the pathway at stage 1**, then run each stage in turn — attaching the
study for that stage as you reach it.
