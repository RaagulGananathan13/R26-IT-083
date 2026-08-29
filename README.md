# R26-IT-083

**Explainable AI System for Cardiovascular Disease Detection and Diagnosis**

Four independently developed research components behind one service and one
clinical console.

> ⚕️ Research prototype. **Not a medical device**, not clinically validated, and
> not for diagnosis or treatment decisions. Every output requires review by a
> qualified clinician.

---

## Contents

- [The components](#the-components)
- [The clinical pathway](#the-clinical-pathway)
- [What each component shows for its answer](#what-each-component-shows-for-its-answer)
- [The shared reliability contract](#the-shared-reliability-contract)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Running it](#running-it)
- [API](#api)
- [Datasets and weights](#datasets-and-weights)
- [Tests](#tests)
- [Research posture](#research-posture)
- [What this system does not establish](#what-this-system-does-not-establish)
- [Prior art to verify before presenting](#prior-art-to-verify-before-presenting)

---

## The components

| # | Owner | Modality | Answers | Dataset |
|---|---|---|---|---|
| **01** | Raagul Gananathan (IT22130020) | Chest radiograph | Cardiomegaly + 7 co-pathologies, Grad-CAM, draft report | MIMIC-CXR |
| **02** | Venushan T | 12-lead ECG | 5 superclasses with conformal rule-in / rule-out triage | PTB-XL |
| **03** | Dilukshan Viyapury (IT22219534) | Echocardiogram | Ejection fraction + 4-class severity grade | EchoNet-Dynamic + CAMUS |
| **04** | Abishnan J (IT22140234) | ED triage record | ACS detection + UA / NSTEMI / STEMI subtyping | MIMIC-IV-ED |

Each keeps its own weights, its own frozen decision rule and its own published
figures. The integration reproduces them unchanged and adds no probabilities of
its own.

### Headline measurements

Every figure below is on that component's own untouched test split.

| Component | Result |
|---|---|
| **01 · Radiograph** | Cardiomegaly **AUROC 0.9189** (95 % CI 0.9112–0.9265), sensitivity 92.3 %, specificity 74.0 %. Projection matters: **AP 0.8224 against PA 0.8864**. |
| **02 · ECG** | **macro-AUROC 0.9343 ± 0.0028**, macro-AUPRC 0.8001 ± 0.0029 over three seeds. Test fold 10, n = 1,711. |
| **03 · Echocardiogram** | **MAE 3.979 EF points**, R² 0.818, 73.0 % accuracy, 73.7 % balanced accuracy, minimum per-class recall 0.723. 99.7 % of predictions land within one severity class and there are **zero Severe↔Normal confusions**. Test n = 1,277. |
| **04 · ED triage** | **AUROC 0.9560** on the intended-use cohort (accuracy 0.9320, balanced accuracy 0.8758, minimum recall 0.8126); 0.9688 across the full ED population. |

### Models per component

| Component | Architectures trained | Note |
|---|---|---|
| **01** | **2** — ConvNeXt-Base 384² classifier, BioBART report generator | Two models, two tasks: one classifies, one writes the report, conditioned on the first |
| **02** | **3** — `resnet`, `resnet_se`, and the `no_se` ablation, three seeds each | See the architecture result below |
| **03** | **2** — R(2+1)D-18 and R3D-18, three seeds each | Both fully trained and scored on the same test split: R(2+1)D **MAE 3.979 / R² 0.818** against R3D **4.033 / 0.812**. Reproduce with `training/run_backbone_ablation.py` |
| **04** | **2** — LightGBM and XGBoost, rank-blended | Both at stage 1 and stage 2, isotonic-calibrated, at three disclosure horizons |

**Component 02 has the sharpest architecture result in the project.** Three
architectures at three seeds each, compared by paired bootstrap on the untouched
test fold, show that its own three additions do not earn their 566 k parameters:

| Comparison | Δ macro-AUROC | p | Verdict |
|---|---|---|---|
| `no_se` − `resnet` | +0.0006 | 0.7410 | the stem and attention pooling change nothing |
| `no_se` − `resnet_se` | +0.0042 | **0.0040** | squeeze-excitation **costs** accuracy |
| `resnet` − `resnet_se` | +0.0036 | **0.0316** | net effect of all three is a loss |

Almost the whole loss sits on hypertrophy (+0.0147 AUROC without SE), and there
is a mechanism rather than a coincidence: LVH is diagnosed from QRS *amplitude*,
squeeze-excitation recalibrates channels by learned importance, and that is an
operation on relative amplitude across leads. Full write-up in
`Component_02/Component_02/audit/architecture_comparison/FINDINGS.md`.

---

## The clinical pathway

A patient arrives with chest pain. The system checks them in **six steps, in
order** — and sometimes stops early, because a result can make the next test
pointless.

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

The routing rules and their guideline citations are in
[`CLINICAL_WORKFLOW.md`](CLINICAL_WORKFLOW.md); a plain-language walkthrough with
the disease classes named is in [`WORKFLOW_README.md`](WORKFLOW_README.md).

### Why Component 04 runs three times

Unstable angina is *defined* by a normal troponin. You cannot separate it from
NSTEMI until the biomarker comes back, so the same model, on the same patient,
gets better purely because more information exists:

| When it runs | Stage-1 AUROC | Unstable angina correctly identified |
|---|---|---|
| Step 1 — at the door | 0.8763 | **37.3 %** |
| Step 4 — after the biomarker | 0.9121 | **58.2 %** |
| Step 6 — workup complete | 0.9560 | **80.0 %** |

That climb is not the model improving. **It is the blood test arriving.**

Figures as published in `Component_04/README.md` §C2. The per-class entries in
`artifacts/reports/evaluation_H*.json` are broken out by decision head
(`cascade_test`, `joint_test`) and will not match this row for row.

---

## What each component shows for its answer

Every component returns a per-case explanation, not just a number. The console
places explainability on the left and the written analysis on the right,
because checking one against the other is the point.

| Component | What you see | What it is |
|---|---|---|
| **01 · Radiograph** | Heat map over the film, original/overlay/heatmap toggle | Grad-CAM on the final convolutional stage, for the named finding |
| **02 · ECG** | A curve over the ten-second strip with the attention peaks marked | Temporal Grad-CAM. Its **shape** is the point: one sharp spike means the call rests on a single complex, a broad plateau means it does not |
| **03 · Echo** | Heat map on the strongest frames, plus importance across the clip | Grad-CAM on the last spatiotemporal convolution, against the **continuous** ejection fraction rather than the grade |
| **04 · Triage** | Signed bars per feature, and a breakdown by evidence channel | SHAP for this one patient, over the two gradient-boosted models behind P(ACS) |

Each carries the limit of what it can support, because a saliency map is the
easiest thing here to over-read:

- **Radiograph** — Grad-CAM repeatability on chest films was measured at SSIM
  0.12. It is a check on *where the model looked*, never localisation evidence.
- **ECG** — the peak times and the curve come from different code paths. That
  they agree is a cross-check, and it is stated so the agreement is checkable.
- **Echo** — the map is computed at **4 × 7 × 7** and interpolated up to
  32 × 112 × 112 for display. The smooth edges are interpolation, not evidence.
  It also names **which clip** it came from: the reported EF is a mean over ten
  clips and both ensemble members, and a single-clip map does not explain a mean.
- **Triage** — averaging SHAP across the two models attributes the mean margin,
  not the calibrated probability. A direction of evidence, not an additive
  decomposition.

For bundled radiographs the console also shows **the report the radiologist
actually dictated** for that exact image, beside the generated one. An upload
the system does not recognise shows no ground truth rather than another
patient's report.

### The one worth watching live

Component 04's channel breakdown is computed per patient, and at stage 1 the
**laboratory channel reads exactly 0.0 %**:

| Horizon | This patient | Published cohort figure |
|---|---|---|
| H = 0 · at the door | **0.000 %** | 0.0 % |
| H = 6 · after the first troponin | 2.500 % | 4.6 % |
| H = 24 · workup complete | 18.980 % | 29.6 % |

The information horizon is enforced per patient rather than asserted from a
table. **A pipeline with a temporal leak cannot produce that zero.**

---

## The shared reliability contract

The four answer different clinical questions from different data and share no
findings — a cardiomegaly probability and an ejection fraction have nothing in
common. What they share is that **each was built around a mechanism that
declines to commit when its own evidence is weak**:

| Component | Its honesty mechanism |
|---|---|
| 01 | Per-projection operating points and selective deferral (AP AUROC 0.8224 vs PA 0.8864) |
| 02 | Quality gate → refusal before any probability exists; conformal zones; guarantee withdrawn on electrode reversal or an out-of-scope rhythm |
| 03 | Split-conformal EF interval, learned aleatoric σ, inter-clip epistemic disagreement |
| 04 | Declared feature-availability horizon, constrained decision layer, clinician referral |

The backend normalises those four vocabularies into one `actionability`
verdict — `actionable` / `caution` / `deferred` / `withheld` / `unavailable` —
so a caller applies one rule across all modalities:

> **Do not act on a result that is not `actionable`.**

Findings marked `withheld` are **not rendered at all**, because showing a
suppressed probability beside a warning invites it to be used anyway. The
console makes the verdict control the visual hierarchy rather than sit in a
corner badge.

---

## Architecture

```
                  ┌──────────────────────────────┐
   browser ───────►  Next.js console (frontend)  │
                  └──────────────┬───────────────┘
                                 │ REST, JSON envelopes
                  ┌──────────────▼───────────────┐
                  │  FastAPI service (cvxai)     │
                  │  ┌────────────────────────┐  │
                  │  │ adapters: cxr ecg echo │  │
                  │  │           triage       │  │
                  │  └───────────┬────────────┘  │
                  │   ModuleSandbox per component│
                  └──────────────┼───────────────┘
                                 │ imports in isolation
            Component_01 … Component_04 (unmodified)
```

**The components are not libraries and were never written to coexist.** Several
define the same top-level module names (`config`, `models`, `predict`,
`inference`), so importing two of them into one process silently gives the
second whichever module the first already registered. `ModuleSandbox` gives each
component its own `sys.path`, its own `sys.modules` namespace and its own
environment, and `verify_owns()` asserts that a name resolves to the component
that is supposed to own it. That check exists because the failure is silent: you
do not get an ImportError, you get another component's model.

Each component returns an **envelope** with the same shape — headline, findings,
reliability verdict, explanation, raw component payload — so the console renders
four very different modalities through one contract.

---

## Repository layout

```
R26-IT-083/
├── backend/          unified FastAPI service (package `cvxai`)
├── frontend/         Next.js clinical console
├── demo/             curated sample studies (gitignored)
├── Component_01/     chest radiograph — model + training code
├── Component_02/     ECG — model + training + audit suite
├── Component_03/     echocardiogram — model + training + ablations
└── Component_04/     ED triage — model + training + leakage audit
```

All four components are plain directories. Component 03 was previously a git
submodule pointing at
[Research_Component_03](https://github.com/Dilukshan285/Research_Component_03)
and has been folded in; that repository remains as its development history.

---

## Running it

Two terminals. The backend must be up first.

```powershell
# terminal 1
cd backend
python -m pip install -r requirements.txt
python run.py --warm                      # http://127.0.0.1:8000

# terminal 2
cd frontend
npm install
npm run dev                               # http://localhost:3001
```

Use `python -m pip`, not bare `pip`: a default Windows Python install puts
`python.exe` on PATH but not the `Scripts\` folder that holds `pip.exe`.

`--warm` loads every component before serving, so the first request of each is
fast. `--reload` auto-restarts on source change and loads lazily instead. **Do
not combine them** — every reload then repays the full model-load cost.

If files move while the server is running, restart it: Python caches module
locations at import time, and a moved component directory produces 500s until
the process is restarted.

`backend/README.md` and `frontend/README.md` carry the detail.

---

## API

Base path `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and per-component readiness |
| `GET` | `/components` · `/components/{id}` | Component metadata and model cards |
| `POST` | `/components/{id}/warm` | Force-load one component |
| `GET` | `/cohorts` | Dataset provenance and measured cohort overlap |
| `POST` | `/cxr/analyze` | Radiograph → findings, Grad-CAM, report |
| `POST` | `/ecg/analyze` | WFDB pair → superclasses, conformal zones, strip |
| `POST` | `/echo/analyze` | Video or `.npy` clip → EF, interval, grade, Grad-CAM |
| `POST` | `/triage/analyze` | ED record → ACS + subtype, SHAP |
| `POST` | `/triage/analyze-pdf` | ED summary PDF → extracted record, then as above |
| `POST` | `/assessment` | Whichever modalities are supplied, all at once |
| `GET` | `/pathway/definition` | The six stages and their routing rules |
| `POST` | `/pathway` | Run the whole pathway from one payload |
| `POST` | `/pathway/stage` | Advance exactly one stage, carrying `context` |

`context` from `/pathway/stage` is opaque by design and is handed straight back
on the next call. Reading it in the client would put a second copy of the
routing rules there.

Interactive docs at `http://127.0.0.1:8000/docs` while the service is running.

---

## Datasets and weights

**None of it is in this repository.**

Three of the four datasets are credentialed. MIMIC-CXR, MIMIC-IV-ED and PTB-XL
are governed by PhysioNet data use agreements; EchoNet-Dynamic and CAMUS carry
their own terms. No images, waveforms, videos, reports or derived datasets are
distributed here, and **neither are model weights**, which are derived from
credentialed data and are equally non-redistributable.

The working tree is tens of gigabytes; this repository is a few megabytes of
source. `.gitignore` is the boundary — read it before adding anything, and
re-check `git status` before a commit that touches a component directory.

To reproduce any component you need your own credentialed access and that
component's own preprocessing pipeline.

### Component 04 splits

There are no `train.csv` / `val.csv` / `test.csv` files. The pipeline keeps one
feature matrix per disclosure horizon plus a single assignment table, and joins
them at load time:

| What | Path |
|---|---|
| Features | `Component_04/artifacts/data/features_H{0,6,24}.parquet` |
| Split | `Component_04/artifacts/data/split_assignment.parquet` |
| Models | `Component_04/artifacts/models/` |
| Raw MIMIC tables | `Component_04/data/processed/` |

`split_assignment.parquet` holds 203,016 stays as `stay_id, subject_id, fold`:
train 142,111 (70 %), val 30,453 (15 %), test 30,452 (15 %). Grouped by patient,
so no `subject_id` crosses folds.

To materialise them as CSV:

```bash
cd Component_04
python src/data/export_splits.py             # all three horizons
python src/data/export_splits.py --horizon 24
```

It exports from the same assignment table the models were trained against, and
verifies the patient grouping before writing.

---

## Tests

```bash
cd backend
python -m pytest tests/ -q                   # 132 passed, 1 skipped
python -m pytest -m "not integration" -q     # skip the ones needing weights
```

Integration tests skip themselves when a component's assets are absent, so the
suite passes on a machine with no credentialed data.

```bash
cd frontend
npx tsc --noEmit && npx next lint --dir src && npm run build
```

---

## Research posture

The components share no findings and no code, and their contributions are graded
differently. What they do share is a habit worth stating as a claim in its own
right:

**Every component measured its own failure and reported it.**

| Component | What it found against itself |
|---|---|
| **01 · Radiograph** | Three interventions were tried to close the AP/PA accuracy gap. All three failed, and the negative result is published rather than buried. The gap is closed by deferring more bedside films to a human — not by claiming the model improved. |
| **02 · ECG** | The conformal guarantee holds marginally and **breaks for identifiable subgroups**. Measured, significance-tested, and fixed with group-conditional calibration — with the coverage cost of that fix also reported. Separately, its own architecture was shown to be worse than the plain baseline it extends. |
| **03 · Echo** | The backbone was inherited from a benchmark. It was tested against the un-factorised baseline at three matched seeds, and the first attempt at that test was found to be confounded and re-run. |
| **04 · Triage** | A previously published AUROC of 0.9889 was traced to temporal leakage, reproduced under controlled conditions, and **retracted**. The honest figure is 0.9560. |

A four-person project where each member independently found and reported a
limitation in their own work is a stronger claim than any single accuracy number
in it.

---

## What this system does not establish

- **No component diagnoses anything.** Four decision-support outputs, each
  requiring clinician review.
- **The four components share no patients.** MIMIC-CXR, PTB-XL,
  EchoNet-Dynamic/CAMUS and MIMIC-IV-ED are four separate cohorts. This is how
  the components *would* compose clinically, justified against published
  guidelines — not a validated end-to-end study on one population.
- **No joint model was trained** across the modalities and no combined accuracy
  is claimed. Every figure belongs to exactly one component.
- **Component 02 recognises five superclasses.** Its `MI` class does not
  separate STEMI from NSTEMI, and atrial fibrillation and other arrhythmias are
  outside the label space entirely.
- **Component 04's UA/NSTEMI boundary rests on ICD coding**, not adjudicated
  labels.
- **Unstable angina does not survive the PDF channel.** The published 80 % recall
  is measured on the 228-feature vector, not on a text document.

---

## Prior art to verify before presenting

Each component's novelty write-up names its closest prior art rather than
leaving a reviewer to find it. **These citations were added from recall and have
not been verified against the published record.** Check each before it goes on a
slide.

| Component | Must cite | Why |
|---|---|---|
| 01 | **Hardt, Price & Srebro, NeurIPS 2016** | Per-group thresholds *are* their post-processing method. Claiming the technique would be fatal. |
| 01 | **Jones et al., ICLR 2021** | Selective classification can *magnify* group disparities — the closest adjacent result, and complementary to this one. |
| 02 | **Vovk, ACML 2012**; **Barber et al., 2021** | Conditional validity, and its impossibility in general, are established theory. The measurement on ECG is what is new. |
| 03 | **Díaz & Marathe, CVPR 2019** | Soft ordinal targets are theirs. Deriving the width from label measurement noise is the refinement. |

---

## Licence

Code: MIT. Data: not licensed here, and not included. See each component's
README for its own terms.
