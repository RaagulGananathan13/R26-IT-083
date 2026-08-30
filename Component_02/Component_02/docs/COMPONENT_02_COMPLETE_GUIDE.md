# Component 02 — the whole thing, in plain English

**ECG Abnormality Detection and Cardiac Risk Reporting** · Venushan T
Part of R26-IT-083, *Explainable AI System for Cardiovascular Disease Detection and Diagnosis*

> ⚕️ Research prototype. Not a medical device, not clinically validated. Every output needs a qualified clinician to review it.

This document explains what Component 02 does, what happens inside it, which file
is responsible for each step, and what every number means. Technical words are
explained the first time they appear.

**Every accuracy figure in this file was recomputed from
`checkpoints/test_logits_seed0.npy` through the shipped calibrator and operating
point — not copied from an older document.** Where a figure differs from another
file in this repository, that is flagged.

---

## Table of contents

1. [What this component does](#1-what-this-component-does)
2. [Words you need first](#2-words-you-need-first)
3. [The five classes](#3-the-five-classes)
4. [**How the system says "this is the disease"**](#4-how-the-system-says-this-is-the-disease)
5. [The complete workflow, step by step](#5-the-complete-workflow-step-by-step)
6. [Which file is responsible for what](#6-which-file-is-responsible-for-what)
7. [How the parts pass information to each other](#7-how-the-parts-pass-information-to-each-other)
8. [How the model was trained](#8-how-the-model-was-trained)
9. [Where hallucination is prevented](#9-where-hallucination-is-prevented)
10. [The report model — old and new](#10-the-report-model--old-and-new)
11. [All the accuracy numbers](#11-all-the-accuracy-numbers)
12. [The research contribution](#12-the-research-contribution)
13. [Limitations](#13-limitations)

---

## 1. What this component does

You give it a 10-second, 12-lead ECG recording. It gives you back:

1. **A decision on each of five heart conditions** — not one label, five separate
   decisions, each of which can be *ruled out*, *referred to a human*, or *ruled in*.
2. **A statistical guarantee** attached to each rule-out: *"when I say no, I am
   wrong at most 5% of the time"* — with the 5% being a number you chose in advance.
3. **An explanation** — which leads drove the decision, which part of the 10
   seconds, and which coronary artery territory that corresponds to.
4. **A written clinical report** that is checked before it is allowed out.
5. **A triage tier** — IMMEDIATE, URGENT, PRIORITY, ROUTINE, or REPEAT ECG.

The thing that makes it different from an ordinary classifier: **it is allowed to
refuse.** If the signal is unreadable it returns no probability at all. If the
evidence is weak it hands that class to a cardiologist instead of guessing.

---

## 2. Words you need first

| Word | What it means, plainly |
|---|---|
| **12-lead ECG** | Twelve simultaneous recordings of the heart's electrical activity, each from a different "viewing angle". Named I, II, III, aVR, aVL, aVF, V1–V6. |
| **Lead** | One of those twelve signals. Not a wire — a viewing angle computed from the wires. |
| **Sampling rate** | How many measurements per second. Ours is 500 Hz, so 10 seconds = 5,000 numbers per lead. |
| **Multi-label** | Each ECG can have several conditions at once, or none. Opposite of "pick one of five". |
| **Sigmoid** | A function that squashes any number into the range 0–1, so it can be read as "how strongly do I think yes". One per class, independently. |
| **Softmax** | The alternative that forces five numbers to add up to 1 — i.e. forces "pick one". **We do not use it.** |
| **Logit** | The raw number the network outputs, before the sigmoid. Can be any value, negative or positive. |
| **Calibration** | Fixing the probabilities so that "70%" actually happens about 70% of the time. A model can rank cases correctly but still print wrong-looking percentages. |
| **Temperature scaling** | The specific calibration method used here — divide the logit by a learned number `T`, add a learned offset. |
| **ECE** (Expected Calibration Error) | How far the printed probabilities are from reality, on average. Lower is better. 0 is perfect. |
| **Conformal prediction** | A method that turns a score into a decision **with a provable error bound**, without assuming anything about the model or the data distribution. |
| **α (alpha)** | The miss-rate budget. `α = 0.05` for MI means: of all real infarctions, at most 5% may be wrongly ruled out. |
| **β (beta)** | The false-alarm budget, the mirror image, for ruling *in*. |
| **λ (lambda)** | A threshold. `λ_out` is the rule-out cut, `λ_in` is the rule-in cut. Computed from the data, not chosen by hand. |
| **Order statistic** | "The m-th smallest value in a sorted list." Conformal thresholds are literally one of these, picked from the calibration scores. |
| **PAC bound** | *Probably Approximately Correct.* A stricter guarantee that holds for **this particular** calibration set with high confidence, rather than merely on average across imaginary repeated calibrations. |
| **δ (delta)** | The confidence level for that PAC bound. `δ = 0.01` means "I am 99% sure this threshold really honours its α". |
| **Recall / sensitivity** | Of all patients who really have the condition, what fraction did we catch? |
| **Specificity** | Of all patients who really don't, what fraction did we correctly clear? |
| **Precision / PPV** | Of everyone we flagged, what fraction really had it? |
| **NPV** | Of everyone we cleared, what fraction really were clear? **This is the number that matters for ruling out.** |
| **AUROC** | Ranking quality, 0.5 = coin flip, 1.0 = perfect. Independent of where you put the threshold. |
| **AUPRC** | Like AUROC but designed for rare conditions. Harder, and more honest on small classes. |
| **F1** | The harmonic mean of precision and recall. Treats a missed heart attack and an unnecessary review as equally bad — which is why we do not lead with it. |
| **Grad-CAM** | An explanation method: which parts of the input the network's own gradients say it leaned on. |
| **Integrated gradients** | Another explanation method, giving each input a **signed** contribution — supporting (+) or opposing (−). |
| **Focal loss** | A training loss that pays more attention to examples the model finds hard, instead of being drowned by easy ones. |
| **EMA** | Exponential moving average of the weights — a smoothed copy that usually generalises a little better than the final weights. |
| **Superclass** | PTB-XL groups many specific diagnoses into 5 broad families. We work at that level. |

---

## 3. The five classes

Defined in exactly one place: **`src/models.py`, line 39** —

```python
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
```

Everything else in the system imports that list. Nothing re-declares it. That is
deliberate: if the order of these five ever disagreed between training and
serving, every probability would be attached to the wrong disease and nothing
would crash.

| Code | Full name | What it means to a patient |
|---|---|---|
| **NORM** | Normal ECG | The trace looks like a healthy heart. |
| **MI** | Myocardial infarction | Heart attack pattern — muscle is dying or has died from a blocked artery. |
| **STTC** | ST/T change | The trace is the wrong *shape* in the segment that reflects recovery. Means the heart is stressed, starved of blood, or affected by a drug or electrolyte problem. |
| **CD** | Conduction disturbance | The heart's electrical wiring is slow or blocked — the signal takes the wrong path or too long. |
| **HYP** | Hypertrophy | The heart muscle wall has thickened, usually from years of working against high pressure. |

Human-readable names live in `src/report.py`:

```python
CLASS_FULL = {
    "NORM": "normal ECG",
    "MI":   "myocardial infarction",
    "STTC": "ST/T change",
    "CD":   "conduction disturbance",
    "HYP":  "ventricular hypertrophy",
}
```

**Important:** these five are *superclasses*. `MI` does not tell you whether it is
a STEMI or an NSTEMI. And atrial fibrillation, the most common arrhythmia in the
world, **is not in this list at all** — see §9 and §13.

---

## 4. How the system says "this is the disease"

This is the part most people get wrong, so it gets its own section.

### It is NOT "pick one of five"

There is **no softmax and no argmax** anywhere in the decision path. The network
produces five numbers, and each one is decided **completely separately** against
**its own two thresholds**.

That means a single ECG can come back as:

- three conditions at once (common — real hearts have several things wrong),
- one condition,
- or none at all.

This is called **multi-label** classification, and it is the right shape for the
problem: a patient can have both a conduction disturbance *and* hypertrophy, and
forcing the model to choose would be clinically wrong.

### The three steps

#### Step 1 — five raw scores → five probabilities

```
network → 5 logits → sigmoid → 5 raw probabilities → temperature scaling → 5 calibrated probabilities
```

**File:** `src/pipeline.py`, lines 142–147.

```python
lg  = self.logits(x)                              # 5 raw numbers
raw = 1.0 / (1.0 + np.exp(-lg))                   # sigmoid, one per class
cal = self.calibrator.predict_proba(lg[None])[0]  # temperature scaling
```

Each class gets its **own** temperature `T` and bias, fitted separately.
Calibration is **monotone** — it never reorders two patients — so AUROC and the
conformal guarantees are mathematically untouched. Only the percentage shown to a
human changes, and it becomes honest.

#### Step 2 — each probability falls into one of three zones

**File:** `src/conformal.py`.

```
        0 ─────────── λ_out ─────────── λ_in ─────────── 1
             RULE OUT        REFER          RULE IN
        (miss rate ≤ α)  (we don't decide) (false alarm ≤ β)
```

| Zone | What the system does | The promise attached |
|---|---|---|
| **RULE OUT** | Says "this condition is not present" | Of all patients who really have it, at most **α** get put here |
| **REFER** | Says "I am not deciding this — a cardiologist must" | No promise needed; a human sees it |
| **RULE IN** | Says "this condition is present" | Of all patients who really don't have it, at most **β** get put here |

The two thresholds are **not chosen by hand and not chosen by maximising F1.**
They are order statistics of the calibration scores:

> Sort the scores of all calibration patients who genuinely **have** the condition.
> Take the m-th smallest, where m comes from α. Use that as `λ_out`.
> Any real positive scoring above it is caught — and the theory bounds how often a
> future positive falls below it.

That is why the guarantee holds **for any model.** A weak model does not break the
promise; it just refers more patients.

#### Step 3 — one structural rule

**File:** `src/report.py`, lines ~131–133.

If **NORM** is ruled in while **any abnormality** is also ruled in, NORM is
demoted to REFER. The system therefore **cannot** print "this ECG is normal" and
"there is a conduction disturbance requiring referral" in the same paragraph.

The old system did exactly that in **99 records (5.8%)** of the test set. It is
now structurally impossible, not filtered afterwards.

### The per-class budgets — this is clinical policy, not a hyperparameter

**File:** `src/conformal.py`, `PRESETS`. Three presets ship; `safety` is default.

| Class | α (miss budget) | β (false-alarm budget) | λ_out | λ_in |
|---|---|---|---|---|
| NORM | 0.20 | 0.20 | 0.1986 | 0.6944 |
| **MI** | **0.05** | 0.10 | 0.0382 | 0.3386 |
| STTC | 0.10 | 0.15 | 0.1934 | 0.3435 |
| CD | 0.10 | 0.15 | 0.1104 | 0.2793 |
| HYP | 0.15 | 0.15 | 0.0506 | 0.1391 |

*(λ values are the shipped `resnet_se`, seed 0, δ = 0.01, read from
`checkpoints/conformal_triage.json`.)*

**Why MI gets the tightest budget:** a missed heart attack can kill within hours.
**Why NORM gets the loosest:** "missing NORM" only means somebody healthy gets an
unnecessary review. The budgets encode that asymmetry openly, so they can be
argued with rather than hidden inside a threshold.

### A worked example — one real record

```
record 00039_hr     quality gate: PASS · SQI 1.00 · HR 61.6 bpm · 12 leads usable

class   calibrated p   λ_out    λ_in    zone       what happens
────────────────────────────────────────────────────────────────────────
NORM        0.021      0.199   0.694    rule_out   listed as excluded
MI          0.599      0.038   0.339    RULE IN    → TRIAGE: IMMEDIATE
STTC        0.144      0.193   0.344    rule_out   listed as excluded
CD          0.203      0.110   0.279    refer      cardiologist review
HYP         0.041      0.051   0.139    rule_out   listed as excluded

headline    "myocardial infarction"
territory   septal → proximal LAD / septal perforators
top leads   V1, V2, V5
timing      Grad-CAM peaks at 1.5 s and 3.8 s
guarantee   "misses at most 5% of infarctions, at 99% confidence"
```

Notice what the clinician receives that a bare probability could never give:
three conditions **positively excluded with a stated miss-rate bound**, one
explicitly handed back as undecided, and the one that is ruled in arriving with a
triage tier, an artery territory, and the leads it rests on.

### One property worth knowing

**The referral rate is not monotone in α.** Loosening α raises `λ_out`; loosening
β lowers `λ_in`. Once `λ_out` passes `λ_in` the two zones **overlap**, and a score
cannot be both ruled in and ruled out — so the overlap must be referred. Very
loose budgets can therefore *increase* referrals. This happens for NORM on
[0.199, 0.694), and the threshold record carries a note saying so.

---

## 5. The complete workflow, step by step

```
 .dat + .hea file
        │
 [1] QUALITY GATE ──────── fail ──► REFUSED. No probabilities exist. Stop.
        │ pass
 [2] PREPROCESS
        │
 [3] CLASSIFY (the network)
        │
 [4] CALIBRATE
        │
 [5] CONFORMAL TRIAGE ─── three zones per class
        │
 [6] XAI (only for classes being reported)
        │
 [7] BUILD REPORT (grounded in Finding objects)
        │
 [8] VERIFY ───────────── fail ──► REPORT WITHHELD. Triage forced to REPEAT ECG.
        │ pass
    result to the clinician

 Running alongside [1]:
    ELECTRODE-REVERSAL CHECK ─┐
    RHYTHM-SCOPE CHECK ───────┴─► withdraws the GUARANTEES, keeps the diagnosis
```

### [1] Quality gate — `src/quality.py`

**Runs before the model. This ordering is the whole design.**

An unreadable ECG must never be able to return a reassuring number, so the gate
comes first and a failure means **no probability is ever computed**.

What it checks:

| Check | Limit | Why |
|---|---|---|
| Shape | must be 2-D, 12 leads | wrong input caught loudly |
| Duration | 5–60 seconds | the archive silently resampled anything, which destroys timing |
| Non-finite samples | zero allowed | NaN/Inf means a broken file |
| Flat leads | at most 2 | more than 2 dead electrodes = not a 12-lead ECG |
| All leads flat | refuse | electrodes disconnected |
| QRS amplitude | ≥ 0.05 mV | below this the trace is essentially flat |
| Max amplitude | ≤ 20 mV | above this is saturation or a unit error (µV file read as mV) |
| Heart rate | 25–250 bpm | outside this the R-peak detection is untrustworthy |

It also **detects and fixes unit errors** (`detect_and_fix_units`) — a file
recorded in microvolts read as millivolts is rescaled and the correction is
stated in the report, not applied silently.

It also computes **SQI** (signal quality index), **heart rate**, **beat count**,
and **R-peak positions**, which are reused later so nothing is computed twice.

> The old system had none of this. An all-zero (disconnected) recording produced
> **"consistent with myocardial infarction"** at probability 0.691.

### [2] Preprocess — `src/preprocess.py`

| Operation | Detail | Why |
|---|---|---|
| Resample | **by sampling rate**, to 500 Hz | The archive resampled by *length*, so a 250 Hz recording was stretched and every interval was wrong. |
| High-pass | 0.5 Hz | removes baseline wander (the trace drifting up and down) while preserving the ST level, which is the MI evidence |
| Low-pass | 40 Hz | removes muscle noise |
| Notch | 50 Hz | mains hum — PTB-XL was recorded in Germany |
| Centre/pad | to exactly 5,000 samples | fixed input size |
| Normalise | per-lead, using frozen statistics from `csv/norm_stats.json` | the model expects the same scaling it was trained on |

Filtering is **zero-phase** (applied forwards then backwards), so it does not
shift the timing of anything — which would corrupt the interval measurements.

### [3] Classify — `src/models.py`

The network. Full architecture in §8.

```python
lg = self.logits(x)   # (5,) raw scores — pipeline.py line 142
```

Takes ~20 ms. It is the fastest part of the whole system.

### [4] Calibrate — `src/calibration.py`

Per-class temperature + bias, fitted on validation fold 9 only.

### [5] Conformal triage — `src/conformal.py`

Turns five calibrated probabilities into five zones. §4 explains it fully.

### [6] XAI — `src/xai.py`

**Only computed for classes actually being reported** — ruled in or referred, and
capped at 3 explanations per record, because integrated gradients is expensive.

Three things are produced:

1. **Temporal Grad-CAM** — a curve over the 10 seconds showing which moments the
   network leaned on. Peaks are extracted (`cam_peaks`). *The shape is the
   information:* one sharp spike means the call rests on a single heartbeat; a
   broad plateau means it does not.
2. **Signed lead attributions** — integrated gradients, giving each of the 12
   leads a signed percentage. Positive = supported the finding, negative = argued
   against it.
3. **Territory localisation** — which coronary artery territory the *supporting*
   leads concentrate in:

   | Territory | Leads | Artery |
   |---|---|---|
   | anterior | V1–V4 | left anterior descending (LAD) |
   | septal | V1, V2 | proximal LAD / septal perforators |
   | lateral | I, aVL, V5, V6 | left circumflex (LCx) |
   | anterolateral | V4–V6, I, aVL | LAD / LCx |
   | inferior | II, III, aVF | right coronary artery (RCA) |

   Only *positive* attribution counts, and the score is normalised by territory
   size so a 5-lead territory is not favoured over a 2-lead one.

   > This is a **lead-group heuristic, not a clinically validated localiser.**
   > Say so.

Thread-safety note: Grad-CAM registers hooks on the model, so `xai.py` holds a
per-model re-entrant lock. The archive had a shared-XAI defect where concurrent
requests corrupted each other's heatmaps.

### [7] Build report — `src/report.py`

See §10.

### [8] Verify — `src/verify.py`

See §9. If it fails, the text is replaced with an explicit withholding notice and
the triage tier is forced to `REPEAT ECG`.

---

## 6. Which file is responsible for what

### The library — `src/` (3,541 lines)

| File | Lines | Responsibility |
|---|---|---|
| **`models.py`** | 447 | **The single source of truth for every architecture.** Defines `CLASS_NAMES`, `LEAD_NAMES`, all three networks and four ablation variants, the focal loss, and `MODEL_REGISTRY`. Nothing else defines a network. |
| **`paths.py`** | 78 | Finds assets. Resolution order: `$ECG_DATA_DIR` → `csv/` → `data/` → `../_archive/data/`. |
| **`signals.py`** | 70 | Reads a WFDB record (`.dat` + `.hea`) or a cached array off disk. |
| **`quality.py`** | 314 | The quality gate. Unit detection, flat/noisy lead detection, R-peak detection, heart rate, SQI, rhythm irregularity. |
| **`preprocess.py`** | 139 | Resample, band-pass, notch, centre/pad, normalise. |
| **`electrodes.py`** | 183 | Exact limb-reversal simulators **and** a physiology-rule detector. |
| **`scope.py`** | 145 | Rhythm-scope check — "is this rhythm outside my five classes?" |
| **`calibration.py`** | 129 | Per-class temperature scaling. |
| **`conformal.py`** | 360 | The three-zone risk-controlled triage. Presets, PAC order statistic, risk–coverage curve. |
| **`xai.py`** | 297 | Grad-CAM, integrated gradients, lead attribution, territory mapping, faithfulness metrics. |
| **`report.py`** | 365 | The grounded report generator. `Finding` and `ClinicalReport` dataclasses, triage tiers. |
| **`verify.py`** | 265 | The safety gate. Finding-preservation check, forbidden terms, overclaim terms. |
| **`pipeline.py`** | 197 | **The single inference entry point.** Wires steps 1–8 together for one model. |
| **`zoo.py`** | 523 | Serves several models at once, each with its own calibrator and thresholds. Consensus rule. |

### Training — `train/`

| File | What it does |
|---|---|
| `preflight.py` | 30-second setup check. Run it before paying for a GPU session. |
| **`train_gpu.py`** | Trains the classifier. Packs data to a memmap, trains, saves weights **and the val/test logits**. |
| **`fit_calibration.py`** | Fits the temperature calibrator and the conformal thresholds on fold 9. No GPU, ~3 min. |
| `compare_architectures.py` | Runs the 3-architecture × 3-seed ablation. |
| `Component02_Colab.ipynb` | The Colab notebook wrapper. |

### Analysis and audit

| File | Question it answers |
|---|---|
| `analysis/01_dataset_deep_audit.py` | Is the dataset what I think it is? Splits, overlap, prevalence, label entanglement. |
| `analysis/02_operating_point.py` | Which threshold policy should ship, and what does it cost? |
| `audit/08_verify_fixes.py` | 26 checks that every known defect stays closed. |
| **`audit/10_conditional_validity.py`** | **Contribution 1** — does the guarantee hold inside patient subgroups? |
| **`audit/11_significance.py`** | Are those subgroup violations statistically real? Holm correction, bootstrap. |
| **`audit/12_electrode_reversal.py`** | **Contribution 2** — what does a swapped cable do to the guarantee? |
| **`audit/13_out_of_scope.py`** | **Contribution 3** — what happens when the disease is not in the label space? |
| `audit/14_multi_model.py` | What does serving two models cost and buy? |
| `audit/15_disagreement_detector.py` | **Contribution 4** — cross-model disagreement as an acquisition check. |
| `audit/architecture_comparison/` | Do the three architectural additions earn their parameters? |
| `audit/legacy/03_report_audit.py` | The audit that ended the BioBART tier. |

### Serving

| File | What it does |
|---|---|
| `backend/server.py` | JSON-only API. `/api/health`, `/api/predict`, `/api/analyze/<id>`, `/api/models`, `/api/demo`. |
| `frontend/` | React 19 + Vite 6 + Tailwind 4 UI. |

---

## 7. How the parts pass information to each other

### The bundle idea

A trained model on its own is not servable. It needs **three artefacts that were
fitted to it specifically**:

```
checkpoints/best_model.pt          the weights          ← resnet_se, seed 0
checkpoints/calibrator.json        temperature + bias   ← fitted to THOSE logits
checkpoints/conformal_triage.json  λ_out / λ_in per class ← fitted to THOSE probabilities
checkpoints/operating_point.json   the single-threshold policy
checkpoints/scope.json             the rhythm-irregularity threshold
```

Pairing a calibrator with a **different** model destroys both the probabilities
and the guarantees **while everything still runs** — no exception, just wrong
numbers. So every artefact carries its own provenance:

```json
"fitted_for": {"model": "resnet_se", "ckpt": "best_model.pt", "filter": true, "seed": 0}
```

and `pipeline.py` (lines ~108–125) checks it at load time:

```python
if want != got:
    warnings.warn(f"{name} was fitted for model={want[0]} filter={want[1]}, "
                  f"but the pipeline is loading model={got[0]} filter={got[1]}. "
                  f"Its output is not valid for this model.")
```

The `filter` flag matters as much as the model name: the baseline was trained on
**unfiltered** signals and the shipped model on **band-passed** ones. A bundle
therefore carries its preprocessing too, and `zoo.py` reads that from the
artefacts rather than assuming it.

### The call chain, end to end

```
backend/server.py
  └── ECGPipeline.analyse(signal, fs)                      src/pipeline.py
        ├── quality.assess(raw, fs)                        → QualityReport
        ├── preprocess.prepare(mv, fs, mean, std)          → (12, 5000) float32
        ├── self.logits(x)                                 → (5,) logits
        ├── calibrator.predict_proba(logits)               → (5,) calibrated
        ├── triage.zones_one(cal)                          → {class: zone}
        ├── xai.explain(model, x, k, name)  ×≤3            → Explanation
        ├── report.build_report(probs, zones, thr, q, exp) → ClinicalReport
        └── verify.verify_report(report)                   → VerifyReport
      returns AnalysisResult
```

### The data objects

| Object | Defined in | Carries |
|---|---|---|
| `QualityReport` | `quality.py` | `acceptable`, `errors[]`, `warnings[]`, `sqi`, `heart_rate_bpm`, `n_beats`, `flat_leads`, `noisy_leads`, `gain_correction`, `electrode_suspect`, `out_of_scope` |
| `ClassThresholds` | `conformal.py` | `alpha`, `beta`, `lambda_out`, `lambda_in`, `n_pos_cal`, `n_neg_cal`, `guarantee_feasible`, `note` |
| `Explanation` | `xai.py` | `cam`, `cam_peaks_s`, `lead_signed`, `lead_magnitude`, `top_leads`, `territory`, `territory_score`, `territory_artery` |
| **`Finding`** | `report.py` | `cls`, `label`, `zone`, `probability`, `lambda_out`, `lambda_in`, `sentence`, `evidence[]`, `territory`, `artery`, `top_leads`, `timing_s` |
| `ClinicalReport` | `report.py` | `triage`, `headline`, `quality_line`, `rhythm_line`, `findings[]`, `ruled_out[]`, `referred[]`, `guarantees[]`, `limitations[]`, `text`, `refused` |
| `AnalysisResult` | `pipeline.py` | everything above plus `probs_raw`, `probs_calibrated`, `zones`, `signal_mv`, `r_peaks` |

**The `Finding` object is the important one.** Every sentence in the report is
emitted *from* a `Finding`, and every `Finding` carries the evidence that produced
it. Nothing can be written that is not traceable back to a measurement, a
conformal decision, or an attribution. That is what makes §9 possible.

---

## 8. How the model was trained

### The data

**PTB-XL** — a public 12-lead ECG dataset from the Physikalisch-Technische
Bundesanstalt, Germany, recorded 1989–1996.

| | Records | Patients |
|---|---|---|
| Official dataset | 21,799 | 18,869 |
| **Used here** | **17,221** | **15,174** |
| Train (folds 1–8) | 13,801 | 12,109 |
| Validation (fold 9) | 1,709 | 1,550 |
| Test (fold 10) | **1,711** | 1,515 |

**Patient overlap between splits: 0 across all three pairs. Duplicate ecg_id
within splits: 0.** Verified by `analysis/01_dataset_deep_audit.py`. This matters
because 1,509 patients (9.9%) have more than one recording — a random row split
would leak the same patient across folds.

**Why 4,578 records were dropped:** only SCP diagnostic codes with
`likelihood == 100` were kept — i.e. only diagnoses the cardiologist was certain
about. That is **21% of PTB-XL**, so **results are not directly comparable to
published benchmarks** that keep everything.

Test-fold class counts: NORM 707, MI 268, STTC 456, CD 483, HYP 132.
Prevalence drift across folds is under 1.5 percentage points.

### The network — `ECGResNetSE`

```
input  (batch, 12, 5000)              12 leads · 10 s · 500 Hz
   │
MultiKernelStem                       stride 2
   ├─ Conv1d kernel=7   → 21 ch       captures P wave / QRS (fast, narrow)
   ├─ Conv1d kernel=15  → 21 ch       middle scale
   └─ Conv1d kernel=31  → 22 ch       captures T wave / ST (slow, wide)
   │  concat → BatchNorm → ReLU → 64 channels
   │
4 × SEResidualBlock                   stride 2 each
   channels  64 → 128 → 256 → 320
   kernels   11 →   7 →   5 →   3
   dropout  0.1  0.1   0.2   0.2
   each block: Conv-BN-ReLU-Dropout-Conv-BN → SEBlock → + skip → ReLU
   │
AttentionPool                         softmax over time
   w = softmax(Conv1d(320 → 1))
   pooled = Σ_t (x · w)               keeps WHERE in the 10 s it happened
   │
head   Linear(320 → 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256 → 5)
   │
output  5 logits
```

**Parameter count: 1,589,588** (verified by loading the checkpoint).

| Piece | Plain explanation | Why it was added |
|---|---|---|
| **Residual block** | Each block learns a *correction* to its input rather than a whole new representation, so gradients reach the early layers. | Standard; lets the network be deep without dying. |
| **Multi-kernel stem** | Three convolutions of different widths in parallel. | A P wave lasts ~80 ms and a T wave ~200 ms. One kernel width cannot see both well. |
| **Squeeze-excitation (SE)** | Looks at all channels together and re-weights them — "for this patient, channel 42 matters more". | Lets the network emphasise leads/filters per case. |
| **Attention pooling** | Instead of averaging over all 5,000 timesteps, learns a weight per timestep. | Average pooling throws away *when* something happened; an infarct is a local event. |

> **The ablation says two of these do nothing and one is harmful.** See §11.

### Training settings — `train/train_gpu.py`

| Setting | Value |
|---|---|
| Hardware | one Colab Pro L4, ≤ 1 GPU-hour |
| Epochs | 40 |
| Batch size | 128 (halves automatically on CUDA OOM, epoch restarts) |
| Optimiser | AdamW, lr 3e-3, weight decay 1e-2 |
| Schedule | OneCycle |
| Precision | AMP (automatic mixed precision) |
| Weight averaging | EMA |
| Seed | 0 (also trained at 1 and 2 for the 3-seed figures) |
| Loss | Multi-label focal loss, γ = 2.0, **α = None** |
| Sampling | Balanced (`WeightedRandomSampler`) |
| Augmentation | amplitude scaling capped at **0.9–1.1×** |

Four of these are fixes for specific audited defects:

1. **`alpha=None` with a balanced sampler.** The old system used *both* a balanced
   sampler and focal-loss alpha weights — correcting class imbalance twice. The
   result was destroyed calibration: HYP was predicted at **4.14×** its true
   prevalence, and the UI printed those numbers to a clinician as "probability %".
   The rule is now enforced in `models.py`: use one or the other, never both.

2. **Augmentation capped at 0.9–1.1×, not 0.8–1.2×.** Hypertrophy is diagnosed by
   **QRS amplitude** (Sokolow-Lyon, Cornell criteria). Wide amplitude jitter
   destroys the very evidence the HYP class depends on.

3. **Band-pass added.** The archive did no filtering at all.

4. **Val and test logits are saved during training.** This means calibration is
   fitted with **no second forward pass**, so there is no chance of the calibrator
   seeing a slightly different model than the one that ships.

Robustness features: `--resume` restores optimiser, scheduler, EMA and RNG state;
`--max-minutes` stops cleanly on a wall-clock budget; a NaN guard stops a diverged
run instead of burning 35 more epochs.

### After training — `train/fit_calibration.py`

Runs on CPU in ~3 minutes and produces the two artefacts the pipeline needs:

1. **`calibrator.json`** — per-class temperature and bias, fitted by minimising
   negative log-likelihood on **fold 9 only**.
2. **`conformal_triage.json`** — `λ_out` and `λ_in` per class, from the PAC order
   statistic at δ = 0.01, computed on **fold 9 only**.

> **The discipline that matters:** both are fitted on validation and *verified* on
> test. Fitting either on test would be exactly the leak the audit warned about.
> Fold 10 is scored **once**, for reporting.

---

## 9. Where hallucination is prevented

"Hallucination" here means: **the system stating something it has no evidence
for.** There are four separate defences, at four different points.

### Defence 1 — the quality gate (`src/quality.py`)

*Prevents: inventing a diagnosis from noise.*

Runs **before** the model, so a refused record produces **no probability at all**.
The old system, given an all-zero recording from disconnected electrodes,
returned *"consistent with myocardial infarction"* at 0.691. That is now
impossible: no probability is computed, and the report says only

> ECG NOT INTERPRETED. The recording failed automated quality control and was
> refused before classification: [reasons]. No diagnostic probabilities are
> reported, because a diagnosis derived from an uninterpretable signal is worse
> than no diagnosis.

### Defence 2 — grounded generation (`src/report.py`)

*Prevents: writing a sentence that no measurement supports.*

Every sentence is emitted **from a `Finding` object**, and every `Finding` carries
its own evidence list. There is no free-text step. The system physically cannot
compose a sentence about something that is not in the findings, because there is
no code path that writes text from anything else.

The conformal zone also **chooses the modality of the language**:

| Zone | Language used |
|---|---|
| RULE IN | *"Findings are consistent with myocardial infarction."* |
| REFER | *"Possible ST/T change (calibrated probability 32%). This falls between the rule-out and rule-in thresholds, so the system does not decide it — cardiologist review is required."* |
| RULE OUT | (no sentence; the class is listed under "ruled out") |

Uncertainty is **stated in the grammar**, not hidden behind a threshold.

### Defence 3 — the structural NORM rule (`src/report.py`)

*Prevents: self-contradiction.*

NORM cannot be ruled in alongside a ruled-in abnormality, and a study is not
called normal while anything is still referred:

> *"No abnormality crossed its rule-in threshold, but this study cannot be called
> normal while conduction disturbance remains unresolved."*

### Defence 4 — the verifier (`src/verify.py`)

*Prevents: anything the first three missed getting out.*

This is the gate, **not** the generator. It checks the finished text against the
structured findings and refuses it if they disagree.

**a) Finding preservation, in both directions.**
Every ruled-in or referred class must be asserted in the text, and nothing may be
asserted that is not in the findings. It handles **negation**, so
*"no evidence of infarction"* is not counted as asserting infarction. Negators
recognised include: `no`, `not`, `without`, `absence of`, `ruled out`, `excluded`,
`negative for`, `cannot be`, `does not`.

**b) Forbidden terms.** A hard list of diagnoses the model has **no output unit
for**. Their appearance is *definitionally* a hallucination:

```
atrial fibrillation · atrial flutter · afib · ventricular tachycardia
ventricular fibrillation · supraventricular tachycardia · pacemaker
paced rhythm · Wolff · Brugada · long QT · torsade · STEMI · NSTEMI
cardiac arrest · asystole · pericarditis · myocarditis
pulmonary embolism · hyperkalaemia
```

**c) Overclaim terms.** Language asserting more certainty than a machine
interpretation can carry: `definitely`, `certainly`, `confirmed diagnosis`,
`rule out entirely`, `no further review`, `guaranteed diagnosis`.

**If verification fails**, the text is replaced with an explicit notice and the
triage tier is forced to `REPEAT ECG`:

```python
rep.text = ("REPORT WITHHELD — the generated text failed automated safety "
            "verification:\n  - " + "\n  - ".join(ver.errors) +
            "\n\nManual interpretation is required.")
rep.triage = "REPEAT ECG"
```

> **Design note worth saying out loud:** `verify.py` is deliberately written as a
> *gate*, not a generator. Any future natural-language layer — a rule realiser, a
> local LLM, a fine-tuned paraphraser — must pass `verify_paraphrase()` or it does
> not ship. Degraded fluency, never degraded safety.

### Defence 5 (partial) — the two "I cannot promise this" checks

These do not prevent a *false statement*; they prevent a **false promise**.

**Electrode reversal — `src/electrodes.py`.** A swapped cable produces a perfectly
clean signal, so the quality gate cannot catch it: on 600 test records it accepted
**587–589 of 600** reversed traces at a mean quality index of **1.000**. But
RA/LA reversal flips at least one label in **86.8%** of records and **voids nine
guarantees at once** — under it, the system fails to recognise 99.6% of normal
ECGs while still printing its promise.

The detector is a **stated physiology rule, not a classifier**: aVR is
predominantly negative and lead I predominantly positive in essentially every
normal heart, so *positive aVR with inverted lead I* is the classic RA/LA
signature. Measured: **70% sensitivity on RA/LA, 61% on RA/LL, 4.5% false
positives.** LA/LL reversal leaves aVR unchanged and is essentially undetectable
this way.

When suspected, the system **still analyses the record** but replaces the
guarantees with:

> *GUARANTEES SUSPENDED — the conformal bounds are calibrated on correctly-placed
> recordings and do not hold here.*

**Rhythm scope — `src/scope.py`.** Atrial fibrillation is *defined* by an
irregularly irregular ventricular response, so the R-R interval series carries the
signal directly — and the pipeline already detects R-peaks for heart rate, so the
feature costs nothing.

| Feature | AUROC (AF/flutter vs rest, test fold) |
|---|---|
| normalised median \|ΔRR\| | **0.912** |
| coefficient of variation of RR | 0.907 |
| pNN50 | 0.892 |

Threshold fitted on the full validation fold at a **5% false-positive budget**
(irr = 0.179). On the unseen test fold: **48.9% sensitivity, 4.8% FPR.**

Critically, it **does not diagnose atrial fibrillation** — the system is not
permitted to name a class it was never trained on. It answers a narrower, honest
question: *is this rhythm outside the region where my guarantee was calibrated?*

---

## 10. The report model — old and new

### There was a real neural report model

**Model C — a BioBART generator** (`GanjinZero/biobart-base`). BioBART is a
biomedical variant of BART, a sequence-to-sequence transformer, pre-trained on
PubMed abstracts and clinical text.

```
ECG signal → ECGBackbone (4-block 1-D CNN)
           → Linear(256 → 768) → LayerNorm → GELU
           → injected into BioBART encoder hidden states
           → BioBART decoder generates English tokens

loss    cross-entropy on token sequences
metric  ROUGE-L vs English-translated cardiologist reports
beams   4 · no_repeat_ngram_size 3
max_len 128 tokens (Tier 3) / 64 (free-text)
```

It ran in two modes: **Tier 3**, paraphrasing a template into "natural prose", and
a **legacy free-text** mode generating straight from signal features.

> ⚠️ **The weights are not in this repository.** `checkpoints_report_gen/best_model.pt`
> is referenced by `reference/legacy_docs/legacy_app.py` and documented in
> `Full_System_Explanation.txt`, but it is not present anywhere in the working
> tree. **The report model cannot be re-run.** What *is* present is its complete
> frozen output on every test record, in
> `reference/audit_dump/audit_real_vs_generated.txt`.

### What the audit found — all 1,711 test records

Reproduce with `audit/legacy/03_report_audit.py`
*(its hard-coded `_archive/` path no longer resolves — repoint it at
`reference/audit_dump/`).*

| Behaviour of the Tier-3 BioBART smoother | Records |
|---|---|
| Byte-identical to its input template | 404 (23.6%) |
| Input with leading characters **clipped off** | 974 (56.9%) |
| Genuinely rewritten | 333 (19.5%) |
| Dropped a clinical concept present in its input | 103 |
| Invented **"atrial fibrillation"** — a class the model cannot produce | **41** |

**Verdict: an identity function plus a truncation bug.** 23.6% + 56.9% = 80.5% of
outputs were the input, unchanged or damaged.

> ⚠️ **Number correction.** `docs/AUDIT_FINDINGS.md` states the hallucination
> count as **42** in two places. Recounting on the shipped dump — strict match,
> whitespace-tolerant match, and any-"afib"-form match all agree — gives **41**.
> **Use 41.**

**All 41 are the identical substitution**, which is a much stronger point than
"it hallucinated randomly":

```
Tier 2:  "ECG shows predominantly normal features. Minor non-specific …"
              ↓
Tier 3:  "Graphic atrial fibrillation. Minor non-specific …"
```

One template opening, one deterministic failure mode. And the classifier had
detected **NORM+STTC on 36 of them**, NORM+MI on 4, NORM+HYP on 1 — so in every
case the paraphraser replaced a hedged normal reading with a named arrhythmia the
upstream model never saw.

### The safety claim that did not hold

The archive defended the smoother with: *"it cannot hallucinate because it never
sees the raw ECG."*

That confuses two different things. Not seeing the signal prevents **inventing
evidence**; it does nothing to stop a seq2seq model **adding, dropping or negating
a finding while rephrasing** — which is precisely what the 103 dropped concepts
and the 41 fabricated atrial fibrillations are.

### The report model's own accuracy on the test fold

Recomputed independently from the dump (standard LCS-F1 ROUGE-L):

| Population | n | Tier 2 template | Tier 3 BioBART | Free-text BioBART |
|---|---|---|---|---|
| **Mean ROUGE-L** | | | | |
| All test records | 1,711 | 0.0895 | 0.0905 | **0.4763** |
| NORM-only records | 695 | 0.1101 | 0.1131 | **0.5868** |
| Records with any abnormality | 1,001 | 0.0761 | 0.0756 | 0.4028 |
| **Median ROUGE-L** | | | | |
| All test records | 1,711 | 0.1026 | 0.1026 | 0.4324 |
| **As recorded in the legacy notes** | | | | |
| All — mean | 1,711 | 0.0657 | 0.0660 | 0.4337 |

*My recomputation runs slightly above the recorded figures — different
tokenisation — but the ordering, the gap and the conclusion are identical.*

**The stratification is the finding.** The free-text tier looks like a huge win at
0.476 against the template's 0.090. Split it by population and the win evaporates:
**0.587 on normal records against 0.403 on abnormal ones.** 16.4% of the reference
reports are a degenerate four-token string in the class of *"sinus rhythm normal
ekg"*, and 65.5% contain the word "normal" at all. The decoder is scoring by
memorising the most common sentence in the corpus. **It scores best exactly where
there is nothing to say, and worst where a report has to carry clinical content.**

*(Component 01 reached the same conclusion from the other direction with a
constant-string control. Two components, two datasets, one result: ROUGE-L does
not measure whether a generated report is clinically right.)*

### Hallucination across all three tiers

| Tier | Records naming a disease the model cannot produce | Rate |
|---|---|---|
| Tier 2 — deterministic template | **0** | 0.00% |
| Tier 3 — BioBART smoother | **41** | 2.40% |
| Free-text BioBART | **141** | 8.24% |

### What no tier ever emitted

| Clinical content | Present in |
|---|---|
| Heart rate (bpm) | **0 / 1,711** |
| Rhythm classification | **0 / 1,711** |
| PR / QRS / QT intervals | **0 / 1,711** |
| QRS axis | **0 / 1,711** |
| Risk score / triage level | **0 / 1,711** |
| Lead-level localisation | **0 / 1,711** |
| Uncertainty statement | **0 / 1,711** |
| Signal-quality statement | **0 / 1,711** |
| Patient age / sex | 192 / 1,711 |

The deliverable was called a *Cardiac Risk Reporting System*, and no tier emitted
a risk stratification, a measured interval, or a signal-quality statement
anywhere. Two further defects: 5.8% of reports asserted NORM **and** an
abnormality in the same paragraph (CD+NORM 55, NORM+STTC 38, MI+NORM 5,
HYP+NORM 1), and **63 distinct reports** covered 1,711 patients — where the
cardiologists wrote **1,055**.

### What replaced it

**Every one of those "0 / 1,711" rows is filled by the new generator.** It emits
heart rate, rhythm regularity, beat count, signal quality index, flat and noisy
leads, gain corrections, the conformal zone per class, the guarantee text, the
territory localisation, and the triage tier.

```
OLD:  template text → BioBART smoother → the clinician
                       (no output check)

NEW:  Finding objects → grounded realiser → verify.py → the clinician
      (evidence attached)  (zone picks the      (gate)     or nothing at all
                            language)
```

**How to frame this so it does not sound like a step backwards:** the contribution
was never the generator. It was discovering that a report generator in a clinical
system needs a **gate**, and building the gate. The seq2seq layer was measured
against its own safety claim, failed on 41 fabricated diagnoses and 103 dropped
findings, and was removed on evidence.

---

## 11. All the accuracy numbers

**All figures below: test fold 10, n = 1,711, patient-disjoint, used once.**
Thresholds chosen on validation fold 9. Shipped model `resnet_se`, seed 0.

### 11.1 Per-class — the shipped operating point

| Class | Prevalence | n pos | AUROC | AUPRC | Accuracy | **Recall** | Specificity | Precision | **NPV** | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| NORM | 41.3% | 707 | 0.9574 | 0.9253 | 0.883 | 0.796 | 0.943 | 0.908 | 0.868 | 0.849 |
| **MI** | 15.7% | 268 | 0.9487 | 0.7833 | 0.884 | **0.836** | 0.893 | 0.591 | **0.967** | 0.692 |
| STTC | 26.7% | 456 | 0.9315 | 0.8280 | 0.868 | 0.803 | 0.892 | 0.729 | 0.926 | 0.764 |
| CD | 28.2% | 483 | 0.9141 | 0.8646 | 0.869 | 0.805 | 0.894 | 0.750 | 0.921 | 0.776 |
| **HYP** | 7.7% | 132 | 0.9085 | 0.5842 | 0.817 | **0.811** | 0.818 | 0.271 | **0.981** | 0.406 |
| **MACRO** | | | **0.9320** | **0.7971** | **0.864** | **0.810** | 0.888 | 0.650 | **0.933** | 0.698 |

✅ **Accuracy ≥ 0.75 on every class.**
✅ **Recall ≥ 0.75 on every class.**
✅ **NPV ≥ 0.86 on every class; macro 0.933.**

Across **3 seeds**: macro-AUROC **0.9343 ± 0.0028**, macro-AUPRC **0.8001 ± 0.0029**.

### 11.2 Three operating points, same weights

The choice between these is a **clinical policy decision**, not a modelling one.

**A · default threshold 0.5** — what an unconfigured model does

| Class | Acc | Recall | Spec | Prec | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| NORM | 0.891 | 0.885 | 0.895 | 0.856 | 0.917 | 0.871 | 626 | 105 | 81 | 899 |
| MI | 0.917 | **0.675** | 0.962 | 0.767 | 0.941 | 0.718 | 181 | 55 | **87** | 1388 |
| STTC | 0.877 | 0.746 | 0.925 | 0.783 | 0.909 | 0.764 | 340 | 94 | 116 | 1161 |
| CD | 0.882 | 0.725 | 0.944 | 0.835 | 0.897 | 0.776 | 350 | 69 | 133 | 1159 |
| HYP | 0.939 | **0.311** | 0.992 | 0.759 | 0.945 | 0.441 | 41 | 13 | **91** | 1566 |
| MACRO | 0.901 | 0.668 | 0.944 | 0.800 | 0.922 | 0.714 | | | | |

**B · F1-optimal** — the conventional choice

| Class | Acc | Recall | Spec | Prec | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| NORM | 0.890 | 0.881 | 0.896 | 0.857 | 0.915 | 0.869 | 623 | 104 | 84 | 900 |
| MI | 0.905 | 0.731 | 0.938 | 0.685 | 0.949 | 0.708 | 196 | 90 | 72 | 1353 |
| STTC | 0.864 | 0.818 | 0.881 | 0.715 | 0.930 | 0.763 | 373 | 149 | 83 | 1106 |
| CD | 0.882 | 0.718 | 0.946 | 0.840 | 0.895 | 0.775 | 347 | 66 | 136 | 1162 |
| HYP | 0.902 | 0.652 | 0.923 | 0.415 | 0.969 | 0.507 | 86 | 121 | 46 | 1458 |
| MACRO | 0.889 | 0.760 | 0.917 | 0.702 | 0.932 | 0.724 | | | | |

**C · recall-first — SHIPPED.** Selected by a PAC conformal lower bound on recall,
sensitivity floor 0.80 on validation.

| Class | Acc | Recall | Spec | Prec | NPV | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| NORM | 0.883 | 0.796 | 0.943 | 0.908 | 0.868 | 0.849 | 563 | 57 | 144 | 947 |
| **MI** | 0.884 | **0.836** | 0.893 | 0.591 | **0.967** | 0.692 | 224 | 155 | **44** | 1288 |
| STTC | 0.868 | 0.803 | 0.892 | 0.729 | 0.926 | 0.764 | 366 | 136 | 90 | 1119 |
| CD | 0.869 | 0.805 | 0.894 | 0.750 | 0.921 | 0.776 | 389 | 130 | 94 | 1098 |
| **HYP** | 0.817 | **0.811** | 0.818 | 0.271 | **0.981** | 0.406 | 107 | 288 | **25** | 1291 |
| MACRO | 0.864 | **0.810** | 0.888 | 0.650 | **0.933** | 0.698 | | | | |

Shipped thresholds (on calibrated probabilities):
`NORM 0.7240 · MI 0.2560 · STTC 0.3755 · CD 0.3395 · HYP 0.0925`

**Read the MI row down the three tables.** At 0.5 the model misses **87 of 268**
infarctions. F1-optimal misses 72. The shipped point misses **44** — at the cost
of 155 false alarms instead of 55.

**What recall costs, stated honestly:**

| Class | F1 (F1-opt) | F1 (shipped) | Recall gain | Precision cost |
|---|---|---|---|---|
| NORM | 0.869 | 0.849 | −0.085 | +0.051 |
| MI | 0.708 | 0.692 | **+0.104** | −0.094 |
| STTC | 0.763 | 0.764 | −0.015 | +0.015 |
| CD | 0.775 | 0.776 | **+0.087** | −0.091 |
| HYP | 0.507 | 0.406 | **+0.159** | −0.145 |

For a rule-out system that is the correct trade: a false alarm costs a
cardiologist's review; a false negative can cost a life.

### 11.3 Calibration

| Class | T | bias | ECE before | ECE after | Over-prediction |
|---|---|---|---|---|---|
| NORM | 0.480 | −0.335 | 0.0941 | 0.0155 | 1.03× → 1.01× |
| MI | 0.670 | −0.009 | 0.0494 | 0.0167 | 1.19× → 1.05× |
| STTC | 0.539 | 0.111 | 0.0964 | 0.0281 | 1.10× → 0.98× |
| CD | 0.521 | −0.465 | 0.1197 | 0.0162 | 1.29× → 0.95× |
| HYP | 0.584 | −1.195 | 0.1289 | 0.0057 | **2.65× → 0.97×** |
| **MACRO** | | | **0.0977** | **0.0164** | |

The old system printed raw sigmoid outputs to a clinician as "probability %" while
HYP was over-predicted **4.14×**.

### 11.4 The conformal layer — the number that actually matters

**A referred case is not a clinical miss: it reached a human.** The only true miss
is a positive that was **ruled out** and therefore never reached anyone.

| Class | α promised | Miss observed | Held? | **Escape rate** | vs baseline FN rate | Ruled out | Referred | Ruled in |
|---|---|---|---|---|---|---|---|---|
| NORM | 0.20 | 0.033 | ✅ | 3.3% | 7.4% | 48.1% | 14.8% | 37.1% |
| **MI** | 0.05 | 0.015 | ✅ | **1.5%** | **29.5%** | 57.6% | 24.1% | 18.3% |
| STTC | 0.10 | 0.092 | ✅ | 9.2% | 14.2% | 60.9% | 7.9% | 31.2% |
| CD | 0.10 | 0.099 | ✅ | 9.9% | 21.9% | 54.6% | 11.8% | 33.5% |
| HYP | 0.15 | 0.121 | ✅ | 12.1% | 39.4% | 69.9% | 12.3% | 17.8% |

**All five guarantees hold on the unseen test fold.**

> **Missed infarctions fall from 29.5% to 1.5%.** The price, stated in the same
> breath: **50.9% of patients** have at least one class deferred to a
> cardiologist; **49.1%** are handled autonomously.

*(Baseline = the archive's F1-tuned single threshold, which had no referral
option, so every false negative escaped.)*

> ⚠️ **Provenance warning.** `audit/results/07_conformal_eval.txt` is headed
> `model : resnet` — the **baseline**, at δ = 0.05. Its numbers are close but not
> identical (MI referral 21.4% vs the shipped 24.1%; autonomous 55.1% vs 49.1%).
> The table above is the **shipped `resnet_se` at δ = 0.01**, read from
> `checkpoints/conformal_triage.json`. If you quote the audit file, say it is the
> baseline.

**Why δ = 0.01 and not 0.05:**

| δ | Guarantees held | MI escape | Referred |
|---|---|---|---|
| 0.05 | 3 / 5 | 2.2% | 45.5% |
| **0.01** | **5 / 5** | **1.5%** | **50.9%** |

At δ = 0.05, CD (0.106 vs 0.10) and HYP (0.152 vs 0.15) were left violated. The
cause is imperfect exchangeability between folds 9 and 10, not a bug in the bound.

### 11.5 The architecture ablation

Three architectures × three seeds, identical script, data and 40-epoch schedule.
Scored once on the untouched test fold by paired bootstrap over identical record
indices.

| Architecture | Components | Params | macro-AUROC | HYP AUROC |
|---|---|---|---|---|
| `resnet` | baseline | 1,018,501 | 0.9440 | 0.9248 |
| `resnet_se_no_se` | stem + attention pooling | 1,536,358 | **0.9446** | **0.9253** |
| `resnet_se` *(shipped)* | stem + attn + **squeeze-excitation** | 1,584,326 | 0.9404 | 0.9106 |

- The stem and attention pooling change **nothing measurable** (p = 0.74).
- Squeeze-excitation **loses 0.0042** macro-AUROC (**p = 0.0040**), and almost all
  of it lands on **hypertrophy** — the one class whose evidence is amplitude,
  which is exactly what an SE block re-weights.

> ⚠️ **You ship the architecture your own ablation ranks worst. Have the answer
> ready.** The shipped bundle is dated 18 Aug; the ablation 26–27 Aug. It
> postdates the calibrator and conformal thresholds, which are fitted per-model
> and would all need refitting. The honest answer: *"the ablation is a late
> finding, the correct next step is refitting the safety layer onto
> `resnet_se_no_se`, and I did not quietly re-label the shipped model to match the
> better result."*

### 11.6 Two-model serving

A class is **ruled out only if every model rules it out**; any disagreement
collapses to REFER. The merged rule-out set is the intersection, so the merged
miss rate is bounded by the **tightest** single-model guarantee.

| Class | Miss `resnet_se` | Miss `resnet` | **Miss merged** | Referred `resnet_se` | **Referred merged** |
|---|---|---|---|---|---|
| NORM | 0.033 | 0.044 | **0.018** | 14.8% | 14.3% |
| MI | 0.015 | 0.007 | **0.000** | 24.1% | 37.3% |
| STTC | 0.092 | 0.077 | **0.055** | 7.9% | 12.3% |
| CD | 0.099 | 0.106 | **0.070** | 11.8% | 17.8% |
| HYP | 0.121 | 0.098 | **0.053** | 12.3% | 19.6% |

The merged rule misses fewer true positives than *either* model on every class,
because the two do not miss the same cases. The price is referrals.

**The number worth reporting:** the two models disagree on at least one class in
**58.9%** of records, and reach **opposite** conclusions — one rules a class in
while the other rules it out — in **10.5%** (180 / 1,711). A single-model
deployment shows the clinician one of those two answers, with a guarantee
attached, and no indication the other exists.

### 11.7 Why HYP F1 is 0.41, and why that is the wrong question

F1 weights a missed infarction and an unnecessary review **equally**. No
cardiology pathway does that — the ESC 0/1 h hs-troponin algorithm and HEART are
governed by **sensitivity and NPV**. HYP F1 is 0.406; **HYP NPV is 0.981**.

Four measured reasons hypertrophy is the hardest class:

| Reason | Evidence |
|---|---|
| **Scarcity** | 1,468 positives, 7.7% prevalence, only **132 in the test fold** |
| **Entanglement** | Of HYP records, **63.8%** also carry STTC, 35.7% also CD, and **0%** are NORM |
| **Voltage = amplitude** | LVH is diagnosed by QRS amplitude; any amplitude normalisation destroys the evidence — which is why augmentation is capped at 0.9–1.1× |
| **Label noise** | ECG criteria for LVH have known low sensitivity against echocardiography, the true reference. The label is a proxy. |

**Published ceiling for HYP F1 on PTB-XL is ≈ 0.54.** Our HYP AUPRC improved
**0.5405 → 0.5842** over the audited baseline. A claim of HYP F1 ≥ 0.75 would
require AUPRC > 0.8 and is not supported by any published result on this dataset.

---

## 12. The research contribution

### The gap

Conformal prediction gives a classifier a **provable** bound: *"when I rule this
out, I am wrong at most α of the time."* It has been applied to ECG (2025) and to
medical imaging.

**But every published result validates the guarantee MARGINALLY — averaged over
the whole population.** A cardiologist never treats the average.

> **Gap: is a conformal ECG guarantee valid *conditionally on the patient*, or only
> on average?**

### The claim

> **A conformal ECG system can satisfy its advertised guarantee exactly — and
> still be unsafe for identifiable groups of patients.**

### Three independent demonstrations

**Axis 1 · conditional on the patient** — `audit/10_conditional_validity.py`,
`audit/11_significance.py`

Thresholds fitted the standard (marginal) way on fold 9; miss rate then measured
**inside each subgroup** of fold 10.

| Class | Promised α | Overall | <50 | 50–69 | ≥70 |
|---|---|---|---|---|---|
| NORM | 0.20 | 0.190 ✓ | 0.103 | 0.228 | **0.330** |
| MI | 0.05 | 0.015 ✓ | 0.000 | 0.011 | 0.019 |
| STTC | 0.10 | 0.092 ✓ | 0.128 | 0.080 | 0.093 |
| CD | 0.10 | 0.099 ✓ | **0.333** | 0.099 | 0.042 |
| HYP | 0.15 | 0.121 ✓ | 0.444¹ | 0.159 | 0.066 |

¹ n = 9 — indicative only.

**Every class passes overall. Two violations survive rigorous testing.**
23 class–subgroup cells were assessed with a Wilson confidence interval, an exact
one-sided binomial test, **Holm correction for 23 comparisons**, and a 2,000-draw
calibration bootstrap that refits the threshold each draw:

| Cell | Promised | Observed | 95% CI | Holm-adj. p | Bootstrap |
|---|---|---|---|---|---|
| **CD, age < 50** | ≤ 10% | **33.3%** (22/66) | [23.2%, 45.3%] | **5.1×10⁻⁶** | violated in **100%** of 2,000 draws |
| **NORM, age ≥ 70** | ≤ 20% | **33.0%** (34/103) | [24.7%, 42.6%] | **2.9×10⁻²** | violated in **100%** of 2,000 draws |

Both intervals lie **entirely above** the promised bound.

**The other 7 apparent excesses did not survive multiple-testing correction and
are reported as noise. Saying so is part of the result.**

> The overall CD figure — 9.9%, comfortably inside the promised 10% — gives no
> hint that a third of under-50 patients with conduction disturbance are missed.
> **Conduction disturbance in the young is not benign:** under 50 it raises
> Brugada, ARVC and inherited conduction disease — causes of sudden cardiac death
> in young adults. That is the group where a miss is least acceptable, and it is
> where the system fails worst.

**The fix: Mondrian (group-conditional) calibration** — one threshold per subgroup.

| | Cells satisfying the bound |
|---|---|
| Marginal calibration (standard practice) | 14 / 23 |
| **Mondrian (group-conditional)** | **22 / 23 (96%)** |

**And its cost — a second finding.** Subgroup calibration needs enough positives
*per group*. ST/T change in under-50s had 42 calibration positives, the PAC bound
became infeasible, and the system returned **λ = −∞**: it can never rule out ST/T
change in a young patient. **Conditional validity costs data, and the groups
needing it most have the least.**

**Axis 2 · conditional on the recording** — `audit/12_electrode_reversal.py`

Covered in §9, Defence 5.

| | Quality gate accepts | Mean quality index |
|---|---|---|
| Correct placement | 589/600 | 1.000 |
| RA/LA reversal | **587/600** | **1.000** |
| RA/LL reversal | **589/600** | **1.000** |
| LA/LL reversal | **589/600** | **1.000** |

| Class | Promised | Correct | RA/LA | RA/LL | LA/LL |
|---|---|---|---|---|---|
| NORM | ≤20% | 16.1% | **99.6%** ❗ | **100%** ❗ | 19.3% |
| MI | ≤5% | 1.0% | 0.0% | 1.0% | **14.4%** ❗ |
| STTC | ≤10% | 11.4% | **70.9%** ❗ | **33.5%** ❗ | 10.1% ❗ |
| HYP | ≤15% | 11.4% | **25.0%** ❗ | **47.7%** ❗ | 4.5% |

**Nine guarantees voided by a cable.**

At a realistic reversal prevalence of 0.4–4%, the **population-level** guarantee
still survives, because 96–99.6% of records are correctly placed. A hospital
auditing across all its ECGs would see nothing wrong. But for **the individual
patient whose electrodes were swapped**, the promise is void — and nothing tells
anyone.

**Detection alone is not the fix, and that is reported:** refusing the detector's
detections restored only **1 of 9** voided guarantees.

**Axis 3 · conditional on the label space** — `audit/13_out_of_scope.py`

There is no sixth output unit. Five sigmoids can each say "not me", but nothing in
the network can say *"none of the above, and here is what it is instead."*

Cardiologist-documented findings in the reference reports of the very records this
system was trained and evaluated on:

| Condition | Records | Share |
|---|---|---|
| Atrial fibrillation | 1,225 | 7.11% |
| Ventricular ectopy | 1,127 | 6.54% |
| Atrial ectopy | 617 | 3.58% |
| Ventricular tachycardia | 30 | 0.17% |
| Atrial flutter | 27 | 0.16% |
| SV tachycardia | 28 | 0.16% |
| Paced rhythm | 12 | 0.07% |
| **Any out-of-scope** | **2,455** | **14.26%** |

**114 test-fold recordings document atrial fibrillation or flutter:**

| | |
|---|---|
| Refused by quality control | **1 / 114** |
| **Carrying a statistical guarantee** | **113 / 114** |
| Guarantees that concern atrial fibrillation | **0** |
| **Reported as a normal ECG** | **2** |

> The system certifies what it can measure while being blind to the finding that
> will cause the stroke. **Every five-superclass PTB-XL paper inherits this
> failure**, because the benchmark scores only the five classes it defined.

**The fix and its cost:** 60 of 114 AF records are now flagged and have the
guarantee withheld. At the 5% false-positive budget, 53/695 test records (7.6%)
have the guarantee withheld. **53 of the 114 are still missed and still receive a
guarantee — the fix is partial, and the report says so.**

### The unification

| Marginal validity holds | Conditional validity fails |
|---|---|
| across the whole population | within a patient subgroup (age) |
| across all recordings | within a mis-acquired recording |
| across the label space | when the disease is not in it |

> **One principle, three independent demonstrations: a conformal guarantee
> averaged over a benchmark says nothing about the patient in front of you.**

### What is honestly mine

| Claim | Status |
|---|---|
| Conformal prediction for ECG | **Not mine** — Ann Noninvasive Electrocardiol 2025, cited |
| XAI → MI subtype localisation | **Not mine** — Strodthoff et al. 2024, cited |
| Mondrian calibration as a method | **Not mine** — Vovk 2003 |
| **Conditional (subgroup) validity of ECG conformal guarantees** | **MINE.** No prior work found. |
| **Evidence that marginal validity hides a 3.3× subgroup violation** | **MINE.** Measured, n reported. |
| **Mondrian calibration for ECG, with its data cost quantified** | **MINE** (ECG application and cost analysis) |
| **Recall-first operating point certified by a conformal lower bound** | **MINE** |

> ⚠️ **Verify before presenting:** the prior-art citations across this project
> were added from recall and have **not** been checked against the published
> record. Check **Vovk (ACML 2012)** and **Barber et al. (2021)** on conditional
> validity and its impossibility in general before they go on a slide.

---

## 13. Limitations

State these before anyone asks. A panel that finds a limitation you hid discounts
everything else you said.

- **PTB-XL only. No external validation.** German cohort, 1989–96.
- **Five superclasses.** Atrial fibrillation and other arrhythmias are **not
  detected** — their absence from a report is **not** evidence of their absence.
  14.3% of the dataset carries a documented finding the label space cannot
  express. The rhythm-scope check flags the irregular ones (48.9% sensitivity,
  4.8% FPR, AUROC 0.912) and withholds the guarantee; **regular** out-of-scope
  rhythms (paced, monomorphic VT, Brugada, long QT) remain silent failures.
- **PR / QRS / QT intervals and QRS axis are not measured.**
- **Labels used only SCP codes with `likelihood == 100`**, dropping 21% of PTB-XL.
  Results are **not directly comparable** to published benchmarks.
- **Territory localisation is a lead-group heuristic**, not clinically validated.
- **The electrode-reversal detector is a physiology rule, not a classifier:**
  ~70% sensitivity on RA/LA, ~61% on RA/LL, 4.5% false positives. LA/LL reversal
  leaves aVR unchanged and is essentially undetectable this way.
- **Every fix is partial, and each partiality is measured:** the scope check
  catches 60 of 114 AF records; the electrode detector restored 1 of 9 voided
  guarantees; Mondrian fixes 22 of 23 cells and makes one **infeasible outright**.
- **~6 s per analysis** (plotting and integrated gradients dominate; the
  classifier itself is ~20 ms).
- **The shipped architecture is not the best one in the ablation.** See §11.5.
- **The report model's weights are not in this repository** and it cannot be
  re-run — only its frozen test-fold output survives. See §10.

---

## Reproducing everything

```bash
# 26 checks that every known defect is closed
python -X utf8 audit/08_verify_fixes.py

# dataset integrity — splits, overlap, prevalence, entanglement
python -X utf8 analysis/01_dataset_deep_audit.py

# the three operating points and what each costs
python -X utf8 analysis/02_operating_point.py

# Contribution 1 — subgroup validity of the guarantees
python -X utf8 audit/10_conditional_validity.py

# statistical significance of the subgroup violations
python -X utf8 audit/11_significance.py

# Contribution 2 — electrode reversal voids the guarantee
python -X utf8 audit/12_electrode_reversal.py

# Contribution 3 — out-of-scope disease gets a guarantee anyway
python -X utf8 audit/13_out_of_scope.py

# multi-model serving — what the two-model rule costs and buys
python -X utf8 audit/14_multi_model.py

# Contribution 4 — cross-model disagreement as an acquisition check
python -X utf8 audit/15_disagreement_detector.py
```

Run the system:

```bash
python -X utf8 backend/server.py     # API on :5000
cd frontend && npm run dev           # UI on :5173
```

---

*Companion document: `COMPONENT_02_DEMO_AND_QA.md` — the 10-minute demo runsheet
and Q&A preparation.*
