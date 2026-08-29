# Component 02 — Complete System Reference

**XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting**
Venushan T · part of the Explainable AI System for Cardiovascular Disease Detection and Diagnosis
Dataset: PTB-XL v1.0.3 (PhysioNet, CC-BY 4.0)

> ⚠️ **AI-generated decision support. NOT a medical device and NOT a diagnosis.**
> Every report requires review by a qualified clinician before any clinical action is taken.

---

This document is the single reference for the whole component: every file explained,
how they connect, the full model specification, every accuracy figure, every known
fault, and the outcome of all eleven experiments in the repository.

Every number is traceable to a named file on disk. The threshold-free metrics in §11
were **recomputed** from `checkpoints/val_logits_seed0.npy` and `test_logits_seed0.npy`;
the parameter counts and tensor shapes in §10 were obtained by instantiating the models.
Nothing here was estimated or supplied from outside the repository.

| | |
|---|---|
| Source lines | 9,048 |
| Python modules | 14 library + 12 scripts |
| Test records | 1,711 (fold 10, never used to select anything) |
| Parameters | 1,584,326 |
| Test macro-AUROC | 0.9320 (seed 0) |
| Test macro-recall | 0.810 |
| Test macro-NPV | 0.933 |

### Contents

1. [What the system is](#1--what-the-system-is)
2. [Repository map](#2--repository-map)
3. [The `src/` library, file by file](#3--the-src-library-file-by-file)
4. [Backend API](#4--backend--backendserverpy)
5. [Frontend](#5--frontend--react-clinical-review-ui)
6. [Training & fitting](#6--training--fitting)
7. [Analysis & audit scripts](#7--analysis--audit-scripts)
8. [Data & shipped artefacts](#8--data--shipped-artefacts)
9. [How it all fits together](#9--how-it-all-fits-together)
10. [The model, in full](#10--the-model-in-full)
11. [Accuracy — every figure](#11--accuracy--every-figure)
12. [Faults found and fixed](#12--faults-found-and-fixed)
13. [Faults that remain](#13--faults-that-remain)
14. [Every outcome, by experiment](#14--every-outcome-by-experiment)
15. [Contradictions to resolve](#15--contradictions-to-resolve-before-writing)
16. [How to run it](#16--how-to-run-it)

---

## 1 · What the system is

A 12-lead ECG goes in as a WFDB `.dat` + `.hea` pair. What comes back is a JSON object
containing five calibrated probabilities, a three-way decision per class carrying a
proven bound on how often that decision is wrong, an explanation of which leads and
which moments drove it, a clinical report that has been machine-checked against those
numbers, and a rendered ECG image.

The five classes are the PTB-XL diagnostic superclasses:

| Code | Meaning | Plain English |
|---|---|---|
| **NORM** | Normal | Nothing wrong |
| **MI** | Myocardial infarction | Heart attack, or scar from an old one |
| **STTC** | ST/T change | Possible reduced blood supply |
| **CD** | Conduction disturbance | Faulty electrical wiring |
| **HYP** | Hypertrophy | Thickened heart muscle wall |

They are **not** native PTB-XL columns. They are derived by mapping SCP-ECG statement
codes through `csv/scp_to_superclass_mapping.json` (44 codes: MI 14, STTC 13, CD 11,
HYP 5, NORM 1), keeping only codes with `likelihood == 100`. A patient can carry more
than one class — the task is per-class binary, not mutually exclusive.

### The three design commitments

Most of the code exists because of these three, not because of the classifier.

1. **It is allowed to refuse.** A quality gate runs *before* the model, so an
   uninterpretable recording never produces a probability at all.
2. **It is allowed to say "I don't know."** Each class lands in one of three zones —
   `rule out` / `refer` / `rule in` — and the outer two carry a finite-sample conformal
   bound on their error rate.
3. **Its own text is checked before release.** Every sentence traces back to a
   structured `Finding`, and a separate verifier blocks the report if the words and the
   numbers disagree.

### The research contribution

Three audits that ask whether the statistical guarantee *survives* — when the population
is split by age and sex, when two electrodes are swapped, and when the patient's actual
disease is not in the label space at all. All three find that it does not. Two ship a
mitigation. See §14.

---

## 2 · Repository map

### Repository root — `RP-Venu/`

| Path | Role | Status |
|---|---|---|
| `Component_02/` | The deliverable. Self-contained; nothing inside reads from outside it on the serving path. | current |
| `_archive/` | The superseded Progress-1 system — `app.py` (server-rendered Flask), `training/` (13 scripts), three checkpoint sets, old documentation. Kept as audit evidence. | superseded |
| `README.md` | ⚠️ Describes the **archive** system: BioBART three-tier reporting, fusion model, macro-F1 0.717, "run `python app.py`". Not updated for Component 02. | **stale** |
| `.env.example` | `PHYSIONET_USER` / `PHYSIONET_PASS` — credentials moved out of source (audit finding E-1). | current |
| `.gitignore` | Excludes weights, WFDB data and packed memmaps; explicitly whitelists the small JSON/CSV artefacts under the comment "these ARE the reproducibility record". | current |

### Inside `Component_02/`

| Folder | Files | Lines | Contents |
|---|---:|---:|---|
| `src/` | 14 | 2,803 | The importable library. This is the system. |
| `backend/` | 2 | 479 | JSON-only Flask API. No HTML, no templates. |
| `frontend/` | 10 | ~800 | React 19 + Vite 6 + Tailwind 4 clinical review UI. |
| `train/` | 4 | 1,092 | Preflight, GPU training, calibration fitting, Colab notebook. |
| `analysis/` | 2 | 491 | Dataset audit and operating-point selection, plus results. |
| `audit/` | 11 | 1,275 | Regression suite + three contribution experiments; `legacy/` holds six audits of the archive. |
| `checkpoints/` | 9 | — | Weights, calibrator, conformal thresholds, operating point, scope threshold, saved logits and labels. |
| `csv/` | 6 | — | Metadata, official split, normalisation stats, SCP→superclass map. |
| `data/` | 17,221×2 | — | PTB-XL WFDB records at 500 Hz + the CC-BY licence text. |
| `docs/` | 5 | 1,860 | Panel answers, audit findings, research contribution, Colab guide. |
| `reference/` | — | — | Superseded checkpoints and legacy documents kept as evidence. |

Line counts from `wc -l`; `frontend/dist/` and `node_modules/` excluded.

---

## 3 · The `src/` library, file by file

Fourteen modules, one responsibility each. Everything imports its constants from
`models.py`; everything that needs a file on disk asks `paths.py`; and `pipeline.py` is
the only thing that runs the stages in order.

### Foundation

#### `src/models.py` — 238 lines

Single source of truth for every network architecture. Exists because the archive
defined the same ResNet in **five separate files**, so any edit had to be made five times
or inference silently diverged from training (audit finding E-10).

| | |
|---|---|
| Constants | `CLASS_NAMES`, `LEAD_NAMES`, `NUM_LEADS=12`, `NUM_CLASSES=5`, `SIGNAL_LENGTH=5000`, `SAMPLING_RATE=500` |
| Baseline | `ResidualBlock`, `ECGResNet` — 1,018,501 params, loads the archive checkpoint bit-exactly |
| Shipped | `SEBlock`, `SEResidualBlock`, `MultiKernelStem`, `AttentionPool`, `ECGResNetSE` — 1,584,326 params |
| Loss | `FocalLoss` — its docstring carries a hard constraint: never combine a non-trivial `alpha` with a balanced sampler |
| Factory | `build_model(name)` — `"resnet"` or `"resnet_se"` |

One subtle detail: `ResidualBlock.forward` uses an out-of-place `out + residual` rather
than the archive's `out += residual`. An in-place op on a tensor carrying a full-backward
hook corrupts Grad-CAM gradients. Numerically identical, functionally not.

#### `src/paths.py` — 78 lines

The only module that knows where data lives. Resolves every asset through one ordered
candidate list, which is what lets the folder be handed to someone else and still run.

- **Order:** `$ECG_DATA_DIR` → `csv/` → `data/` → `assets/` → `../_archive/data/`
- **API:** `find` (or None), `require` (or exit printing the list it searched),
  `signals_cache`, `describe`

#### `src/signals.py` — 70 lines

Loads one test-set ECG by id. Prefers the derived `signals_cache/<id>.npy`
(240 KB/record, 3.9 GB total) but falls back to reading the original WFDB record
(120 KB/record, 1.93 GB total). Shipping the WFDB halves the deliverable and hands on
authoritative data rather than a derived artefact; the ~4 ms read cost is negligible
against a ~6 s analysis.

#### `src/__init__.py` — 23 lines

Package exports — the public surface another component would import. `__version__ = "0.2.0"`.

---

### Gate — everything that can stop a record before the model sees it

#### `src/quality.py` — 314 lines

The gate, and the fix for the single most dangerous defect in the archive (C-1): an
all-zero signal produced *"MI 0.691, STTC 0.602 → consistent with myocardial
infarction"*. Nothing reaches the model now without passing these checks, and a record
that fails is **refused, not diagnosed**.

| | |
|---|---|
| Order | shape → duration → finite → flat leads → units/gain → amplitude → noise → rhythm → electrodes → scope → SQI |
| Bounds | duration 5–60 s · \|x\| ≤ 20 mV · p95 \|x\| ∈ [0.15, 6.0] mV · ≤ 2 flat leads · HR 25–250 bpm · ≥ 3 QRS |
| Units | `detect_and_fix_units` accepts **only** powers of 1000 (µV, V, nV) and only when the rescaled amplitude lands comfortably inside the band. An arbitrary factor that happens to fit is a gain *fault* and is refused, not silently corrected. |
| Peaks | `detect_r_peaks` — Pan-Tompkins-style: 5–15 Hz band-pass, differentiate, square, 120 ms moving-average integrate, `find_peaks` with a 250 ms refractory (240 bpm ceiling). No external dependency. |
| SQI | `1 − (0.20×flat + 0.08×noisy + 0.15 if gain corrected + 0.10 if no HR)`, clipped to [0,1] |

The check order is load-bearing and documented in the source. Flat-lead detection uses a
**relative** criterion and runs **before** unit inference — dead leads drag the amplitude
percentile down, so running the unit check first made a record with eight disconnected
leads look like a gain error instead.

#### `src/electrodes.py` — 183 lines

Limb-electrode reversal screening. Physiology, not machine learning. A reversed recording
is perfectly clean — correct amplitude, no noise, no flat leads — so no signal-quality
metric can see it, yet the waveform is a mathematically transformed version of the truth.

Limb reversals are **exact** linear maps because the augmented leads are derived.
Precordial leads are unaffected: Wilson's central terminal `(RA+LA+LL)/3` is invariant
under a limb swap.

| Reversal | Effect |
|---|---|
| **RA/LA** | I → −I, II ↔ III, aVR ↔ aVL, aVF unchanged |
| **RA/LL** | I → −III, II → −II, III → −I, aVR ↔ aVF, aVL unchanged |
| **LA/LL** | I ↔ II, III → −III, aVL ↔ aVF, **aVR unchanged** — which is why this one is undetectable by polarity rules |

**Detection rules:** aVR polarity > 0.15 *and* lead-I polarity < −0.10 → RA/LA ·
aVR > 0.30 alone → unspecified limb problem · aVF < −0.35 with non-negative aVR → RA/LL

**Policy:** raises suspicion, never asserts. Does not touch `acceptable` or `sqi` — a
4.5 % false-positive rate makes automatic refusal the wrong default. It suspends the
guarantee instead.

#### `src/scope.py` — 145 lines

Detects disease *outside* the label space. The model has no output unit for atrial
fibrillation, so when one walks in the network does not abstain — softmax has no "none of
the above". It redistributes the evidence across the five classes it does have, and the
report prints a statistical guarantee next to it.

| | |
|---|---|
| Features | `cv` (coefficient of variation of RR), `rmssd`, `pnn50`, `irr` = median \|ΔRR\| ÷ median RR |
| Gate | needs ≥ 5 R-peaks and ≥ 4 usable RR intervals in the 24–240 bpm window; otherwise returns "in scope, too few beats to assess" |
| Threshold | `irr > 0.1792` (from `checkpoints/scope.json`; module fallback 0.179) → out of scope |
| Cost | Free — the pipeline already detects R-peaks to compute heart rate. |

Note the deliberate coupling: every detail sentence this module emits also carries the
disclaimer wording, so `verify.py` can distinguish an admission of ignorance from a
hallucinated diagnosis. Without that, the safety verifier would flag the system for
admitting its own limits.

---

### Signal & probability

#### `src/preprocess.py` — 139 lines

Deterministic conditioning, shared by training and serving — the module header says
plainly: *import it, do not reimplement it*. Fixes E-5 (the archive resampled *any* length
to 5000 samples with no sampling-rate check, compressing a 30 s strip 3× and distorting
heart rate silently) and the total absence of filtering.

| | |
|---|---|
| Chain | resample by **rate** → high-pass 0.5 Hz (Butterworth 3) → low-pass 40 Hz (Butterworth 4) → 50 Hz notch (Q = 30) → centre-crop or zero-pad to 5000 → normalise |
| Filters | all zero-phase `filtfilt`; follows the AHA/ACC diagnostic-ECG recommendation |
| Normalise | per-record median removal (robust DC offset) **then** global per-lead mean/std from `norm_stats.json` |
| Augment | noise σ0.05 @50 % · scale 0.9–1.1 @50 % · time roll ±250 @30 % · 1–2 lead dropout @25 % · baseline wander @20 % |

The scale band is narrow on purpose. Hypertrophy is diagnosed from QRS **amplitude**
(Sokolow-Lyon, Cornell), and the audit measured the classifier as highly gain-sensitive
(×10 → HYP 0.998). Wide scale augmentation would teach it to ignore exactly the evidence
HYP depends on.

#### `src/calibration.py` — 129 lines

Per-class temperature scaling (Guo et al., ICML 2017), fitted by LBFGS on the validation
fold only. Fixes C-6: the archive applied `WeightedRandomSampler` oversampling **and**
focal-loss alpha weights together, correcting class imbalance twice, so the sigmoid
outputs were not posterior probabilities — yet the UI printed them to a clinician as
"probability %". HYP was predicted at 4.14× its prevalence.

| | |
|---|---|
| Form | `p = σ(z / T_c + b_c)` — one temperature and one bias per class |
| Property | Monotone, so AUROC and the conformal bounds are untouched. Only the numbers shown to a human become honest. |
| Provenance | `save(..., fitted_for=)` records which model these belong to. A calibrator paired with a different model produces nonsense while everything still "runs" — it happened twice during development. |
| Metrics | `expected_calibration_error` (10 bins), `calibration_report` (ECE, Brier, mean p, prevalence, over-prediction ratio, macro ECE) |

#### `src/conformal.py` — 360 lines

**Research contribution 1.** Replaces a single F1-tuned threshold with a *pair* of
conformal thresholds per class and an abstention zone between them. The archive's F1
threshold missed 29.5 % of infarctions and reported them as normal with no indication
anything was uncertain.

```
score < λ_out          → RULE OUT   guaranteed miss rate ≤ α
λ_out ≤ score < λ_in   → REFER      defer to a cardiologist
score ≥ λ_in           → RULE IN    guaranteed false-alarm rate ≤ β
```

| | |
|---|---|
| Theory | One-sided split conformal. Conditioning on `Y_k = 1` keeps exchangeability inside the positive subset, so `P(S_{n+1} < S_(m)) ≤ m/(n+1)`. The bound holds for **any** model — a weak one simply defers more often. |
| PAC | `_pac_order_statistic(n, α, δ)` — the coverage of the k-th order statistic is `Beta(k, n−k+1)`, so it takes the largest k with `P(Beta ≤ α) ≥ 1−δ`. Strictly more conservative than the marginal `⌊α(n+1)⌋`, and it is what makes the guarantee survive a **single** realisation. Falls back to the marginal bound if SciPy is unavailable. |
| Infeasible | When `m < 1` it returns `−inf` and says so — "this class can never be ruled out" — rather than silently pretending. |
| Overlap | If `λ_out > λ_in` the two zones overlap and a score would satisfy both bounds at once. The only safe resolution is to refer the entire overlap; the code swaps the two and records a note. This happens for NORM in the shipped model. |
| Presets | `safety` (shipped) / `balanced` / `throughput` — described in the source as the system's clinical **policy**, "meant to be argued about, not hidden" |

---

### Explanation, report, gate

#### `src/xai.py` — 297 lines

Grad-CAM, signed integrated gradients, and the mapping from lead attributions onto
coronary territories — which is what turns XAI into **report content** rather than a
picture beside the report.

| | |
|---|---|
| C-5 fix | The archive kept **one** module-level Grad-CAM object storing activations and gradients as instance state, under Flask's threaded server. Four concurrent calls were tested; three returned corrupted heatmaps. Hooks are now created and removed inside a per-call context manager, and a `WeakKeyDictionary` of per-model `RLock`s serialises the 60 ms critical section. |
| 7.4 fix | The archive applied `.abs()` before summing attributions over time, discarding sign — a lead *arguing against* the diagnosis was shown to the clinician as "important". Sign is now preserved and reported separately from magnitude. |
| IG | `steps = 64` — the audit measured completeness error 1.3 % at 30 steps and 0.2 % at 100; 64 is the knee. Ranking is stable (Spearman 0.999 vs 200 steps). |
| Localise | Only **positive** attribution counts, and each territory's share is normalised by its lead count so five-lead anterolateral is not favoured over two-lead septal. Requires ≥ 1.35× the uniform baseline before it will name a territory at all. |
| Evaluation | `deletion_insertion_auc` vs a random-order baseline, and `qrs_alignment` — the fraction of Grad-CAM mass inside a ±60 ms window around each R-peak |

**Territories:** anterior V1–V4 (LAD) · septal V1–V2 (proximal LAD) · lateral I/aVL/V5/V6
(LCx) · anterolateral V4–V6/I/aVL · inferior II/III/aVF (RCA)

#### `src/report.py` — 365 lines

Grounded report generation. Every sentence is emitted from a `Finding` object and every
`Finding` carries the evidence that produced it, so nothing can be written that is not
traceable to a measurement, a conformal decision, or an attribution.

| | |
|---|---|
| Zones → language | The conformal zone chooses the **modality** of the sentence. Rule-in asserts; refer says "Possible X (calibrated probability N %) … the system does not decide it". |
| C-4 fix | NORM is structurally mutually exclusive with any ruled-in abnormality — if an abnormality is ruled in, NORM is demoted to `REFER` before any text is generated. The archive emitted both sentences in the same paragraph for 99 of 1,711 records. |
| Triage | MI ruled in → `IMMEDIATE` · MI referred → `URGENT` · CD/STTC ruled in → `PRIORITY` · nothing crossing → `PRIORITY` · NORM ruled in → `ROUTINE` · gate failed → `REPEAT ECG` |
| Guarantees | Emitted per ruled-out class — but **suppressed entirely** if the electrodes are suspect or the rhythm is out of scope, replaced by an explicit withdrawal sentence. |
| Refusal path | Returns "ECG NOT INTERPRETED", no probabilities, and the reason: *"a diagnosis derived from an uninterpretable signal is worse than no diagnosis"*. |

#### `src/verify.py` — 265 lines

The safety gate. Fixes C-3. The archive claimed its BioBART tier "cannot hallucinate
because it never sees the raw ECG" — a claim that confuses two things. Not seeing the
signal prevents inventing *evidence*; it does not stop a seq2seq model adding, dropping or
negating a *finding* while rephrasing. Measured on 1,711 records, it dropped a clinical
concept in 103 and inserted "atrial fibrillation" in 42.

| | |
|---|---|
| Checks | 1 no invented diagnosis · 2 no NORM + abnormality contradiction · 3 overclaiming language · 4 mandatory disclaimer and triage tier · 5 every ruled-in finding actually appears · 6 referred classes marked uncertain |
| Negation | `_negated()` is **directional within a clause**. A fixed character window was too short ("RULED OUT: normal ECG, ST/T change …" put the negator far back); a whole-clause scan was too coarse ("meets criteria for a normal study; no abnormality crossed…" put a negator *after* the term). Only preceding text inside the clause counts. |
| Allow-list | `SCOPE_DISCLAIMERS` — contexts where naming an out-of-scope diagnosis is the **opposite** of hallucinating it. Without it the scope warning would be rejected as a hallucination. |
| Paraphrase | `verify_paraphrase` enforces bidirectional containment — may not add a class the source didn't assert, may not drop one it did. Any future language layer must pass through it. `safe_paraphrase` returns the template text on failure: degraded fluency, never degraded safety. |

#### `src/pipeline.py` — 197 lines

The single inference entry point. Nothing may bypass this order.

```
 1 quality gate    qc.assess(signal_raw, fs)  →  if not acceptable: RETURN refused report
 2 preprocess      pp.prepare(...)            →  (12, 5000) normalised
 3 classify        model(x)                   →  5 raw logits
 4 calibrate       calibrator.predict_proba   →  5 honest probabilities
 5 conformal       triage.zones_one(cal)      →  rule_out / refer / rule_in per class
 6 explain         explain(...) × max 3       →  Grad-CAM + signed IG + territory
 7 report          build_report(...)          →  grounded text + guarantees
 8 verify          verify_report(rep)         →  if failed: WITHHOLD text, triage = REPEAT ECG
```

| | |
|---|---|
| Provenance | `from_checkpoint` warns loudly if the calibrator or conformal thresholds were fitted for a different model or a different filter setting. |
| XAI budget | Explanations are computed only for classes actually being reported (rule-in or refer, excluding NORM), capped at three. If none qualify it explains the argmax class. |
| Threading | No mutable module-level state and no shared XAI object (audit C-5). |

---

## 4 · Backend — `backend/server.py`

479 lines. Serves **data only** — no HTML, no templates — so any client can consume it.
The React app in `frontend/` is one such client.

### Startup is a safety check

`_startup()` builds the pipeline and then **refuses to start** if the calibrator or the
conformal thresholds were fitted for a different model or a different filter setting. The
old app defaulted to the archive checkpoint while the shipped calibrator belonged to the
retrained model — a pairing that silently produces invalid probabilities and void
guarantees while everything appears to work. `ECG_ALLOW_MISMATCH=1` downgrades it to a
warning, and the error message prints the exact `fit_calibration.py` command that fixes it.

### Endpoints

| Method | Route | Behaviour |
|---|---|---|
| `GET` | `/api/health` | Model name, checkpoint, whether calibration and conformal are loaded, PAC δ, class list, per-class thresholds and calibration counts, test-set size, whether browsing is enabled, the disclaimer. |
| `GET` | `/api/patients/<class>` | Browse the bundled test fold, filtered by superclass, with optional `?q=` substring search over id and reference report. Capped at 500 rows. |
| `POST` | `/api/analyze/<ecg_id>` | Analyse a bundled record. Additionally returns `referenceReport`, `groundTruth` and patient age/sex — labelled in the UI as "dataset ground truth, not shown in deployment". |
| `POST` | `/api/demo` | Random record from the test fold. |
| `POST` | `/api/predict` | **The integration point.** `multipart/form-data` with `dat_file` + `hea_file`. Validates extensions, requires a shared base name, sanitises with `secure_filename`, reads via `wfdb.rdrecord` inside a `TemporaryDirectory`, rejects anything that is not 12-lead, and reorders leads to the standard order if the header names them out of order (with a warning). |

Configure with env vars: `HOST`, `PORT`, `CORS_ORIGINS`, `ECG_CKPT`, `ECG_MODEL`,
`ECG_FILTER`. Defaults are correct for the shipped model. Binds `127.0.0.1` unless `HOST`
is set explicitly (E-9). Upload cap 25 MB.

> ✅ **Resolved open question.** The write-up flags as unknown whether `/api/predict`
> persists uploaded files. It does not — the files are written inside a
> `tempfile.TemporaryDirectory()` context and the directory is deleted when the block
> exits, before the signal is analysed. Nothing is retained on disk.

### Response shape (abbreviated)

```jsonc
{
  "triage": "IMMEDIATE",              // IMMEDIATE|URGENT|PRIORITY|ROUTINE|REPEAT ECG
  "electrode": { "suspected": false, "reversal": null, "confidence": 0.0, "reasons": [] },
  "scope":     { "outOfScope": false, "reason": null, "confidence": 0.0, "detail": [] },
  "headline": "myocardial infarction",
  "refused": false,
  "classes": [ { "name": "MI", "probability": 59.9, "zone": "rule_in",
                 "ruleOut": 3.8, "ruleIn": 33.9, "alpha": 0.05 } ],
  "reportText": "...",
  "findings": [ /* structured, each with its evidence */ ],
  "ruledOut": ["NORM", "STTC"],
  "guarantees": ["... miss-rate bound of 5% ..."],
  "quality": { "sqi": 1.0, "heart_rate_bpm": 61.6, "errors": [], "warnings": [] },
  "verification": { "passed": true, "errors": [] },
  "explanation": { "territory": "septal", "artery": "proximal LAD",
                   "topLeads": ["V1","V2","V5"], "peaksSeconds": [1.5, 3.8] },
  "leads": [ { "name": "V1", "signed": 18.6 } ],
  "ecgImage": "<base64 png>"
}
```

### Four response fields a caller MUST respect

| Field | When true | Required caller behaviour |
|---|---|---|
| `refused` | quality gate failed | No probabilities are returned. **Do not treat as normal.** Show `quality.errors` and ask for a repeat ECG. |
| `verification.passed = false` | generated text failed the safety check | Text was withheld. Do not display `reportText` as a report. |
| `electrode.suspected` | limb reversal suspected | Probabilities still valid to show, but **the statistical guarantees do not apply**. Prompt for a repeat. |
| `scope.outOfScope` | irregularly irregular R-R | Guarantees withheld and an arrhythmia has **not** been excluded. Do not present the five-class result as complete. |

Documented as a contract, but **nothing enforces it at the protocol level** — a downstream
integrator that ignores these presents an unsafe result as a clean one.

### ECG rendering

`render_ecg()` draws on standard ECG paper rather than plotting a curve: 25 mm/s,
10 mm/mV, 3×4 lead layout plus a full-duration lead-II rhythm strip, a 1 mV / 200 ms
calibration pulse, fine 0.04 s × 0.1 mV and bold 0.20 s × 0.5 mV grids, and a dark-theme
variant selected by `?theme=dark`. The Grad-CAM overlay is drawn as a thin band **beneath**
each row, never over the trace — a wash across the waveform obscures the exact morphology a
reader needs. Returned as a base64 PNG in `ecgImage`. This is also the slowest step in the
request.

---

## 5 · Frontend — React clinical review UI

Ten source files. Its own design brief, stated in `index.css`: *read as clinical review
software, not a dashboard. Light by default; colour carries meaning only; numbers
monospace, tabular, right-aligned; dense over airy; no emoji, no gradients, no decorative
shadows; the ECG trace is the primary artefact and gets the most space.*

| File | Lines | Responsibility |
|---|---:|---|
| `main.jsx` | 10 | React 19 root mount in StrictMode. |
| `App.jsx` | 152 | All state: health, selected class, patient list, active result, busy/error flags, theme. One `run()` helper wraps every analysis call. Composes the layout. |
| `api.js` | 36 | Five thin fetch wrappers. Appends `?theme=` from the current DOM class so the server renders the ECG to match. Turns non-JSON responses into "Is the backend running?". |
| `index.css` | 78 | The design system: `ink-50…950` slate scale, four severity colours (`crit` #b91c1c, `urgent` #c2410c, `review` #a16207, `clear` #15803d), `.panel` / `.panel-head` / `.num` / `.tag` / `.kv` primitives. |
| `components/ui.jsx` | 65 | `ZONE` and `TRIAGE` lookup tables, `Panel`, `Tag`, `Spinner`, `Empty`. |
| `components/Worklist.jsx` | 149 | Left rail. Class filter with live counts, id/report search, dense study rows, random-study button, drag-and-drop `.dat` + `.hea` importer. |
| `components/PatientBanner.jsx` | 85 | Sticky identification band with a severity spine coloured by triage tier — so a reviewer can never be looking at the wrong patient's data. |
| `components/EcgViewer.jsx` | 51 | The trace, first and largest. Fit-width / full-size toggle, PNG download, peak-attention timestamps. |
| `components/Interpretation.jsx` | 227 | Three exports. `DecisionTable` renders each class as a bar showing where the probability sits **between** λ_out and λ_in — that position *is* the decision. `Measurements` shows rate, complexes, duration, SQI, usable leads, gain, honest "not measured" rows for intervals and axis, plus the scope and electrode warning blocks. `LeadEvidence` is a signed diverging bar chart — red supports, green argues against. |
| `components/ReportCard.jsx` | 155 | Structured / plain-text toggle. Shows the withheld-report banner when verification fails, then interpretation with per-finding evidence lines, excluded classes, statistical basis, limitations, and the dataset reference report in a visually separated block. |
| `vite.config.js` | 18 | Dev server on 5173, proxies `/api` → `127.0.0.1:5000` so the browser never sees a cross-origin request. |

---

## 6 · Training & fitting

#### `train/preflight.py` — 232 lines

Run before training. Every check corresponds to a way a paid Colab run silently wastes
compute units: wrong paths crash at minute four, missing packed data falls back to
per-file reads at 6× the time, too-large batch OOMs at epoch one, no GPU attached trains
for eight hours, a small disk crashes at the checkpoint write, a corrupt memmap trains on
garbage undetected.

#### `train/train_gpu.py` — 528 lines

The classifier trainer, written around a one-GPU-hour budget.

| | |
|---|---|
| Pack | `--pack` converts 17,221 per-record `.npy` files into three float16 memmaps. **Refuses to proceed if any cached signal is missing** — the archive's silent zero-fill bug (E-4) is exactly how corrupt records enter training. |
| Imbalance | Balanced sampler **XOR** focal alpha, never both (C-6). |
| Resilience | `--resume` restores optimizer, scheduler, EMA and RNG state · CUDA OOM halves the batch and rebuilds OneCycleLR rather than dying · NaN guard stops a diverged run · `--max-minutes` hard wall-clock budget · atomic checkpoint writes · SIGINT finishes the epoch and saves |
| Workers | `_worker_init` gives every DataLoader worker its own RNG stream. Without it all workers produce identical augmentations every epoch, making augmentation a no-op after epoch 1. |
| Outputs | `best_model.pt`, `last.pt`, and — critically — `val_logits_seed0.npy`, `test_logits_seed0.npy`, `val_labels.npy`, `test_labels.npy`, so calibration needs no second inference pass |

#### `train/fit_calibration.py` — 332 lines

Turns the trained checkpoint into the deployed system. No GPU, ~3 minutes on CPU. Fits
the temperature calibrator and the conformal thresholds on **fold 9 only**, then verifies
both on fold 10.

- **Writes:** `checkpoints/calibrator.json`, `checkpoints/conformal_triage.json`,
  `audit/results/07_conformal_eval.txt/.json`
- **Guard:** refuses to run if the packed data's filter flag disagrees with `--filter`.
  Feeding filtered data to a model trained unfiltered shifted macro-ECE from 0.183 to
  0.209 *silently* before this check existed. `ECG_NO_PACKED=1` is the escape hatch — and
  re-opens the hole.

#### `train/Component02_Colab.ipynb`

The Colab entry point described in `docs/COLAB_GUIDE.md`. Its organising principle: *do
the slow, boring part on your laptop; do only the GPU part on Colab.* Reading 17,221 small
files from Drive would dominate training time — you would pay GPU rates to wait on network I/O.

---

## 7 · Analysis & audit scripts

### `analysis/` — reads only bundled assets, never `../_archive`

**`01_dataset_deep_audit.py`** (265 lines) — answers every dataset question a panel can
ask, with evidence: provenance, split integrity (patient leakage, fold assignment,
duplicates), label structure (prevalence, co-occurrence, NORM exclusivity), signal
integrity, demographics (the age-300 sentinel, imputation artefacts), and a dedicated
section measuring *why* hypertrophy is the hardest class.

**`02_operating_point.py`** (226 lines) — selects and certifies the shipped operating
point. Per class it takes the highest threshold whose **validation** recall clears a
sensitivity floor, then reports the full confusion profile on the untouched test fold.
Three policies printed side by side: default 0.5, F1-optimal, recall-first (shipped).
Fold 10 is scored once and never used to select anything.

### `audit/` — current suite

**`08_verify_fixes.py`** (215 lines) — the regression suite. Re-runs the exact adversarial
inputs that broke the archive and shows the new behaviour side by side. 26 checks; run it
after any change.

**`10_conditional_validity.py`** (264 lines) — **Contribution experiment 1.** A conformal
guarantee is marginal: it holds on average over the whole distribution. A clinician does
not treat "the whole distribution". Section A measures the realised miss rate inside each
sex and age subgroup and refits Mondrian (group-conditional) thresholds as the repair.
Section B tests whether the quality gate itself is label-dependent, which would break
exchangeability and void the guarantee via the system's own safety mechanism.

**`11_significance.py`** (244 lines) — closes the "isn't 33 % of 66 patients just chance?"
question three independent ways: Wilson intervals on every subgroup rate, exact one-sided
binomial tests with Holm correction across all 23 cells, and a calibration-draw bootstrap
that resamples fold 9, refits the threshold, and re-measures on fold 10 — the conformal
analogue of a multi-seed run.

**`12_electrode_reversal.py`** (304 lines) — **Contribution experiment 2.** Prior work on
lead reversal detects it and measures accuracy loss under it. Neither asks what it does to
a statistical *guarantee*. Six stages: is it visible to the gate, how many diagnoses
change, does the guarantee survive, can a physiology detector catch it, does refusing
detections restore validity, and what detection sensitivity would actually be enough at
realistic reversal prevalence.

**`13_out_of_scope.py`** (248 lines) — **Contribution experiment 3.** Measures how much
disease in PTB-XL the five-class label space cannot express, what the system reports for
atrial fibrillation, how many of those reports carry a guarantee, and whether a
validation-fitted rhythm check can withhold it. Also fits and saves `checkpoints/scope.json`.

> **A defect found and fixed inside this script.** The shipped threshold must not depend on
> `--n`. Fitting it on a subsample made a smaller run silently degrade the deployed system —
> a `--n 400` run pushed sensitivity from 71 % to 28 %. The threshold is now always fitted
> on the full validation fold; `--n` only limits the test-side evaluation.

**`audit/legacy/`** — `01_data_audit`, `02_model_audit`, `03_report_audit`, `05_xai_audit`,
`06_leakage_audit`, `09_verify_transfer`. Audits of the superseded system. They require
`../_archive` to run, which is why they are quarantined here.

---

## 8 · Data & shipped artefacts

### Dataset

| Property | Value |
|---|---|
| Source | PTB-XL v1.0.3 (PhysioNet), CC-BY 4.0 — redistribution permitted with attribution |
| Format | WFDB `.dat` + `.hea`, 12-lead, 500 Hz, 10 s |
| Official size | 21,799 records / 18,869 patients |
| **Used here** | **17,221 records / 15,174 patients** — 4,578 dropped (21.0 %) |
| Drop rule | Only SCP codes with `likelihood == 100` retained |
| Split | Official `strat_fold`: 1–8 train / 9 val / 10 test — patient-disjoint, **0 overlap on all three pairs** |

### Split & prevalence

| Split | Records | Patients | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|---:|---:|
| train (1–8) | 13,801 | 12,109 | 40.77 % | 17.56 % | 26.48 % | 27.61 % | 8.71 % |
| val (9) | 1,709 | 1,550 | 40.90 % | 16.56 % | 26.62 % | 28.38 % | 7.84 % |
| test (10) | 1,711 | 1,515 | 41.32 % | 15.66 % | 26.65 % | 28.23 % | 7.71 % |

Prevalence drift under 1.5 pp across folds — no split-induced bias. 17.2 % of records
carry more than one label; mean 1.209 labels/record; **0** records carry NORM together
with an abnormality.

### Shipped artefacts in `checkpoints/`

| File | Contents |
|---|---|
| `best_model.pt` | Weights + `epoch 13`, `best_auroc 0.940342` (validation), `model_name resnet_se`, full training args, `ema False` |
| `calibrator.json` | Five temperatures, five biases, `fitted_for` provenance block |
| `conformal_triage.json` | α, β, δ = 0.01, λ_out, λ_in, calibration counts and feasibility per class |
| `operating_point.json` | Policy `recall_first`, floor 0.80, report floor 0.75, five thresholds, and the full test confusion matrix per class |
| `scope.json` | `irr_threshold 0.17920692444965153`, `fpr_budget 0.05`, fitted on val fold 9, `n_val 1696` |
| `val/test_logits_seed0.npy`, `val/test_labels.npy` | Saved model outputs — every downstream fit and audit reuses these instead of re-running inference. They are also what made the metrics in §11 independently verifiable. |

---

## 9 · How it all fits together

There are two flows through this repository, and they meet only at `checkpoints/`.

### Offline — build the artefacts

```
csv/*.csv + signals_cache/            metadata + 17,221 signals
    │
    ├─ train_gpu.py --pack ─────────► data/{train,val,test}_X.npy   float16 memmaps
    │                                 (band-pass + normalise baked in)
    │
    ├─ train_gpu.py ────────────────► checkpoints/best_model.pt
    │        40-epoch OneCycle              val/test_logits_seed0.npy
    │        AdamW · focal · AMP            val/test_labels.npy
    │
    ├─ fit_calibration.py ──────────► checkpoints/calibrator.json
    │        fold 9 only                    checkpoints/conformal_triage.json
    │                                       audit/results/07_conformal_eval.*
    │
    ├─ analysis/02_operating_point ─► checkpoints/operating_point.json
    │
    └─ audit/13_out_of_scope ───────► checkpoints/scope.json
```

Every threshold in the system — decision thresholds, conformal λ, the rhythm-scope `irr`
cut-off — is fitted on **fold 9 and only fold 9**. Fold 10 is scored once, for reporting.
That discipline is what makes the numbers in §11 honest.

### Online — serve one request

```
browser  ─ drag .dat/.hea ─►  Worklist.jsx
                                   │ api.js  POST /api/predict?theme=…
                            Vite proxy :5173 → :5000
                                   ▼
                              server.py  predict()
                                   │ secure_filename · TemporaryDirectory · wfdb.rdrecord
                                   │ 12-lead check · lead reorder
                                   ▼
                        ECGPipeline.analyse()   ◄── the eight stages, §3
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          quality gate FAILS              gate PASSES
                    │                             │
          refused report, no             preprocess → classify → calibrate
          probabilities at all           → conformal → XAI → report → verify
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                        server.py  analyse() wrapper
                        + scope/electrode blocks
                        + render_ecg() → base64 PNG
                                   ▼
                              JSON response
                                   ▼
              App.jsx → PatientBanner · EcgViewer · DecisionTable
                        · Measurements · ReportCard · LeadEvidence
```

### Dependency shape

- `models.py` is the root — every other module imports its constants from there, which is
  what keeps training and serving in agreement.
- `paths.py` is the only module that touches the filesystem layout, so the folder can be
  relocated or shipped without `data/` and still work through the upload path.
- `pipeline.py` is the only orchestrator. `server.py`, `08_verify_fixes.py` and
  `13_out_of_scope.py` all go through it, so a change to the stage order changes every
  consumer at once.
- `quality.py` imports `electrodes` and `scope` lazily inside `try/except` — those checks
  are advisory and must never be able to break the gate itself.
- `verify.py` imports from `report.py`, not the reverse. The generator does not know what
  the checker will accept, which is the point.

---

## 10 · The model, in full

### Architecture — `ECGResNetSE`

Shapes verified by running the model.

| Stage | Operation | Output |
|---|---|---:|
| input | normalised, band-passed 12-lead signal | (12, 5000) |
| `MultiKernelStem` | three parallel Conv1d, kernels 7 / 15 / 31, stride 2 → concat → BN → ReLU. Short and long receptive fields because P/QRS and T/ST live at different scales. | (64, 2500) |
| block 1 | `SEResidualBlock` 64→64, k = 11, stride 2, dropout 0.1 | (64, 1250) |
| block 2 | `SEResidualBlock` 64→128, k = 7, stride 2, dropout 0.1 | (128, 625) |
| block 3 | `SEResidualBlock` 128→256, k = 5, stride 2, dropout 0.2 | (256, 313) |
| block 4 | `SEResidualBlock` 256→320, k = 3, stride 2, dropout 0.2 — this is `cam_layer`, where Grad-CAM hooks attach | (320, 157) |
| `AttentionPool` | Conv1d(320→1, k = 1) → softmax over time → weighted sum. Global average pooling throws away *when* something happened; this keeps it. | (320,) |
| head | Linear 320→256 → BN → ReLU → Dropout 0.3 → Linear 256→5 | (5,) |
| **output** | five **independent** sigmoid logits — multi-label, not softmax | (5,) |

Inside each `SEResidualBlock`: Conv → BN → ReLU → Dropout → Conv → BN → squeeze-excitation
→ add projection skip → ReLU. The SE block averages over time, passes through
`Linear(C → C/8) → ReLU → Linear(C/8 → C) → sigmoid`, and gates the channels — letting the
network re-weight leads and filters per record.

### Parameter counts — measured, not quoted

| Model | Parameters | BN buffers | state_dict total |
|---|---:|---:|---:|
| ECGResNet (baseline) | 1,018,501 | 4,109 | 1,022,610 |
| **ECGResNetSE (shipped)** | **1,584,326** | 5,262 | 1,589,588 |

The two numbers differ because `state_dict` includes BatchNorm running statistics, which
are buffers rather than learnable parameters. This is why both figures appear in the
project's own documentation.

### Training configuration — read from the shipped checkpoint

| Setting | Value | Note |
|---|---|---|
| Optimiser | AdamW | lr 3e-3, weight decay 1e-2 |
| Schedule | OneCycleLR | pct_start 0.25, div_factor 20, final_div_factor 200 |
| Loss | FocalLoss | γ = 2.0, label smoothing 0.02, `alpha = None` |
| Imbalance | sampler | `WeightedRandomSampler` ON, focal alpha OFF — never both |
| Precision | bf16 AMP | grad clip 1.0 |
| Batch / epochs | 128 / 40 | patience 0 — early stopping is disabled on purpose, because OneCycle's annealing tail is where the model consolidates |
| Seed | 0 | the only seed whose checkpoint exists on disk |
| Budget | 60 min | ⚠️ CLI default is 75; the shipped run used 60 |
| **Best epoch** | **13** | val macro-AUROC 0.940342 — EMA weights were evaluated and *not* kept |

### The safety layer's shipped parameters

| Class | T | bias | α | β | λ_out | λ_in | n pos cal | decision thr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 0.4279 | +0.3028 | 0.20 | 0.20 | 0.1986 | 0.6944 | 699 | 0.7240 |
| MI | 0.5939 | −1.0544 | 0.05 | 0.10 | 0.0382 | 0.3386 | 283 | 0.2560 |
| STTC | 0.5479 | +0.2045 | 0.10 | 0.15 | 0.1934 | 0.3435 | 455 | 0.3755 |
| CD | 0.5593 | +0.2181 | 0.10 | 0.15 | 0.1104 | 0.2793 | 485 | 0.3395 |
| HYP | 0.6271 | −0.9508 | 0.15 | 0.15 | 0.0506 | 0.1391 | 134 | 0.0925 |

PAC δ = 0.01 · preset `safety` · all five classes feasible · NORM carries a note that its
rule-out and rule-in regions overlap on [0.199, 0.694) and the overlap is referred rather
than decided. Rhythm-scope `irr` threshold 0.1792 at a 5 % false-positive budget.

The α values are the system's **clinical policy**, not a tuning artefact. MI gets the
tightest miss budget because a missed infarction kills; NORM gets the loosest because
"missing NORM" only costs an unnecessary review. δ = 0.01 rather than the code's 0.05
default because at 0.05 two of five guarantees failed on the test realisation —
CD 0.106 vs 0.10 and HYP 0.152 vs 0.15.

---

## 11 · Accuracy — every figure

### Threshold-free discrimination, recomputed from the shipped logits

Seed 0, recomputed from `checkpoints/*_logits_seed0.npy`.

| Class | prev | val AUROC | val AUPRC | test AUROC | test AUPRC | test F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 41.3 % | 0.9694 | 0.9564 | 0.9574 | 0.9253 | 0.8647 |
| MI | 15.7 % | 0.9464 | 0.8027 | 0.9487 | 0.7833 | 0.6915 |
| STTC | 26.7 % | 0.9319 | 0.8199 | 0.9315 | 0.8280 | 0.7420 |
| CD | 28.2 % | 0.9334 | 0.8792 | 0.9141 | 0.8646 | 0.7557 |
| HYP | 7.7 % | 0.9206 | 0.5633 | 0.9085 | 0.5842 | 0.5348 |
| **MACRO** | — | **0.9403** | **0.8043** | **0.9320** | **0.7971** | **0.7177** |

Val macro-AUROC 0.9403 matches `best_auroc` in the checkpoint exactly. Test macro-AUROC
0.9320 matches the seed-0 value quoted in `docs/RESEARCH_CONTRIBUTION.md`
(0.9320 / 0.9374 / 0.9335 across three seeds), which independently corroborates the
0.9343 ± 0.0028 headline even though the other two seeds' artefacts are not on disk.

> **Read AUPRC, not AUROC, for HYP.** HYP AUROC 0.909 reads as strong and is misleading at
> 7.7 % prevalence. Its AUPRC is **0.584**. That is still a real improvement — the audited
> baseline scored 0.5405 — and it sits above the ≈0.54 published norm for this class on
> PTB-XL, but it is the number to quote.

### Shipped operating point — recall-first, test fold 10 (n = 1,711)

Thresholds selected on validation only.

| Class | Acc | Recall | Spec | NPV | Prec | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 0.883 | 0.796 | 0.943 | 0.868 | 0.908 | 0.849 | 563 | 57 | 144 | 947 |
| MI | 0.884 | 0.836 | 0.893 | **0.967** | 0.591 | 0.692 | 224 | 155 | 44 | 1288 |
| STTC | 0.868 | 0.803 | 0.892 | 0.926 | 0.729 | 0.764 | 366 | 136 | 90 | 1119 |
| CD | 0.869 | 0.805 | 0.894 | 0.921 | 0.750 | 0.776 | 389 | 130 | 94 | 1098 |
| HYP | 0.817 | 0.811 | 0.818 | **0.981** | 0.271 | 0.406 | 107 | 288 | 25 | 1291 |
| **MACRO** | **0.864** | **0.810** | **0.888** | **0.933** | **0.650** | **0.698** | | | | |

**Every class clears both clinical targets: accuracy ≥ 0.75 and recall ≥ 0.75.**

### What the recall-first choice costs — three policies on the same fold

| Policy | Acc | Recall | Spec | Prec | NPV | F1 | BAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| A · default 0.5 | 0.901 | 0.668 | 0.944 | 0.800 | 0.922 | 0.714 | 0.806 |
| B · F1-optimal | 0.889 | 0.760 | 0.917 | 0.702 | 0.932 | **0.724** | 0.839 |
| **C · recall-first (shipped)** | 0.864 | **0.810** | 0.888 | 0.650 | **0.933** | 0.698 | 0.849 |

Per-class cost of B → C: HYP recall +0.159 for precision −0.145; MI +0.104 for −0.094;
CD +0.087 for −0.091. Recall is bought with precision — for a rule-out system that is the
correct trade, because a false alarm costs a review and a false negative can cost a life.

**On F1:** F1 weights a missed infarction and an unnecessary review *equally*. No
cardiology pathway does that — the ESC 0/1h hs-troponin algorithm and HEART are governed
by sensitivity and NPV. HYP F1 is 0.41; HYP NPV is 0.981. The second number is the one
that decides whether you can rule hypertrophy out.

### Calibration — before and after temperature scaling

| Class | ECE before | ECE after | over-prediction before | after |
|---|---:|---:|---:|---:|
| NORM | 0.1027 | 0.0160 | 0.94× | 1.01× |
| MI | 0.1138 | 0.0153 | 1.72× | 1.04× |
| STTC | 0.0751 | 0.0183 | 1.05× | 0.98× |
| CD | 0.0794 | 0.0265 | 1.08× | 0.98× |
| HYP | 0.1032 | 0.0150 | 2.31× | 1.01× |
| **MACRO** | **0.0948** | **0.0182** | | |

A 5.2× reduction in macro ECE. Because temperature scaling is monotone, AUROC and the
conformal bounds are unchanged — only the numbers shown to a human become honest. The
archive's equivalent figures were far worse: macro ECE 0.183 with HYP over-predicted 4.14×.

### Conformal triage on the unseen test fold

| Class | α | miss | held | β | false alarm | held | rule out | refer | rule in |
|---|---:|---:|:--:|---:|---:|:--:|---:|---:|---:|
| NORM | 0.20 | 0.033 | ✅ | 0.20 | 0.062 | ✅ | 48.1 % | 14.8 % | 37.1 % |
| MI | 0.05 | 0.015 | ✅ | 0.10 | 0.071 | ✅ | 57.6 % | 24.1 % | 18.3 % |
| STTC | 0.10 | 0.092 | ✅ | 0.15 | 0.124 | ✅ | 60.9 % | 7.9 % | 31.2 % |
| CD | 0.10 | 0.099 | ✅ | 0.15 | 0.138 | ✅ | 54.6 % | 11.8 % | 33.5 % |
| HYP | 0.15 | 0.121 | ✅ | 0.15 | 0.128 | ✅ | 69.9 % | 12.3 % | 17.8 % |

All ten guarantees hold marginally. CD at 0.099 against a promised 0.10 is the narrowest
margin in the table.

### Clinical escape rate — true positives ruled out, never reaching a human

| Class | this system | archive baseline FN rate | reduction |
|---|---:|---:|---:|
| NORM | 3.3 % | 7.4 % | −4.1 pp |
| MI | 1.5 % | 29.5 % | **−28.0 pp** |
| STTC | 9.2 % | 14.2 % | −5.0 pp |
| CD | 9.9 % | 21.9 % | −12.0 pp |
| HYP | 12.1 % | 39.4 % | −27.3 pp |

The baseline had no referral option, so every false negative escaped. Here: **49.1 %** of
patients are handled autonomously (no class deferred) at **70.5 %** exact-match accuracy on
that subset (n = 840), against 62.1 % exact-match over *all* patients for the baseline.
50.9 % are referred.

### Against the baseline architecture

| Model | params | macro AUROC | macro AUPRC | macro F1 |
|---|---:|---:|---:|---|
| ECGResNet (baseline, 1 seed) | 1,018,501 | 0.9297 | 0.7864 | 0.7172 [0.6971, 0.7365] |
| **ECGResNetSE (3 seeds, as reported)** | 1,584,326 | 0.9343 ± 0.0028 | 0.8001 ± 0.0029 | — |
| Δ | | +0.0046 · 1.6σ | +0.0137 · 4.7σ | |

The repo's own honest reading, which should be preserved in any write-up: **the AUROC gain
is within run-to-run noise and must not be claimed.** Only the AUPRC gain is real
(t = 8.2, p ≈ 0.015). The comparison is also asymmetric — 3 seeds against a 1-seed point
estimate.

---

## 12 · Faults found and fixed

Twelve defects were found in the previous system. All are closed, and
`audit/08_verify_fixes.py` re-runs the exact inputs that broke it: **26 passed, 0 failed.**

### Critical defects

| ID | What the archive did | Fix | Evidence |
|---|---|---|---|
| **C-1** | Flatline → "MI 0.691, consistent with myocardial infarction". Gaussian noise → "MI 0.629, CD 0.657". Microvolt file → "STTC 1.000, HYP 1.000". Inverted ECG → "MI 0.818". | `src/quality.py` refuses uninterpretable records before the model runs | 8 adversarial inputs, all handled correctly |
| **C-2** | The fusion model consumed `report_en` — the very text the labels were derived from. Report text *alone* reached macro-AUROC 0.8872 vs 0.9567 for the full model. | Model withdrawn; the ECG-only model is the deliverable | 4-arm ablation in `06_leakage_audit` |
| **C-3** | The BioBART tier was an identity function 80.5 % of the time, corrupted text in 56.9 % of cases, dropped a clinical concept in 103 records, and invented "atrial fibrillation" in 42. | BioBART removed entirely; `src/verify.py` gates all text | 200/200 reports verified |
| **C-4** | 99 of 1,711 reports asserted a normal ECG *and* an abnormality in the same paragraph. | Structurally impossible in `src/report.py` — NORM is demoted before text generation | 0 contradictions in 200 records |
| **C-5** | One shared Grad-CAM object under a threaded server. 3 of 4 concurrent calls returned corrupted heatmaps. | Per-call hooks + per-model `RLock` | 4 concurrent calls match the sequential result exactly |
| **C-6** | Oversampling *and* focal alpha applied together. HYP predicted at 4.14× its prevalence; macro ECE 0.183. | Sampler XOR alpha, plus post-hoc temperature scaling | macro ECE 0.0948 → 0.0182; HYP 2.31× → 1.01× |

### Engineering defects

| ID | Defect | Fix |
|---|---|---|
| **E-1** | PhysioNet password in plaintext in source | Moved to `PHYSIONET_USER` / `PHYSIONET_PASS`; `.gitignore` blocks `.env`, `*credential*`, `*secret*`, `*.pem`, `*.key` |
| **E-2** | Repository rooted at the home directory | RP-Venu is its own repository |
| **E-3** | No upload sanitisation — path traversal via filename | `secure_filename`, extension checks, base-name match, 25 MB cap |
| **E-4** | Failed downloads silently zero-filled into training data | `pack()` refuses to build on any missing record |
| **E-5** | Any-length signal resampled to 5000 samples, distorting heart rate | Resample by *rate*; centre-crop or pad by *length*; duration outside 5–60 s refused |
| **E-9** | Server bound to all interfaces by default | Binds 127.0.0.1 unless `HOST` is set explicitly |
| **E-10** | The same ResNet defined in five files | `src/models.py` is the single source of truth |
| **7.4** | Lead saliency took `abs()` before summing, so a lead arguing *against* the diagnosis displayed as "important" | Sign preserved; signed and magnitude reported separately |

---

## 13 · Faults that remain

Ordered by how much they matter for a patient.

### Clinical

| Fault | Measured | Status |
|---|---|---|
| **The label space is blind to arrhythmia.** No output unit for atrial fibrillation — the most common serious rhythm disorder, associated with roughly one ischaemic stroke in four. | 14.3 % of PTB-XL records carry a documented finding the five classes cannot express. 113 of 114 test-fold AF/flutter recordings received a statistical guarantee; 2 were reported as normal. | partly mitigated |
| **The rhythm-scope gate catches under half.** It flags irregular rhythms only. | 48.9 % sensitivity at 4.8 % FPR (AUROC 0.912). Misses 53 of 114, including ECG 15796 — the very case reported as normal. Paced rhythms, monomorphic VT, Brugada and long QT keep a regular beat and stay invisible. | **open** |
| **The guarantee is not conditionally valid.** It holds on average and fails inside specific subgroups. | CD in patients under 50: promised ≤ 10 % missed, delivered **33.3 %**. NORM in patients over 70: promised ≤ 20 %, delivered **33.0 %**. Both survive Holm correction and 100 % of 2,000 bootstrap calibration draws. | Mondrian repair implemented, not shipped |
| **Electrode reversal voids the guarantee silently.** A miswired recording is clean, passes every quality metric, and changes the diagnosis. | Up to 87 % of diagnoses flip. 7 guarantees voided. Detector reaches 65.5 % / 60.5 % sensitivity on RA/LA and RA/LL, 4.0 % on LA/LL, at a 4.5 % FPR. Refusing detected reversals restored **0 of 7**. | warning shipped, not repaired |
| **Hypertrophy generates a heavy referral burden.** | 288 false positives against 107 true positives — precision 0.271. Bought deliberately for recall 0.811 and NPV 0.981. | by design |
| **No intervals, no axis.** PR, QRS, QT and QRS axis are not measured — things a cardiologist reads routinely. Surfaced honestly in the UI as literal "not measured" rows. | — | **open** |
| **Territory localisation is unvalidated.** The lead-group → coronary-artery map is a heuristic. It agreed with the cardiologist's own report on the one worked example (record 271, "anteroseptal"), which is an anecdote, not evidence. | — | **open** |

### Generalisation

| Fault | Detail |
|---|---|
| **No external validation** | Single German cohort, 1989–1996. Nothing is known about performance on any other population. This is the limitation most likely to be raised first. |
| **Results not comparable to published benchmarks** | The `likelihood == 100` filter dropped 21 % of PTB-XL — precisely the ambiguous cases. The task here is *easier* than the standard benchmark, and the repository says so explicitly rather than hiding it. |
| **50 Hz notch is hard-coded** | `NOTCH_HZ = 50.0` because PTB-XL is German. A 60 Hz recording from the US or Japan would be filtered incorrectly. |
| **Quality gate is near-inert on PTB-XL** | Median SQI 1.000 across the dataset, 0 records below 1.0. The gate only earns its place in deployment, so its value is argued rather than demonstrated on this data. |
| **Fairness axes limited to age and sex** | Race, ethnicity, socioeconomic status and geography are not in PTB-XL and cannot be audited. No audit by device, recording site, comorbidity, or signal-quality band either. |

### Engineering & reproducibility

| Fault | Detail |
|---|---|
| **No tests, no CI** | Zero test files, no `pytest` config, no workflow. `08_verify_fixes.py` is a script, not a test suite, and emits no machine-readable pass/fail alongside its text output. |
| **No dependency pinning** | `requirements.txt` uses only `>=`. No lockfile, no `pyproject.toml`, no Docker image. A future resolve may not reproduce these numbers. Capture `pip freeze` now. |
| **Everything downstream of training is single-seed** | Calibration, conformal thresholds, the operating point and all three contribution audits use seed 0 only. |
| **Latency ~6 s per analysis** | The classifier itself is ~20 ms. Matplotlib rendering and 64-step integrated gradients dominate. |
| **The frontend has no test coverage** | No test file, no runner. |
| **No figure assets exist** | A recursive search for `*.png / *.svg / *.pdf` returns nothing. Every paper figure must be drawn from scratch — but the `.json` result files hold the underlying numbers, so plotting scripts can be written without re-running any experiment. |
| **Git history** | At the time of this reading the working tree is untracked with no commits — no version history, no attribution record, no rollback point for ~2 GB of work. |

---

## 14 · Every outcome, by experiment

### Contribution 1 — does the guarantee hold inside subgroups?

Marginal thresholds fitted on all of fold 9; miss rate then measured inside each subgroup
of fold 10. (`audit/results/10_conditional_validity.txt`)

| Class | α | overall | male | female | <50 | 50–69 | ≥70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORM | 0.20 | 0.190 | 0.158 | 0.225 | 0.103 | 0.228 | **0.330** |
| MI | 0.05 | 0.015 | 0.020 | 0.008 | 0.000 | 0.011 | 0.019 |
| STTC | 0.10 | 0.092 | 0.103 | 0.083 | 0.128 | 0.080 | 0.093 |
| CD | 0.10 | 0.099 | 0.085 | 0.117 | **0.333** | 0.099 | 0.042 |
| HYP | 0.15 | 0.121 | 0.182 | 0.061 | 0.444 | 0.159 | 0.066 |

**9 subgroup violations** at n ≥ 15. Mondrian (group-conditional) calibration held in
**22 of 23** cells versus **14 of 23** for marginal — but STTC in under-50s had only 42
calibration positives and the PAC bound became infeasible, returning λ = −∞: that class can
never be ruled out in a young patient. **Conditional validity costs data, and the groups
needing it most have the least.**

#### Which violations are real? (23 cells tested)

| Cell | promised | observed | 95 % CI | exact p | Holm p | P(violate) |
|---|---:|---:|---|---:|---:|---:|
| **CD · age < 50** | ≤ 0.10 | 0.333 (22/66) | [0.232, 0.453] | 2.23e−07 | **5.14e−06** | 100 % |
| **NORM · age ≥ 70** | ≤ 0.20 | 0.330 (34/103) | [0.247, 0.426] | 1.32e−03 | **2.91e−02** | 100 % |

Both confidence intervals lie **entirely** above the promised bound. The seven other
apparent excesses did not survive multiple-testing correction and are reported as noise —
saying so is part of the result.

The overall CD figure, 9.9 % and comfortably inside its bound, gives **no hint** that a
third of under-50 patients with conduction disturbance are missed. Clinically this is the
worst possible group: conduction disturbance under 50 can mean **Brugada syndrome or
ARVC** — inherited conditions that cause sudden cardiac death in young, otherwise healthy
people.

#### Section B — is the quality gate label-dependent?

Tested because if sick patients have worse signals, the gate removes a label-dependent
subset and the system's own safety mechanism voids its own guarantee. Result: mean SQI is
1.0000 for both positive and negative cases in all five classes, 0 of 5 classes differ
significantly. Under simulated deployment corruption (a disconnected lead plus an EMG burst
on a random 25 %), 581 of 600 records are still accepted and the largest prevalence shift
is +0.0050. **Small but non-zero** — the correct fix is to calibrate *through* the same gate.

### Contribution 2 — electrode reversal

(`audit/results/12_electrode_reversal.txt`)

| Question | Answer |
|---|---|
| **1 · Is reversal visible to the quality gate?** | **No.** 198/200, 198/200, 198/200 and 197/200 records accepted under correct placement, RA/LA, RA/LL and LA/LL respectively. This is the premise of the whole finding. |
| **2 · How many diagnoses change?** | Any label flips in **86.0 %** (RA/LA), **87.0 %** (RA/LL) and 43.5 % (LA/LL). MI specifically flips in 54.5 % / 57.5 % / 22.5 %. |
| **3 · Does the guarantee survive?** | **7 violations introduced.** NORM under RA/LA: promised ≤ 20 %, observed **98.8 %**. NORM under RA/LL: **100 %**. STTC under RA/LA: 62.0 % against ≤ 10 %. HYP under RA/LL: 53.3 % against ≤ 15 %. |
| **4 · Can physiology catch it?** | Partly. 65.5 % on RA/LA, 60.5 % on RA/LL, **4.0 % on LA/LL** (which leaves aVR unchanged and is essentially undetectable this way), at a 4.5 % false-positive rate. |
| **5 · Does refusing detections restore validity?** | **0 of 7.** The 30–40 % of reversed records that slip through are enough to keep every bound void. |
| **6 · What detection would be enough?** | Mostly **0 %** — and that is the finding. At a realistic 0.4–4 % reversal prevalence the *population*-level guarantee survives, so a hospital auditing across all its ECGs would see nothing wrong. For the individual patient whose electrodes were swapped, the promise is void and nothing tells anyone. Three cells read "impossible": STTC's miss rate on *correctly* placed ECGs in this subsample is already 12.0 % against a promised 10 %, so no acquisition check can rescue it — that cell needs recalibration. |

> **The unification the repository draws.** One principle, two demonstrations. Marginal
> validity holds across the whole population but fails within a patient subgroup (age); it
> holds across all recordings but fails within a mis-acquired one. A conformal guarantee
> averaged over a population says nothing about the patient in front of you. The remedy is
> the same in both cases — condition the guarantee on what actually varies: by patient
> stratum (Mondrian) and by verified acquisition.

### Contribution 3 — disease outside the label space

(`audit/results/13_out_of_scope.txt`)

| Finding | Value |
|---|---:|
| Atrial fibrillation in the dataset | 1,225 · 7.11 % |
| Ventricular ectopy | 1,127 · 6.54 % |
| Atrial ectopy | 617 · 3.58 % |
| Flutter / paced / SVT / VT | 27 · 12 · 28 · 30 |
| **Any out-of-scope finding** | **2,455 · 14.26 %** |
| AF or flutter in the test fold | 114 |
| … carrying a statistical guarantee | **113 / 114** |
| … reported as NORMAL | **2** — ECG 15796 (ROUTINE, 4 guarantees attached) and ECG 18266 (URGENT, 3 guarantees) |
| Rhythm check: sensitivity / FPR / AUROC | 48.9 % / 4.8 % / 0.912 |
| Cost of withholding | 53 of 695 records (7.6 %) lose the guarantee — 22 genuinely out of scope, 31 over-caution |

Not one of those 113 guarantees concerns atrial fibrillation, because **no guarantee about
atrial fibrillation exists**. The system certifies what it can measure while being blind to
the finding that will cause the stroke.

Withholding does **not** reduce diagnostic output — the classifier still reports its five
classes. What is withheld is the *claim*, because that claim was never true for these
records. Every paper using the five PTB-XL superclasses inherits this failure; none report
it, because the benchmark only scores the five classes it defined.

### Regression suite & dataset integrity

| Experiment | Headline outcome |
|---|---|
| `08_verify_fixes` | **26 passed, 0 failed.** All 9 adversarial inputs handled; NORM/abnormality contradiction structurally impossible; hallucinated and finding-dropping paraphrases both rejected while a faithful one is accepted; 4 concurrent Grad-CAM calls match; macro ECE 0.0948 → 0.0182; every report contains heart rate, triage, quality, guarantee, limitations and disclaimer. 100 % of 200 test reports pass verification with zero contradictions, and **175 distinct reports** were produced across 200 patients — against 63 distinct across 1,711 for the archive. |
| `01_dataset_deep_audit` | **PASS.** Patient-disjoint official split, 0 leakage on all three pairs, 0 duplicate ids, prevalence drift under 1.5 pp, 0 records with zero labels, 0 records labelled NORM together with an abnormality. Signals: 0 wrong shape, 0 NaN/Inf, 0 fully flat leads in the scanned sample, 1 saturated. Median SQI 1.000. Declared caveats: the 21 % `likelihood==100` drop, the age-300 sentinel (255 records, PTB-XL anonymises age > 89), and constant-imputed height/weight. |
| `01_dataset_deep_audit` §6 | **Why HYP is hard**, measured four ways: **scarcity** (1,468 positives, 8.5 %, only 132 in test); **entanglement** (63.8 % of HYP also carry STTC, 35.7 % also CD, 0 % are NORM); **voltage is amplitude**, and any amplitude normalisation destroys the evidence; **label noise** — ECG criteria for LVH have known low sensitivity against echocardiography, so the label is a proxy. |

### Audits of the superseded system — kept as the "before"

| Audit | Outcome |
|---|---|
| `02_model_audit` | Every published figure reproduced to four decimals (Δ +0.0000) — the archive's reported numbers were *honest*, which is worth saying plainly. Macro AUROC 0.9297, AUPRC 0.7864, F1 0.7172. Threshold optimism only +0.0106. Batch-size sensitivity 1.2e−07, 0 decision flips. Patient leakage 0/1711. But: six degenerate inputs all produced confident diagnoses, and 99 of 1,711 reports contained the NORM/abnormality contradiction. |
| `06_leakage_audit` | Four-arm ablation. Full fusion 0.9567 · text zeroed 0.9072 · signal zeroed 0.8904 · **report text alone 0.8872**. The cardiologist's free-text report by itself nearly matched the full model — the labels were derived from that same text. Model withdrawn. |
| `03_report_audit` | Tier 3 byte-identical to Tier 2 in 23.6 % of records, a leading-character truncation in 56.9 %, genuinely rewritten in 19.5 %. Dropped concepts in 103 records, added none. 63 distinct reports for 1,711 patients (cardiologists wrote 1,055). Heart rate, rhythm, intervals, axis, triage, localisation, uncertainty and signal quality appeared in **0 of 1,711** reports. |
| `04_runtime_audit` | 26.3 s cold start, 1,567 MB RSS. Single-ECG latency 11.77 s: matplotlib 5.22 s, BioBART 2.55 s, legacy free generation 1.49 s (documented as "ablation only" yet executed on every request), IG 1.83 s, classifier 0.02 s. No lead-order check, no sampling-rate check, no quality check, no filename sanitisation. Grad-CAM race confirmed. |
| `05_xai_audit` | Grad-CAM is faithful: deletion AUC 0.3933 vs random 0.5368; insertion 0.5751 vs 0.5039. Adebayo model-randomisation check passes (Spearman 0.163 against a random head). IG ranking stable — Spearman 0.999 between 30 and 200 steps, top-1 lead agreement 19/20. **But this audit ran against the archive** (it references `app.py` and steps = 30; the current `src/xai.py` ships steps = 64), and its 35.6 % completeness figure is a known ill-conditioned measurement. Re-run before quoting any XAI number. |

### Risk–coverage — how much can safely be ruled out at each budget

MI, the class that matters most (`audit/results/07_conformal_eval.txt`):

| α | λ_out | ruled out | observed miss |
|---:|---:|---:|---:|
| 0.01 | 0.0113 | 43.7 % | 0.4 % |
| 0.02 | 0.0320 | 55.7 % | 1.1 % |
| **0.05** | **0.0730** | **64.9 %** | **3.7 %** |
| 0.10 | 0.1268 | 70.8 % | 7.5 % |
| 0.20 | 0.2988 | 79.9 % | 19.4 % |
| 0.30 | 0.4536 | 84.9 % | 30.2 % |

The full curve exists for all five classes. This is the deliverable a clinician can
actually reason about — not "F1 = 0.69" but *"at a 5 % miss budget I can safely clear 65 %
of patients"*. The bold row is the shipped α.

---

## 15 · Contradictions to resolve before writing

Places where two files in the repository disagree. None invalidate the work; all of them
would be found by a reviewer.

| Conflict | Where | Resolution |
|---|---|---|
| **Electrode-reversal sample size: 200 vs 600.** The on-disk results file is an n = 200 run (198/200, 197/200), while `docs/PANEL_ANSWERS.md` reports n = 600 (589/600, 587/600) — and the results file's own **summary** block says "7 of 600 records", contradicting its own header. Sensitivities also differ: 65.5 %/60.5 % on disk vs "~70 %/~61 %" in the README, and the diagnosis-flip and miss-rate tables differ throughout. | `12_electrode_reversal.txt` vs `PANEL_ANSWERS.md` vs `README.md` | Re-run at `--n 600` and regenerate every quoted number, or restate the docs at n = 200. The script's default is already 600. |
| **XAI audit is stale.** References `app.py` and steps = 30; no `app.py` exists in Component_02 and `src/xai.py` ships steps = 64. Its 35.6 % completeness figure is ill-conditioned; the true median is ≈1.3 % at 30 steps. | `05_xai_audit.txt` vs `src/xai.py` | Re-run against `src/xai.py` before quoting any XAI number. |
| **PAC δ default: 0.05 vs 0.01.** Code defaults are 0.05 (`ConformalTriage.__init__`, `fit_calibration --delta`); the shipped artefact and every audit script use 0.01. | `src/conformal.py` vs `conformal_triage.json` | The shipped artefact is authoritative. Update the code defaults so a rerun reproduces the shipped system. |
| **Validation set size: 1,709 vs 1,696.** | `csv/val.csv` vs `checkpoints/scope.json` | Most likely the 13 records where `rr_features` returned `None` for too few R-peaks. Confirm and state it. |
| **Training budget: 75 vs 60 minutes.** CLI default vs the value recorded in the shipped checkpoint's args. | `train_gpu.py` vs `best_model.pt` | Quote 60 — that is what actually ran. |
| **Validation 0.9403 vs test 0.9343.** Not a contradiction — different splits — but trivially misquoted. | `best_model.pt` vs `README.md` | State the split explicitly wherever either number appears. |
| **The 3-seed headline has no results artefact.** 0.9343 ± 0.0028 appears only as prose in four files; only seed 0's checkpoint is on disk. | four docs | Partly mitigated: seed 0 recomputes to **0.9320** here, matching the per-seed list exactly. Re-run seeds 1 and 2 capturing stdout to `audit/results/`, or drop the ± claim. **Highest-priority item.** |
| **Folder size understated.** The README says `data/` is 1.93 GB and the total ≈2.0 GB; the packed training memmaps add roughly another 2 GB it does not count. `.gitignore` is the only file that gets this right. | `README.md` vs `.gitignore` | Correct the README and note the memmaps are regenerable via `train_gpu.py --pack`. |
| **"All four verified standalone"** — but seven commands are listed above that line. | `README.md` | Stale count; say seven. |
| **The top-level README describes the superseded system** — BioBART three-tier reporting, the fusion model as a headline feature, macro-F1 0.717, and `cd _archive && python app.py` as the quick start. | `/README.md` | Rewrite it to point at Component_02, or mark it explicitly as the Progress-1 record. |

> ✅ **One question this reading closed.** `POST /api/predict` does **not** persist uploaded
> files. They are written into a `tempfile.TemporaryDirectory()` and removed when the block
> exits, before analysis begins. The write-up listed this as unresolved.

---

## 16 · How to run it

All commands run from **inside** the `Component_02/` folder.

```bash
cd Component_02
pip install -r requirements.txt
```

```bash
# Terminal 1 — API on :5000
python -X utf8 backend/server.py
```

```bash
# Terminal 2 — UI on :5173
cd frontend
npm install          # first time only
npm run dev
```

Open **http://localhost:5173**

> Seeing `can't open file '...Component_02\Component_02\backend\server.py'`?
> You are already inside `Component_02/` — drop the prefix.

**Requirements:** Python 3.10+, Node 18+ (Vite 6 is pinned for Node 20.16 compatibility).
No GPU needed for inference.

### Reproduce every experiment

```bash
python -X utf8 audit/08_verify_fixes.py             # 26 regression checks
python -X utf8 analysis/01_dataset_deep_audit.py    # dataset integrity
python -X utf8 analysis/02_operating_point.py       # operating point
python -X utf8 audit/10_conditional_validity.py     # contribution 1
python -X utf8 audit/11_significance.py             # significance of contribution 1
python -X utf8 audit/12_electrode_reversal.py       # contribution 2
python -X utf8 audit/13_out_of_scope.py             # contribution 3
```

### Retrain

```bash
python train/preflight.py                    # 30 s, catches every setup mistake
python train/train_gpu.py --pack             # once, CPU is fine (~12 min)
python train/train_gpu.py --epochs 40
python train/fit_calibration.py --model resnet_se \
    --ckpt checkpoints/best_model.pt --filter --from-logits --delta 0.01
```

Full GPU procedure in `docs/COLAB_GUIDE.md`.

### Self-contained

Nothing here reads from outside this folder. `src/paths.py` resolves assets in order:
`$ECG_DATA_DIR` → `csv/` → `data/` → `../_archive/data/`.

| | |
|---|---|
| Total | ~4.1 GB measured (1.9 GB dataset + ~2 GB regenerable packed memmaps) |
| Without `data/` | ~20 MB — the upload path still works fully |
| Needs `data/` | only to browse the built-in test set |

Send the whole folder minus `frontend/node_modules/`.

---

## Related documents

| Document | For |
|---|---|
| `START_HERE.md` | Plain-English walkthrough, no maths assumed |
| `README.md` | Run instructions and the integration API contract |
| `docs/PANEL_ANSWERS.md` | Research contribution, novelty, gap, anticipated Q&A |
| `docs/AUDIT_FINDINGS.md` | The 12 defects found in the previous system, in full |
| `docs/RESEARCH_CONTRIBUTION.md` | Long-form write-up (§0 is plain English) |
| `docs/CONTRIBUTION_FINAL.md` | Contribution with prior-art positioning |
| `docs/COLAB_GUIDE.md` | Retraining on a GPU |
| `component02-writeup.md` | Evidence-based technical dossier with per-claim source citations |
| `docs/system-reference.html` | This document, as a styled standalone page |

---

*Compiled by reading every source file in `Component_02/` plus the top-level repository.
Every number is traceable to a named file on disk; the threshold-free metrics in §11 were
recomputed from the saved logits, and the parameter counts and tensor shapes in §10 by
instantiating the models. Nothing here was estimated or supplied from outside the repository.*

**AI-generated decision support. NOT a medical device and NOT a diagnosis.**
Every report the system produces requires review by a qualified clinician.
