# Component 02 — ECG Abnormality Detection & Cardiac Risk Reporting

**Venushan T** · part of the Explainable AI System for Cardiovascular Disease
Detection and Diagnosis.

Takes a 12-lead ECG, classifies it into 5 diagnostic superclasses, explains the
decision, and produces a verified clinical report with a statistical guarantee on
missed cases.

> ⚠️ **AI-generated decision support. NOT a medical device and NOT a diagnosis.**
> Every report requires review by a qualified clinician.

> 📖 **New to this project? Read [START_HERE.md](START_HERE.md) first** — a
> plain-English walkthrough with no maths background assumed.

---

## Run it

**All commands run from inside this `Component_02/` folder.**

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

**Requirements:** Python 3.10+, Node 18+ (Vite 6 is pinned for Node 20.16
compatibility). No GPU needed for inference.

---

## Integration API

The backend is **JSON only** — no HTML, no templates. Point any client at it.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | model info, class list, thresholds, readiness |
| `GET` | `/api/patients/<class>` | browse the built-in test set |
| `POST` | `/api/analyze/<ecg_id>` | analyse a bundled record |
| `POST` | `/api/predict` | **analyse an uploaded `.dat` + `.hea` pair** |
| `POST` | `/api/demo` | random record |

`/api/predict` is the integration point. Send `multipart/form-data` with
`dat_file` and `hea_file`. Add `?theme=dark` to render the ECG plot dark.

**Response shape** (abbreviated):

```jsonc
{
  "triage": "IMMEDIATE",              // IMMEDIATE|URGENT|PRIORITY|ROUTINE|REPEAT ECG
  "electrode": {                       // limb-electrode reversal screening
    "suspected": false,
    "reversal": null,                  // "RA/LA" | "RA/LL" | null
    "confidence": 0.0,
    "reasons": []
  },
  "scope": {                           // is the rhythm inside the label space?
    "outOfScope": false,
    "reason": null,
    "confidence": 0.0,
    "detail": []
  },
  "headline": "myocardial infarction",
  "refused": false,                    // true = signal not interpretable
  "classes": [                         // one per superclass
    { "name": "MI", "probability": 59.9, "zone": "rule_in",
      "ruleOut": 3.8, "ruleIn": 33.9, "alpha": 0.05 }
  ],
  "reportText": "...",                 // full clinical report
  "findings": [ /* structured, each with its evidence */ ],
  "ruledOut": ["NORM", "STTC"],
  "guarantees": ["... miss-rate bound of 5% ..."],
  "quality": { "sqi": 1.0, "heart_rate_bpm": 61.6, "errors": [], "warnings": [] },
  "verification": { "passed": true, "errors": [] },
  "explanation": { "territory": "septal", "artery": "proximal LAD",
                   "topLeads": ["V1","V2","V5"], "peaksSeconds": [1.5,3.8] },
  "leads": [ { "name": "V1", "signed": 18.6 } ],
  "ecgImage": "<base64 png>"
}
```

**Four fields the caller must respect:**

- `refused: true` → the signal failed quality control. **No probabilities are
  returned.** Do not treat this as "normal". Show `quality.errors` and ask for a
  repeat ECG.
- `verification.passed: false` → the generated text failed the safety check and
  was withheld. Do not display `reportText` as a report.
- `electrode.suspected: true` → a limb-electrode reversal is suspected. The
  probabilities are still returned but **the statistical guarantees do not apply
  to this record**. Surface the warning and prompt for a repeat ECG.
- `scope.outOfScope: true` → the rhythm appears to lie outside the five classes
  (irregularly irregular R-R, characteristic of atrial fibrillation or flutter).
  **The guarantees are withheld and an arrhythmia has NOT been excluded.** Do not
  present the five-class result as a complete interpretation.

Configure with env vars: `HOST`, `PORT`, `CORS_ORIGINS`, `ECG_CKPT`,
`ECG_MODEL`, `ECG_FILTER`. Defaults are correct for the shipped model.

---

## Results

**Test fold 10 (n = 1,711), never used to select anything.**

| Class | Accuracy | Recall | Specificity | NPV | Precision | F1 |
|---|---|---|---|---|---|---|
| NORM | 0.883 | 0.796 | 0.943 | 0.868 | 0.908 | 0.849 |
| MI | 0.884 | 0.836 | 0.893 | **0.967** | 0.591 | 0.692 |
| STTC | 0.868 | 0.803 | 0.892 | 0.926 | 0.729 | 0.764 |
| CD | 0.869 | 0.805 | 0.894 | 0.921 | 0.750 | 0.776 |
| HYP | 0.817 | 0.811 | 0.818 | **0.981** | 0.271 | 0.406 |
| **MACRO** | **0.864** | **0.810** | 0.888 | **0.933** | 0.650 | 0.698 |

Accuracy ≥ 0.75 and recall ≥ 0.75 on **every** class.
Threshold-free, 3 seeds: **macro-AUROC 0.9343 ± 0.0028**, **macro-AUPRC 0.8001 ± 0.0029**.

Thresholds are chosen on validation (fold 9) using a PAC conformal lower bound on
recall. Fold 10 is scored once, for reporting only.

**On F1:** this is a *rule-out* system, so it is tuned for sensitivity and NPV,
the metrics every cardiology rule-out pathway uses. HYP F1 is 0.41 but its NPV is
0.981 — see `docs/PANEL_ANSWERS.md` §2 for why hypertrophy is the hardest class
and what the published ceiling is.

---

## Pipeline

```
quality gate ──► preprocess ──► classify ──► calibrate ──► conformal triage
  │        │                                                      │
  │   electrode + scope checks                                    ▼
  │   (withhold guarantees)              XAI ──► report ──► verify
REFUSED if uninterpretable
```

The quality gate runs **before** the classifier, so a refused record never
produces a probability. A suspected **electrode reversal** does not refuse the
record — it is high-quality, just wired wrong — but it **withdraws the
statistical guarantees**, which are calibrated on correctly-placed recordings.
The same applies when the **rhythm falls outside the five classes** (e.g. atrial
fibrillation): the diagnosis is still reported, but no conformal bound covers a
class that is not in the label space, so the claim is withheld.

| Module | Responsibility |
|---|---|
| `src/models.py` | Architecture registry (single definition, ablation variants) |
| `src/paths.py`, `src/signals.py` | Asset resolution; reads WFDB or cache |
| `src/quality.py` | Flatline / noise / unit / duration / rhythm gate |
| `src/electrodes.py` | Limb-electrode reversal detection (physiology rules) |
| `src/scope.py` | Rhythm-scope check: is this disease inside the label space? |
| `src/preprocess.py` | Band-pass, resample by rate, normalise |
| `src/calibration.py` | Per-class temperature scaling |
| `src/conformal.py` | Risk-controlled triage (rule-out / refer / rule-in) |
| `src/xai.py` | Thread-safe Grad-CAM, signed IG, territory mapping |
| `src/report.py`, `src/verify.py` | Grounded report + safety gate |
| `src/pipeline.py` | The single inference entry point (one model) |
| `src/zoo.py` | Serving several models at once, each with its own safety layer |

---

## Using more than one model

Two models are served, each with **its own** calibrator and conformal
thresholds. They are not interchangeable: the Progress-1 baseline was trained on
unfiltered signals and the shipped model on band-passed ones, so each bundle
also carries the preprocessing it was fitted with. `src/zoo.py` reads that from
the artefacts rather than assuming it.

```
checkpoints/best_model.pt              resnet_se  (default, filter=on)
checkpoints/calibrator.json
checkpoints/conformal_triage.json

checkpoints/resnet/best_model.pt       resnet     (baseline, filter=off)
checkpoints/resnet/calibrator.json
checkpoints/resnet/conformal_triage.json
```

```python
from src.zoo import ModelZoo

zoo = ModelZoo.discover()            # finds every bundle on disk
print(zoo.describe())

zoo.analyse(sig, fs=500)                      # default model
zoo.analyse(sig, fs=500, model="resnet")      # a named model
cons = zoo.consensus(sig, fs=500)             # every serveable model
print(cons.summary(), cons.escalate)
```

Over the API:

```bash
curl  localhost:5000/api/models                       # what is loaded
curl -X POST "localhost:5000/api/analyze/9?model=resnet"
curl -X POST "localhost:5000/api/analyze/9?compare=1" # adds a `consensus` block
```

### The decision rule when they disagree

A class is **ruled out only if every model rules it out**; any disagreement
collapses to REFER. This is not a heuristic — the merged rule-out set is the
intersection of the individual ones, so

```
P(merged rules out | Y=1)  <=  min_m P(model m rules out | Y=1)  <=  min_m alpha_m
```

the merged miss rate is bounded by the *tightest* single-model guarantee. On the
unseen test fold (`audit/results/14_multi_model.txt`, 1,711 records):

| Class | Miss resnet_se | Miss resnet | **Miss merged** | Referred resnet_se | **Referred merged** |
|---|---|---|---|---|---|
| NORM | 0.033 | 0.044 | **0.018** | 14.8% | 14.3% |
| MI | 0.015 | 0.007 | **0.000** | 24.1% | 37.3% |
| STTC | 0.092 | 0.077 | **0.055** | 7.9% | 12.3% |
| CD | 0.099 | 0.106 | **0.070** | 11.8% | 17.8% |
| HYP | 0.121 | 0.098 | **0.053** | 12.3% | 19.6% |

The merged rule misses fewer true positives than *either* model on every class,
because the two models do not miss the same cases. The price is referrals.

**The number worth reporting:** the two models disagree on at least one class in
**58.9%** of records, and reach **opposite** conclusions — one rules a class in
while the other rules it out — in **10.5%** (180 / 1,711). A single-model
deployment shows the clinician one of those two answers, with a guarantee
attached, and no indication the other exists.

### Adding a third model

```bash
python -X utf8 train/train_gpu.py --list-models        # what can be trained
python -X utf8 train/train_gpu.py --model resnet_se_no_se --seed 0
python -X utf8 train/fit_calibration.py --model resnet_se_no_se     --ckpt checkpoints/resnet_se_no_se/best_model.pt     --out-dir checkpoints/resnet_se_no_se --tag no_se --delta 0.01
```

`--out-dir` and `--tag` are required for any model after the first: without them
the new calibrator overwrites the previous model's, and the second model's
logits land under filenames that still imply the first.

---

## Folder map

```
Component_02/
├── README.md              this file
├── requirements.txt
├── src/                   library — import this to embed the pipeline
├── backend/server.py      JSON API
├── frontend/              React 19 + Vite 6 + Tailwind 4
├── checkpoints/           model, calibrator, conformal thresholds, operating point
├── csv/                   metadata, official split, norm_stats.json
├── data/                  1.93 GB — PTB-XL records (WFDB) + licence
├── analysis/              dataset audit + operating-point selection
├── audit/                 regression suite + the contribution experiment
│   └── legacy/            audits of the superseded system (need ../_archive)
├── docs/                  panel answers, audit findings, training guide
└── reference/             superseded artefacts kept as audit evidence
```

---

## Verify it works

```bash
# 26 checks that every known defect is closed
python -X utf8 audit/08_verify_fixes.py

# dataset integrity
python -X utf8 analysis/01_dataset_deep_audit.py

# operating point
python -X utf8 analysis/02_operating_point.py

# research contribution 1 — subgroup validity of the guarantees
python -X utf8 audit/10_conditional_validity.py

# statistical significance of the subgroup violations
python -X utf8 audit/11_significance.py

# research contribution 2 — electrode reversal voids the guarantee
python -X utf8 audit/12_electrode_reversal.py

# research contribution 3 — out-of-scope disease gets a guarantee anyway
python -X utf8 audit/13_out_of_scope.py

# multi-model serving + what the two-model rule costs and buys
python -X utf8 audit/14_multi_model.py

# research contribution 4 — cross-model disagreement as an acquisition check
python -X utf8 audit/15_disagreement_detector.py
```

All verified standalone with no `_archive/` present.

---

## Self-contained

Nothing here reads from outside this folder. Verified by copying it to an empty
directory and running the full suite.

| | |
|---|---|
| Total | ~2.0 GB (1.93 GB is the dataset) |
| Without `data/` | ~20 MB — upload path still works fully |
| Needs `data/` | only to browse the built-in test set |

`src/paths.py` resolves assets in order:
`$ECG_DATA_DIR` → `csv/` → `data/` → `../_archive/data/`

Send the whole folder minus `frontend/node_modules/`.

---

## Documentation

| Document | For |
|---|---|
| `docs/HOW_IT_WORKS.md` | **Start here if you are new** — the whole system explained from scratch, no ML background assumed |
| `docs/PANEL_ANSWERS.md` | **Research contribution, novelty, gap, Q&A** |
| `docs/AUDIT_FINDINGS.md` | The 12 defects found in the previous system |
| `docs/RESEARCH_CONTRIBUTION.md` | Long-form write-up (§0 is plain English) |
| `docs/CONTRIBUTION_FINAL.md` | Contribution with prior-art positioning |
| `docs/COLAB_GUIDE.md` | Retraining on a GPU |

---

## Limitations

- Trained on PTB-XL only (German cohort, 1989–96). **No external validation.**
- Recognises 5 superclasses. **Atrial fibrillation and other arrhythmias are not
  detected** — their absence from a report is not evidence of their absence.
  14.3% of the dataset carries a documented finding the label space cannot
  express. The rhythm-scope check flags the irregular ones (48.9% sensitivity,
  4.8% FPR at a 5% false-positive budget; AUROC 0.912) and withholds the
  guarantee; regular out-of-scope rhythms (paced, monomorphic VT, Brugada,
  long QT) remain silent failures.
- PR/QRS/QT intervals and QRS axis are not measured.
- Labels used only SCP codes with `likelihood == 100`, dropping 21% of PTB-XL.
  Results are **not directly comparable** to published benchmarks.
- Territory localisation is a lead-group heuristic, not clinically validated.
- The electrode-reversal detector is a physiology rule, not a classifier: ~70%
  sensitivity on RA/LA, ~61% on RA/LL, 4.5% false positives. LA/LL reversal
  leaves aVR unchanged and is essentially undetectable this way.
- ~6 s per analysis (plotting and integrated gradients dominate; the classifier
  itself is ~20 ms).
