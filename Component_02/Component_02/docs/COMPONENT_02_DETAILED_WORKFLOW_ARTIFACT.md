# Component 02 — Detailed Workflow and Code Map

This document is a practical architecture guide for the ECG abnormality detection system in Component 02. It explains:

- how the project is organized
- which files matter most
- what each module does
- how the full flow works from ECG upload to final clinical report
- which code paths are critical for safety, performance, and explainability
- how to reason about this project as a full working system

---

## 1. What this project actually does

This system takes a 12-lead ECG signal and turns it into:

1. a class prediction (normal, myocardial infarction, ST/T change, conduction disturbance, hypertrophy)
2. calibrated probabilities
3. risk-based triage zones (`rule_out`, `refer`, `rule_in`)
4. explainability (why the model decided this)
5. a clinical-style report
6. a safety verification step to prevent hallucinated text

The project is not just a model. It is a full end-to-end pipeline with:

- signal quality checking
- preprocessing and filtering
- model inference
- probability calibration
- conformal risk guarantees
- explainability and anatomical localization
- clinical reporting
- verification gate
- API + frontend serving layer

---

## 2. High-level project structure

Inside Component_02/Component_02:

- `src/` — main Python logic
- `backend/` — Flask JSON API
- `frontend/` — React UI
- `train/` — model training and calibration scripts
- `analysis/` — dataset audits and operating point analysis
- `audit/` — research validation and regression tests
- `checkpoints/` — model weights and fitted calibrators
- `csv/` — metadata and mapping files
- `data/` — ECG dataset files
- `docs/` — research and usage documentation

The main working logic is in the `src/` package. The backend just exposes it through HTTP endpoints.

---

## 3. The most important files and their role

### 3.1 Core model definitions

#### `src/models.py`
This file is the single source of truth for the model architecture.

Why it matters:
- The whole project references architecture definitions from here.
- It avoids duplicated model code across files.
- It contains both the baseline ResNet and the improved `resnet_se` model.
- It defines class names, lead names, sampling rate, signal length, and model registry.

Important constants:

- `CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]`
- `LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]`
- `NUM_LEADS = 12`
- `NUM_CLASSES = 5`
- `SIGNAL_LENGTH = 5000`
- `SAMPLING_RATE = 500`

Important classes:

- `ResidualBlock`
- `ECGResNet`
- `SEBlock`
- `SEResidualBlock`
- `MultiKernelStem`
- `SingleKernelStem`
- `AttentionPool`
- `ECGResNetSE`

Important function:

- `build_model(name)`
- `resolve_model_name()`

This is where you go when you want to understand how the ECG model is coded and what architecture is active.

---

### 3.2 Signal and dataset access

#### `src/paths.py`
This file decides where artifacts live on disk.

Importance:
- Handles dataset files and checkpoints
- resolves files from environment, CSVs, data folders, and fallback locations
- allows the app to be moved elsewhere without breaking paths

Important functions:

- `find(...)`
- `require(...)`
- `describe()`
- `signals_cache` logic

It acts like the project’s filesystem hub.

#### `src/signals.py`
This loads ECG records from WFDB `.dat` + `.hea` files or a cached numpy representation.

Importance:
- used to fetch records for analysis
- can read from `signals_cache` if available
- falls back to standard data files

This is the file that turns raw ECG data into numpy arrays.

---

### 3.3 Safety gate before classification

#### `src/quality.py`
This is probably the most important file for safe operation.

It prevents garbage signals from reaching the classifier.

Key purpose:
- reject flat signals
- reject impossible durations
- reject impossible amplitudes
- reject unrealistic units/gain values
- detect noisy leads
- estimate heart rate and rhythm quality
- detect electrode reversal suspicion
- detect out-of-scope rhythm

Important data class:

- `QualityReport`

Important functions:

- `detect_and_fix_units()`
- `detect_r_peaks()`
- `assess()`

Critical policy:
- if `report.acceptable` is false, the model never runs
- this is the first guardrail against bad AI decisions

This file fixes one of the biggest prior risks: the system refusing to diagnose an all-zero or obviously broken signal instead of forcing a model output.

---

### 3.4 Electrode and scope checks

#### `src/electrodes.py`
This performs limb-electrode reversal detection.

Why it matters:
- a reversed lead placement can produce a valid-looking ECG but wrong physiology
- the system does not automatically refuse it
- instead it flags suspicion and removes guarantee strength

This is a clinical safety layer, not a model output.

Key idea:
- polarity relationships between leads can identify RA/LA or RA/LL reversal patterns
- the system raises suspicion but does not silently accept it as valid

#### `src/scope.py`
This checks whether the rhythm is in the model’s valid disease space.

Why it matters:
- the model is trained on only five classes
- if the ECG is actually something like atrial fibrillation, the model still may produce a softmax output but that output is not valid as a clinical statement
- this module flags out-of-scope rhythms and withholds guarantee claims

Critical idea:
- not all ECGs are interpretable under this label system
- the report must say the rhythm was outside the model’s scope, rather than misleadingly presenting a normal five-class diagnosis

---

### 3.5 Preprocessing and normalization

#### `src/preprocess.py`
This is the “clean-up” stage before modeling.

It takes a raw ECG, does deterministic conditioning, and prepares it for inference.

Important pipeline stages:

- resample by rate
- high-pass / low-pass filtering
- notch filtering
- crop or pad to fixed length
- normalize per lead

Meaning:
- the raw ECG is not fed directly to the model
- the signal is standardized in a reproducible way

This file is important because the training and serving preprocessing must match. If they do not match, the model output becomes unreliable.

---

### 3.6 Probability calibration and conformal guarantees

#### `src/calibration.py`
This corrects model overconfidence.

Why it matters:
- neural networks often output probabilities that are too confident
- the project needed realistic calibrated probabilities

Important class:

- `TemperatureCalibrator`

Important logic:
- fits per-class temperature scaling
- transforms logits into calibrated probabilities
- stores provenance metadata (`fitted_for`) so a mismatched model/calibrator is rejected

This fixes the issue where the raw model output was more confident than the true probabilities.

#### `src/conformal.py`
This is the risk-control logic that makes the project more clinically meaningful.

This is not just a class probability threshold. It defines a mathematically controlled triage boundary.

Important constants:

- `RULE_OUT`
- `REFER`
- `RULE_IN`

Important concept:
- rule-out threshold is selected to satisfy a miss-rate budget
- rule-in threshold is selected to satisfy a false-alarm budget
- the report says not only what the model thinks, but the guarantee behind it

This is a major research contribution.

Key idea:
- `rule_out`: confident absence
- `refer`: uncertain, human review required
- `rule_in`: confident presence

The thresholds are tied to a clinical safety policy with per-class alpha/beta settings.

---

### 3.7 Explainability and report generation

#### `src/xai.py`
This is the explainability layer.

It explains why the model made the classification using:

- Grad-CAM for temporal importance
- signed integrated gradients for lead-level attribution
- territory localization for known ECG regions

Important classes:

- `Explanation`

Important functions:

- `grad_cam()`
- `integrated_gradients()`
- `lead_attributions()`
- `localise()`
- `cam_peaks()`
- `explain()`

This is the architectural glue between the model and the report. It is what turns a black-box prediction into a clinically meaningful explanation.

#### `src/report.py`
This turns structured predictions and explanations into a final human-readable report.

It builds a `ClinicalReport` with:

- triage level
- headline
- quality line
- rhythm line
- findings
- ruled-out classes
- referred classes
- guarantees
- limitations
- final report text

This is where the project converts raw scores and explanations into a clinical summary.

#### `src/verify.py`
This is the strict safety gate.

It checks whether the generated text is consistent with the structured findings.

Why it matters:
- prevents hallucinated diagnoses such as atrial fibrillation when the model cannot produce that class
- ensures rules are kept consistent
- prevents contradictions like saying “normal” and “MI” in the same report

This file is crucial because it keeps the narrative output honest.

---

### 3.8 Inference pipeline

#### `src/pipeline.py`
This is the central inference engine.

It orchestrates the exact sequence:

1. quality gate
2. preprocess
3. model logits
4. sigmoid/raw probabilities
5. calibration
6. conformal triage
7. XAI explanation
8. report generation
9. safety verification

Important class:

- `AnalysisResult`
- `ECGPipeline`

Important method:

- `analyse()`
- `logits()`

This is the actual brain of the system.

It is the file that says: the model is not allowed to run before quality gating and the report is not allowed to leave before verification.

---

### 3.9 Multi-model orchestration

#### `src/zoo.py`
This allows serving multiple models at once, each with its own safety layer.

Why it matters:
- not all models use the same preprocessing or calibration
- there may be a default model and a baseline model
- this file keeps each model bundle separate

Important classes:

- `ModelAssets`
- `ModelBundle`
- `ClassDisagreement`

Important logic:
- discover models from `checkpoints/`
- load model-specific calibrators and thresholds
- manage cross-model agreement decisions

This is the multi-model variant of the single `ECGPipeline`.

---

## 4. The full runtime workflow

This is the actual execution sequence of the system.

### Step 1: A signal enters the system
The user uploads an ECG in `.dat` + `.hea` format or selects a bundled dataset sample.

The backend receives the request in `backend/server.py`.

### Step 2: Server startup loads safe model bundles
On startup, the API loads:
- checkpoint
- calibrator
- conformal thresholds
- normalization stats

It enforces provenance checks to ensure the calibrator and thresholds match the model and preprocessing mode.

This is important because mismatched calibrators are dangerous.

### Step 3: Input signal is validated
In `src/quality.py`, the signal is checked:
- shape
- duration
- finite values
- flat leads
- unit/gain correctness
- noise
- heart rate plausibility
- rhythm plausibility

If the quality check fails, the project does not classify the signal.

This is the “refuse unsafe inputs” layer.

### Step 4: Preprocessing prepares the signal
If the ECG passes quality checks, `src/preprocess.py` cleans and standardizes it:
- resample if needed
- filter noise
- center and normalize
- pad or crop to fixed size

The input is transformed to a standard representation expected by the network.

### Step 5: Model computes logits
The model in `src/models.py` runs the prepared ECG input and returns logits.

This is the raw scoring stage.

### Step 6: Probabilities are calibrated
`src/calibration.py` transforms the logits using temperature scaling.

This ensures the output probability is more honest and less overconfident.

### Step 7: Conformal triage decides rule-out / refer / rule-in
`src/conformal.py` applies class-specific thresholds.

For each class:
- below `lambda_out` => `rule_out`
- between `lambda_out` and `lambda_in` => `refer`
- above `lambda_in` => `rule_in`

This is where the model does not just guess — it can abstain and request review.

### Step 8: XAI explains the decision
`src/xai.py` computes attributions and localizes evidence:
- which leads mattered
- which times of the signal mattered
- which anatomical territory is implicated

This turns the prediction from a number into a clinically interpretable explanation.

### Step 9: Structured report is created
`src/report.py` converts the decision and explanations into:
- triage category
- headline
- finding sentences
- evidence-backed report body

This is a deterministic, structured clinical narrative.

### Step 10: Report verification blocks unsafe text
`src/verify.py` compares the final text with the actual findings.

It rejects:
- hallucinated diseases
- contradictions
- missing findings
- missing disclaimer
- unsupported text

If verification fails, the system withholds the report or uses a minimal fallback safe message.

### Step 11: JSON response is returned to the frontend
`backend/server.py` packages the result and sends JSON to the client.

### Step 12: Frontend displays the result
`frontend/src` renders the ECG plot, feature explanation, decision summary, and report.

---

## 5. The backend API and its role

### `backend/server.py`
This is the HTTP layer exposing the system.

It is not the main intelligence. It is the interface to the model and safety stack.

Important endpoints:

- `GET /api/health`
- `GET /api/patients/<class>`
- `POST /api/analyze/<ecg_id>`
- `POST /api/predict`
- `POST /api/demo`

Core responsibilities:

- load the pipeline and model zoo
- validate uploaded files
- call the model pipeline
- encode ECG plot as base64
- return structured JSON to the browser

It enforces startup checks so the app does not run with mismatched calibrators or invalid model-safety pairs.

---

## 6. How the frontend fits into the flow

The frontend is under `frontend/src/` and is a UI layer.

Main responsibilities:

- upload `.dat` + `.hea` files
- show ECG waveform and signal quality details
- display class probabilities
- render triage or rule-out / rule-in decision
- display explanations and territory mapping
- show generated clinical report

This is a view layer. The heavy logic is in Python.

---

## 7. Training and calibration scripts

### `train/preflight.py`
Checks the project environment and data assumptions before training.

### `train/train_gpu.py`
Main training script for the model.

### `train/fit_calibration.py`
Fits the temperature calibrator and conformal thresholds.

### `train/Component02_Colab.ipynb`
Notebook version for running in Colab or an external environment.

These scripts are important because the model, calibrators, and thresholds all need to be trained and fitted consistently.

---

## 8. Analysis and audit files

### `analysis/01_dataset_deep_audit.py`
Audits dataset quality and label distribution.

### `analysis/02_operating_point.py`
Evaluates the operating point and risk thresholds.

These are research and validation scripts used to choose class thresholds, inspect data quality, and determine behavior before shipping the model.

### `audit/`
This folder contains a large set of validation and research scripts. They exist to test the project under stress conditions.

Examples:

- `08_verify_fixes.py`
- `10_conditional_validity.py`
- `11_significance.py`
- `12_electrode_reversal.py`
- `13_out_of_scope.py`

These are critical to understand the project’s research quality and safety story.

The audit folder is where the project documents:
- prior failures
- why the old system was unsafe
- how the new pipeline fixed those bugs
- statistical validity of the conformal guarantees

---

## 9. Files that are most important to understand first

If you want the real heart of the project, start with these files in order:

1. `src/models.py` — model architecture
2. `src/quality.py` — safety gate
3. `src/preprocess.py` — conditioning
4. `src/calibration.py` — probability calibration
5. `src/conformal.py` — guaranteed triage logic
6. `src/xai.py` — explanations
7. `src/report.py` — clinical narrative assembly
8. `src/verify.py` — safety verification
9. `src/pipeline.py` — end-to-end inference flow
10. `backend/server.py` — API interface

These cover the full system without needing to read every administrative file.

---

## 10. What makes this project different from a basic ML classifier

A basic classifier would do:

- input signal
- output class label
- maybe probability score

This project does much more:

- refuses invalid ECGs
- calibrates probabilities
- gives uncertainty-aware triage
- explains predictions with wave and lead attribution
- writes a report tied to evidence
- verifies the text before release
- enforces safety rules on the final output

This is why the project is more than “an ECG model”; it is a safety-constrained decision-support pipeline.

---

## 11. The actual workflow in plain English

Here is the full conceptual story:

A patient’s ECG enters the system.

The system first checks whether the signal is real and interpretable. If the recording is flat, noisy, impossible, or wrong in duration/units, it refuses to process it.

Then it cleans the signal and standardizes it to match training assumptions.

Then the model predicts which diagnostic superclass is most likely.

Then the system recalibrates the raw confidence values so they are honest.

Then it decides whether a class is ruled out, uncertain, or ruled in based on statistical guarantees.

Then it explains which leads and times drove the decision and localizes the likely anatomy.

Then it writes a clinical report grounded in those findings.

Finally, it automatically checks the report to ensure it does not claim anything unsupported, contradict itself, or invent diagnoses that the model cannot produce.

Only then does the result go to the frontend or another client.

---

## 12. Why the project is careful about guarantees

The project does not try to pretend that a model prediction is perfect.

Instead, it tries to answer clinical questions honestly:

- when can we rule something out?
- when should a doctor review the case?
- when do we need to withhold a claim?

The `conformal` logic gives a finite-sample bound on error rates.

That is why this project is stronger than a standard ECG classifier: it is designed around risk control, not just top-1 accuracy.

---

## 13. The few critical design rules to remember

These are the essential principles this project follows:

1. A bad ECG should be refused, not diagnosed.
2. A model is not allowed to pretend certainty without a valid guarantee.
3. Explanations must be tied to evidence.
4. Generated text must be verified before release.
5. Safety matters more than polished narration.
6. The model pipeline and calibration pipeline must align exactly.
7. Different models each need their own valid safety layer.

---

## 14. Recommended reading order for a new developer

If you are new to this codebase, read in this order:

1. `README.md`
2. `START_HERE.md`
3. `SYSTEM_README.md`
4. `src/models.py`
5. `src/quality.py`
6. `src/preprocess.py`
7. `src/calibration.py`
8. `src/conformal.py`
9. `src/xai.py`
10. `src/report.py`
11. `src/verify.py`
12. `src/pipeline.py`
13. `backend/server.py`
14. `frontend/src/App.jsx` and `frontend/src/api.js`

This is the shortest route to understanding the actual system.

---

## 15. Final summary

Component 02 is not a single script. It is a complete ECG decision-support system with:

- strict signal quality validation
- robust preprocessing
- deep-learned classification
- calibration
- probabilistic triage with guarantees
- explainability
- report generation
- automated safety verification
- backend and frontend integration

The most important code is concentrated in `src/`, especially:

- `models.py`
- `quality.py`
- `preprocess.py`
- `calibration.py`
- `conformal.py`
- `xai.py`
- `report.py`
- `verify.py`
- `pipeline.py`

Those files form the real operating system of the project.

This system is designed to do two unusual things well:

- it can say “I cannot interpret this safely”
- it can say “I am uncertain and this should be reviewed by a clinician”

Those two behaviors are central to the project’s design and are the main difference from a standard classifier.

---

End of artifact.
