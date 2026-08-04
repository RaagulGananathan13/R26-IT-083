# Explainable AI System for Cardiovascular Disease Detection and Diagnosis

A final-year research project built around a shared idea: cardiovascular AI is only clinically useful if it can show its work. Across four independent components, the system takes different clinical data modalities — chest X-rays, 12-lead ECGs, echocardiogram videos, and emergency triage data — and pairs every prediction with an explanation a clinician can actually verify (Grad-CAM, SHAP, Integrated Gradients, wall-motion/segmentation maps) instead of an opaque score.

Each component is an independently built, independently trained, and independently deployed system. This document summarizes all four and links out to their individual, full READMEs for setup and usage details.

> ⚕️ **Disclaimer:** All four components are research prototypes built for an undergraduate final-year project. None are approved medical devices, none have undergone clinical validation, and none should be used as the sole basis for diagnosis or treatment. Every output requires review by a qualified clinician.

---

## Components at a Glance

| # | Component | Author | Student ID | Input Data | Core Task | XAI Method(s) |
|---|---|---|---|---|---|---|
| 1 | Cardiomegaly Detection with XAI and Automatic Report Generation | Raagul Gananathan | IT22130020 | Chest X-ray (MIMIC-CXR) | Multi-label detection of 8 chest pathologies + automated report generation | Grad-CAM |
| 2 | XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting System | Venushan T | IT22082824 | 12-lead ECG (PTB-XL) | 5-class ECG superclass classification + automated clinical report generation | Grad-CAM (temporal) + Integrated Gradients (spatial) |
| 3 | EchoStrat — Cardiac Function Assessment and Ejection Fraction Prediction via Spatiotemporal Explainable AI | Dilukshan | IT22219534 | Echocardiogram video (EchoNet-Dynamic) | Ejection fraction (EF) prediction + 4-class heart failure severity classification | Grad-CAM, wall motion maps, LV segmentation |
| 4 | XAI-Based Acute Coronary Syndrome (ACS) Detection and ACS Type Classification | Abishnan J | IT22140234 | Multimodal triage data — vitals, labs, demographics, clinical text, ECG markers (MIMIC-IV-ED) | Binary ACS detection + subtype classification (UA / NSTEMI / STEMI) | SHAP + NLP highlighting |

---

## Component 01 — Cardiomegaly Detection with XAI and Automatic Report Generation

**Author:** Raagul Gananathan (IT22130020)

Takes a frontal chest X-ray and does three things: predicts cardiomegaly plus seven co-occurring pathologies (edema, pleural effusion, atelectasis, consolidation, lung opacity, pneumonia, pneumothorax), highlights *where* on the image the model looked via Grad-CAM, and drafts a radiology report in clinical language — while auditing itself for reliability.

**Architecture:** A shared ConvNeXt-Base (384×384, ImageNet-pretrained) vision backbone feeds two heads — a multi-label classifier (8 pathologies) and a BioBART-v2-base decoder that generates a "FINDINGS / IMPRESSION" style report from the same visual features (via `inputs_embeds`, not `encoder_outputs`, which was found to matter significantly). Grad-CAM is computed on the final convolutional stage.

**Dataset:** MIMIC-CXR / MIMIC-CXR-JPG (PhysioNet, credentialed), patient-disjoint split — 36,362 train / 4,474 val / 4,722 test, frontal views only (AP and PA).

**Key results (test set, n = 4,722):**
- Cardiomegaly: AUROC **0.9189**, sensitivity 92.3%, accuracy 83.2%
- Mean AUROC across 8 pathologies: **0.8554**
- Report generator: CheXbert micro-F1(14) **0.5939**, zero fabricated prior-study references

**Research contribution — Acquisition fairness:** An audit found the classifier is significantly worse on AP (bedside) films than PA (standing) films (AUROC gap 0.0639, 95% CI [0.0491, 0.0790]) — and AP films come from the sickest patients. Per-projection threshold adjustment cut TPR disparity by 73.3% at zero accuracy cost, but the underlying AUROC gap proved irreducible at the model level (tested via adversarial invariance and conditional heads, both ineffective).

**Tech stack:** PyTorch, torchvision, Hugging Face Transformers, FastAPI, React, Google Colab (NVIDIA L4).

**License:** Code under MIT License. MIMIC-CXR data is credentialed and not redistributed — reproducing this component requires the user's own PhysioNet credentialed access.

📄 Full details: `IT22130020_Raagul` component README (includes full model architecture diagrams, ablations, and the "What Didn't Work" section).

---

## Component 02 — XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting System

**Author:** Venushan T (IT22082824)

Classifies 12-lead ECG signals into 5 diagnostic superclasses and generates a hallucination-free clinical report, via a zero-setup Flask web interface for uploading raw `.dat`/`.hea` WFDB files.

**Architecture — Three-Tier Hybrid Reporting Pipeline:**
1. **Classification:** A custom 1D ResNet processes all 12 leads to output probabilities for 5 superclasses (NORM, MI, STTC, CD, HYP).
2. **Structured reporting:** A deterministic template engine generates clinical findings strictly from the classifier's output (avoiding LLM hallucination).
3. **Language smoothing:** A constrained BioBART decoder converts the structured template into natural clinical prose without seeing the raw signal, so it cannot invent findings.

**Explainability:** Grad-CAM (temporal) shows *when* in the 10-second signal an abnormality was detected; Integrated Gradients (spatial) ranks *which* of the 12 leads contributed most.

The full research model also explores multimodal fusion — 1D ECG signal (CNN) + patient demographics (MLP) + prior clinical reports (ClinicalBERT embeddings).

**Dataset:** PTB-XL (PhysioNet), 21,837 twelve-lead ECG recordings across 5 superclasses.

**Key results (test set, 1,711 records):**

| Class | F1 |
|---|---|
| Normal (NORM) | 0.87 |
| ST/T Change (STTC) | 0.77 |
| Conduction Disturbance (CD) | 0.75 |
| Myocardial Infarction (MI) | 0.68 |
| Hypertrophy (HYP) | 0.49 |
| **Macro F1** | **0.717** |

**Tech stack:** PyTorch, Flask, Transformers, Pandas, SciPy, WFDB.

**Running locally:** `cd _archive && python app.py`, then open `http://localhost:5000`.

📄 Full details: `IT22082824_Venushan__T` component README.

---

## Component 03 — EchoStrat: Cardiac Function Assessment and Ejection Fraction Prediction via Spatiotemporal Explainable AI

**Author:** Dilukshan (IT22219534)

Assesses cardiac function from Apical Four-Chamber (A4C) echocardiogram videos, predicting the exact ejection fraction (EF) percentage, a 4-class ACC/AHA 2022 severity grade (Severely Reduced / Reduced / Mildly Reduced / Normal), and flagging urgent referrals when EF < 35%. Motivated by the ±10% inter-observer variability between cardiologists on manual EF estimation, which becomes clinically dangerous near the 35% ICD-eligibility threshold.

**Architecture:** An R(2+1)D-18 backbone (double-pretrained: Kinetics-400 → EchoNet-Dynamic → EchoStrat) feeds a 3-head ensemble — a classification head (severity), a regression head (continuous EF%), and a CORN (Conditional Ordinal Regression) head for ordinal boundary consistency, combined via a 35%/25%/40% weighted ensemble. Explainability comes from Grad-CAM heatmaps, wall motion intensity maps, and LV segmentation.

**Notable methods:** boundary-weighted CORN loss near clinical EF thresholds (30/40/50%), ordinal-aware mixup/CutMix that only blends adjacent severity classes, 8-way test-time augmentation, and stochastic weight averaging.

**Dataset:** EchoNet-Dynamic (Stanford University), 10,030 A4C echo videos.

**Key results (test set):**

| Metric | Value |
|---|---|
| Balanced Accuracy | 69.62% |
| AUROC (Macro) | 84.72% |
| Normal (C3) Accuracy | 90.75% |
| Severely Reduced (C0) Accuracy | 83.95% |
| Mildly Reduced (C2) Accuracy | 55.63% |
| Reduced (C1) Accuracy | 48.15% |
| EF MAE | ~7% |

**Tech stack:** PyTorch/torchvision (training), FastAPI (backend `/predict`), React + Vite + Tailwind CSS (frontend).

**Running:** Backend — `cd Backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000`; Frontend — `cd Frontend && npm run dev`, then open `http://localhost:5173`.

📄 Full details: `IT22219534_Dilukshan` component README.

---

## Component 04 — XAI-Based Acute Coronary Syndrome (ACS) Detection and ACS Type Classification

**Author:** Abishnan J (IT22140234)

A two-stage hierarchical pipeline for emergency triage. Fuses 29 clinical features across 5 modalities — structured vitals, laboratory biomarkers, patient demographics, unstructured clinician-entered text (NLP), and physiological ECG signal markers — into a real-time clinical decision-support tool.

**Pipeline:**
1. **Stage 1 — Binary Gatekeeper:** An XGBoost binary classifier predicts ACS presence (yes/no) with high sensitivity, using a 52-dimensional fused feature vector, tuned via `scale_pos_weight` for severe class imbalance.
2. **Stage 2 — Subtype Classifier:** If ACS is detected, a multiclass XGBoost model differentiates Unstable Angina (UA), NSTEMI, and STEMI. A custom "STEMI-Boost" tiered probability heuristic overrides baseline probabilities when ST-Elevation plus critically high Troponin are both present, targeting 100% STEMI safety recall.

**Explainability:** SHAP (SHapley Additive exPlanations) computed directly on the XGBoost trees for feature-level attribution (e.g., troponin, prior MI, chief-complaint text), plus NLP highlighting of triage text.

**Dataset:** MIMIC-IV-ED (PhysioNet/MIT) — structured triage vitals/labs/demographics, unstructured chief-complaint text, and paired ECG waveforms (`.dat`/`.hea`).

**Key results:**

| Stage | Metric | Value |
|---|---|---|
| Stage 1 (Binary ACS) | AUC-ROC | 0.9841 |
| Stage 1 (Binary ACS) | Balanced Accuracy | 77.0% |
| Stage 1 (Binary ACS) | Negative Predictive Value | 99.93% |
| Stage 2 (Subtype) | STEMI Safety Recall | 100% |

**Tech stack:** XGBoost, SHAP, scikit-learn, WFDB, NeuroKit2, TF-IDF/NLTK (ML); FastAPI, Uvicorn (backend); React, Vite, Tailwind CSS (frontend).

**Running:** Backend — `cd Component_4 && python app.py` (serves at `http://localhost:8000`); Frontend — `cd Component_4/frontend && npm run dev` (serves at `http://localhost:5173`).

**License:** MIT License (code). Research/academic use only — not an approved diagnostic tool.

📄 Full details: `IT22140234_Abishnan` component README.

---

## Common Design Themes Across Components

- **Different data modalities, one philosophy:** imaging (X-ray, echo video), signal (ECG), and structured/text (triage data) are each handled by a modality-appropriate model, but every component pairs its prediction with a feature- or region-level explanation rather than a bare score.
- **Guarding against black-box and hallucination risk:** Components 1 and 2 both explicitly avoid free-text LLM generation from raw data — Component 1 conditions its report generator on a shared, verified visual backbone, and Component 2 uses a deterministic "classifier-first, template-second" pipeline so its language model cannot invent findings.
- **Safety-first thresholding:** Components 2, 3, and 4 all build in explicit clinical safety margins — urgent referral flags (Component 3, EF < 35%), 100% STEMI safety recall (Component 4), and sensitivity-first operating points (Components 1 and 2).
- **Deployment:** all four ship as web applications (Flask or FastAPI backends with browser-based interfaces) rather than requiring local software installation by the clinician.

---

## Repository Structure

Each component lives in its own top-level folder with its own README, dependencies, training scripts, and web application:

```
project-root/
├── Component_01/   # Raagul — Cardiomegaly Detection with XAI
├── Component_02/   # Venushan — ECG Abnormality Detection (referred to as "_archive" in its own README)
├── Component_03/   # Dilukshan — EchoStrat (Ejection Fraction / Severity)
└── Component_04/   # Abishnan — ACS Detection & Type Classification
```

Refer to each component's own README for prerequisites, installation, dataset access instructions (several datasets — MIMIC-CXR, MIMIC-IV-ED — require credentialed PhysioNet access), training steps, and API usage.

---

## Datasets Used

| Component | Dataset | Source |
|---|---|---|
| 1 | MIMIC-CXR / MIMIC-CXR-JPG | PhysioNet (credentialed) |
| 2 | PTB-XL | PhysioNet |
| 3 | EchoNet-Dynamic | Stanford University |
| 4 | MIMIC-IV-ED | PhysioNet / MIT (credentialed) |

---

## Note on Licensing

Component 1's code is MIT-licensed; its underlying MIMIC-CXR data is credentialed and not redistributed. Component 4 is released under the MIT License for academic/research use. Component 3's own README states it was developed "as part of the FedMed Federated Healthcare Intelligence System" rather than referencing this project's title directly — worth confirming with Dilukshan whether that's the intended framing before this combined README is finalized. Component 2's README does not state a license.

---

## Disclaimer

All four components are research prototypes developed for an undergraduate final-year research project. None have undergone clinical validation, none are approved medical devices, and none should be used as the sole basis for diagnosis, treatment, or patient management. All outputs require review by a qualified clinician.
