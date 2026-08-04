# XAI-Based Acute Coronary Syndrome (ACS) Detection and ACS Type Classification

**Component:** ACS Detection + ACS Type Classification with Explainable AI

**Name:** Abishnan J

**IT Number:** IT22140234

---

## Project Description

Cardiovascular diseases remain one of the leading causes of mortality worldwide. Among them, Acute Coronary Syndrome (ACS) is a time-critical medical emergency that encompasses unstable angina (UA), non-ST elevation myocardial infarction (NSTEMI), and ST elevation myocardial infarction (STEMI). Early identification of ACS is clinically challenging because symptoms such as chest pain, nausea, diaphoresis, and shortness of breath often overlap with non-cardiac conditions.

For this research, my specific component focuses on building a multimodal, two-stage hierarchical Artificial Intelligence pipeline for ACS Detection and Subtype Classification.

Instead of relying on single data points, my system fuses 29 distinct clinical features across 5 modalities: structured vitals, laboratory biomarkers, patient demographics, unstructured clinician-entered text (processed via NLP), and physiological ECG signal markers.

When emergency triage data is entered into the system, my component performs the following:

Stage 1 (Binary Gatekeeper): Instantly predicts whether the patient is experiencing ACS (YES / NO) with high sensitivity to ensure critical cases are not missed.

Stage 2 (Subtype Classification): If ACS is detected, the model classifies the exact medical emergency (Unstable Angina, NSTEMI, or STEMI) to dictate the specific clinical treatment pathway.

Explainable AI (XAI) Integration: To solve the "black-box" problem in medical AI, my component utilizes SHAP (SHapley Additive exPlanations) and NLP highlighting. It provides clinicians with real-time, transparent reasoning, explicitly showing which vitals, lab results, or textual symptoms drove the diagnosis.

Ultimately, my component acts as the core clinical decision support engine, deployed as a real-time web application to assist cardiologists and triage nurses in making faster, safer, and more accurate life-saving decisions.

---

## Objectives

-Build a hierarchical classification pipeline using XGBoost to first detect Acute Coronary Syndrome (ACS), and then classify it into three critical subtypes (UA, NSTEMI, STEMI), targeting high Balanced Accuracy and 100% STEMI safety recall.
-Train a multimodal fusion model that ingests 29 clinical features—including structured triage vitals, laboratory biomarkers, and TF-IDF encoded chief complaint text—to produce a real-time clinical risk score.
-Integrate SHAP (SHapley Additive exPlanations) to produce transparent, feature-level attributions that highlight exact clinical markers (e.g., troponin levels, high pain scores), so clinicians can verify the model's reasoning.
-Differentiate between critical overlapping conditions (Unstable Angina vs. NSTEMI vs. STEMI) alongside the primary ACS detection using custom probability heuristics (STEMI-Boost).
-Deploy everything as a usable real-time web application (using FastAPI and React) with a clean, dark-themed clinical triage interface.
-Make the system's outputs interpretable and trustworthy enough to serve as a rapid decision-support tool for cardiologists and triage nurses in high-stress emergency departments.

---

## Technologies Used


--AI and Machine Learning

-Python 3.12+
-XGBoost (Hierarchical tree-based classification for binary and multiclass prediction)
-SHAP (SHapley Additive exPlanations for model interpretability and feature attribution)
-scikit-learn (Group-stratified splitting, class weighting, AUROC, and Balanced Accuracy metrics)
-WFDB & NeuroKit2 (Physiological signal parsing and ECG waveform feature extraction)
-NLTK / TF-IDF Vectorizer (Text processing and encoding for clinical chief complaints)
-NumPy, Pandas, Matplotlib, Seaborn (Data manipulation, feature matrices, and visualization)

**Backend**

-FastAPI 0.110 (REST API with /predict and /predict-with-ecg endpoints)
-Uvicorn 0.29 (ASGI web server)
-python-multipart (File upload handling for .dat and .hea ECG waveforms)
-joblib (Model serialization, configuration, and weights loading)

**Frontend**

- React 18.x (Clinical UI rendering)
-Vite 5.x (Build tool, dev server, and API proxy)
-Tailwind CSS 3.x (Utility-first styling for dark-mode dashboard)

**Training**

- Local High-Performance CPU Environment / Google Colab
-XGBoost hist tree method (Histogram-based algorithm optimized for highly accelerated training on large tabular datasets)
-Patient-level Group Stratified Splitting (Ensuring 0% data leakage across train, validation, and test splits)
-Advanced Class Weighting (Utilizing scale_pos_weight and custom probability thresholds to handle the severe class imbalance of STEMI cases)

**Dataset**

- MIMIC-IV-ED (PhysioNet / MIT)
-Structured tabular clinical data (triage vitals, laboratory biomarkers, patient demographics)
-Unstructured clinician-entered triage notes (chiefcomplaint column)
-Paired physiological ECG waveforms (processed from .dat and .hea WFDB files)

---



## How It Works
The system uses two hierarchical models and a centralized feature engineering pipeline that work together:

## Model 1 — Binary Gatekeeper (XGBoost Classifier)

Takes the fused 52-dimensional multimodal feature vector (derived from 13 base clinical inputs plus raw ECG data) and runs it through a highly sensitive XGBoost binary classifier. This model acts as the triage gatekeeper, outputting a primary ACS prediction (positive/negative) with a baseline confidence score. It explicitly handles severe class imbalances using scale_pos_weight to ensure rare but critical cardiac events are not overlooked.

SHAP (SHapley Additive exPlanations) is computed directly on this model's decision trees. The exact marginal contribution of each feature (e.g., prior_mi, troponin_max, cc_chest_pain) is extracted to produce a transparent feature attribution report, allowing the clinician to visually confirm exactly why the AI triggered an ACS alert.

## Model 2 — Subtype Classifier (XGBoost Multiclass)

If Model 1 detects ACS, the same feature vector is immediately passed to the Stage 2 Multiclass XGBoost model. This model differentiates between three highly overlapping critical conditions: Unstable Angina (UA), NSTEMI, and STEMI.

Because missing a STEMI is fatal, this stage incorporates a Tiered Probability Heuristic (STEMI-Boost). If the raw ECG parser detects ST-Elevation combined with critically high serial Troponin levels, the system actively overrides standard statistical probabilities, dynamically boosting the STEMI confidence score to ensure an immediate "Cath Lab" activation protocol.

The Shared Multimodal Fusion Pipeline

## A key design choice is that both models do not rely on raw, unstructured data. The centralized pipeline processes the data first:

NLP Text Mining: The raw "Chief Complaint" text is passed through regex and TF-IDF to extract weighted symptoms (chest pain, diaphoresis, radiating pain).
Signal Parsing: The uploaded .dat/.hea files are parsed via wfdb and neurokit2 to extract QRS Duration, LBBB, and ST-segment changes.

## Installation Steps

### Prerequisites

- Python 3.12 or newer
- Node.js 20 or newer with npm
- Git

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Component_4
```

### 2. Set up the backend

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Place model weights

Ensure the trained XGBoost models are located in the correct directories:

```text
models/
    stage1/
        xgb_stage1.joblib
    stage2/
        xgb_stage2.joblib
```

### 4. Set up the frontend

```bash
cd frontend
npm install
```

---

## Usage Instructions

### Running the app

Open two terminals.

**Terminal 1 — Backend:**

```bash
cd Component_4
python app.py
```

This starts the FastAPI server at `http://localhost:8000`.

**Terminal 2 — Frontend:**

```bash
cd Component_4/frontend
npm run dev
```

This starts the React dev server at `http://localhost:5173`. Open that URL in your browser.

### Using the interface

1. Input the patient's triage vitals, laboratory biomarkers, and demographics into the respective fields.
2. Type the patient's exact symptoms into the "Chief Complaint" text box.
3. (Optional) Toggle the specific ECG findings if an ECG has been performed.
4. Click **"Run ACS Detection"**.
5. You'll see:
   - A diagnosis card showing the final ACS prediction (e.g., STEMI, NSTEMI, UA, No ACS).
   - The clinical risk level and actionable medical directive.
   - The specific probabilities for each subtype.

### Training the models yourself

The training scripts are optimized for high-performance CPU execution on tabular data. 

**Model 1 & 2** (`train_xgboost_acs_component4_FIXED.py`): Run this script to execute the patient-level stratified splitting and train both the binary ACS detector and the multiclass subtyper.

```bash
python train_xgboost_acs_component4_FIXED.py
```

### API endpoint

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"vitals": {"heartrate": 110, "sbp": 90}, "demographics": {"age": 65}, "chief_complaint": "chest pain"}'
```

Returns JSON with the final diagnosis prediction, risk level, subtype probabilities, and recommended clinical action.

---


## Features

**Classification**
- Binary ACS detection with an AUROC of 0.98 on the test set.
- Multi-class differentiation of 3 critical subtypes (Unstable Angina, NSTEMI, STEMI).
- Tiered probability heuristics (STEMI-Boost) ensuring 100% Safety Recall for critical cases.

**Explainability**
- SHAP (SHapley Additive exPlanations) extracted directly from XGBoost decision trees.
- Feature-level attributions showing exactly which clinical markers (e.g., troponin) drove the diagnosis.
- NLP-based identification highlighting crucial symptoms from the triage text.

**Multimodal Fusion & Text Mining**
- 5 distinct modalities (vitals, labs, demographics, text, ECG) fused into a 52-dimensional feature vector.
- Custom regex and TF-IDF pipeline parsing raw "Chief Complaint" text into clinical risk scores.
- Automated parsing of QRS Duration, LBBB, and ST-segment changes from raw `.dat`/`.hea` ECG signal files.

**Web Interface**
- Dynamic risk-level badging (Low, Moderate, High, Critical) with color-coded probability bars.
- Real-time inference results displayed in a dark-themed medical dashboard layout.
- Immediate, actionable clinical directives (e.g., "IMMEDIATE CATH LAB ACTIVATION").
- Responsive design tailored for emergency department screens.

**Training**
- XGBoost `hist` tree method optimized for highly accelerated execution on tabular data.
- Strict patient-level (`subject_id`) grouped stratified splitting, completely eliminating data leakage.
- Advanced class weighting via `scale_pos_weight` to address the severe imbalance of STEMI cases.
- Missing-data-aware processing pipeline to handle routinely missing biomarkers (Troponin, BNP).

---


## Project Structure

```text
Component_4/
|
|-- README.md                          # this file
|-- train_xgboost_acs_component4_FIXED.py # training script for binary and multiclass XGBoost models
|-- feature_engineering.py             # multimodal feature extraction (NLP, Vitals, Labs)
|-- ecg_parser.py                      # wfdb parser for raw .dat/.hea ECG waveforms
|-- deep_e2e_test.py                   # end-to-end safety & testing script
|-- verify_ecg.py                      # clinical threshold verification script
|
|-- app.py                             # FastAPI server and inference routes (Backend)
|
|-- frontend/                          # React application (Frontend)
|   |-- src/
|   |   |-- App.jsx                    # main React dashboard component
|   |   |-- index.css                  # Tailwind styling directives
|   |   |-- main.jsx                   # React entry point
|   |-- package.json
|   |-- vite.config.js
|   |-- tailwind.config.js
|
|-- data/                              # Dataset directory
|   |-- splits/                        # Patient-level stratified training/testing splits
|
|-- models/                            # Trained model weights
|   |-- stage1/                        # Binary ACS detection model (xgb_stage1.joblib)
|   |-- stage2/                        # Subtype classification model (xgb_stage2.joblib)
|
|-- demo_ecg/                          # Sample ECG waveform files for UI testing
|   |-- STEMI/                         # .dat and .hea files
|   |-- NSTEMI/
|   |-- UA/
|   |-- No_ACS/
```

---

## Model Performance

**Binary ACS Gatekeeper (Stage 1)**

- AUC-ROC: 0.9841
- Balanced Accuracy: 77.0%
- Negative Predictive Value (NPV): 99.93% — extremely reliable when confirming "No ACS" for safe discharge
- Architecture: XGBoost Binary Classifier with 52-dimensional multimodal inputs
- Training: `hist` tree method, optimized with `scale_pos_weight` to handle severe class imbalances and patient-level cross-validation

**ACS Subtype Classifier (Stage 2)**

- STEMI Safety Recall: 100% — aggressively tuned to never miss a critical, life-threatening STEMI case
- Subtypes Classified: Unstable Angina (UA), NSTEMI, STEMI
- Architecture: XGBoost Multi-Class Classifier (`multi:softprob` objective)
- Training: Incorporates a custom "STEMI-Boost" tiered probability heuristic, actively overriding baseline probabilities when critical ST-Elevation and Troponin thresholds are met


## Novelty
* First explainable multimodal ACS triage system.
* Uses real clinician-entered chief complaint text.
* Dual prediction: ACS detection + ACS type classification.
* Modality attribution for clinician trust.
* Missing-data-aware biomarker prediction.

---

## Future Work
* ECG waveform integration.
* Transformer-based multimodal learning.
* External hospital validation.

---

## References

1.MIMIC-IV-ED — Johnson, A.E.W. et al. (2023). MIMIC-IV Emergency Department Database.
2.XGBoost — Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of KDD 2016.
3.SHAP — Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
4.Term frequency–inverse document frequency — Ramos, J. (2003). Using TF-IDF to Determine Word Relevance in Document Queries.
5.Acute Coronary Syndrome — Amsterdam, E.A. et al. (2014). AHA/ACC Guideline for the Management of Patients with Non-ST-Elevation Acute Coronary Syndromes.
6.ST-elevation myocardial infarction — O’Gara, P.T. et al. (2013). ACCF/AHA Guideline for the Management of STEMI.
7.Natural Language Processing — Jurafsky, D. and Martin, J.H. (2023). Speech and Language Processing.
8.Class imbalance — Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique

**Frameworks and tools:** 
-Python
-XGBoost
-scikit-learn
-SHAP
-TF-IDF
-Flask
-React
-Vite
-Tailwind CSS
-Visual Studio Code

**Dataset:** 
-MIMIC-IV-ED
-Linked hospital records from MIMIC-IV
-Linked ICU records from MIMIC-IV

---

## License

-This project was developed for academic and research purposes as part of a university final-year research project.

-The project focuses on **Acute Coronary Syndrome detection and ACS subtype classification using explainable artificial intelligence and multimodal clinical data.

-This project is released under the MIT License.


**Medical disclaimer:** 
-This system is a research prototype designed for educational and experimental purposes only. It is not an approved clinical diagnostic tool and must not be used as the sole basis for medical diagnosis, treatment decisions, or patient management.