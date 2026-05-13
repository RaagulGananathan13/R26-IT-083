 XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting System



**Author**: Venushan T  
**Project Type**: Research Project (RP)  
**Focus**: Clinical-Grade 12-Lead ECG Classification & Automated Report Generation  
**Dataset**: PTB-XL (PhysioNet)

---

## 📖 Overview

Cardiovascular disease is the leading cause of death globally, yet skilled cardiologists to interpret ECGs are scarce in many regions. Existing AI tools often act as "black boxes" or utilize free-text LLMs that can hallucinate non-existent clinical findings—creating unacceptable patient safety risks.

This project is a **clinically trustworthy, explainable, automated ECG analysis system** that operates entirely via a zero-setup web interface. It classifies 12-lead ECG signals into 5 diagnostic superclasses and generates **hallucination-free** clinical reports while explicitly explaining *which* parts of the signal drove the diagnosis.

---

## 🌟 Key Features & Novelty

1. **Three-Tier Hybrid Reporting Pipeline**: 
   - Instead of relying on an LLM to freely generate text from raw signals (which causes hallucinations), this system uses a "classifier-first, template-second" architecture. 
   - It fills deterministic templates based on CNN outputs (Zero Hallucination Risk) and optionally uses a constrained BioBART model to smooth the text into natural clinical prose.
2. **First-Class Explainable AI (XAI)**:
   - **Grad-CAM (Temporal)**: Highlights exactly *when* in the 10-second signal the model detected abnormalities.
   - **Integrated Gradients (Spatial)**: Ranks exactly *which* of the 12 leads contributed most to the diagnosis.
3. **Multi-Modal Fusion**:
   - The full research model fuses 1D ECG signals (CNN), Patient Demographics (MLP), and prior Clinical Reports (ClinicalBERT embeddings) for maximum accuracy.
4. **Zero-Setup Clinical Web Interface**:
   - Built with Flask, the system allows clinicians to drag-and-drop raw `.dat` and `.hea` WFDB files and receive instant visual analysis without installing any heavy medical software.

---

## 🧠 Architecture Pipeline

1. **Preprocessing**: 12-lead, 500Hz signals are normalized using pre-computed statistics.
2. **Classification (Tier 1)**: A custom **1D ResNet** processes all leads simultaneously to output probabilities for 5 superclasses (NORM, MI, STTC, CD, HYP).
3. **Structured Reporting (Tier 2)**: A deterministic template engine generates clinical findings based strictly on the ResNet's output.
4. **Language Smoothing (Tier 3)**: A **BioBART** decoder translates the structured template into natural clinical prose without seeing the raw signal, ensuring it cannot invent findings.

---

## 📊 Dataset & Performance

Trained on the **PTB-XL** dataset featuring 21,837 clinical 12-lead ECG recordings categorized into 5 superclasses.

**ResNet-1D Classifier Results (Test Set - 1711 records):**
- **Normal (NORM)**: F1 - 0.87
- **Myocardial Infarction (MI)**: F1 - 0.68 
- **ST/T Change (STTC)**: F1 - 0.77
- **Conduction Disturbance (CD)**: F1 - 0.75
- **Hypertrophy (HYP)**: F1 - 0.49
- **Macro F1**: 0.717

---

## 🚀 Quick Start (Running Locally)

To launch the web interface and explore the test set or upload your own `.dat`/`.hea` records:

1. Ensure you have the required dependencies installed (PyTorch, Flask, Transformers, Pandas, SciPy, WFDB).
2. Navigate to the archive folder:
   ```bash
   cd _archive
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to:
   **http://localhost:5000**

---
*Developed as a comprehensive research implementation exploring the intersection of time-series classification, natural language generation, and explainable AI in healthcare.*
