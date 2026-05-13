# Project Overview

This repository contains two main projects currently undergoing restructuring:

## 1. Clinical ECG Analysis System (`_archive/app.py`)
A comprehensive hybrid AI pipeline for analyzing 12-lead Electrocardiograms (ECGs).
- **Tier 1**: ResNet-1D CNN classifier for extracting features and detecting abnormalities.
- **Tier 2**: Deterministic Template Engine for generating clinically grounded, hallucination-free sentences.
- **Tier 3**: BioBART sequence-to-sequence model used as a smoother for natural language polishing.
- **Explainable AI (XAI)**: Integrated Gradients for lead-wise saliency and 1D Grad-CAM for temporal attention visualization.

To run the ECG Analysis System:
```bash
cd _archive
python app.py
```
Then navigate to `http://localhost:5000`.

## 2. Worksheet Generator (`_archive/index.html`)
A dynamic web-based worksheet generator that allows educators to create math worksheets, trace pages, and equations. It supports uploading custom templates and generating multiple randomized variations.

## Frontend (`frontend/`)
A React + Vite frontend that is currently being set up for the new project architecture.
