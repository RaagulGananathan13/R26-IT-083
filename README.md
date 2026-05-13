# Explainable AI System for Cardiovascular Disease Detection and Diagnosis

**Component:** Cardiomegaly Detection with XAI and Automatic Report Generation

**Name:** Raagul Gananathan

**IT Number:** IT22130020

---

## Project Description

Cardiovascular diseases are the leading cause of death globally, with around 17.9 million deaths every year. Cardiomegaly — an enlarged heart visible on chest X-rays — is one of the earliest indicators of conditions like heart failure and cardiomyopathy. Catching it early makes a real difference in patient outcomes, but manual reading of X-rays is slow, subjective, and depends heavily on the radiologist's experience.

This project builds an AI system that does four things when you give it a chest X-ray:

1. **Tells you if cardiomegaly is present or not**, with a confidence score.
2. **Detects co-existing conditions** — pleural effusion, pulmonary edema, pneumothorax, atelectasis, consolidation, lung opacity, and pneumonia — through multi-label classification, since these pathologies frequently occur alongside cardiomegaly.
3. **Shows you exactly where the model is looking** using a GradCAM heatmap overlay on the X-ray.
4. **Writes a radiology report** describing the findings, similar to what a radiologist would write.

The primary focus is cardiomegaly, but the model doesn't stop there. In clinical practice, an enlarged heart rarely exists in isolation — patients often present with fluid in the lungs (edema), fluid around the lungs (pleural effusion), or collapsed lung tissue (atelectasis). Detecting these together gives a much more complete picture.

The key idea is that it's not just a black-box classifier. The GradCAM heatmap and the generated report together act as explanations — a clinician can look at the heatmap to verify the model is focusing on the heart region, and read the report to see if the textual description makes clinical sense. This is what makes it an *explainable* AI system.

The whole thing runs as a web app — a React frontend where you upload the X-ray, and a FastAPI backend that runs both models and returns the results.

---

## Objectives

- Build a binary classifier for cardiomegaly using ConvNeXt-Base pretrained on ImageNet-22K, targeting high AUC on the test set.
- Train a report generation model that takes a chest X-ray and produces a radiology report (Impression + Findings) using a ConvNeXt encoder paired with a BART decoder.
- Integrate GradCAM to produce visual explanations that highlight the cardiac region, so clinicians can verify the model's reasoning.
- Detect co-occurring conditions (edema, pleural effusion, atelectasis, etc.) alongside the primary cardiomegaly diagnosis.
- Deploy everything as a usable web application with a clean clinical interface.
- Make the system's outputs interpretable and trustworthy enough to serve as a decision-support tool.

---

## Technologies Used

**AI and Deep Learning**

- Python 3.11+
- PyTorch 2.x with CUDA support
- torchvision (image transforms, ConvNeXt-Base backbone)
- timm (pretrained model weights — `convnext_base.fb_in22k_ft_in1k`)
- HuggingFace Transformers (BART tokenizer and decoder)
- scikit-learn (AUC-ROC, confusion matrix, classification report)
- rouge-score (ROUGE-1/2/L for evaluating generated reports)
- OpenCV (GradCAM heatmap colorization)
- NumPy, Pandas, Matplotlib, Pillow

**Backend**

- FastAPI 0.110 (REST API with `/predict` endpoint)
- Uvicorn 0.29 (ASGI server)
- python-multipart (file upload handling)

**Frontend**

- React 19.x
- Vite 6.x (build tool and dev server)
- Tailwind CSS 4.x

**Training**

- Google Colab with NVIDIA T4 GPU (15GB VRAM)
- Mixed precision training (FP16 via PyTorch AMP)
- Gradient accumulation for the report model (effective batch size 16)

**Dataset**

- MIMIC-CXR (PhysioNet / MIT)
- 384x384 grayscale chest X-ray images
- Paired radiology reports with `report_text`, `findings_text`, and `impression_text` columns

---

## How It Works

The system has two trained models that work together:

**Model 1 — Image Classifier (ConvNeXt-Base)**

Takes a 384x384 chest X-ray, runs it through a ConvNeXt-Base backbone (with the early stages frozen and the later stages fine-tuned), and outputs a cardiomegaly prediction (positive/negative) with a confidence score. It also detects 7 other pathologies (edema, pleural effusion, atelectasis, consolidation, lung opacity, pneumonia, pneumothorax) through a multi-label classification head.

GradCAM is computed on the last convolutional stage (`features[7]`). The gradients of the cardiomegaly output are backpropagated to produce a heatmap showing which spatial regions contributed most to the prediction. This heatmap is overlaid on the original X-ray so you can visually confirm the model is focusing on the heart.

**Model 2 — Report Generator (ConvNeXt + BART)**

Uses the same ConvNeXt backbone (frozen, loaded from Model 1's trained weights) as a vision encoder. The 12x12 spatial feature map is flattened into 144 visual tokens, projected from 1024 dimensions to 768 dimensions through a learned linear layer with LayerNorm and GELU activation, and then fed into a BART decoder as encoder outputs.

BART generates the report text using beam search (4 beams) with no-repeat trigram blocking. The raw output goes through a regex-based cleaning pipeline that strips out training artifacts — things like "compared to the previous exam from ___" or "findings discussed with referring physician at 3:45 PM" — which are present in the MIMIC-CXR training data but meaningless for new patients.

**The Shared Encoder**

A key design choice is that Model 2's vision encoder is not trained from scratch. It loads the exact convolutional weights from Model 1's trained checkpoint and freezes them. This means both the classifier and the report generator are looking at the same visual features — if the classifier says cardiomegaly is present, the report generator is working from the same visual evidence.

---

## Installation Steps

### Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer with npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CardioVision-XAI.git
cd CardioVision-XAI
```

### 2. Set up the backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Place model weights

You need the trained checkpoint files. Place them like this:

```
ckpt_image_model/
    best.pth          <-- classifier weights

ckpt_report_model/
    best.pth          <-- report generator weights
```

Then update the paths in `backend/inference.py` — look for `CKPT_DIR_IMG` and `CKPT_DIR_REP` near the top of the file and set them to wherever you put the checkpoints.

### 4. Set up the frontend

```bash
cd ../frontend
npm install
```

### 5. (Optional) Place dataset CSVs for ground truth comparison

If you want the app to show the original radiologist's report alongside the AI-generated one, place the CSV files:

```
cardiomegaly_dataset/
    cardio_train.csv
    cardio_val.csv
    cardio_test.csv
```

---

## Usage Instructions

### Running the app

Open two terminals.

**Terminal 1 — Backend:**

```bash
cd backend
python app.py
```

This starts the FastAPI server at `http://localhost:8000`.

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

This starts the React dev server at `http://localhost:5173`. Open that URL in your browser.

### Using the interface

1. Drag and drop a frontal chest X-ray image (PNG or JPG) onto the upload zone, or click to browse.
2. Wait a few seconds for both models to process the image.
3. You'll see:
   - A diagnosis card showing cardiomegaly detected/not detected with confidence percentage.
   - A GradCAM heatmap overlaid on the original X-ray.
   - A generated radiology report split into Impression and Findings sections.
   - Any secondary findings (edema, pleural effusion) detected by the multi-label classifier.
4. You can toggle between the cleaned AI report, the raw model output, and the ground truth (if the image is from the test set) using the view switcher in the report panel.
5. Use the copy button to copy the report text to clipboard.

### Training the models yourself

Both training scripts are designed for Google Colab with a T4 GPU. Each file has sections marked with `# %%` — copy each section into a separate Colab cell and run them in order.

**Model 1** (`Model1_Image_Classifier.py`): Upload `cardio_image_384/` folder to Google Drive, set runtime to GPU T4, run all cells. Trains in roughly 3-4 hours.

**Model 2** (`Model2_Report_Generator.py`): Upload both the image folder and the CSV files to Google Drive, set runtime to GPU T4, run all cells. Trains in roughly 20+ hours (can be resumed from checkpoint if Colab disconnects).

### API endpoint

```bash
curl -X POST http://localhost:8000/predict -F "file=@chest_xray.png"
```

Returns JSON with prediction, confidence, base64-encoded GradCAM heatmap, generated report text, and co-pathology findings.

---

## Features

**Classification**
- Binary cardiomegaly detection with AUC-ROC of 0.92 on the test set.
- Multi-label detection of 8 thoracic conditions simultaneously.
- Confidence scoring with visual meter.

**Explainability**
- GradCAM heatmaps targeting the last convolutional stage of ConvNeXt, highlighting the cardiac region.
- Generated radiology reports that describe what the model "sees" in natural language.
- Negation-aware co-pathology extraction from the report text (correctly handles phrases like "no consolidation, effusion, or pneumothorax").

**Report Generation**
- 144 spatial visual tokens from ConvNeXt projected into BART's embedding space.
- Beam search decoding with trigram blocking for fluent, non-repetitive output.
- Post-processing pipeline with 25+ regex patterns to clean training artifacts.
- Three-way report viewer: AI Report / Raw Output / Ground Truth toggle.

**Web Interface**
- Drag-and-drop image upload.
- Real-time inference results displayed in a clean dashboard layout.
- Copy-to-clipboard for reports.
- Responsive design.

**Training**
- Mixed precision (FP16) with gradient scaling.
- Cosine warmup learning rate schedule.
- Early stopping with configurable patience.
- Checkpoint saving and resume support.
- Differential learning rates for the report model (projection layer trains 20x faster than BART).

---

## Project Structure

```
Component_1/
|
|-- README.md                          # this file
|-- Model1_Image_Classifier.py         # training script for ConvNeXt classifier
|-- Model2_Report_Generator.py         # training script for ConvNeXt + BART report generator
|
|-- backend/
|   |-- app.py                         # FastAPI server and routes
|   |-- inference.py                   # model loading, GradCAM, report generation, NLP
|   |-- requirements.txt              # Python dependencies
|
|-- frontend/
|   |-- src/
|   |   |-- App.jsx                    # main React component
|   |   |-- App.css                    # global styles
|   |   |-- components/
|   |       |-- Header.jsx             # app header
|   |       |-- UploadZone.jsx         # drag-and-drop upload
|   |       |-- ResultsPanel.jsx       # diagnosis card + co-pathology chips
|   |       |-- GradCamViewer.jsx      # heatmap overlay viewer
|   |       |-- ReportViewer.jsx       # report display with view toggle
|   |-- package.json
|   |-- vite.config.js
|
|-- cardio_image_384/                  # CXR images (384x384 PNG)
|   |-- train/positive/ and negative/  # 36,938 images (balanced)
|   |-- val/positive/ and negative/    # 4,550 images
|   |-- test/positive/ and negative/   # 4,786 images
|
|-- cardiomegaly_dataset/              # CSV files with labels and reports
|   |-- cardio_train.csv
|   |-- cardio_val.csv
|   |-- cardio_test.csv
|
|-- ckpt_image_model/                  # classifier checkpoint (best.pth)
|-- ckpt_report_model/                 # report generator checkpoint (best.pth)
```

---

## Model Performance

**Image Classifier (Model 1)**

- AUC-ROC: 0.9179
- Accuracy: 84%
- Positive recall (sensitivity): 89% — catches most cardiomegaly cases
- Negative precision: 88% — reliable when it says "no cardiomegaly"
- Architecture: ConvNeXt-Base with ImageNet-22K pretraining
- Training: 30 epochs, batch size 32, AdamW optimizer, cosine warmup LR

**Report Generator (Model 2)**

- Best validation loss: 1.34 (cross-entropy)
- ROUGE-1: 0.293 (word overlap with ground truth)
- ROUGE-2: 0.102 (phrase overlap)
- ROUGE-L: 0.175 (sentence structure similarity)
- Architecture: Frozen ConvNeXt encoder + BART-base decoder
- Training: 20 epochs, batch size 4 with 4-step gradient accumulation, differential LR

---

## References

1. Selvaraju, R.R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
2. Liu, Z. et al. (2022). "A ConvNet for the 2020s." CVPR 2022.
3. Lewis, M. et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." ACL 2020.
4. Johnson, A.E.W. et al. (2019). "MIMIC-CXR, a De-identified Publicly Available Database of Chest Radiographs with Free-text Reports." Scientific Data, 6(317).
5. Irvin, J. et al. (2019). "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI 2019.
6. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning."
7. van der Velden, B.H.M. et al. (2022). "Explainable Artificial Intelligence (XAI) in Deep Learning-based Medical Image Analysis." Medical Image Analysis, 79, 102470.

**Frameworks and tools:** [PyTorch](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/), [timm](https://github.com/huggingface/pytorch-image-models), [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/), [Vite](https://vitejs.dev/)

**Dataset:** [MIMIC-CXR (PhysioNet)](https://physionet.org/content/mimic-cxr/2.0.0/)

---

## License

This project was built for academic purposes as a university final-year project. It is released under the MIT License.

**Medical disclaimer:** This is a research prototype. It is not a clinical diagnostic tool and should not be used for making medical decisions. All outputs should be reviewed by qualified medical professionals.
