# 🫀 Explainable AI System for Cardiovascular Disease Diagnosis

## Component 03 — EchoStrat: Cardiac Function Assessment and Ejection Fraction Prediction via Spatiotemporal Explainable AI

---

## 📋 Project Description

EchoStrat is a deep learning-based clinical decision support system that automatically assesses cardiac function from echocardiogram (heart ultrasound) videos. The system takes an Apical Four-Chamber (A4C) echo video as input and produces:

- **Ejection Fraction (EF)** — the percentage of blood the left ventricle pumps out per heartbeat
- **4-Class Severity Classification** — based on ACC/AHA 2022 clinical guidelines
- **Explainable AI Visualizations** — GradCAM heatmaps, wall motion maps, and LV segmentation

The system addresses a critical clinical challenge: manual EF estimation from echocardiograms suffers from ±10% inter-observer variability between cardiologists. Around the 35% EF threshold — where a patient may require an Implantable Cardioverter Defibrillator (ICD) — this disagreement becomes medically dangerous. EchoStrat provides a consistent, explainable, AI-driven second opinion.

---

## 🎯 Objectives

1. **Automated Severity Classification** — Classify echo videos into 4 ACC/AHA heart failure severity grades:
   - Severely Reduced (EF < 30%)
   - Reduced (EF 30–40%)
   - Mildly Reduced (EF 40–50%)
   - Normal (EF ≥ 50%)

2. **Continuous EF Prediction** — Predict the exact ejection fraction percentage for clinical tracking.

3. **Clinical Explainability** — Provide visual explanations (GradCAM, Wall Motion, LV Segmentation) so cardiologists can verify AI decisions.

4. **Urgent Referral Flagging** — Automatically flag patients with EF < 35% for immediate cardiology review.

5. **Ordinal-Aware Learning** — Ensure the AI understands that misclassifying a severe case as normal is far more dangerous than misclassifying it as reduced.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EchoStrat Pipeline                               │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Dataset     │    │ Preprocessing│    │     Training (V8)       │   │
│  │ EchoNet-      │───▶│ Cardiac-cycle│───▶│ R(2+1)D-18 Backbone    │   │
│  │ Dynamic       │    │ alignment    │    │ + 3-Head Ensemble       │   │
│  │ (10,030 A4C)  │    │ + Corner mask│    │ + CORN/Focal/Center    │   │
│  └──────────────┘    └──────────────┘    └──────────┬───────────────┘   │
│                                                      │                   │
│                                          ┌───────────▼───────────┐      │
│                                          │   FastAPI Backend     │      │
│                                          │   /predict endpoint   │      │
│                                          │   + GradCAM XAI       │      │
│                                          │   + Wall Motion XAI   │      │
│                                          │   + LV Segmentation   │      │
│                                          └───────────┬───────────┘      │
│                                                      │                   │
│                                          ┌───────────▼───────────┐      │
│                                          │   React + Vite        │      │
│                                          │   Clinical Dashboard  │      │
│                                          │   + EF Display        │      │
│                                          │   + Severity Cards    │      │
│                                          │   + XAI Visualizations│      │
│                                          │   + Urgent Referral   │      │
│                                          └───────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```
Input: Echo Video [Batch, 3, 32, 112, 112]
                    ↓
         R(2+1)D-18 Backbone (EchoNet-Dynamic Pretrained)
              Frozen: stem + layer1
              Trainable: layer2 + layer3 + layer4
                    ↓
              512-dim features
                    ↓
            ProjectionBlock (512 → 512)
                    ↓
           FeatureTransform (512 → 512)
                    ↓
         ┌──────────┼──────────┐
         ↓          ↓          ↓
    cls_head     reg_head   corn_head
   (512→256→4)  (512→1)   (512→256→3)
    Severity     EF %      Ordinal
    Classes     Value     Boundaries
         ↓          ↓          ↓
         └──────────┼──────────┘
                    ↓
        3-Head Ensemble (35%/25%/40%)
                    ↓
         Final Prediction + XAI
```

### Three Task Heads

| Head | Architecture | Output | Purpose |
|------|-------------|--------|---------|
| **Classification** | DeeperHead (512→256→4) | 4-class probabilities | Severity classification |
| **Regression** | Linear (512→1) | Continuous EF% | Ejection fraction prediction |
| **CORN** | DeeperHead (512→256→3) | 3 boundary decisions | Ordinal rank consistency |

### Pretrained Backbone (Double Pretraining)

```
Kinetics-400 (Generic Video) → EchoNet-Dynamic (Cardiac Video) → EchoStrat V8 (Severity)
```

---

## 🔬 Technical Novelty

1. **First research to combine CORN (Conditional Ordinal Regression) with cardiac-specific R(2+1)D-18 spatiotemporal features** for EF-based severity stratification.

2. **Novel ordinal-aware mixup augmentation** that constrains cross-class blending based on clinical severity distance — only adjacent severity classes are mixed.

3. **Boundary-weighted CORN loss** that applies stronger learning signal to samples near clinical EF thresholds (30%, 40%, 50%).

4. **Three-head ensemble inference** blending classification, regression, and ordinal probabilities for robust clinical decision support.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Balanced Accuracy | **69.62%** |
| AUROC (Macro) | **84.72%** |
| Severely Reduced (C0) Accuracy | **83.95%** |
| Reduced (C1) Accuracy | **48.15%** |
| Mildly Reduced (C2) Accuracy | **55.63%** |
| Normal (C3) Accuracy | **90.75%** |
| EF MAE | ~7% |

---

## 💻 Technologies Used

### Training Pipeline
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Core language |
| PyTorch | 2.0+ | Deep learning framework |
| torchvision | 0.15+ | R(2+1)D-18 backbone |
| NumPy | 1.24+ | Array operations |
| pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | Metrics (balanced accuracy, AUROC, F1) |
| OpenCV | 4.8+ | Video processing |
| tqdm | 4.65+ | Progress bars |

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.100+ | REST API server |
| Uvicorn | 0.23+ | ASGI server |
| Pillow | 10.0+ | Image encoding |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19.x | UI framework |
| Vite | 8.x | Build tool |
| Tailwind CSS | 4.x | Styling |
| Framer Motion | 12.x | Animations |
| Lucide React | 1.x | Icons |

### Dataset
| Dataset | Source | Size |
|---------|--------|------|
| EchoNet-Dynamic | Stanford University | 10,030 A4C echo videos |

---

## 📁 Project Structure

```
Component_03/
└── Dilukshan/
    ├── Dataset/                          # EchoNet-Dynamic dataset
    │   ├── FileList.csv                  # Video metadata (EF, split, ESV, EDV)
    │   ├── VolumeTracings.csv            # LV tracings (ED/ES frame indices)
    │   └── Videos/                       # Raw .avi echo videos
    │
    ├── Training/                         # Model training pipeline
    │   ├── config/
    │   │   └── settings.py               # Clinical thresholds, paths, constants
    │   ├── training/
    │   │   ├── config.py                 # Hyperparameters (LR, loss weights, SWA)
    │   │   ├── losses.py                 # OrdinalFocalLoss, CORNLoss, CenterLoss
    │   │   ├── augmentation.py           # Ordinal-aware Mixup + CutMix
    │   │   ├── tta.py                    # 8-way TTA + 3-head ensemble evaluation
    │   │   ├── checkpoint.py             # Model save/load utilities
    │   │   └── model.py                  # Compatibility layer → imports main model
    │   ├── utils/
    │   │   ├── dataset.py                # EchoDataset + ENS class weighting
    │   │   └── logger.py                 # Training logger
    │   ├── model.py                      # V8 EchoNet architecture (3 heads)
    │   ├── preprocess_echo.py            # Video → tensor preprocessing pipeline
    │   ├── train_classifier.py           # Main training script
    │   ├── outputs/
    │   │   ├── checkpoints/best_model.pt # Trained model weights
    │   │   ├── training_results_v8.json  # Training metrics
    │   │   └── step3_tensors/            # Preprocessed video tensors
    │   └── weights/                      # Pretrained backbone weights
    │
    ├── Backend/                          # FastAPI inference server
    │   ├── main.py                       # API endpoints (/predict)
    │   ├── xai_gradcam.py                # GradCAM heatmap generation
    │   ├── xai_wall_motion.py            # Wall motion map + LV segmentation
    │   ├── xai_preprocessing.py          # Video preprocessing for inference
    │   ├── xai_encoding.py               # Frame → base64 encoding
    │   └── backend_utils.py              # Shared utilities
    │
    ├── Frontend/                         # React clinical dashboard
    │   ├── src/
    │   │   ├── App.jsx                   # Main application component
    │   │   └── index.css                 # Tailwind CSS styles
    │   ├── index.html                    # Entry point
    │   └── package.json                  # Dependencies
    │
    ├── demo_videos/                      # Sample videos for demonstration
    ├── export_demo_videos.py             # Script to generate demo videos
    └── verify_demos.py                   # Demo video verification
```

---

## ⚙️ Installation Steps

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- NVIDIA GPU with CUDA support (recommended, 8GB+ VRAM)
- EchoNet-Dynamic dataset (download from [Stanford](https://echonet.github.io/dynamic/))

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Component_03/Dilukshan
```

### 2. Set Up Training Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn opencv-python tqdm pillow fastapi uvicorn python-multipart openpyxl
```

### 3. Set Up Frontend
```bash
cd Frontend
npm install
cd ..
```

### 4. Download Dataset
- Download EchoNet-Dynamic from [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)
- Place `FileList.csv`, `VolumeTracings.csv`, and `Videos/` folder inside `Dataset/`

### 5. Preprocess Videos
```bash
cd Training
python preprocess_echo.py
```
This converts 10,030 raw .avi videos into standardized 32-frame tensor clips.

### 6. Train the Model
```bash
python train_classifier.py
```
Training takes approximately 6–8 hours on a single NVIDIA GPU (RTX 3060 or better).

---

## 🚀 Usage Instructions

### Start the Backend Server
```bash
cd Backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Start the Frontend Dashboard
```bash
cd Frontend
npm run dev
```

### Access the Application
Open your browser and navigate to: `http://localhost:5173`

### How to Use
1. Click **"Upload Echo Video"** on the dashboard
2. Select an `.avi`, `.mp4`, or `.npy` echo video file
3. Wait for AI analysis (2–5 seconds)
4. View results:
   - **EF Value** — predicted ejection fraction percentage
   - **Severity Class** — Severely Reduced / Reduced / Mildly Reduced / Normal
   - **Probability Distribution** — confidence across all 4 classes
   - **GradCAM Heatmap** — where the AI focused attention
   - **Wall Motion Map** — which cardiac walls are contracting
   - **LV Segmentation** — identified left ventricle region
   - **Urgent Referral Alert** — triggered when EF < 35%

---

## ✨ Features

### Core Clinical Features
- ✅ 4-class heart failure severity classification (ACC/AHA 2022)
- ✅ Continuous ejection fraction (EF%) prediction
- ✅ Automatic urgent referral flagging (EF < 35%)
- ✅ Class probability distribution display

### Explainable AI (XAI)
- ✅ GradCAM heatmaps — spatial attention visualization
- ✅ Wall motion intensity maps — cardiac contraction visualization
- ✅ LV segmentation mask — left ventricle localization

### Training Pipeline
- ✅ R(2+1)D-18 backbone with EchoNet-Dynamic pretraining
- ✅ Three-head ensemble (Classification + Regression + CORN)
- ✅ Ordinal-aware focal loss with distance weighting
- ✅ CORN loss for rank-consistent ordinal regression
- ✅ Center loss for intra-class feature compactness
- ✅ Cardiac-cycle-aligned video preprocessing
- ✅ 8-way test-time augmentation (TTA)
- ✅ Stochastic Weight Averaging (SWA)
- ✅ Ordinal-aware mixup and CutMix augmentation
- ✅ ENS (Effective Number of Samples) class balancing

---

## 🔮 Future Work

- Add **temporal attention maps** to visualize which cardiac phases influenced each AI prediction
- Generate **automated clinical reports** summarizing AI findings for cardiologist review

---

## 👨‍💻 Author

| Name | Student ID | Component |
|------|-----------|-----------|
| **Dilukshan** | IT22219534 | Component 03 — EchoStrat |

---

## 📸 Screenshots

### Clinical Dashboard
*Upload an echo video and receive AI-powered cardiac assessment with explainable visualizations.*

### GradCAM Heatmap
*Red/yellow regions show where the AI focused — should highlight the left ventricle walls.*

### Wall Motion Map
*Color overlay showing cardiac wall contraction intensity — green/yellow = active motion, blue = minimal motion.*

### LV Segmentation
*Identified left ventricle region highlighted for clinical verification.*

---

## 📚 References

| # | Paper | Used For |
|---|-------|----------|
| 1 | Ouyang D, et al. "Video-based AI for beat-to-beat assessment of cardiac function." *Nature*, 2020. | EchoNet-Dynamic dataset + pretrained weights |
| 2 | Tran D, et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition." *CVPR*, 2018. | R(2+1)D-18 backbone architecture |
| 3 | Shi X, et al. "CORN — Conditional Ordinal Regression for Neural Networks." *Pattern Recognition*, 2021. | CORN loss for ordinal boundaries |
| 4 | Lin TY, et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017. | Focal loss for class imbalance |
| 5 | Wen Y, et al. "A Discriminative Feature Learning Approach for Deep Face Recognition." *ECCV*, 2016. | Center loss for feature compactness |
| 6 | Zhang H, et al. "mixup: Beyond Empirical Risk Minimization." *ICLR*, 2018. | Mixup augmentation |
| 7 | Yun S, et al. "CutMix: Regularization Strategy to Train Strong Classifiers." *ICCV*, 2019. | CutMix augmentation |
| 8 | Cui Y, et al. "Class-Balanced Loss Based on Effective Number of Samples." *CVPR*, 2019. | ENS class weighting |
| 9 | Selvaraju RR, et al. "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*, 2017. | GradCAM explainability |

---

## 📄 License

This project is developed for academic and research purposes as part of the FedMed Federated Healthcare Intelligence System. The EchoNet-Dynamic dataset is used under the terms specified by Stanford University.

---

<p align="center">
  <b>EchoStrat — Making cardiac AI transparent, reliable, and clinically actionable.</b>
</p>
