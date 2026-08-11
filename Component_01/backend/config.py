"""
Configuration — Component_01 v2 backend.

This is a SEPARATE deployment from ../../backend. Nothing in the original
system is read or written. It serves the retrained models:

    classifier       Stage 5   mean AUROC 0.8554  (was 0.8251)
    report generator Stage 11  CheXbert F1 0.5939 (was ROUGE-L 0.2740,
                               which sat BELOW the 0.2769 constant baseline)

and adds per-projection operating points, which the original does not have.
"""
from pathlib import Path

# Component_01/Component_01/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent                       # Component_01/

# ---------------------------------------------------------------------------
# Weights. Stage 11 is preferred; Stage 4 is the fallback if it has not been
# downloaded from Drive yet. The service prints which one it actually loaded --
# silently serving the older generator would misrepresent the system.
# ---------------------------------------------------------------------------
CLASSIFIER_WEIGHTS = PROJECT_ROOT / "checkpoints" / "stage5" / "best.pt"
REPORTGEN_STAGE11 = PROJECT_ROOT / "checkpoints" / "stage11" / "best.pt"
REPORTGEN_STAGE4 = PROJECT_ROOT / "checkpoints" / "stage4" / "best.pt"

THRESHOLDS_JSON = Path(__file__).resolve().parent / "thresholds.json"

# Stage 13 selective-deferral policy. Optional: if it is absent the service runs
# with deferral disabled rather than failing. Produced by stage13_deferral.py.
DEFERRAL_POLICY_JSON = PROJECT_ROOT / "reports" / "stage13" / "deferral_policy.json"
TEST_MANIFEST = PROJECT_ROOT / "training_manifest" / "manifest_test.csv"
TEST_IMAGE_DIR = REPO_ROOT / "data" / "output" / "cardio_image_384"

# Ground truth comes from the ORIGINAL, UNTOUCHED dataset -- the radiologist's
# text exactly as dictated, prior-study references and all. Deliberately NOT
# training_manifest/manifest_test.csv, whose `report` column is the Stage-1
# CLEANED target. Showing the cleaned version as "ground truth" would compare
# the model against our own preprocessing rather than against the radiologist.
# Opened READ-ONLY; nothing is ever written back to it.
#
# This is a verified byte-identical copy of
#   <repo>/data/output/cardiomegaly_dataset/cardio_test.csv
# kept INSIDE the project so the folder is self-contained and can be handed to
# someone else without the parent directory. The original is untouched.
#
# Falls back to the out-of-tree original if the local copy is absent, so an
# existing checkout that never made the copy keeps working.
ORIGINAL_TEST_CSV = PROJECT_ROOT / "review_cases" / "cardio_test.csv"
if not ORIGINAL_TEST_CSV.exists():
    ORIGINAL_TEST_CSV = REPO_ROOT / "data" / "output" / "cardiomegaly_dataset" / "cardio_test.csv"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
IMG_SIZE = 384
DECODER_NAME = "GanjinZero/biobart-v2-base"
NUM_VISUAL_TOKENS = 144                                # 12x12 grid

LABEL_COLS = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
              "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]
NUM_LABELS = len(LABEL_COLS)

# Shown prominently in the dashboard; the rest are still computed and returned.
PRIMARY_PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion"]
PRIMARY_INDICES = [LABEL_COLS.index(p) for p in PRIMARY_PATHOLOGIES]

# NOTE ON PREPROCESSING
# The original backend normalised with ImageNet mean/std. That is wrong for
# grayscale radiographs -- measured 4.4x worse variance across images than raw
# pixels. This deployment imports cxr_transforms and uses per-image z-score,
# the same transform the models were trained with. Any mismatch here silently
# degrades every prediction, so the transform is never redefined locally.

# ---------------------------------------------------------------------------
# Generation (Stage 4B ablation: greedy beat beam-4 on 5 of 7 metrics)
# ---------------------------------------------------------------------------
GEN_NUM_BEAMS = 1
GEN_MAX_TOKENS = 192
GEN_MIN_TOKENS = 24
GEN_NO_REPEAT_NGRAM = 3

# ---------------------------------------------------------------------------
# Measured performance, surfaced in /api/health and the UI footer
# ---------------------------------------------------------------------------
MODEL_STATS = {
    "classifier": {
        "mean_auroc": 0.8554,
        "cardiomegaly_auroc": 0.9189,
        "cardiomegaly_sensitivity": 0.923,
        "cardiomegaly_ci": [0.9112, 0.9265],
    },
    "report_generator": {
        "chexbert_micro_f1_14": 0.5939,
        "cardiomegaly_report_f1": 0.8287,
        "rouge_l": 0.2896,
        "constant_string_baseline": 0.2641,
        "prior_hallucination_rate": 0.0,
    },
    "test_set_n": 4722,
}

# AP/PA disparity, measured. Drives the reliability flag.
PROJECTION_AUROC = {"AP": 0.8224, "PA": 0.8864}
PROJECTION_GAP = 0.0639

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173",
                "http://localhost:5174", "http://127.0.0.1:5174"]
