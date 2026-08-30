"""
ECG Arrhythmia Analysis — Web Interface
Flask backend serving the classification + report generation + XAI pipeline.

Pipeline:
  Tier 1 — ResNet CNN classifier       → structured probabilities
  Tier 2 — Template engine             → clinically-grounded sentences (zero hallucination risk)
  Tier 3 — BioBART smoother (optional) → natural language polish (constrained paraphrase)
  Legacy — BioBART free generation     → kept for research/ablation comparison only

Usage: python app.py
Then open http://localhost:5000 in your browser.
"""

import os, sys, json, io, base64, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from report_templates import build_structured_report, format_smoother_prompt

# ── Config ────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
CLASSIFIER_CKPT = os.path.join(WORK_DIR, "checkpoints_ecg_only", "best_model.pt")
REPORT_CKPT = os.path.join(WORK_DIR, "checkpoints_report_gen", "best_model.pt")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_FULL = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
CNN_CHANNELS = [64, 128, 192, 256]
CNN_KERNELS = [15, 7, 5, 3]

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
#  These classes define the neural network architectures used for:
#    - ECG signal classification (ECGResNet)
#    - Feature extraction backbone (ECGBackbone)
#    - Report generation via BioBART (ECGReportModel)
# ══════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """
    A single residual block for 1D convolution on ECG signals.
    Uses a skip (shortcut) connection to prevent vanishing gradients.
    If input/output dimensions differ, a 1x1 conv adjusts the skip path.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN + Skip -> ReLU
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2  # same-padding to preserve temporal resolution
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        # Skip connection: identity if dims match, else 1x1 conv to match
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch))

    def forward(self, x):
        residual = self.skip(x)              # shortcut path
        out = F.relu(self.bn1(self.conv1(x)))  # first conv + activation
        out = self.dropout(out)               # regularisation
        out = self.bn2(self.conv2(out))        # second conv
        out += residual                       # add skip connection
        return F.relu(out)                    # final activation


class ECGResNet(nn.Module):
    """
    PRIMARY CLASSIFIER — 1D ResNet for 12-lead ECG classification.
    This is the model used in the web interface for real-time inference.
    Input:  (batch, 12 leads, 5000 time-steps)
    Output: (batch, 5 logits) — one per superclass [NORM, MI, STTC, CD, HYP]
    Architecture: 4 ResidualBlocks -> Global Avg Pool -> FC head -> 5 outputs
    """
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = 12  # 12 ECG leads as input channels
        # Build 4 residual blocks with increasing channels and decreasing kernels
        for i, (out_ch, ks) in enumerate(zip(CNN_CHANNELS, CNN_KERNELS)):
            drop = 0.1 if i < 2 else 0.2  # more dropout in deeper layers
            blocks.append(ResidualBlock(in_ch, out_ch, ks, stride=2, dropout=drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)  # stacked ResidualBlocks
        self.pool = nn.AdaptiveAvgPool1d(1)     # collapse time dim -> single value per channel
        # Classifier head: 256 -> 128 -> 5 (with BN, ReLU, Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, x):
        feat = self.backbone(x)            # extract features through residual blocks
        feat = self.pool(feat).squeeze(-1)  # global average pooling -> (batch, 256)
        return self.classifier(feat)        # output 5 class logits (sigmoid applied later)


class ECGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = 12
        for i, (out_ch, ks) in enumerate(zip(CNN_CHANNELS, CNN_KERNELS)):
            drop = 0.1 if i < 2 else 0.2
            blocks.append(ResidualBlock(in_ch, out_ch, ks, stride=2, dropout=drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)


class ECGReportModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        from transformers import BartForConditionalGeneration
        from transformers.modeling_outputs import BaseModelOutput
        self.BaseModelOutput = BaseModelOutput
        self.ecg_backbone = ECGBackbone()
        self.projection = nn.Sequential(
            nn.Linear(256, 768), nn.LayerNorm(768), nn.GELU())
        self.bart = BartForConditionalGeneration.from_pretrained(
            "GanjinZero/biobart-base", torch_dtype=torch.float32)
        self.tokenizer = tokenizer

    def generate_report(self, signal, max_length=64, num_beams=4):
        """
        LEGACY (Research Only) — Free generation from raw ECG signal.
        WARNING: This path CAN hallucinate clinical findings because BioBART
        generates freely from CNN features without any template constraint.
        Kept for ablation comparison only — NOT used for clinical decisions.
        """
        with torch.no_grad():
            feat = self.ecg_backbone(signal)  # extract CNN features from raw signal
        feat = feat.permute(0, 2, 1)          # reshape for projection
        hidden = self.projection(feat)        # project to BioBART's 768-dim space
        enc_out = self.BaseModelOutput(last_hidden_state=hidden)  # wrap as encoder output
        attn = torch.ones(signal.size(0), hidden.size(1), dtype=torch.long)
        # BioBART decoder generates text freely from projected ECG features
        gen = self.bart.generate(encoder_outputs=enc_out, attention_mask=attn,
                                  max_length=max_length, num_beams=num_beams,
                                  early_stopping=True, no_repeat_ngram_size=3)
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Fix known BioBART text artifacts — applied as final pass."""
        import re as _re
        text = text.replace("ST-seal", "ST-segment")
        text = text.replace("STsegment", "ST-segment")
        text = text.replace("ST- segment", "ST-segment")
        text = text.replace(".Clinical", ". Clinical")
        text = text.replace(".Cl ", ". Clinical ")
        text = text.replace("Cl correlation", "Clinical correlation")
        text = text.replace("record keeping", "correlation")
        text = text.replace("  ", " ")  # collapse double spaces
        # Fix any period directly touching a capital letter (no space)
        text = _re.sub(r'\.(?=[A-Z])', '. ', text)
        # Fix any remaining double spaces introduced by the regex
        text = _re.sub(r' {2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _fix_smoothed_output(smoothed: str, template: str) -> str:
        """
        Post-processing fix for BioBART decoder artifacts (V2).

        Fixes truncation (first 1-3 chars clipped) and garbled starts.
        Text typo cleanup is handled separately by _clean_text().
        """
        smoothed = smoothed.strip()
        if not smoothed:
            return template

        # ── Case 1: Clean start matching template ─────────────────
        if smoothed[0].isupper() and template.startswith(smoothed[:10]):
            return smoothed

        # ── Case 2: Truncated suffix — restore clipped prefix ─────
        template_lower = template.lower()
        smoothed_lower = smoothed.lower()
        for offset in range(1, min(30, len(template))):
            fragment = template_lower[offset:offset + 15]
            if fragment and smoothed_lower.startswith(fragment):
                return template[:offset] + smoothed.lstrip(" -")

        # ── Case 3: Garbled start — fall back to template ─────────
        return template

    def smooth_report(self, structured_text: str, max_length=128, num_beams=4) -> str:
        """
        Tier 3: BioBART as a constrained paraphraser (smoother).

        Input  : structured_text — the template-generated report (Tier 2 output)
        Output : natural clinical prose, constrained to the same findings

        Unlike generate_report(), the model cannot hallucinate new findings because
        it never sees the raw ECG — it only sees the already-verified structured text.
        """
        inputs = self.tokenizer(
            structured_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        )
        with torch.no_grad():
            gen = self.bart.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
            )
        raw = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        fixed = self._fix_smoothed_output(raw, structured_text)
        return self._clean_text(fixed)


# ── Grad-CAM (XAI Method 1) ──────────────────────────────────────
# Answers: WHERE IN TIME did the model look?
# Highlights which temporal regions of the ECG were most important
# for the classification decision (shown as red heatmap on the plot).
class GradCAM1D:
    """
    1D Gradient-weighted Class Activation Mapping.
    Hooks into the last ResidualBlock to capture feature maps + gradients.
    Produces a heatmap showing which time regions influenced the prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        # Register hooks on the target layer to capture activations & gradients
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o): self.activations = o.detach()     # save forward activations
    def _save_grad(self, m, gi, go): self.gradients = go[0].detach()  # save backward gradients

    def generate(self, signal, class_idx):
        """Generate a 1D Grad-CAM heatmap for a specific target class."""
        self.model.eval()
        signal.requires_grad_(True)
        logits = self.model(signal)           # forward pass
        self.model.zero_grad()
        logits[0, class_idx].backward()       # backprop for target class only
        # Weight each feature map channel by its average gradient
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)  # weighted sum of feature maps
        cam = F.relu(cam)                     # keep only positive contributions
        cam = cam / (cam.max() + 1e-8)        # normalise to [0, 1]
        return cam.squeeze().detach().numpy()  # return as numpy array


# ── Integrated Gradients (XAI Method 2) ──────────────────────────
# Answers: WHICH LEADS were most important for the prediction?
# Ranks all 12 ECG leads by their contribution percentage.
def compute_lead_saliency(model, signal, class_idx, steps=30):
    """
    Compute per-lead importance using Integrated Gradients.
    Interpolates from a zero baseline to the real signal in 30 steps,
    accumulates gradients, and attributes importance to each of the 12 leads.
    Returns: array of 12 values summing to 100% (one per lead).
    """
    model.eval()
    baseline = torch.zeros_like(signal)  # flat-line baseline (no cardiac activity)
    grads = []
    # Integrate gradients along the interpolation path
    for i in range(1, steps + 1):
        alpha = float(i) / steps
        scaled = baseline + alpha * (signal - baseline)  # interpolated input
        scaled.requires_grad_(True)
        logits = model(scaled)
        model.zero_grad()
        logits[0, class_idx].backward()  # backprop for target class
        grads.append(scaled.grad.detach())
    avg = torch.stack(grads).mean(dim=0)      # average gradients across all steps
    ig = (signal - baseline) * avg            # final attribution = (input - baseline) * avg_grad
    imp = ig.squeeze().abs().sum(dim=1).numpy()  # sum over time per lead
    imp = imp / (imp.sum() + 1e-8) * 100      # normalise to percentages
    return imp  # 12 values: [Lead_I%, Lead_II%, ..., V6%]


# ── ECG Plot ──────────────────────────────────────────────────────
def make_ecg_plot(signal_raw, cam):
    """Generate a clean 12-lead ECG plot with Grad-CAM overlay (imshow, fast)."""
    cam_up = np.interp(np.linspace(0, 1, signal_raw.shape[1]),
                       np.linspace(0, 1, len(cam)), cam)
    time = np.arange(signal_raw.shape[1]) / 500.0

    fig, axes = plt.subplots(2, 6, figsize=(20, 7), facecolor="white")
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#fafafa")
        lead = signal_raw[i]
        # Grad-CAM overlay via imshow (single call, instant)
        extent = [time[0], time[-1], lead.min(), lead.max()]
        ax.imshow(cam_up[np.newaxis, :], aspect="auto", extent=extent,
                  cmap="Reds", alpha=0.35, interpolation="bilinear",
                  vmin=0, vmax=1)
        ax.plot(time, lead, color="#1e3a5f", linewidth=0.6)
        ax.set_title(LEAD_NAMES[i], fontsize=9, fontweight="600",
                     color="#1e293b", pad=3)
        ax.tick_params(labelsize=6, colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")
            spine.set_linewidth(0.5)
        ax.set_xlim(0, time[-1])
        if i >= 6:
            ax.set_xlabel("s", fontsize=7, color="#64748b")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════
#  LOAD MODELS (once at startup)
# ══════════════════════════════════════════════════════════════════
print("Loading models...")

with open(os.path.join(DATA_DIR, "norm_stats.json")) as f:
    norm_stats = json.load(f)
sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
sig_std = np.array(norm_stats["signal_std"], dtype=np.float32)

# Classifier
classifier = ECGResNet()
cls_state = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
classifier.load_state_dict(cls_state["model_state"])
classifier.eval()
thresholds = cls_state["optimal_thresholds"]
gradcam = GradCAM1D(classifier, classifier.backbone[-1])
print(f"  Classifier loaded (AUROC {cls_state['best_auroc']:.4f})")

# Report generator
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
report_model = ECGReportModel(tokenizer)
rpt_state = torch.load(REPORT_CKPT, map_location="cpu", weights_only=False)
report_model.load_state_dict(rpt_state["model_state"])
report_model.eval()
print(f"  Report generator loaded (ROUGE-L {rpt_state['best_rougeL']:.4f})")

# Test set + class index
import pandas as pd
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Build class index: {"NORM": [{"ecg_id": 9, "report": "sinus rhythm...", "age": 55, "sex": "M"}, ...]}
CLASS_INDEX = {}
for cls in CLASS_NAMES:
    col = f"label_{cls}"
    sub = test_df[test_df[col] == 1][["ecg_id", "report_en", "age", "sex"]].copy()
    sub["sex"] = sub["sex"].map({0: "M", 1: "F"})
    CLASS_INDEX[cls] = sub.replace({np.nan: None}).to_dict(orient="records")

print(f"  Ready! Test set: {len(test_df)} patients")
for cls in CLASS_NAMES:
    print(f"    {cls}: {len(CLASS_INDEX[cls])} records")


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


def run_analysis(signal_raw):
    """
    CORE FUNCTION — Run the full three-tier hybrid pipeline on a (5000, 12) ECG signal.

    This is the main analysis function called for every ECG inference request.
    It orchestrates: classification -> XAI -> report generation -> visualisation.

    Returns three report variants:
      templateReport — Tier 2: deterministic, classifier-grounded (ZERO hallucination risk)
      smoothedReport — Tier 3: BioBART paraphrase of template (constrained, low risk)
      report         — Legacy: BioBART free generation from raw signal (research/ablation only)
    """
    signal_norm = (signal_raw - sig_mean) / sig_std
    signal_t = signal_norm.T  # (12, 5000)
    tensor = torch.from_numpy(signal_t).unsqueeze(0).float()

    # ── Tier 1: Classification ────────────────────────────────────────────────
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()

    # Find target class for XAI
    detected_idx = [(i, probs[i]) for i in range(5) if probs[i] >= thresholds[i]]
    target = max(detected_idx, key=lambda x: x[1])[0] if detected_idx else np.argmax(probs)

    # ── XAI ──────────────────────────────────────────────────────────────────
    cam = gradcam.generate(tensor.clone(), target)
    lead_sal = compute_lead_saliency(classifier, tensor.clone(), target)

    # ── Tier 2: Template report (hallucination-free) ─────────────────────────
    structured = build_structured_report(
        probs=list(probs),
        thresholds=list(thresholds),
        class_names=CLASS_NAMES,
    )
    template_report = structured["report_text"]

    # ── Tier 3: BioBART smoother (constrained paraphrase of template) ─────────
    try:
        smoothed_report = report_model.smooth_report(template_report)
    except Exception:
        smoothed_report = template_report   # fallback: just use template

    # ── Legacy: BioBART free generation (kept for ablation) ──────────────────
    try:
        free_reports = report_model.generate_report(tensor)
        free_report  = free_reports[0]
    except Exception:
        free_report = "(free generation unavailable)"

    # ── ECG Plot ─────────────────────────────────────────────────────────────
    ecg_img = make_ecg_plot(signal_raw.T, cam)

    # ── Build JSON response ──────────────────────────────────────────────────
    classes = []
    for i, cls in enumerate(CLASS_NAMES):
        conf = structured["confidence_map"][cls]
        classes.append({
            "name":        cls,
            "fullName":    CLASS_FULL[cls],
            "probability": round(float(probs[i]) * 100, 1),
            "threshold":   round(float(thresholds[i]) * 100, 1),
            "detected":    bool(probs[i] >= thresholds[i]),
            "tier":        conf["tier"],   # "high" | "medium" | None
        })

    leads = []
    sorted_idx = np.argsort(lead_sal)[::-1]
    for idx in sorted_idx:
        leads.append({
            "name":       LEAD_NAMES[idx],
            "importance": round(float(lead_sal[idx]), 1),
        })

    return {
        "classes":          classes,
        "templateReport":   template_report,    # Tier 2 — safe primary report
        "smoothedReport":   smoothed_report,    # Tier 3 — natural language polish
        "report":           free_report,        # Legacy — for ablation/comparison
        "detectedLabels":   structured["detected_labels"],
        "hasAbnormality":   structured["has_abnormality"],
        "leads":            leads,
        "ecgImage":         ecg_img,
        "gradcamTarget":    CLASS_NAMES[target],
    }


@app.route("/demo", methods=["POST"])
def demo():
    """Pick a random test patient and analyze."""
    row = test_df.sample(1).iloc[0]
    return _analyze_row(row)


@app.route("/patients/<class_name>", methods=["GET"])
def get_patients(class_name):
    """Return list of test patients for a given class."""
    if class_name not in CLASS_INDEX:
        return jsonify({"error": f"Unknown class: {class_name}"}), 400
    return jsonify({"class": class_name, "patients": CLASS_INDEX[class_name]})


@app.route("/analyze/<int:ecg_id>", methods=["POST"])
def analyze_by_id(ecg_id):
    """Analyze a specific ECG by ID (from the test set browser)."""
    row = test_df[test_df["ecg_id"] == ecg_id]
    if row.empty:
        return jsonify({"error": f"ECG ID {ecg_id} not found in test set"}), 404
    return _analyze_row(row.iloc[0])


def _analyze_row(row):
    """Shared helper: load signal for a DataFrame row and run full analysis."""
    ecg_id = int(row["ecg_id"])
    sig_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
    signal_raw = np.load(sig_path).astype(np.float32)

    result = run_analysis(signal_raw)
    result["patientId"] = ecg_id
    result["referenceReport"] = str(row.get("report_en", ""))

    # Ground truth labels
    label_cols = [f"label_{c}" for c in CLASS_NAMES]
    truth = []
    for i, cls in enumerate(CLASS_NAMES):
        truth.append({"name": cls, "actual": bool(row[label_cols[i]] == 1)})
    result["groundTruth"] = truth
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle uploaded .dat + .hea files."""
    import wfdb

    if "dat_file" not in request.files or "hea_file" not in request.files:
        return jsonify({"error": "Please upload both .dat and .hea files"}), 400

    dat_file = request.files["dat_file"]
    hea_file = request.files["hea_file"]

    with tempfile.TemporaryDirectory() as tmpdir:
        hea_name = hea_file.filename
        record_name = os.path.splitext(hea_name)[0]
        dat_file.save(os.path.join(tmpdir, dat_file.filename))
        hea_file.save(os.path.join(tmpdir, hea_file.filename))

        try:
            record = wfdb.rdrecord(os.path.join(tmpdir, record_name))
            signal_raw = record.p_signal.astype(np.float32)
            if signal_raw.shape[1] != 12:
                return jsonify({"error": f"Expected 12 leads, got {signal_raw.shape[1]}"}), 400
            if signal_raw.shape[0] != 5000:
                from scipy.signal import resample
                signal_raw = resample(signal_raw, 5000, axis=0).astype(np.float32)
        except Exception as e:
            return jsonify({"error": f"Failed to read ECG file: {str(e)}"}), 400

    result = run_analysis(signal_raw)
    result["patientId"] = record_name
    return jsonify(result)


if __name__ == "__main__":
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
