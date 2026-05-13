"""
ECG Analysis Pipeline — Complete Inference + XAI Visualization

Combines both trained models:
  1. ECG Classifier    → Disease probabilities (NORM, MI, STTC, CD, HYP)
  2. Report Generator  → English clinical text
  3. Grad-CAM 1D       → Which time segments triggered the diagnosis
  4. Lead Saliency     → Which of the 12 leads mattered most

Usage:
  python predict.py                     # Random test patient
  python predict.py --ecg_id 12345      # Specific patient
  python predict.py --random 5          # 5 random patients
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from report_templates import build_structured_report

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
NORM_STATS_PATH = os.path.join(DATA_DIR, "norm_stats.json")
CLASSIFIER_CKPT = os.path.join(WORK_DIR, "checkpoints_ecg_only", "best_model.pt")
REPORT_CKPT = os.path.join(WORK_DIR, "checkpoints_report_gen", "best_model.pt")
OUTPUT_DIR = os.path.join(WORK_DIR, "predictions")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
NUM_LEADS = 12
CNN_CHANNELS = [64, 128, 192, 256]
CNN_KERNELS = [15, 7, 5, 3]
CLASSIFIER_HIDDEN = 128
DROPOUT = 0.3
NUM_CLASSES = 5


# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (must match training scripts exactly)
# ══════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ECGResNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = NUM_LEADS
        for i, (out_ch, ks) in enumerate(zip(CNN_CHANNELS, CNN_KERNELS)):
            drop = 0.1 if i < 2 else 0.2
            blocks.append(ResidualBlock(in_ch, out_ch, ks, stride=2, dropout=drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(CNN_CHANNELS[-1], CLASSIFIER_HIDDEN),
            nn.BatchNorm1d(CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, NUM_CLASSES),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).squeeze(-1)
        return self.classifier(feat)


class ECGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = NUM_LEADS
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
            nn.Linear(CNN_CHANNELS[-1], 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )
        self.bart = BartForConditionalGeneration.from_pretrained(
            "GanjinZero/biobart-base", torch_dtype=torch.float32
        )
        self.tokenizer = tokenizer

    def get_encoder_outputs(self, signal):
        with torch.no_grad():
            feat = self.ecg_backbone(signal)
        feat = feat.permute(0, 2, 1)
        hidden = self.projection(feat)
        return self.BaseModelOutput(last_hidden_state=hidden)

    def generate_report(self, signal, max_length=64, num_beams=4):
        """Tier Legacy: free generation from raw ECG signal (kept for ablation comparison)."""
        encoder_outputs = self.get_encoder_outputs(signal)
        batch_size = signal.size(0)
        seq_len = encoder_outputs.last_hidden_state.size(1)
        attn_mask = torch.ones(batch_size, seq_len, dtype=torch.long,
                               device=signal.device)
        generated = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attn_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

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
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
#  GRAD-CAM 1D
# ══════════════════════════════════════════════════════════════════
class GradCAM1D:
    """Grad-CAM for 1D CNN — highlights which time segments matter."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, signal, class_idx):
        self.model.eval()
        signal.requires_grad_(True)
        logits = self.model(signal)

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().numpy()


# ══════════════════════════════════════════════════════════════════
#  LEAD SALIENCY (Integrated Gradients)
# ══════════════════════════════════════════════════════════════════
def compute_lead_saliency(model, signal, class_idx, steps=50):
    """Compute per-lead importance using Integrated Gradients."""
    model.eval()
    baseline = torch.zeros_like(signal)
    scaled_inputs = [baseline + (float(i) / steps) * (signal - baseline)
                     for i in range(1, steps + 1)]

    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad_(True)
        logits = model(scaled)
        model.zero_grad()
        logits[0, class_idx].backward()
        grads.append(scaled.grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    ig = (signal - baseline) * avg_grads  # (1, 12, 5000)

    # Sum over time to get per-lead importance
    lead_importance = ig.squeeze().abs().sum(dim=1).numpy()  # (12,)
    lead_importance = lead_importance / (lead_importance.sum() + 1e-8) * 100
    return lead_importance


# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def create_analysis_plot(signal_raw, cam, lead_saliency, probs,
                         thresholds, report_text, ecg_id, output_path):
    """Create a comprehensive analysis figure."""

    # Upsample CAM to match signal length
    cam_upsampled = np.interp(
        np.linspace(0, 1, signal_raw.shape[1]),
        np.linspace(0, 1, len(cam)),
        cam
    )
    time_axis = np.arange(signal_raw.shape[1]) / 500.0  # seconds

    fig = plt.figure(figsize=(20, 14), facecolor="white")

    # Use explicit subplot positions: rows 1-2 for leads, row 3 for bars, row 4 for report
    gs = gridspec.GridSpec(4, 6, height_ratios=[2.5, 2.5, 2, 1.2],
                           hspace=0.4, wspace=0.4)

    fig.suptitle(f"ECG Analysis — Patient #{ecg_id}",
                 fontsize=18, fontweight="bold", y=0.98)

    # ── Top 2 rows: 12-lead ECG with Grad-CAM overlay (imshow) ──
    for i in range(12):
        row = i // 6
        col = i % 6
        ax = fig.add_subplot(gs[row, col])

        lead_signal = signal_raw[i]
        ax.plot(time_axis, lead_signal, color="#1a1a2e", linewidth=0.6, alpha=0.9)

        # Efficient Grad-CAM overlay using imshow instead of 5000 axvspans
        extent = [time_axis[0], time_axis[-1], lead_signal.min(), lead_signal.max()]
        ax.imshow(cam_upsampled[np.newaxis, :], aspect="auto", extent=extent,
                  cmap="Reds", alpha=0.3, interpolation="bilinear",
                  vmin=0, vmax=1)

        title_color = "darkred" if lead_saliency[i] > 12 else "black"
        ax.set_title(f"{LEAD_NAMES[i]} ({lead_saliency[i]:.1f}%)",
                     fontsize=9, fontweight="bold", color=title_color)
        ax.set_xlim(0, time_axis[-1])
        ax.tick_params(labelsize=6)
        if row == 1:
            ax.set_xlabel("Time (s)", fontsize=7)

    # ── Row 3 left: Classification probabilities ──
    ax_bar = fig.add_subplot(gs[2, :3])
    colors_bar = ["#e74c3c" if probs[i] >= thresholds[i] else "#2ecc71"
                  for i in range(NUM_CLASSES)]
    bars = ax_bar.barh(CLASS_NAMES, probs * 100, color=colors_bar, edgecolor="white")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Probability (%)", fontsize=10)
    ax_bar.set_title("Disease Classification", fontsize=12, fontweight="bold")
    for i, bar in enumerate(bars):
        label = "DETECTED" if probs[i] >= thresholds[i] else "normal"
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{probs[i]*100:.1f}% ({label})",
                    va="center", fontsize=8, fontweight="bold")

    # ── Row 3 right: Lead saliency bar ──
    ax_lead = fig.add_subplot(gs[2, 3:])
    sorted_idx = np.argsort(lead_saliency)[::-1]
    sorted_names = [LEAD_NAMES[i] for i in sorted_idx]
    sorted_vals = lead_saliency[sorted_idx]
    lead_colors = ["#e74c3c" if v > 12 else "#3498db" for v in sorted_vals]
    ax_lead.barh(sorted_names, sorted_vals, color=lead_colors, edgecolor="white")
    ax_lead.set_xlabel("Importance (%)", fontsize=10)
    ax_lead.set_title("Lead Saliency (Integrated Gradients)", fontsize=12,
                      fontweight="bold")
    ax_lead.invert_yaxis()

    # ── Row 4: Generated report ──
    ax_report = fig.add_subplot(gs[3, :])
    ax_report.axis("off")
    detected = [CLASS_NAMES[i] for i in range(NUM_CLASSES)
                if probs[i] >= thresholds[i]]
    detected_str = ", ".join(detected) if detected else "No abnormality detected"
    report_box = (
        f"━━━ HYBRID CLINICAL REPORT (Classifier-Grounded) ━━━\n\n"
        f"Confirmed findings: {detected_str}\n\n"
        f"{report_text}\n"
    )
    ax_report.text(0.02, 0.95, report_box, transform=ax_report.transAxes,
                   fontsize=10, verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4fd",
                             edgecolor="#2196F3", linewidth=1.5))

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    return output_path


# ══════════════════════════════════════════════════════════════════
#  MAIN PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════
def predict(ecg_id, classifier, report_model, gradcam, norm_stats, thresholds):
    """Run full analysis on one patient."""

    # 1. Load and preprocess signal
    sig_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
    signal_raw = np.load(sig_path).astype(np.float32)  # (5000, 12)

    sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
    sig_std = np.array(norm_stats["signal_std"], dtype=np.float32)
    signal_norm = (signal_raw - sig_mean) / sig_std
    signal_norm = signal_norm.T  # (12, 5000)

    signal_tensor = torch.from_numpy(signal_norm).unsqueeze(0)  # (1, 12, 5000)

    # 2. Classification
    classifier.eval()
    with torch.no_grad():
        logits = classifier(signal_tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()

    # 3. Tier 2 — Template report (hallucination-free, classifier-grounded)
    structured = build_structured_report(
        probs=list(probs),
        thresholds=list(thresholds),
        class_names=CLASS_NAMES,
    )
    template_report = structured["report_text"]

    # 4. Tier 3 — BioBART smoother (constrained paraphrase of template)
    try:
        smoothed_report = report_model.smooth_report(template_report)
    except Exception as e:
        smoothed_report = f"(smoother unavailable: {e})"

    # 5. Legacy — BioBART free generation (ablation/comparison only)
    try:
        free_reports = report_model.generate_report(signal_tensor)
        free_report  = free_reports[0]
    except Exception as e:
        free_report = f"(free generation unavailable: {e})"

    # 6. Find the most important predicted class for Grad-CAM
    predicted = [(i, probs[i]) for i in range(NUM_CLASSES)
                 if probs[i] >= thresholds[i]]
    if predicted:
        target_class = max(predicted, key=lambda x: x[1])[0]
    else:
        target_class = np.argmax(probs)

    # 7. Grad-CAM
    cam = gradcam.generate(signal_tensor.clone(), target_class)

    # 8. Lead saliency (Integrated Gradients)
    lead_saliency = compute_lead_saliency(classifier, signal_tensor.clone(),
                                           target_class)

    # 9. Console output
    print(f"\n{'='*60}")
    print(f"  Patient #{ecg_id}")
    print(f"{'='*60}")
    print(f"  {'Class':<8} {'Prob':>8} {'Thresh':>8}  {'Result'}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")
    for i, cls in enumerate(CLASS_NAMES):
        status = "★ DETECTED" if probs[i] >= thresholds[i] else "  normal"
        tier   = structured["confidence_map"][cls]["tier"] or "-"
        print(f"  {cls:<8} {probs[i]*100:>7.1f}% {thresholds[i]:>8.3f}  {status}  [{tier}]")

    print(f"\n  Lead Saliency (top 5):")
    sorted_idx = np.argsort(lead_saliency)[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"    {rank+1}. Lead {LEAD_NAMES[idx]:<4} — {lead_saliency[idx]:.1f}%")

    print(f"\n  ━━ Tier 2 — Template Report (safe, grounded) ━━")
    print(f"    {template_report}")

    print(f"\n  ━━ Tier 3 — Smoothed Report (BioBART paraphrase) ━━")
    print(f"    {smoothed_report}")

    print(f"\n  ━━ Legacy — Free Generation (ablation only) ━━")
    print(f"    {free_report}")

    # 10. Save visualization (uses template report as primary text)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"ecg_{ecg_id}_analysis.png")
    create_analysis_plot(
        signal_raw.T, cam, lead_saliency, probs,
        thresholds, template_report, ecg_id, output_path
    )
    print(f"\n  Visualization saved: {output_path}")
    print(f"{'='*60}")

    return probs, template_report, smoothed_report, free_report, cam, lead_saliency


def main():
    parser = argparse.ArgumentParser(description="ECG Analysis Pipeline")
    parser.add_argument("--ecg_id", type=int, default=None,
                        help="Specific patient ECG ID")
    parser.add_argument("--random", type=int, default=1,
                        help="Number of random test patients to analyze")
    args = parser.parse_args()

    print("=" * 60)
    print("  ECG Analysis Pipeline")
    print("  Classifier + Report Generator + Grad-CAM + Lead Saliency")
    print("=" * 60)

    # ── Load norm stats ──
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    # ── Load classifier ──
    print("  Loading ECG Classifier...")
    classifier = ECGResNet()
    cls_state = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
    classifier.load_state_dict(cls_state["model_state"])
    classifier.eval()
    thresholds = cls_state["optimal_thresholds"]
    print(f"    Loaded (epoch {cls_state['epoch']+1}, "
          f"AUROC {cls_state['best_auroc']:.4f})")

    # ── Setup Grad-CAM on last residual block ──
    target_layer = classifier.backbone[-1]  # Last ResidualBlock
    gradcam = GradCAM1D(classifier, target_layer)

    # ── Load report generator ──
    print("  Loading Report Generator...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
    report_model = ECGReportModel(tokenizer)
    rpt_state = torch.load(REPORT_CKPT, map_location="cpu", weights_only=False)
    report_model.load_state_dict(rpt_state["model_state"])
    report_model.eval()
    print(f"    Loaded (epoch {rpt_state['epoch']+1}, "
          f"ROUGE-L {rpt_state['best_rougeL']:.4f})")

    # ── Select patients ──
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    if args.ecg_id:
        ecg_ids = [args.ecg_id]
    else:
        ecg_ids = test_df["ecg_id"].sample(n=args.random, random_state=None).tolist()

    # ── Run predictions ──
    for ecg_id in ecg_ids:
        predict(ecg_id, classifier, report_model, gradcam,
                norm_stats, thresholds)

    print(f"\n  All visualizations saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
