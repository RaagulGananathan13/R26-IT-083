import os, json, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
NORM_STATS_PATH = os.path.join(DATA_DIR, "norm_stats.json")
CLASSIFIER_CKPT = os.path.join(WORK_DIR, "checkpoints_ecg_only", "best_model.pt")
REPORT_CKPT = os.path.join(WORK_DIR, "checkpoints_report_gen", "best_model.pt")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CNN_CHANNELS = [64, 128, 192, 256]
CNN_KERNELS = [15, 7, 5, 3]

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch))

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
        in_ch = 12
        for i, (out_ch, ks) in enumerate(zip(CNN_CHANNELS, CNN_KERNELS)):
            drop = 0.1 if i < 2 else 0.2
            blocks.append(ResidualBlock(in_ch, out_ch, ks, stride=2, dropout=drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).squeeze(-1)
        return self.classifier(feat)

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
        with torch.no_grad():
            feat = self.ecg_backbone(signal)
        feat = feat.permute(0, 2, 1)
        hidden = self.projection(feat)
        enc_out = self.BaseModelOutput(last_hidden_state=hidden)
        attn = torch.ones(signal.size(0), hidden.size(1), dtype=torch.long)
        gen = self.bart.generate(encoder_outputs=enc_out, attention_mask=attn,
                                  max_length=max_length, num_beams=num_beams,
                                  early_stopping=True, no_repeat_ngram_size=3)
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def smooth_report(self, structured_text, max_length=128, num_beams=4):
        inputs = self.tokenizer(structured_text, return_tensors="pt",
                                max_length=256, truncation=True)
        with torch.no_grad():
            gen = self.bart.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length, num_beams=num_beams,
                early_stopping=True, no_repeat_ngram_size=3,
                length_penalty=1.2)
        raw = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        return self._fix_smoothed_output(raw, structured_text)

    @staticmethod
    def _fix_smoothed_output(smoothed, template):
        """Post-processing fix for BioBART decoder artifacts (V2)."""
        import re as _re
        smoothed = smoothed.strip()
        if not smoothed:
            return template
        # Fix text typos
        smoothed = smoothed.replace("ST-seal", "ST-segment")
        smoothed = smoothed.replace("STsegment", "ST-segment")
        smoothed = smoothed.replace(".Clinical", ". Clinical")
        smoothed = smoothed.replace(".Cl ", ". Clinical ")
        smoothed = smoothed.replace("Cl correlation", "Clinical correlation")
        smoothed = smoothed.replace("record keeping", "correlation")
        smoothed = _re.sub(r'\.(?=[A-Z])', '. ', smoothed)
        # Case 1: Clean start
        if smoothed[0].isupper() and template.startswith(smoothed[:10]):
            return smoothed
        # Case 2: Truncated suffix — restore prefix
        template_lower = template.lower()
        smoothed_lower = smoothed.lower()
        for offset in range(1, min(30, len(template))):
            fragment = template_lower[offset:offset + 15]
            if fragment and smoothed_lower.startswith(fragment):
                return template[:offset] + smoothed.lstrip(" -")
        # Case 3: Garbled start — fallback to template
        return template

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep Audit Export")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of signals to process (0 for all)")
    args = parser.parse_args()

    print("Loading datasets and models...")
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
    sig_std  = np.array(norm_stats["signal_std"],  dtype=np.float32)

    classifier = ECGResNet()
    cls_state = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
    classifier.load_state_dict(cls_state["model_state"])
    classifier.eval()
    thresholds = cls_state["optimal_thresholds"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
    report_model = ECGReportModel(tokenizer)
    rpt_state = torch.load(REPORT_CKPT, map_location="cpu", weights_only=False)
    report_model.load_state_dict(rpt_state["model_state"])
    report_model.eval()

    from report_templates import build_structured_report

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    if args.limit > 0:
        test_df = test_df.head(args.limit)
    
    n_patients = len(test_df)
    
    output_file = os.path.join(WORK_DIR, "audit_real_vs_generated.txt")
    print(f"Generating reports and saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("ECG REPORT DEEP AUDIT: REAL VS GENERATED\n")
        f.write("="*80 + "\n\n")

        for idx in tqdm(range(n_patients), desc="Processing ECGs"):
            row = test_df.iloc[idx]
            ecg_id = int(row["ecg_id"])
            
            # Ground truth
            ref = str(row.get("report_en", ""))
            if ref == "nan" or ref.strip() == "":
                ref = "normal ecg"
                
            # Prepare signal
            sig_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
            if not os.path.exists(sig_path):
                continue
            signal = np.load(sig_path).astype(np.float32)
            signal = (signal - sig_mean) / sig_std
            signal_t = signal.T
            tensor = torch.from_numpy(signal_t).unsqueeze(0).float()
            
            # Classify
            with torch.no_grad():
                logits = classifier(tensor)
                probs = torch.sigmoid(logits).squeeze().numpy()
            
            detected_classes = [CLASS_NAMES[i] for i in range(5) if probs[i] >= thresholds[i]]
            
            # Tier 2: Template
            structured = build_structured_report(
                probs=list(probs),
                thresholds=list(thresholds),
                class_names=CLASS_NAMES,
            )
            template_text = structured["report_text"]
            
            # Tier 3: Smoothed
            try:
                smoothed_text = report_model.smooth_report(template_text)
            except Exception:
                smoothed_text = template_text
                
            # Legacy
            try:
                free_text = report_model.generate_report(tensor)[0]
            except Exception:
                free_text = "(generation failed)"
                
            f.write(f"--- ECG ID: {ecg_id} ---\n")
            f.write(f"Detected Labels: {', '.join(detected_classes) if detected_classes else 'None'}\n")
            f.write(f"[REAL REPORT]\n{ref}\n\n")
            f.write(f"[GENERATED (TIER 2 - TEMPLATE)]\n{template_text}\n\n")
            f.write(f"[GENERATED (TIER 3 - SMOOTHED)]\n{smoothed_text}\n\n")
            f.write(f"[GENERATED (LEGACY FREE-TEXT)]\n{free_text}\n")
            f.write("="*80 + "\n\n")
            f.flush()

    print(f"Deep audit complete. File saved to: {output_file}")

if __name__ == "__main__":
    main()
