"""
evaluate_hybrid.py — Full Backend Evaluation of the Three-Tier Hybrid Pipeline

Tests:
  1. Classifier accuracy   — AUROC, F1, TP/TN/FP/FN per class (ResNet CNN)
  2. Template report safety — Verifies the template NEVER mentions undetected conditions
  3. ROUGE/BLEU scores      — Compares all three report tiers against reference reports:
                                Tier 2 (Template), Tier 3 (Smoothed), Legacy (Free BioBART)
  4. Hallucination audit    — Counts how often each tier mentions diseases the classifier did NOT detect
  5. Sample comparisons     — Prints side-by-side for manual inspection

Usage:
  python evaluate_hybrid.py                  # Full test set (1711 patients)
  python evaluate_hybrid.py --samples 50     # Quick test on 50 patients
  python evaluate_hybrid.py --samples 200    # Medium test
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORK_DIR, "data")
SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
NORM_STATS_PATH = os.path.join(DATA_DIR, "norm_stats.json")
CLASSIFIER_CKPT = os.path.join(WORK_DIR, "checkpoints_ecg_only", "best_model.pt")
REPORT_CKPT = os.path.join(WORK_DIR, "checkpoints_report_gen", "best_model.pt")
RESULTS_DIR = os.path.join(WORK_DIR, "evaluation_results")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = 5
CNN_CHANNELS = [64, 128, 192, 256]
CNN_KERNELS = [15, 7, 5, 3]

# Disease keywords for hallucination audit
DISEASE_KEYWORDS = {
    "NORM": ["normal", "within normal limits", "no significant abnormalities", "sinus rhythm"],
    "MI":   ["myocardial infarction", "st-segment changes", "st segment", "infarction", "ischemia", "ischaemia"],
    "STTC": ["st/t", "st-segment and t-wave", "t-wave changes", "t wave", "st change", "repolarization"],
    "CD":   ["conduction", "bundle branch", "block", "conduction delay", "conduction disturbance"],
    "HYP":  ["hypertrophy", "voltage criteria", "ventricular hypertrophy", "lvh", "rvh"],
}


# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (must match training scripts)
# ══════════════════════════════════════════════════════════════════
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
        """Legacy free generation from raw ECG signal."""
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
        """Tier 3: BioBART as constrained paraphraser."""
        inputs = self.tokenizer(structured_text, return_tensors="pt",
                                max_length=256, truncation=True)
        with torch.no_grad():
            gen = self.bart.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length, num_beams=num_beams,
                early_stopping=True, no_repeat_ngram_size=3,
                length_penalty=1.2)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
#  TEXT METRICS
# ══════════════════════════════════════════════════════════════════
def compute_text_metrics(predictions, references):
    """ROUGE-1, ROUGE-2, ROUGE-L, BLEU."""
    from rouge_score import rouge_scorer
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    refs_tok = [[ref.split()] for ref in references]
    preds_tok = [pred.split() for pred in predictions]
    bleu = corpus_bleu(refs_tok, preds_tok, smoothing_function=smooth)

    return {
        "rouge1": float(np.mean(rouge_scores["rouge1"])),
        "rouge2": float(np.mean(rouge_scores["rouge2"])),
        "rougeL": float(np.mean(rouge_scores["rougeL"])),
        "bleu":   float(bleu),
    }


# ══════════════════════════════════════════════════════════════════
#  HALLUCINATION AUDIT
# ══════════════════════════════════════════════════════════════════
def hallucination_audit(reports, detected_labels_list):
    """
    Count how many reports mention a disease that the classifier did NOT detect.

    Args:
        reports              : list[str] — generated report texts
        detected_labels_list : list[list[str]] — per-patient detected class names

    Returns dict with per-class and total hallucination counts.
    """
    hallucinations = {cls: 0 for cls in CLASS_NAMES if cls != "NORM"}
    total_hallucinated = 0
    total_patients = len(reports)

    for report, detected in zip(reports, detected_labels_list):
        report_lower = report.lower()
        for cls in CLASS_NAMES:
            if cls == "NORM":
                continue
            if cls not in detected:
                # This class was NOT detected — check if the report mentions it
                for keyword in DISEASE_KEYWORDS[cls]:
                    if keyword in report_lower:
                        hallucinations[cls] += 1
                        total_hallucinated += 1
                        break  # one keyword match per class is enough

    return {
        "per_class":     hallucinations,
        "total":         total_hallucinated,
        "total_patients": total_patients,
        "rate":          round(total_hallucinated / max(total_patients, 1) * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid ECG Pipeline")
    parser.add_argument("--samples", type=int, default=0,
                        help="Number of test samples (0 = all)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  HYBRID PIPELINE — FULL BACKEND EVALUATION")
    print("=" * 70)

    # ── Load norm stats ──
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
    sig_std  = np.array(norm_stats["signal_std"],  dtype=np.float32)

    # ── Load classifier ──
    print("\n  Loading ECG Classifier...")
    classifier = ECGResNet()
    cls_state = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=False)
    classifier.load_state_dict(cls_state["model_state"])
    classifier.eval()
    thresholds = cls_state["optimal_thresholds"]
    print(f"    Loaded (epoch {cls_state['epoch']+1}, AUROC {cls_state['best_auroc']:.4f})")
    print(f"    Thresholds: {dict(zip(CLASS_NAMES, [f'{t:.3f}' for t in thresholds]))}")

    # ── Load report generator ──
    print("\n  Loading Report Generator (BioBART)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
    report_model = ECGReportModel(tokenizer)
    rpt_state = torch.load(REPORT_CKPT, map_location="cpu", weights_only=False)
    report_model.load_state_dict(rpt_state["model_state"])
    report_model.eval()
    print(f"    Loaded (epoch {rpt_state['epoch']+1}, ROUGE-L {rpt_state['best_rougeL']:.4f})")

    # ── Load template engine ──
    from report_templates import build_structured_report

    # ── Load test set ──
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    if args.samples > 0:
        test_df = test_df.sample(n=min(args.samples, len(test_df)),
                                 random_state=42).reset_index(drop=True)
    n_patients = len(test_df)
    print(f"\n  Test set: {n_patients} patients")

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 1: CLASSIFIER EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SECTION 1: CLASSIFIER PERFORMANCE (ResNet-1D CNN)")
    print(f"{'='*70}")

    all_probs = []
    all_labels = []
    label_cols = [f"label_{c}" for c in CLASS_NAMES]

    for idx in tqdm(range(n_patients), desc="  Classifying"):
        row = test_df.iloc[idx]
        ecg_id = int(row["ecg_id"])
        sig_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
        signal = np.load(sig_path).astype(np.float32)
        signal = (signal - sig_mean) / sig_std
        signal = signal.T  # (12, 5000)
        tensor = torch.from_numpy(signal).unsqueeze(0)

        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.sigmoid(logits).squeeze().numpy()
        all_probs.append(probs)
        all_labels.append(np.array([row[c] for c in label_cols], dtype=np.float32))

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Per-class metrics
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    print(f"\n  {'Class':<8} {'AUROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} "
          f"{'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Acc%':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    aurocs, f1s, precs, recs = [], [], [], []
    classifier_results = {}

    for i, cls in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = 0.5
        pred = (all_probs[:, i] >= thresholds[i]).astype(int)
        true = all_labels[:, i].astype(int)

        f1  = f1_score(true, pred, zero_division=0)
        pre = precision_score(true, pred, zero_division=0)
        rec = recall_score(true, pred, zero_division=0)

        tp = int(np.sum((pred == 1) & (true == 1)))
        tn = int(np.sum((pred == 0) & (true == 0)))
        fp = int(np.sum((pred == 1) & (true == 0)))
        fn = int(np.sum((pred == 0) & (true == 1)))
        acc = (tp + tn) / n_patients * 100

        aurocs.append(auc); f1s.append(f1); precs.append(pre); recs.append(rec)
        classifier_results[cls] = {
            "auroc": round(auc, 4), "f1": round(f1, 4),
            "precision": round(pre, 4), "recall": round(rec, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": round(acc, 1),
        }

        print(f"  {cls:<8} {auc:>8.4f} {f1:>8.4f} {pre:>8.4f} {rec:>8.4f} "
              f"{tp:>6} {tn:>6} {fp:>6} {fn:>6} {acc:>6.1f}%")

    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Macro':<8} {np.mean(aurocs):>8.4f} {np.mean(f1s):>8.4f} "
          f"{np.mean(precs):>8.4f} {np.mean(recs):>8.4f}")

    classifier_results["macro"] = {
        "auroc": round(float(np.mean(aurocs)), 4),
        "f1":    round(float(np.mean(f1s)),    4),
        "precision": round(float(np.mean(precs)), 4),
        "recall":    round(float(np.mean(recs)),  4),
    }

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 2: THREE-TIER REPORT GENERATION + EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SECTION 2: THREE-TIER REPORT GENERATION")
    print(f"{'='*70}")

    template_reports  = []
    smoothed_reports  = []
    free_reports      = []
    reference_reports = []
    detected_labels_all = []

    t_start = time.time()

    for idx in tqdm(range(n_patients), desc="  Generating reports"):
        row = test_df.iloc[idx]
        ecg_id = int(row["ecg_id"])
        probs = all_probs[idx]

        # Reference report
        ref = str(row.get("report_en", ""))
        if ref == "nan" or ref.strip() == "":
            ref = "normal ecg"
        reference_reports.append(ref)

        # Tier 2: Template
        structured = build_structured_report(
            probs=list(probs),
            thresholds=list(thresholds),
            class_names=CLASS_NAMES,
        )
        template_reports.append(structured["report_text"])
        detected_labels_all.append(structured["detected_labels"])

        # Tier 3: Smoothed (BioBART paraphrase of template)
        try:
            smoothed = report_model.smooth_report(structured["report_text"])
        except Exception:
            smoothed = structured["report_text"]
        smoothed_reports.append(smoothed)

        # Legacy: Free BioBART generation
        sig_path = os.path.join(SIGNAL_CACHE, f"{ecg_id}.npy")
        signal = np.load(sig_path).astype(np.float32)
        signal = (signal - sig_mean) / sig_std
        tensor = torch.from_numpy(signal.T).unsqueeze(0).float()
        try:
            free = report_model.generate_report(tensor)[0]
        except Exception:
            free = "(generation failed)"
        free_reports.append(free)

    gen_time = time.time() - t_start
    print(f"\n  Generation time: {gen_time:.1f}s ({gen_time/n_patients:.2f}s per patient)")

    # ── Compute ROUGE/BLEU for each tier ──
    print(f"\n  Computing ROUGE / BLEU metrics...")

    tier2_metrics = compute_text_metrics(template_reports, reference_reports)
    tier3_metrics = compute_text_metrics(smoothed_reports, reference_reports)
    free_metrics  = compute_text_metrics(free_reports,     reference_reports)

    print(f"\n  {'Metric':<12} {'Tier 2':>12} {'Tier 3':>12} {'Legacy':>12}")
    print(f"  {'':12}  {'Template':>12} {'Smoothed':>12} {'Free Gen':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for metric in ["rouge1", "rouge2", "rougeL", "bleu"]:
        print(f"  {metric.upper():<12} {tier2_metrics[metric]:>12.4f} "
              f"{tier3_metrics[metric]:>12.4f} {free_metrics[metric]:>12.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 3: HALLUCINATION AUDIT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SECTION 3: HALLUCINATION AUDIT")
    print(f"  (Does the report mention diseases the classifier did NOT detect?)")
    print(f"{'='*70}")

    tier2_halluc = hallucination_audit(template_reports, detected_labels_all)
    tier3_halluc = hallucination_audit(smoothed_reports, detected_labels_all)
    free_halluc  = hallucination_audit(free_reports,     detected_labels_all)

    print(f"\n  {'Disease':<8} {'Tier 2':>12} {'Tier 3':>12} {'Legacy':>12}")
    print(f"  {'':8}  {'Template':>12} {'Smoothed':>12} {'Free Gen':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for cls in CLASS_NAMES:
        if cls == "NORM":
            continue
        t2 = tier2_halluc["per_class"][cls]
        t3 = tier3_halluc["per_class"][cls]
        fr = free_halluc["per_class"][cls]
        print(f"  {cls:<8} {t2:>12} {t3:>12} {fr:>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':<8} {tier2_halluc['total']:>12} "
          f"{tier3_halluc['total']:>12} {free_halluc['total']:>12}")
    print(f"  {'RATE':<8} {tier2_halluc['rate']:>11.1f}% "
          f"{tier3_halluc['rate']:>11.1f}% {free_halluc['rate']:>11.1f}%")

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 4: SAMPLE COMPARISONS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SECTION 4: SAMPLE COMPARISONS (5 examples)")
    print(f"{'='*70}")

    sample_indices = np.random.RandomState(42).choice(n_patients, min(5, n_patients), replace=False)
    sample_comparisons = []

    for idx in sample_indices:
        ecg_id = int(test_df.iloc[idx]["ecg_id"])
        probs = all_probs[idx]
        detected = detected_labels_all[idx]

        print(f"\n  ┌─ ECG #{ecg_id} ──────────────────────────────────────")
        print(f"  │ Detected:  {', '.join(detected) if detected else 'None'}")
        print(f"  │ Probs:     {' '.join(f'{CLASS_NAMES[i]}={probs[i]*100:.0f}%' for i in range(5))}")
        print(f"  │")
        print(f"  │ Reference: {reference_reports[idx][:120]}")
        print(f"  │")
        print(f"  │ Tier 2:    {template_reports[idx][:120]}")
        print(f"  │ Tier 3:    {smoothed_reports[idx][:120]}")
        print(f"  │ Legacy:    {free_reports[idx][:120]}")
        print(f"  └─────────────────────────────────────────────────")

        sample_comparisons.append({
            "ecg_id": ecg_id,
            "detected": detected,
            "probs": {CLASS_NAMES[i]: round(float(probs[i])*100, 1) for i in range(5)},
            "reference": reference_reports[idx],
            "tier2_template": template_reports[idx],
            "tier3_smoothed": smoothed_reports[idx],
            "legacy_free": free_reports[idx],
        })

    # ═══════════════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    results = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_patients": n_patients,
        "generation_time_sec": round(gen_time, 1),
        "classifier": classifier_results,
        "report_metrics": {
            "tier2_template": tier2_metrics,
            "tier3_smoothed": tier3_metrics,
            "legacy_free":    free_metrics,
        },
        "hallucination_audit": {
            "tier2_template": tier2_halluc,
            "tier3_smoothed": tier3_halluc,
            "legacy_free":    free_halluc,
        },
        "sample_comparisons": sample_comparisons,
    }

    results_path = os.path.join(RESULTS_DIR, "hybrid_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Classifier Macro AUROC:  {classifier_results['macro']['auroc']:.4f}")
    print(f"  Classifier Macro F1:     {classifier_results['macro']['f1']:.4f}")
    print(f"")
    print(f"  Template   ROUGE-L:      {tier2_metrics['rougeL']:.4f}")
    print(f"  Smoothed   ROUGE-L:      {tier3_metrics['rougeL']:.4f}")
    print(f"  Free Gen   ROUGE-L:      {free_metrics['rougeL']:.4f}")
    print(f"")
    print(f"  Hallucination Rate:")
    print(f"    Template:   {tier2_halluc['rate']:.1f}%  ({tier2_halluc['total']}/{n_patients})")
    print(f"    Smoothed:   {tier3_halluc['rate']:.1f}%  ({tier3_halluc['total']}/{n_patients})")
    print(f"    Free Gen:   {free_halluc['rate']:.1f}%  ({free_halluc['total']}/{n_patients})")
    print(f"")
    print(f"  Full results saved to: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
