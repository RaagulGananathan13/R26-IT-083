"""
PTB-XL ECG Report Generator — CNN Encoder + BioBART Decoder
Input:  12-lead ECG signal (500Hz, 5000 samples)
Output: English clinical report text

Architecture:
  - Frozen 1D ResNet CNN encoder (reused from ECG-only classifier)
  - Projection layer: CNN features (256-dim) → BART hidden (768-dim)
  - BioBART decoder: generates English medical text word-by-word

Features:
  - Reuses pre-trained ECG encoder (no re-training needed)
  - Teacher forcing during training
  - ROUGE-L + BLEU evaluation per epoch
  - Resume from checkpoint
  - Beam search generation during evaluation

Usage: python -X utf8 train_report_gen.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
class Config:
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(WORK_DIR, "data")
    SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
    NORM_STATS = os.path.join(DATA_DIR, "norm_stats.json")
    ECG_CHECKPOINT = os.path.join(WORK_DIR, "checkpoints_ecg_only", "best_model.pt")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints_report_gen")

    # Signal
    SIGNAL_LENGTH = 5000
    NUM_LEADS = 12

    # ECG Encoder (must match train_ecg_only.py)
    CNN_CHANNELS = [64, 128, 192, 256]
    CNN_KERNELS = [15, 7, 5, 3]

    # BioBART
    BART_MODEL = "GanjinZero/biobart-base"
    MAX_REPORT_LEN = 64       # max tokens for reports
    FREEZE_DECODER_LAYERS = 4  # freeze bottom N decoder layers

    # Training
    BATCH_SIZE = 8
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 20
    PATIENCE = 5
    NUM_WORKERS = 0
    GRAD_CLIP = 1.0

    # Evaluation
    NUM_BEAMS = 4
    EVAL_SAMPLES = 200  # samples per epoch for fast ROUGE eval


# ══════════════════════════════════════════════════════════════════
#  FROZEN ECG ENCODER (from train_ecg_only.py)
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)


class ECGBackbone(nn.Module):
    """Just the CNN backbone (no classifier head). Returns (B, 256, L)."""

    def __init__(self, config):
        super().__init__()
        channels = config.CNN_CHANNELS
        kernels = config.CNN_KERNELS
        blocks = []
        in_ch = config.NUM_LEADS
        for i, (out_ch, ks) in enumerate(zip(channels, kernels)):
            drop = 0.1 if i < 2 else 0.2
            blocks.append(ResidualBlock(in_ch, out_ch, ks,
                                        stride=2, dropout=drop))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)  # (B, 256, L) where L≈313


def load_frozen_encoder(config):
    """Load pre-trained ECG encoder and freeze it."""
    backbone = ECGBackbone(config)

    # Load weights from classifier checkpoint
    ckpt = torch.load(config.ECG_CHECKPOINT, map_location="cpu",
                      weights_only=False)
    model_state = ckpt["model_state"]

    # Extract only backbone weights
    backbone_state = {}
    for k, v in model_state.items():
        if k.startswith("backbone."):
            backbone_state[k] = v

    backbone.load_state_dict(backbone_state)

    # Freeze all parameters
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    print(f"  Loaded frozen ECG encoder from epoch {ckpt['epoch']+1}")
    return backbone


# ══════════════════════════════════════════════════════════════════
#  REPORT GENERATION MODEL
# ══════════════════════════════════════════════════════════════════
class ECGReportModel(nn.Module):
    """CNN Encoder → Projection → BioBART Decoder → Text."""

    def __init__(self, config, tokenizer):
        super().__init__()
        from transformers import BartForConditionalGeneration
        from transformers.modeling_outputs import BaseModelOutput

        self.BaseModelOutput = BaseModelOutput

        # Frozen ECG backbone
        self.ecg_backbone = load_frozen_encoder(config)

        # Projection: (B, L, 256) → (B, L, 768)
        self.projection = nn.Sequential(
            nn.Linear(config.CNN_CHANNELS[-1], 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

        # BioBART (force float32 — safetensors stores as float16)
        print(f"  Loading {config.BART_MODEL}...")
        self.bart = BartForConditionalGeneration.from_pretrained(
            config.BART_MODEL, torch_dtype=torch.float32
        )

        # Freeze BART's own encoder (we don't use it)
        for p in self.bart.model.encoder.parameters():
            p.requires_grad = False

        # Freeze bottom N decoder layers
        n_freeze = config.FREEZE_DECODER_LAYERS
        for i in range(n_freeze):
            for p in self.bart.model.decoder.layers[i].parameters():
                p.requires_grad = False
            # But UNFREEZE cross-attention (needs to learn CNN features)
            for p in self.bart.model.decoder.layers[i].encoder_attn.parameters():
                p.requires_grad = True
            for p in self.bart.model.decoder.layers[i].encoder_attn_layer_norm.parameters():
                p.requires_grad = True

        self.tokenizer = tokenizer
        self.config = config

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params:    {total - trainable:,}")

    def get_encoder_outputs(self, signal):
        """Run ECG through CNN and project to BART dimension."""
        with torch.no_grad():
            feat = self.ecg_backbone(signal)  # (B, 256, L)

        feat = feat.permute(0, 2, 1)          # (B, L, 256)
        hidden = self.projection(feat)         # (B, L, 768)

        return self.BaseModelOutput(last_hidden_state=hidden)

    def forward(self, signal, labels, decoder_attention_mask=None):
        """Training forward pass with teacher forcing."""
        encoder_outputs = self.get_encoder_outputs(signal)

        # Create encoder attention mask (all ones — no padding in CNN output)
        batch_size = signal.size(0)
        seq_len = encoder_outputs.last_hidden_state.size(1)
        encoder_attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=signal.device
        )

        outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs  # .loss is the cross-entropy loss

    def generate_report(self, signal, max_length=64, num_beams=4):
        """Generate text from ECG signal using beam search."""
        encoder_outputs = self.get_encoder_outputs(signal)

        batch_size = signal.size(0)
        seq_len = encoder_outputs.last_hidden_state.size(1)
        encoder_attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=signal.device
        )

        generated = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════
class ECGReportDataset(Dataset):
    def __init__(self, csv_path, config, norm_stats, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.tokenizer = tokenizer

        self.sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
        self.sig_std = np.array(norm_stats["signal_std"], dtype=np.float32)

        # Clean reports
        self.df["report_clean"] = self.df["report_en"].fillna("").astype(str)
        self.df["report_clean"] = self.df["report_clean"].apply(
            lambda x: x.strip() if x.strip() else "normal ecg"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = int(row["ecg_id"])

        # Load signal
        sig_path = os.path.join(self.config.SIGNAL_CACHE, f"{ecg_id}.npy")
        signal = np.load(sig_path).astype(np.float32)
        signal = (signal - self.sig_mean) / self.sig_std
        signal = signal.T  # (12, 5000)

        # Tokenize report
        report = row["report_clean"]
        tokens = self.tokenizer(
            report,
            max_length=self.config.MAX_REPORT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Labels: set padding tokens to -100 (ignored by loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return (
            torch.from_numpy(signal),
            labels,
            attention_mask,
        )


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════
def compute_text_metrics(predictions, references):
    """Compute ROUGE-L and BLEU scores."""
    from rouge_score import rouge_scorer
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                       use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # BLEU (corpus-level)
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    refs_tok = [[ref.split()] for ref in references]
    preds_tok = [pred.split() for pred in predictions]
    bleu = corpus_bleu(refs_tok, preds_tok, smoothing_function=smooth)

    return {
        "rouge1": float(np.mean(rouge_scores["rouge1"])),
        "rouge2": float(np.mean(rouge_scores["rouge2"])),
        "rougeL": float(np.mean(rouge_scores["rougeL"])),
        "bleu": float(bleu),
    }


# ══════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # Norm stats
        with open(config.NORM_STATS) as f:
            self.norm_stats = json.load(f)

        # Tokenizer
        from transformers import AutoTokenizer
        print("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BART_MODEL)

        # Datasets
        self.train_ds = ECGReportDataset(
            os.path.join(config.DATA_DIR, "train.csv"),
            config, self.norm_stats, self.tokenizer
        )
        self.val_ds = ECGReportDataset(
            os.path.join(config.DATA_DIR, "val.csv"),
            config, self.norm_stats, self.tokenizer
        )
        self.test_ds = ECGReportDataset(
            os.path.join(config.DATA_DIR, "test.csv"),
            config, self.norm_stats, self.tokenizer
        )

        self.train_loader = DataLoader(
            self.train_ds, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=config.NUM_WORKERS,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS,
        )

        # Model
        self.model = ECGReportModel(config, self.tokenizer).to(self.device)

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.MAX_EPOCHS
        )

        # State
        self.start_epoch = 0
        self.best_rougeL = 0.0
        self.patience_counter = 0
        self.history = []

        # Resume
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_rougeL = ckpt["best_rougeL"]
            self.patience_counter = ckpt["patience_counter"]
            self.history = ckpt.get("history", [])
            print(f"  Resumed from epoch {self.start_epoch}, "
                  f"best ROUGE-L: {self.best_rougeL:.4f}")
        else:
            print("  Starting fresh training.")

    def train(self):
        config = self.config
        print(f"\n{'='*65}")
        print(f"  TRAINING — ECG Report Generator")
        print(f"{'='*65}")
        print(f"  Epochs:     {config.MAX_EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  LR:         {config.LR}")
        print(f"  Train:      {len(self.train_ds):,}")
        print(f"  Val:        {len(self.val_ds):,}")
        print(f"  Test:       {len(self.test_ds):,}")

        t_start = time.time()

        for epoch in range(self.start_epoch, config.MAX_EPOCHS):
            t_epoch = time.time()

            # — Train —
            train_loss = self._train_epoch(epoch)

            # — Validate (loss) —
            val_loss = self._val_loss()

            # — ROUGE evaluation (on subset for speed) —
            metrics = self._evaluate_rouge(n_samples=config.EVAL_SAMPLES)

            lr_now = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_epoch
            self.scheduler.step()

            rougeL = metrics["rougeL"]
            improved = ""
            if rougeL > self.best_rougeL:
                self.best_rougeL = rougeL
                self.patience_counter = 0
                improved = " ★ BEST"
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "best_rougeL": self.best_rougeL,
                }, os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
            else:
                self.patience_counter += 1

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            })

            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_rougeL": self.best_rougeL,
                "patience_counter": self.patience_counter,
                "history": self.history,
            }, os.path.join(config.CHECKPOINT_DIR, "checkpoint.pt"))

            print(f"  Epoch {epoch+1:>2}/{config.MAX_EPOCHS} │ "
                  f"Train: {train_loss:.4f} │ Val: {val_loss:.4f} │ "
                  f"ROUGE-L: {rougeL:.4f} │ BLEU: {metrics['bleu']:.4f} │ "
                  f"LR: {lr_now:.6f} │ {elapsed:.0f}s{improved}")

            if self.patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - t_start
        print(f"\n  Training complete in {total_time/60:.1f} minutes")
        print(f"  Best ROUGE-L: {self.best_rougeL:.4f}")

        self.final_evaluation()

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.ecg_backbone.eval()  # Keep frozen encoder in eval mode
        running_loss = 0

        pbar = tqdm(self.train_loader,
                    desc=f"  Epoch {epoch+1:>2} [Train]",
                    bar_format='{l_bar}{bar:25}{r_bar}', leave=False)

        for signal, labels, dec_mask in pbar:
            signal = signal.to(self.device)
            labels = labels.to(self.device)
            dec_mask = dec_mask.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(signal, labels, dec_mask)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.GRAD_CLIP
            )
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(self.train_loader)

    def _val_loss(self):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for signal, labels, dec_mask in self.val_loader:
                signal = signal.to(self.device)
                labels = labels.to(self.device)
                dec_mask = dec_mask.to(self.device)
                outputs = self.model(signal, labels, dec_mask)
                running_loss += outputs.loss.item()
        return running_loss / len(self.val_loader)

    def _evaluate_rouge(self, n_samples=200):
        """Generate reports for N samples and compute ROUGE/BLEU."""
        self.model.eval()
        predictions, references = [], []

        # Sample from val set
        indices = np.random.choice(len(self.val_ds), min(n_samples, len(self.val_ds)),
                                    replace=False)

        with torch.no_grad():
            for i in tqdm(range(0, len(indices), self.config.BATCH_SIZE),
                          desc="  Evaluating", leave=False):
                batch_idx = indices[i:i + self.config.BATCH_SIZE]
                signals = []
                refs = []
                for idx in batch_idx:
                    sig, _, _ = self.val_ds[idx]
                    signals.append(sig)
                    refs.append(self.val_ds.df.iloc[idx]["report_clean"])

                signal_batch = torch.stack(signals).to(self.device)
                generated = self.model.generate_report(
                    signal_batch,
                    max_length=self.config.MAX_REPORT_LEN,
                    num_beams=self.config.NUM_BEAMS,
                )
                predictions.extend(generated)
                references.extend(refs)

        return compute_text_metrics(predictions, references)

    def final_evaluation(self):
        config = self.config
        print(f"\n{'='*65}")
        print(f"  FINAL TEST SET EVALUATION")
        print(f"{'='*65}")

        # Load best model
        best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        print(f"  Loaded best model (epoch {state['epoch']+1})")

        self.model.eval()
        predictions, references = [], []

        with torch.no_grad():
            for signal, labels, dec_mask in tqdm(
                self.test_loader, desc="  Generating reports", leave=False
            ):
                signal = signal.to(self.device)
                generated = self.model.generate_report(
                    signal,
                    max_length=config.MAX_REPORT_LEN,
                    num_beams=config.NUM_BEAMS,
                )
                predictions.extend(generated)

                # Get reference reports
                batch_start = len(references)
                for i in range(signal.size(0)):
                    idx = batch_start + i
                    if idx < len(self.test_ds):
                        references.append(
                            self.test_ds.df.iloc[idx]["report_clean"]
                        )

        # Compute metrics
        metrics = compute_text_metrics(predictions, references)

        print(f"\n  Test Results:")
        print(f"  {'Metric':<12} {'Score':>8}")
        print(f"  {'-'*12} {'-'*8}")
        print(f"  {'ROUGE-1':<12} {metrics['rouge1']:>8.4f}")
        print(f"  {'ROUGE-2':<12} {metrics['rouge2']:>8.4f}")
        print(f"  {'ROUGE-L':<12} {metrics['rougeL']:>8.4f}")
        print(f"  {'BLEU':<12} {metrics['bleu']:>8.4f}")

        # Show 5 example predictions
        print(f"\n  Sample Predictions (5 examples):")
        print(f"  {'─'*60}")
        for i in range(min(5, len(predictions))):
            print(f"  Reference:  {references[i][:80]}")
            print(f"  Generated:  {predictions[i][:80]}")
            print(f"  {'─'*60}")

        # Save results
        results = {
            "model": "ECG Report Generator (CNN + BioBART)",
            "best_epoch": state["epoch"] + 1,
            "best_val_rougeL": state["best_rougeL"],
            "test_metrics": metrics,
            "sample_predictions": [
                {"reference": references[i], "generated": predictions[i]}
                for i in range(min(20, len(predictions)))
            ],
            "history": self.history,
        }
        results_path = os.path.join(config.CHECKPOINT_DIR, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PTB-XL ECG Report Generator")
    print("  ResNet 1D CNN → BioBART Decoder → English Clinical Report")
    print("=" * 65)

    config = Config()

    # Verify pre-trained ECG encoder exists
    if not os.path.exists(config.ECG_CHECKPOINT):
        print(f"\n  ERROR: Pre-trained ECG encoder not found!")
        print(f"  Expected: {config.ECG_CHECKPOINT}")
        print(f"  Run 'python train_ecg_only.py' first!")
        sys.exit(1)

    # Verify data
    for f in ["train.csv", "val.csv", "test.csv"]:
        if not os.path.exists(os.path.join(config.DATA_DIR, f)):
            print(f"\n  ERROR: Missing {f}. Run prepare_data.py first!")
            sys.exit(1)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
