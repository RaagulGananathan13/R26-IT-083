"""
PTB-XL ECG-Only Classifier — ResNet 1D CNN
Input:  12-lead ECG signal (500Hz, 5000 samples)
Output: 5 superclass probabilities (NORM, MI, STTC, CD, HYP)

Features:
  - ResNet-style 1D CNN with residual connections (stronger than plain CNN)
  - Signal augmentation (noise, scaling, time-shift)
  - Focal Loss with per-class alpha weights
  - WeightedRandomSampler for class imbalance
  - Cosine Annealing LR with warm restarts
  - Gradient clipping, Dropout, BatchNorm, Weight Decay
  - Early stopping (patience=15)
  - Per-epoch per-class AUROC + F1
  - Resume from checkpoint
  - Final test evaluation with exact TP/TN/FP/FN counts

Usage: python -X utf8 train_ecg_only.py
"""

import os, sys, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
class Config:
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(WORK_DIR, "data")
    SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
    NORM_STATS = os.path.join(DATA_DIR, "norm_stats.json")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints_ecg_only")

    # Signal
    SIGNAL_LENGTH = 5000
    NUM_LEADS = 12

    # Architecture
    NUM_CLASSES = 5
    CNN_CHANNELS = [64, 128, 192, 256]
    CNN_KERNELS = [15, 7, 5, 3]
    CLASSIFIER_HIDDEN = 128
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 32
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 15
    NUM_WORKERS = 0
    GRAD_CLIP = 1.0

    # Class imbalance
    POS_WEIGHT = [1.45, 4.69, 2.78, 2.62, 10.48]
    FOCAL_GAMMA = 2.0
    CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

    # Augmentation
    AUG_NOISE_STD = 0.05
    AUG_SCALE_RANGE = (0.8, 1.2)
    AUG_SHIFT_MAX = 100


# ══════════════════════════════════════════════════════════════════
#  DATASET WITH SIGNAL AUGMENTATION
# ══════════════════════════════════════════════════════════════════
class ECGDataset(Dataset):
    def __init__(self, csv_path, config, norm_stats, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.is_train = is_train
        self.label_cols = [f"label_{c}" for c in config.CLASS_NAMES]

        self.sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
        self.sig_std = np.array(norm_stats["signal_std"], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def augment(self, signal):
        """Random augmentations applied ONLY during training."""
        # 1. Gaussian noise (50% chance)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config.AUG_NOISE_STD, signal.shape)
            signal = signal + noise.astype(np.float32)

        # 2. Random amplitude scaling (50% chance)
        if np.random.random() < 0.5:
            lo, hi = self.config.AUG_SCALE_RANGE
            scale = np.random.uniform(lo, hi)
            signal = signal * scale

        # 3. Random circular time shift (30% chance)
        if np.random.random() < 0.3:
            shift = np.random.randint(-self.config.AUG_SHIFT_MAX,
                                       self.config.AUG_SHIFT_MAX)
            signal = np.roll(signal, shift, axis=0)

        return signal.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = int(row["ecg_id"])

        # Load and normalize signal
        sig_path = os.path.join(self.config.SIGNAL_CACHE, f"{ecg_id}.npy")
        signal = np.load(sig_path).astype(np.float32)       # (5000, 12)
        signal = (signal - self.sig_mean) / self.sig_std

        # Augment during training only
        if self.is_train:
            signal = self.augment(signal)

        signal = signal.T  # (12, 5000) — channels first for Conv1d

        labels = np.array([row[c] for c in self.label_cols], dtype=np.float32)

        return torch.from_numpy(signal), torch.from_numpy(labels)


# ══════════════════════════════════════════════════════════════════
#  RESNET 1D CNN (STRONGER THAN PLAIN CNN)
# ══════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """1D Residual block with skip connection."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        # Skip connection (match dimensions if needed)
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
    """ResNet-style 1D CNN for 12-lead ECG classification."""

    def __init__(self, config):
        super().__init__()
        channels = config.CNN_CHANNELS   # [64, 128, 192, 256]
        kernels = config.CNN_KERNELS     # [15, 7, 5, 3]

        # Build residual blocks with increasing channels and stride-2 downsampling
        blocks = []
        in_ch = config.NUM_LEADS  # 12

        for i, (out_ch, ks) in enumerate(zip(channels, kernels)):
            drop = 0.1 if i < 2 else 0.2
            blocks.append(ResidualBlock(in_ch, out_ch, ks,
                                        stride=2, dropout=drop))
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], config.CLASSIFIER_HIDDEN),
            nn.BatchNorm1d(config.CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.CLASSIFIER_HIDDEN, config.NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, 12, 5000)
        feat = self.backbone(x)         # (B, 256, ~312)
        feat = self.pool(feat)          # (B, 256, 1)
        feat = feat.squeeze(-1)         # (B, 256)
        logits = self.classifier(feat)  # (B, 5)
        return logits


# ══════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                  reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * 1.0
        return (focal_weight * alpha_t * bce).mean()


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════
def compute_all_metrics(probs, labels, class_names, thresholds=None):
    """Compute per-class and macro AUROC + F1."""
    if thresholds is None:
        thresholds = [0.5] * len(class_names)

    results = {}
    aurocs, f1s = [], []

    for i, cls in enumerate(class_names):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        pred = (probs[:, i] >= thresholds[i]).astype(int)
        f1 = f1_score(labels[:, i], pred, zero_division=0)
        results[cls] = {"auroc": auc, "f1": f1}
        aurocs.append(auc)
        f1s.append(f1)

    results["macro_auroc"] = float(np.mean(aurocs))
    results["macro_f1"] = float(np.mean(f1s))
    return results


def optimize_thresholds(probs, labels, num_classes=5):
    """Find per-class threshold that maximizes F1 on validation set."""
    thresholds = []
    for i in range(num_classes):
        precision, recall, thresh = precision_recall_curve(
            labels[:, i], probs[:, i]
        )
        f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_vals)
        best_t = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
        thresholds.append(best_t)
    return thresholds


# ══════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # Load norm stats
        with open(config.NORM_STATS) as f:
            self.norm_stats = json.load(f)

        # Datasets
        self.train_ds = ECGDataset(
            os.path.join(config.DATA_DIR, "train.csv"),
            config, self.norm_stats, is_train=True
        )
        self.val_ds = ECGDataset(
            os.path.join(config.DATA_DIR, "val.csv"),
            config, self.norm_stats, is_train=False
        )
        self.test_ds = ECGDataset(
            os.path.join(config.DATA_DIR, "test.csv"),
            config, self.norm_stats, is_train=False
        )

        # Weighted sampler for training
        sample_weights = []
        for _, row in self.train_ds.df.iterrows():
            labs = [row[f"label_{c}"] for c in config.CLASS_NAMES]
            w = max((pw for l, pw in zip(labs, config.POS_WEIGHT) if l == 1),
                    default=1.0)
            sample_weights.append(w)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        self.train_loader = DataLoader(
            self.train_ds, batch_size=config.BATCH_SIZE,
            sampler=sampler, num_workers=config.NUM_WORKERS,
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
        self.model = ECGResNet(config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())

        # Loss, optimizer, scheduler
        alpha = torch.tensor(config.POS_WEIGHT, dtype=torch.float32)
        self.criterion = FocalLoss(alpha, config.FOCAL_GAMMA).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # State
        self.start_epoch = 0
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.history = []

        # Try resume
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_auroc = ckpt["best_auroc"]
            self.patience_counter = ckpt["patience_counter"]
            self.history = ckpt.get("history", [])
            print(f"  Resumed from epoch {self.start_epoch}, "
                  f"best AUROC: {self.best_auroc:.4f}")
        else:
            print(f"  Starting fresh training.")

        print(f"  Model parameters: {total_params:,}")

    # ── Training Loop ─────────────────────────────────────────────
    def train(self):
        config = self.config
        print(f"\n{'='*65}")
        print(f"  TRAINING (ECG Signal Only — ResNet 1D CNN)")
        print(f"{'='*65}")
        print(f"  Epochs:     {config.MAX_EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  LR:         {config.LR}")
        print(f"  Device:     {self.device}")
        print(f"  Train:      {len(self.train_ds):,}")
        print(f"  Val:        {len(self.val_ds):,}")
        print(f"  Test:       {len(self.test_ds):,}")

        t_start = time.time()

        for epoch in range(self.start_epoch, config.MAX_EPOCHS):
            t_epoch = time.time()

            # — Train —
            train_loss = self._run_epoch(self.train_loader, train=True,
                                         desc=f"Epoch {epoch+1:>2} [Train]")

            # — Validate —
            val_loss, val_probs, val_labels = self._run_epoch(
                self.val_loader, train=False, desc=f"Epoch {epoch+1:>2} [Val]"
            )

            # — Metrics —
            metrics = compute_all_metrics(
                val_probs, val_labels, config.CLASS_NAMES
            )
            val_auroc = metrics["macro_auroc"]
            lr_now = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_epoch

            self.scheduler.step()

            # — Check improvement —
            improved = ""
            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
                self.patience_counter = 0
                improved = " ★ BEST"

                opt_thresh = optimize_thresholds(val_probs, val_labels)
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "best_auroc": self.best_auroc,
                    "optimal_thresholds": opt_thresh,
                }, os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
            else:
                self.patience_counter += 1

            # — Save checkpoint for resume —
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "macro_auroc": val_auroc,
                "per_class": {c: metrics[c] for c in config.CLASS_NAMES},
            })

            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_auroc": self.best_auroc,
                "patience_counter": self.patience_counter,
                "history": self.history,
            }, os.path.join(config.CHECKPOINT_DIR, "checkpoint.pt"))

            # — Print epoch summary —
            print(f"  Epoch {epoch+1:>2}/{config.MAX_EPOCHS} │ "
                  f"Train: {train_loss:.4f} │ Val: {val_loss:.4f} │ "
                  f"AUROC: {val_auroc:.4f} │ "
                  f"LR: {lr_now:.6f} │ {elapsed:.0f}s{improved}")

            # — Per-class breakdown —
            auc_parts = " ".join(
                f"{c}={metrics[c]['auroc']:.3f}" for c in config.CLASS_NAMES
            )
            f1_parts = " ".join(
                f"{c}={metrics[c]['f1']:.3f}" for c in config.CLASS_NAMES
            )
            print(f"    AUROC: {auc_parts}")
            print(f"    F1:    {f1_parts}")

            # — Early stopping —
            if self.patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {config.PATIENCE} epochs)")
                break

        total_time = time.time() - t_start
        print(f"\n  Training complete in {total_time/60:.1f} minutes")
        print(f"  Best validation AUROC: {self.best_auroc:.4f}")

        # — Final test evaluation —
        self.final_evaluation()

    # ── Run one epoch ─────────────────────────────────────────────
    def _run_epoch(self, loader, train=True, desc=""):
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        all_probs, all_labels = [], []

        pbar = tqdm(loader, desc=f"  {desc}",
                    bar_format='{l_bar}{bar:25}{r_bar}', leave=False)

        for signal, labels in pbar:
            signal = signal.to(self.device)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()
                logits = self.model(signal)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP
                )
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(signal)
                    loss = self.criterion(logits, labels)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if not train:
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = running_loss / len(loader)

        if train:
            return avg_loss
        else:
            probs = np.concatenate(all_probs)
            labs = np.concatenate(all_labels)
            return avg_loss, probs, labs

    # ── Final Test Evaluation ─────────────────────────────────────
    def final_evaluation(self):
        config = self.config
        print(f"\n{'='*65}")
        print(f"  FINAL TEST SET EVALUATION")
        print(f"{'='*65}")

        # Load best model
        best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        opt_thresh = state["optimal_thresholds"]
        print(f"  Loaded best model (epoch {state['epoch']+1})")

        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for signal, labels in self.test_loader:
                signal = signal.to(self.device)
                logits = self.model(signal)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        n = len(labels)

        # ── Default thresholds (0.5) ──
        metrics_05 = compute_all_metrics(
            probs, labels, config.CLASS_NAMES
        )
        print(f"\n  Test Results (threshold=0.5)")
        print(f"  {'Class':<8} {'AUROC':>8} {'F1':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8}")
        for cls in config.CLASS_NAMES:
            m = metrics_05[cls]
            print(f"  {cls:<8} {m['auroc']:>8.4f} {m['f1']:>8.4f}")
        print(f"  {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Macro':<8} {metrics_05['macro_auroc']:>8.4f} "
              f"{metrics_05['macro_f1']:>8.4f}")

        # ── Optimized thresholds ──
        metrics_opt = compute_all_metrics(
            probs, labels, config.CLASS_NAMES, opt_thresh
        )
        print(f"\n  Test Results (optimized thresholds)")
        print(f"  {'Class':<8} {'AUROC':>8} {'F1':>8} {'Thresh':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i, cls in enumerate(config.CLASS_NAMES):
            m = metrics_opt[cls]
            print(f"  {cls:<8} {m['auroc']:>8.4f} {m['f1']:>8.4f} "
                  f"{opt_thresh[i]:>8.3f}")
        print(f"  {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Macro':<8} {metrics_opt['macro_auroc']:>8.4f} "
              f"{metrics_opt['macro_f1']:>8.4f}")

        # ── Exact prediction counts ──
        print(f"\n  Exact Prediction Counts (optimized thresholds)")
        print(f"  {'Class':<8} {'TP':>6} {'TN':>6} {'FP':>6} "
              f"{'FN':>6} {'Acc%':>7}")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

        for i, cls in enumerate(config.CLASS_NAMES):
            pred = (probs[:, i] >= opt_thresh[i]).astype(int)
            true = labels[:, i].astype(int)
            tp = int(np.sum((pred == 1) & (true == 1)))
            tn = int(np.sum((pred == 0) & (true == 0)))
            fp = int(np.sum((pred == 1) & (true == 0)))
            fn = int(np.sum((pred == 0) & (true == 1)))
            acc = (tp + tn) / n * 100
            print(f"  {cls:<8} {tp:>6} {tn:>6} {fp:>6} {fn:>6} {acc:>6.1f}%")

        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        print(f"  TP = Correctly found disease | TN = Correctly ruled out")
        print(f"  FP = False alarm             | FN = Missed disease")

        # ── Save results ──
        results = {
            "model": "ECG-Only ResNet 1D CNN",
            "best_epoch": state["epoch"] + 1,
            "best_val_auroc": state["best_auroc"],
            "test_default": metrics_05,
            "test_optimized": metrics_opt,
            "optimal_thresholds": {
                c: t for c, t in zip(config.CLASS_NAMES, opt_thresh)
            },
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
    print("  PTB-XL ECG-Only Classifier")
    print("  ResNet 1D CNN → 5 Superclasses (NO Text, NO Demographics)")
    print("=" * 65)

    config = Config()

    # Verify data exists
    for f in ["train.csv", "val.csv", "test.csv"]:
        if not os.path.exists(os.path.join(config.DATA_DIR, f)):
            print(f"\n  ERROR: Missing {f}. Run prepare_data.py first!")
            sys.exit(1)

    if not os.path.exists(config.NORM_STATS):
        print(f"\n  ERROR: Missing norm_stats.json. Run prepare_data.py first!")
        sys.exit(1)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
