"""
PTB-XL Multi-Modal ECG Training Script
1D CNN (signal) + MLP (demographics) + Text Projector (ClinicalBERT) + Fusion

Features: Focal loss, per-class thresholds, resume support, progress bars,
           early stopping, per-class AUROC/F1 reporting.

Usage: python train.py
Resume: python train.py  (automatically resumes from last checkpoint)
"""
import os
import sys
import json
import time
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
    # Paths
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(WORK_DIR, "data")
    SIGNAL_CACHE = os.path.join(DATA_DIR, "signals_cache")
    TEXT_CACHE = os.path.join(DATA_DIR, "text_cache")
    NORM_STATS = os.path.join(DATA_DIR, "norm_stats.json")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Model
    SIGNAL_LENGTH = 5000    # 500Hz × 10s
    NUM_LEADS = 12
    NUM_CLASSES = 5
    CNN_HIDDEN = 256
    DEMO_HIDDEN = 64
    TEXT_HIDDEN = 128
    FUSION_HIDDEN = 256
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 32
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 50
    PATIENCE = 10           # Early stopping patience
    NUM_WORKERS = 0         # 0 for Windows compatibility

    # Class weights (from training set)
    POS_WEIGHT = [1.45, 4.69, 2.78, 2.62, 10.48]

    # Focal loss
    FOCAL_GAMMA = 2.0

    CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════
class PTBXLDataset(Dataset):
    """Loads cached signals, text embeddings, and demographics."""

    def __init__(self, csv_path, config, norm_stats):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.norm_stats = norm_stats
        self.label_cols = [f"label_{c}" for c in config.CLASS_NAMES]
        self.demo_cols = ["age", "sex", "height", "weight",
                          "height_missing", "weight_missing"]

        # Pre-compute demographic means/stds for normalization
        self.demo_mean = np.array([
            norm_stats["demographics"]["age"]["mean"],
            0.5,  # sex is binary, center at 0.5
            norm_stats["demographics"]["height"]["mean"],
            norm_stats["demographics"]["weight"]["mean"],
            0.0, 0.0,  # missing flags are already 0/1
        ], dtype=np.float32)
        self.demo_std = np.array([
            norm_stats["demographics"]["age"]["std"],
            0.5,  # sex
            norm_stats["demographics"]["height"]["std"],
            norm_stats["demographics"]["weight"]["std"],
            1.0, 1.0,  # missing flags
        ], dtype=np.float32)

        # Signal normalization (per-lead)
        self.sig_mean = np.array(norm_stats["signal_mean"], dtype=np.float32)
        self.sig_std = np.array(norm_stats["signal_std"], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = int(row["ecg_id"])

        # Signal: (1000, 12) → normalize → transpose to (12, 1000)
        sig_path = os.path.join(self.config.SIGNAL_CACHE, f"{ecg_id}.npy")
        signal = np.load(sig_path).astype(np.float32)
        signal = (signal - self.sig_mean) / self.sig_std
        signal = signal.T  # (12, 1000) — channels first for Conv1d

        # Demographics: 6 features, normalized
        demo = np.array([row[c] for c in self.demo_cols], dtype=np.float32)
        demo = (demo - self.demo_mean) / self.demo_std

        # Text embedding: (768,)
        text_path = os.path.join(self.config.TEXT_CACHE, f"{ecg_id}.npy")
        text_emb = np.load(text_path).astype(np.float32)

        # Labels: (5,)
        labels = np.array([row[c] for c in self.label_cols], dtype=np.float32)

        return (
            torch.from_numpy(signal),
            torch.from_numpy(demo),
            torch.from_numpy(text_emb),
            torch.from_numpy(labels),
        )


# ══════════════════════════════════════════════════════════════════
#  MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════
class ECGEncoder(nn.Module):
    """5-layer 1D CNN for 12-lead ECG signals (500Hz, 5000 samples)."""

    def __init__(self, config):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: 12 → 32
            nn.Conv1d(config.NUM_LEADS, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 5000 → 2500
            nn.Dropout(0.1),

            # Block 2: 32 → 64
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 2500 → 1250
            nn.Dropout(0.1),

            # Block 3: 64 → 128
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1250 → 625
            nn.Dropout(0.2),

            # Block 4: 128 → 192
            nn.Conv1d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 625 → 312
            nn.Dropout(0.2),

            # Block 5: 192 → 256
            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 312 → 1 (global average pooling)
        )

    def forward(self, x):
        # x: (B, 12, 1000)
        x = self.conv_blocks(x)  # (B, 256, 1)
        return x.squeeze(-1)     # (B, 256)


class DemographicEncoder(nn.Module):
    """MLP for demographic features (age, sex, height, weight, flags)."""

    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.DEMO_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.mlp(x)  # (B, 64)


class TextProjector(nn.Module):
    """Projects ClinicalBERT embeddings to fusion dimension."""

    def __init__(self, config):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(768, config.TEXT_HIDDEN),
            nn.LayerNorm(config.TEXT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.projector(x)  # (B, 128)


class MultiModalFusionModel(nn.Module):
    """Full multi-modal model: CNN + MLP + Text → Fusion → 5 classes."""

    def __init__(self, config):
        super().__init__()
        self.ecg_encoder = ECGEncoder(config)
        self.demo_encoder = DemographicEncoder(config)
        self.text_projector = TextProjector(config)

        fusion_in = config.CNN_HIDDEN + config.DEMO_HIDDEN + config.TEXT_HIDDEN
        # 256 + 64 + 128 = 448

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.FUSION_HIDDEN),
            nn.BatchNorm1d(config.FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FUSION_HIDDEN, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.NUM_CLASSES),
            # No sigmoid here — using BCEWithLogitsLoss / FocalLoss
        )

    def forward(self, signal, demo, text_emb):
        ecg_feat = self.ecg_encoder(signal)       # (B, 256)
        demo_feat = self.demo_encoder(demo)        # (B, 64)
        text_feat = self.text_projector(text_emb)  # (B, 128)

        fused = torch.cat([ecg_feat, demo_feat, text_feat], dim=1)  # (B, 448)
        logits = self.fusion(fused)  # (B, 5)
        return logits


# ══════════════════════════════════════════════════════════════════
#  FOCAL LOSS
# ══════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification with class weights."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape (num_classes,)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
            focal_weight = focal_weight * alpha_t

        return (focal_weight * bce).mean()


# ══════════════════════════════════════════════════════════════════
#  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")

        # Load normalization stats
        with open(config.NORM_STATS, "r") as f:
            self.norm_stats = json.load(f)

        # Datasets
        self.train_ds = PTBXLDataset(
            os.path.join(config.DATA_DIR, "train.csv"), config, self.norm_stats
        )
        self.val_ds = PTBXLDataset(
            os.path.join(config.DATA_DIR, "val.csv"), config, self.norm_stats
        )
        self.test_ds = PTBXLDataset(
            os.path.join(config.DATA_DIR, "test.csv"), config, self.norm_stats
        )

        # Strategy 5: WeightedRandomSampler — oversample minority classes
        train_df = self.train_ds.df
        class_weights = config.POS_WEIGHT  # [1.45, 4.69, 2.78, 2.62, 10.48]
        sample_weights = []
        for _, row in train_df.iterrows():
            labels = [row[f'label_{c}'] for c in config.CLASS_NAMES]
            # Weight each sample by its rarest positive class
            weight = max(
                (w for l, w in zip(labels, class_weights) if l == 1),
                default=1.0
            )
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=config.BATCH_SIZE,
            sampler=sampler,  # replaces shuffle=True
            num_workers=config.NUM_WORKERS, pin_memory=False,
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
        self.model = MultiModalFusionModel(config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {total_params:,}")

        # Loss
        alpha = torch.tensor(config.POS_WEIGHT, dtype=torch.float32).to(self.device)
        self.criterion = FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)

        # Optimizer + Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Tracking
        self.start_epoch = 0
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [],
                        "val_auroc": [], "lr": []}
        self.optimal_thresholds = [0.5] * config.NUM_CLASSES

        # Checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def save_checkpoint(self, epoch, is_best=False):
        """Save training state for resume."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_auroc": self.best_auroc,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "optimal_thresholds": self.optimal_thresholds,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, "last_checkpoint.pt")
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pt")
            torch.save(state, best_path)

    def load_checkpoint(self):
        """Resume from last checkpoint if available."""
        path = os.path.join(self.config.CHECKPOINT_DIR, "last_checkpoint.pt")
        if not os.path.exists(path):
            return False

        print(f"  Resuming from checkpoint...")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch = state["epoch"] + 1
        self.best_auroc = state["best_auroc"]
        self.patience_counter = state["patience_counter"]
        self.history = state["history"]
        self.optimal_thresholds = state.get("optimal_thresholds", [0.5] * 5)
        print(f"  Resumed at epoch {self.start_epoch}, "
              f"best AUROC: {self.best_auroc:.4f}")
        return True

    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader,
                    desc=f"  Epoch {epoch+1:>2}/{self.config.MAX_EPOCHS} [Train]",
                    bar_format='{l_bar}{bar:25}{r_bar}',
                    leave=False)

        for signal, demo, text_emb, labels in pbar:
            signal = signal.to(self.device)
            demo = demo.to(self.device)
            text_emb = text_emb.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(signal, demo, text_emb)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader, desc="Val"):
        """Evaluate on a dataset. Returns loss, predictions, targets."""
        self.model.eval()
        all_logits = []
        all_labels = []
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  {'':>25}[{desc}]",
                    bar_format='{l_bar}{bar:25}{r_bar}', leave=False)

        for signal, demo, text_emb, labels in pbar:
            signal = signal.to(self.device)
            demo = demo.to(self.device)
            text_emb = text_emb.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(signal, demo, text_emb)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.sigmoid(all_logits).numpy()
        all_labels = all_labels.numpy()

        avg_loss = running_loss / n_batches
        return avg_loss, all_probs, all_labels

    def compute_metrics(self, probs, labels, thresholds=None):
        """Compute per-class AUROC and F1."""
        if thresholds is None:
            thresholds = [0.5] * self.config.NUM_CLASSES

        metrics = {}
        aurocs = []
        f1s = []

        for i, cls in enumerate(self.config.CLASS_NAMES):
            try:
                auc = roc_auc_score(labels[:, i], probs[:, i])
            except ValueError:
                auc = 0.5
            pred = (probs[:, i] >= thresholds[i]).astype(int)
            f1 = f1_score(labels[:, i], pred, zero_division=0)

            metrics[cls] = {"auroc": auc, "f1": f1}
            aurocs.append(auc)
            f1s.append(f1)

        metrics["macro_auroc"] = np.mean(aurocs)
        metrics["macro_f1"] = np.mean(f1s)
        return metrics

    def optimize_thresholds(self, probs, labels):
        """Find optimal per-class thresholds on validation set."""
        thresholds = []
        for i in range(self.config.NUM_CLASSES):
            precision, recall, thresh = precision_recall_curve(
                labels[:, i], probs[:, i]
            )
            f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_vals)
            if best_idx < len(thresh):
                thresholds.append(float(thresh[best_idx]))
            else:
                thresholds.append(0.5)
        return thresholds

    def print_metrics(self, metrics, title=""):
        """Pretty-print metrics table."""
        print(f"\n  {title}")
        print(f"  {'Class':<8} {'AUROC':>8} {'F1':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8}")
        for cls in self.config.CLASS_NAMES:
            m = metrics[cls]
            print(f"  {cls:<8} {m['auroc']:>8.4f} {m['f1']:>8.4f}")
        print(f"  {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Macro':<8} {metrics['macro_auroc']:>8.4f} "
              f"{metrics['macro_f1']:>8.4f}")

    def train(self):
        """Full training loop with early stopping and checkpointing."""
        print(f"\n{'='*65}")
        print(f"  TRAINING START")
        print(f"{'='*65}")
        print(f"  Epochs:     {self.config.MAX_EPOCHS}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  LR:         {self.config.LR}")
        print(f"  Device:     {self.device}")
        print(f"  Train set:  {len(self.train_ds):,}")
        print(f"  Val set:    {len(self.val_ds):,}")
        print(f"  Test set:   {len(self.test_ds):,}")

        # Try to resume
        resumed = self.load_checkpoint()
        if not resumed:
            print(f"  Starting fresh training.")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.MAX_EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_probs, val_labels = self.evaluate(self.val_loader, "Val")
            val_metrics = self.compute_metrics(val_probs, val_labels)
            val_auroc = val_metrics["macro_auroc"]

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auroc"].append(val_auroc)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            improved = ""
            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
                self.patience_counter = 0
                self.optimal_thresholds = self.optimize_thresholds(
                    val_probs, val_labels
                )
                self.save_checkpoint(epoch, is_best=True)
                improved = " ★ BEST"
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)

            print(f"  Epoch {epoch+1:>2}/{self.config.MAX_EPOCHS} │ "
                  f"Train: {train_loss:.4f} │ "
                  f"Val: {val_loss:.4f} │ "
                  f"AUROC: {val_auroc:.4f} │ "
                  f"LR: {current_lr:.6f} │ "
                  f"{epoch_time:.0f}s{improved}")

            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {self.config.PATIENCE} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\n  Training complete in {total_time/60:.1f} minutes")
        print(f"  Best validation AUROC: {self.best_auroc:.4f}")

        # Final evaluation on test set
        self.final_evaluation()

    def final_evaluation(self):
        """Load best model and evaluate on test set."""
        print(f"\n{'='*65}")
        print(f"  FINAL TEST SET EVALUATION")
        print(f"{'='*65}")

        # Load best model
        best_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pt")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device,
                               weights_only=False)
            self.model.load_state_dict(state["model_state"])
            self.optimal_thresholds = state.get("optimal_thresholds",
                                                 [0.5] * 5)
            print(f"  Loaded best model (epoch {state['epoch']+1})")

        # Test evaluation
        test_loss, test_probs, test_labels = self.evaluate(
            self.test_loader, "Test"
        )

        # Default thresholds (0.5)
        metrics_default = self.compute_metrics(test_probs, test_labels)
        self.print_metrics(metrics_default, "Test Results (threshold=0.5)")

        # Optimized thresholds
        metrics_opt = self.compute_metrics(
            test_probs, test_labels, self.optimal_thresholds
        )
        self.print_metrics(metrics_opt, "Test Results (optimized thresholds)")

        print(f"\n  Optimal thresholds per class:")
        for i, cls in enumerate(self.config.CLASS_NAMES):
            print(f"    {cls}: {self.optimal_thresholds[i]:.3f}")

        # Save results
        results = {
            "test_loss": test_loss,
            "default_metrics": {
                k: v if not isinstance(v, dict) else v
                for k, v in metrics_default.items()
            },
            "optimized_metrics": {
                k: v if not isinstance(v, dict) else v
                for k, v in metrics_opt.items()
            },
            "optimal_thresholds": {
                cls: self.optimal_thresholds[i]
                for i, cls in enumerate(self.config.CLASS_NAMES)
            },
            "training_history": self.history,
        }
        results_path = os.path.join(
            self.config.CHECKPOINT_DIR, "test_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PTB-XL Multi-Modal ECG Classifier")
    print("  1D CNN + Demographics MLP + ClinicalBERT + Fusion")
    print("=" * 65)

    config = Config()

    # Verify data is prepared
    required = [
        os.path.join(config.DATA_DIR, "train.csv"),
        os.path.join(config.DATA_DIR, "val.csv"),
        os.path.join(config.DATA_DIR, "test.csv"),
        config.NORM_STATS,
    ]
    for path in required:
        if not os.path.exists(path):
            print(f"\n  ERROR: Missing {path}")
            print(f"  Run 'python prepare_data.py' first!")
            sys.exit(1)

    # Check signal cache
    train_df = pd.read_csv(os.path.join(config.DATA_DIR, "train.csv"))
    sample_id = train_df["ecg_id"].iloc[0]
    if not os.path.exists(os.path.join(config.SIGNAL_CACHE, f"{sample_id}.npy")):
        print(f"\n  ERROR: Signal cache not found.")
        print(f"  Run 'python prepare_data.py' first!")
        sys.exit(1)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
