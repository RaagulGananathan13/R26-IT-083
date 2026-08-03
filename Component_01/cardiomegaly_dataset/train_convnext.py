"""
Cardiomegaly Binary Classification — ConvNeXt-Base Training Script
==================================================================
Author  : Senior AI Engineer
Purpose : Fine-tune ConvNeXt-Base for Cardiomegaly detection on CXR images.
Dataset : 384x384 grayscale PNGs, ImageFolder layout (positive/negative).
Features: Mixed-precision, cosine-warm LR, resume from checkpoint, best-model
          saving, TensorBoard logging, full test evaluation with metrics.
"""

import os, sys, json, time, random, argparse, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

try:
    import timm
except ImportError:
    print("timm not found. Install: pip install timm")
    sys.exit(1)

# ──────────────────────────── CONFIGURATION ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ConvNeXt-Base Cardiomegaly Trainer")
    # Paths
    p.add_argument("--data_dir", type=str,
                   default="C:/Users/94775/Desktop/Component_1/cardio_image_384")
    p.add_argument("--ckpt_dir", type=str,
                   default="C:/Users/94775/Desktop/Component_1/cardio_image_384/.ckpt")
    # Training hyper-parameters
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--min_lr",       type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs",type=int,   default=3)
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early stopping patience (0=disabled)")
    # Model
    p.add_argument("--model_name", type=str, default="convnext_base.fb_in22k_ft_in1k")
    p.add_argument("--drop_rate",  type=float, default=0.3)
    p.add_argument("--img_size",   type=int,   default=384)
    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    # Workers
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()

# ──────────────────────────── SEED ────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ──────────────────────────── LOGGER ────────────────────────────

def setup_logger(ckpt_dir: str) -> logging.Logger:
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "training.log")
    logger = logging.getLogger("CardioTrainer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ──────────────────────────── DATA ────────────────────────────

def build_transforms(img_size: int):
    """
    Train: strong augmentations tailored for CXR.
    Val/Test: deterministic resize + normalise only.
    Grayscale images are replicated to 3 channels for ConvNeXt.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, eval_tf


def build_loaders(data_dir, img_size, batch_size, num_workers):
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx

# ──────────────────────────── MODEL ────────────────────────────

def build_model(model_name: str, drop_rate: float, num_classes: int = 2):
    """
    Load ConvNeXt-Base pretrained on ImageNet-22k → fine-tuned on IN-1k.
    Replace the classifier head with: LayerNorm → Dropout → Linear(1024→2).
    Freeze stem + first 2 stages, fine-tune stages 2-3 + head.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    in_features = model.num_features  # 1024 for ConvNeXt-Base

    model.head = nn.Identity()  # remove original head

    classifier = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=drop_rate),
        nn.Linear(in_features, num_classes),
    )

    # Freeze stem + stages 0-1 (early feature extractors)
    for name, param in model.named_parameters():
        if any(k in name for k in ["stem", "stages.0.", "stages.1."]):
            param.requires_grad = False

    full_model = nn.Sequential(model, classifier)
    return full_model, in_features


# ──────────────────────────── LR SCHEDULER ────────────────────────────

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup — step-level granularity."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ──────────────────────────── TRAIN / EVAL ────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, logger, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {scheduler.get_lr():.2e}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    total = len(all_labels)
    avg_loss = running_loss / total
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    metrics = {
        "loss":      avg_loss,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc":   roc_auc_score(all_labels, all_probs),
    }
    return metrics, all_labels, all_preds


# ──────────────────────────── CHECKPOINT ────────────────────────────

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_auc"], ckpt.get("history", [])


# ──────────────────────────── MAIN ────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logger(args.ckpt_dir)

    # ── Device ──
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — training will be VERY slow on CPU!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                     f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")

    # ── Data ──
    logger.info("Loading datasets ...")
    train_loader, val_loader, test_loader, class_map = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers)
    logger.info(f"Class mapping: {class_map}")
    logger.info(f"Train: {len(train_loader.dataset)} | "
                f"Val: {len(val_loader.dataset)} | "
                f"Test: {len(test_loader.dataset)}")

    # ── Model ──
    logger.info(f"Building model: {args.model_name}")
    model, in_features = build_model(args.model_name, args.drop_rate, num_classes=2)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters — Total: {total:,} | Trainable: {trainable:,} "
                f"({100*trainable/total:.1f}%)")

    # ── Loss / Optimizer / Scaler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # ── Scheduler ──
    steps_per_epoch = len(train_loader)
    total_steps   = args.epochs * steps_per_epoch
    warmup_steps  = args.warmup_epochs * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, args.min_lr)

    # ── Resume ──
    start_epoch = 1
    best_val_auc = 0.0
    history = []
    no_improve_count = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        start_epoch, best_val_auc, history = load_checkpoint(
            args.resume, model, optimizer, scaler, device)
        start_epoch += 1
        scheduler.current_step = (start_epoch - 1) * steps_per_epoch
        logger.info(f"Resumed at epoch {start_epoch} | Best AUC so far: {best_val_auc:.4f}")

    # ── Paths ──
    best_model_path = os.path.join(args.ckpt_dir, "best_model.pth")
    last_model_path = os.path.join(args.ckpt_dir, "last_checkpoint.pth")
    history_path    = os.path.join(args.ckpt_dir, "history.json")

    # ── Training Loop ──
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, logger, epoch)

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        ep_time = time.time() - ep_start
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 5),
            "val_loss":   round(val_metrics["loss"], 5),
            "val_acc":    round(val_metrics["accuracy"], 5),
            "val_auc":    round(val_metrics["auc_roc"], 5),
            "val_f1":     round(val_metrics["f1"], 5),
            "lr":         round(scheduler.get_lr(), 8),
            "time_sec":   round(ep_time, 1),
        }
        history.append(record)

        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"  Val Loss:  {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}\n"
            f"  Val AUC:   {val_metrics['auc_roc']:.4f} | Val F1:  {val_metrics['f1']:.4f}\n"
            f"  Epoch Time: {ep_time:.1f}s | LR: {scheduler.get_lr():.2e}"
        )

        # Save checkpoint state
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_auc": best_val_auc,
            "history": history,
            "args": vars(args),
        }

        # Best model?
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            ckpt_state["best_val_auc"] = best_val_auc
            save_checkpoint(ckpt_state, best_model_path)
            logger.info(f"  ★ NEW BEST AUC: {best_val_auc:.4f} — saved to {best_model_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"  No improvement for {no_improve_count} epoch(s).")

        # Always save last checkpoint (for resume)
        save_checkpoint(ckpt_state, last_model_path)

        # Save history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if args.patience > 0 and no_improve_count >= args.patience:
            logger.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    total_time = time.time() - t0
    logger.info(f"\nTraining complete in {total_time/60:.1f} minutes.")
    logger.info(f"Best Validation AUC: {best_val_auc:.4f}")

    # ── Test Evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION (using best model)")
    logger.info("=" * 60)

    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device)

    logger.info(f"Test Loss:      {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    logger.info(f"Test AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(test_labels, test_preds, target_names=['Negative','Positive'])}")

    # Save test results
    test_results = {
        "test_metrics": {k: round(v, 5) for k, v in test_metrics.items()},
        "confusion_matrix": cm.tolist(),
        "best_val_auc": round(best_val_auc, 5),
        "total_training_time_min": round(total_time / 60, 2),
    }
    results_path = os.path.join(args.ckpt_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"\nTest results saved to {results_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
