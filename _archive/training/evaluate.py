"""Quick evaluation — loads best checkpoint and shows per-class metrics."""
import os, json, torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from train import Config, PTBXLDataset, MultiModalFusionModel
from torch.utils.data import DataLoader

config = Config()
with open(config.NORM_STATS) as f:
    norm_stats = json.load(f)

# Load best model
best = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
                  map_location="cpu", weights_only=False)
print(f"Best model from epoch {best['epoch']+1}, AUROC: {best['best_auroc']:.4f}\n")

model = MultiModalFusionModel(config)
model.load_state_dict(best["model_state"])
model.eval()

# Evaluate on val and test
for split in ["val", "test"]:
    ds = PTBXLDataset(os.path.join(config.DATA_DIR, f"{split}.csv"), config, norm_stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for sig, demo, text, labels in loader:
            logits = model(sig, demo, text)
            all_probs.append(torch.sigmoid(logits).numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    print(f"{'='*50}")
    print(f"  {split.upper()} SET ({len(ds):,} records)")
    print(f"{'='*50}")
    print(f"  {'Class':<8} {'AUROC':>8} {'F1@0.5':>8} {'Pos':>6} {'Neg':>6}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    aurocs, f1s = [], []
    for i, cls in enumerate(config.CLASS_NAMES):
        auc = roc_auc_score(labels[:, i], probs[:, i])
        pred = (probs[:, i] >= 0.5).astype(int)
        f1 = f1_score(labels[:, i], pred, zero_division=0)
        pos = int(labels[:, i].sum())
        neg = len(labels) - pos
        aurocs.append(auc)
        f1s.append(f1)
        print(f"  {cls:<8} {auc:>8.4f} {f1:>8.4f} {pos:>6} {neg:>6}")

    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Macro':<8} {np.mean(aurocs):>8.4f} {np.mean(f1s):>8.4f}")
    print()
