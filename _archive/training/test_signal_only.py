"""Tests the existing Multi-Modal model using ONLY ECG signals (zeroing out text and demographics)."""
import os, json, torch, numpy as np
from train import Config, PTBXLDataset, MultiModalFusionModel
from torch.utils.data import DataLoader

config = Config()
with open(config.NORM_STATS) as f:
    norm_stats = json.load(f)

# Load the exact model you already trained
best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
state = torch.load(best_path, map_location="cpu", weights_only=False)
thresholds = state["optimal_thresholds"]

model = MultiModalFusionModel(config)
model.load_state_dict(state["model_state"])
model.eval()

ds = PTBXLDataset(os.path.join(config.DATA_DIR, "test.csv"), config, norm_stats)
loader = DataLoader(ds, batch_size=64, shuffle=False)

all_probs, all_labels = [], []
with torch.no_grad():
    for sig, demo, text, labels in loader:
        # ZERO OUT DEMOGRAPHICS AND TEXT SO IT ONLY USES THE SIGNAL
        zero_demo = torch.zeros_like(demo)
        zero_text = torch.zeros_like(text)
        
        logits = model(sig, zero_demo, zero_text)
        all_probs.append(torch.sigmoid(logits).numpy())
        all_labels.append(labels.numpy())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

print("\n" + "="*70)
print(f"  BLIND TEST: EXISTING MODEL USING **ONLY ECG SIGNAL**")
print("="*70)
print("  Class    | True Pos | True Neg | False Pos | False Neg | Accuracy")
print("  " + "-"*68)

for i, cls in enumerate(config.CLASS_NAMES):
    pred = (probs[:, i] >= thresholds[i]).astype(int)
    true = labels[:, i].astype(int)
    
    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    acc = (tp + tn) / len(ds) * 100
    
    print(f"  {cls:<8} | {tp:>8} | {tn:>8} | {fp:>9} | {fn:>9} | {acc:>7.1f}%")
print("="*70 + "\n")
