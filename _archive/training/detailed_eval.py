"""Prints exact Correct/Wrong counts for the multi-modal model."""
import os, json, torch, numpy as np
from train import Config, PTBXLDataset, MultiModalFusionModel
from torch.utils.data import DataLoader

config = Config()
with open(config.NORM_STATS) as f:
    norm_stats = json.load(f)

# Load best model and thresholds
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
        logits = model(sig, demo, text)
        all_probs.append(torch.sigmoid(logits).numpy())
        all_labels.append(labels.numpy())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

print("\n" + "="*70)
print(f"  MULTI-MODAL EXACT PREDICTION COUNTS (TEST SET: {len(ds)} patients)")
print("="*70)
print("  Class    | True Pos | True Neg | False Pos | False Neg | Accuracy")
print("  " + "-"*68)

for i, cls in enumerate(config.CLASS_NAMES):
    pred = (probs[:, i] >= thresholds[i]).astype(int)
    true = labels[:, i].astype(int)
    
    tp = np.sum((pred == 1) & (true == 1))  # Correctly predicted YES
    tn = np.sum((pred == 0) & (true == 0))  # Correctly predicted NO
    fp = np.sum((pred == 1) & (true == 0))  # Wrongly predicted YES (False Alarm)
    fn = np.sum((pred == 0) & (true == 1))  # Wrongly predicted NO (Missed it)
    
    acc = (tp + tn) / len(ds) * 100
    
    print(f"  {cls:<8} | {tp:>8} | {tn:>8} | {fp:>9} | {fn:>9} | {acc:>7.1f}%")

print("  " + "-"*68)
print("  True Pos  = Patient HAD the disease, AI correctly found it.")
print("  True Neg  = Patient DID NOT have the disease, AI correctly ignored it.")
print("  False Pos = WRONG (AI said YES, but patient was healthy).")
print("  False Neg = WRONG (AI said NO, but patient was sick).")
print("="*70 + "\n")
