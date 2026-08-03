
# %% Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install transformers rouge-score -q
import torch
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB")
print(f"Free: {(torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated())/1024**3:.1f}GB")

# %% Cell 2: Config
import os, math
DRIVE      = '/content/drive/MyDrive/Component_1'
IMG_DIR    = f'{DRIVE}/cardio_image_384'
CSV_DIR    = f'{DRIVE}'
CKPT_DIR   = f'{DRIVE}/ckpt_report_model'
CLASSIFIER = f'{DRIVE}/ckpt_image_model/best.pth'   # Frozen ConvNeXt weights
os.makedirs(CKPT_DIR, exist_ok=True)

BART_MODEL  = 'facebook/bart-base'
IMG_SIZE    = 384
MAX_LEN     = 512
NUM_VISUAL  = 144    # 12x12 spatial features from ConvNeXt

# T4 GPU optimized: batch=4, accum=4 → effective batch=16
CFG = dict(
    batch_size=4, accum_steps=4, epochs=30,
    lr=5e-5, weight_decay=0.01,
    warmup_epochs=2, patience=7, seed=42, workers=4,
    num_beams=4,
)
RESUME = f'{CKPT_DIR}/last_checkpoint.pth'

# %% Cell 3: Imports
import csv, json, time, random, warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from rouge_score import rouge_scorer
warnings.filterwarnings('ignore')

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_all(CFG['seed'])
device = torch.device('cuda')

# %% Cell 4: Dataset
class CXRReportDataset(Dataset):
    def __init__(self, split, tokenizer, transform=None):
        df = pd.read_csv(f'{CSV_DIR}/cardio_{split}.csv', low_memory=False)
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            did = str(row['dicom_id'])
            cls_dir = 'positive' if int(row['Cardiomegaly']) == 1 else 'negative'
            img_path = f'{IMG_DIR}/{split}/{cls_dir}/{did}.png'
            report = str(row['report_text']).strip()
            if not os.path.exists(img_path) or len(report) < 5 or report == 'nan':
                skipped += 1
                continue
            self.samples.append((img_path, report))

        if skipped:
            print(f"  {split}: skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, report = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Tokenize → labels only (BART handles decoder_input_ids internally)
        enc = self.tokenizer(
            report, max_length=MAX_LEN, truncation=True,
            padding='max_length', return_tensors='pt',
        )
        labels = enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return img, labels

# %% Cell 5: Transforms
def get_transforms(split):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == 'train':
        return transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03),
                                    scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])

# %% Cell 6: Model — ConvNeXt Encoder + BART Decoder
class CXRReportGenerator(nn.Module):
    
    def __init__(self, classifier_path, bart_name=BART_MODEL):
        super().__init__()

        # Vision encoder (frozen ConvNeXt features from classifier)
        self.vision = self._load_vision(classifier_path)
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

        # Projection: 1024 → BART dim (768)
        self.vision_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Loads the pretrained BART model for text generation.
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)  

    def _load_vision(self, path):
        # Load ConvNeXt-Base features from the classifier checkpoint.
        base = models.convnext_base(weights=None)
        features = base.features
        if os.path.exists(str(path)):
            ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
            feat_state = {
                k.replace('features.', ''): v
                for k, v in ckpt['model'].items() if k.startswith('features.')
            }
            features.load_state_dict(feat_state)
            print('  → Vision encoder loaded from classifier checkpoint.')
        else:
            print(f'  Classifier checkpoint not found at {path}, using random weights.')
        return features

    def _encode_vision(self, images):
        with torch.no_grad():
            feats = self.vision(images)              # (B, 1024, 12, 12)
        B = feats.shape[0]
        feats = feats.flatten(2).transpose(1, 2)     # (B, 144, 1024)
        feats = self.vision_proj(feats)               # (B, 144, 768)
        return feats

    def forward(self, images, labels):
        # Training: pass only labels. BART creates decoder_input_ids internally.
        vis = self._encode_vision(images)
        enc_mask = torch.ones(vis.shape[0], NUM_VISUAL,
                              device=vis.device, dtype=torch.long)

        outputs = self.bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=vis),
            attention_mask=enc_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate_report(self, images, tokenizer, fast=False):
        """Generate reports. fast=True uses greedy (for val), False uses beam (for test)."""
        self.eval()
        vis = self._encode_vision(images)
        enc_mask = torch.ones(vis.shape[0], NUM_VISUAL,
                              device=vis.device, dtype=torch.long)

        gen_kwargs = dict(
            encoder_outputs=BaseModelOutput(last_hidden_state=vis),
            attention_mask=enc_mask,
            max_length=200 if fast else MAX_LEN,
            no_repeat_ngram_size=3,
        )
        if fast:
            gen_kwargs['num_beams'] = 1            # greedy (fast)
            gen_kwargs['do_sample'] = False
        else:
            gen_kwargs['num_beams'] = CFG['num_beams']   # beam search (quality)
            gen_kwargs['early_stopping'] = True
            gen_kwargs['length_penalty'] = 1.0

        ids = self.bart.generate(**gen_kwargs)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

# %% Cell 7: ROUGE Evaluation Helper
def compute_rouge(preds, refs):
    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        for k in scores:
            scores[k].append(s[k].fmeasure)
    return {k: np.mean(v) for k, v in scores.items()}

# %% Cell 8: Build Everything
print("Loading BART tokenizer...")
tokenizer = BartTokenizer.from_pretrained(BART_MODEL)

print("Loading datasets...")
train_ds = CXRReportDataset('train', tokenizer, get_transforms('train'))
val_ds   = CXRReportDataset('val',   tokenizer, get_transforms('val'))
test_ds  = CXRReportDataset('test',  tokenizer, get_transforms('test'))
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

train_loader = DataLoader(train_ds, CFG['batch_size'], shuffle=True,
                          num_workers=CFG['workers'], pin_memory=True,
                          drop_last=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, CFG['batch_size'], shuffle=False,
                          num_workers=CFG['workers'], pin_memory=True,
                          persistent_workers=True)
test_loader  = DataLoader(test_ds, CFG['batch_size'], shuffle=False,
                          num_workers=CFG['workers'], pin_memory=True)

print("\nBuilding model...")
model = CXRReportGenerator(CLASSIFIER, BART_MODEL).to(device)
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {train_p/1e6:.1f}M | Frozen: {frozen_p/1e6:.1f}M")

# Differential LR: projection 20x higher so it quickly adapts to BART's space
optimizer = optim.AdamW([
    {'params': model.vision_proj.parameters(), 'lr': CFG['lr'] * 20},  # 1e-3
    {'params': model.bart.parameters(), 'lr': CFG['lr']},              # 5e-5
], weight_decay=CFG['weight_decay'])
print(f"LR: projection={CFG['lr']*20:.1e}, BART={CFG['lr']:.1e}")

steps_per_epoch = len(train_loader) // CFG['accum_steps']
warmup_steps = CFG['warmup_epochs'] * steps_per_epoch
total_steps = CFG['epochs'] * steps_per_epoch

def lr_fn(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
scaler = torch.amp.GradScaler('cuda')

# Resume
start_ep, best_rougeL, hist, noimp = 1, 0.0, [], 0
if RESUME and os.path.isfile(RESUME):
    ck = torch.load(RESUME, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    scaler.load_state_dict(ck['scaler'])
    start_ep = ck['epoch'] + 1
    best_rougeL = ck['best_rougeL']
    hist = ck.get('history', [])
    print(f"Resumed ep {start_ep} | Best ROUGE-L: {best_rougeL:.4f}")

# %% Cell 9: Train & Eval Functions
def train_one_epoch(model, loader, epoch):
    model.train()
    model.vision.eval()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f'  Train E{epoch:02d}', leave=False)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            loss = model(imgs, labels).loss / CFG['accum_steps']

        scaler.scale(loss).backward()

        if (step + 1) % CFG['accum_steps'] == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * CFG['accum_steps']
        n += 1
        pbar.set_postfix_str(f"loss={loss.item()*CFG['accum_steps']:.4f}", refresh=False)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, max_gen=50, fast=True):
    """Evaluate. fast=True uses greedy on 50 samples (val), False uses beam on more (test)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_refs = [], []

    pbar = tqdm(loader, desc='  Eval      ', leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            total_loss += model(imgs, labels).loss.item()
        n += 1

        if len(all_preds) < max_gen:
            preds = model.generate_report(imgs, tokenizer, fast=fast)
            all_preds.extend(preds)
            ref_ids = labels.clone()
            ref_ids[ref_ids == -100] = tokenizer.pad_token_id
            all_refs.extend(tokenizer.batch_decode(ref_ids, skip_special_tokens=True))

    rouge = compute_rouge(all_preds[:max_gen], all_refs[:max_gen])
    return total_loss / max(n, 1), rouge, all_preds[:3], all_refs[:3]

# %% Cell 10: Training Loop
print('='*72)
print('  CXR REPORT GENERATOR — TRAINING (ConvNeXt + BART)')
print('='*72)
t0 = time.time()

for ep in range(start_ep, CFG['epochs'] + 1):
    et0 = time.time()
    train_loss = train_one_epoch(model, train_loader, ep)
    for _ in range(steps_per_epoch):
        scheduler.step()

    val_loss, rouge, spred, sref = evaluate(model, val_loader)
    lr = optimizer.param_groups[0]['lr']
    dt = time.time() - et0
    rL = rouge['rougeL']

    hist.append({
        'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss,
        'rouge1': rouge['rouge1'], 'rouge2': rouge['rouge2'],
        'rougeL': rL, 'lr': lr,
    })

    improved = ''
    if rL > best_rougeL:
        best_rougeL = rL
        noimp = 0
        improved = ' ** BEST'
        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_rougeL': best_rougeL, 'history': hist,
        }, f'{CKPT_DIR}/best.pth')
    else:
        noimp += 1

    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n  E{ep:02d} | TrL={train_loss:.4f} VlL={val_loss:.4f} "
          f"| R1={rouge['rouge1']:.4f} R2={rouge['rouge2']:.4f} RL={rL:.4f} "
          f"| LR={lr:.1e} | {dt:.0f}s | {vram:.1f}GB{improved}")

    if ep % 5 == 0 or improved:
        for i in range(min(2, len(spred))):
            print(f"\n  REF: {sref[i][:200]}")
            print(f"  GEN: {spred[i][:200]}")

    torch.save({
        'epoch': ep, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_rougeL': best_rougeL, 'history': hist,
    }, f'{CKPT_DIR}/last_checkpoint.pth')

    with open(f'{CKPT_DIR}/history.json', 'w') as f:
        json.dump(hist, f, indent=2)

    if CFG['patience'] > 0 and noimp >= CFG['patience']:
        print(f"\n  Early stopping ({CFG['patience']} epochs no improvement)")
        break

total = time.time() - t0
print(f"\n{'='*72}")
print(f"  DONE | {total/60:.1f} min | Best ROUGE-L: {best_rougeL:.4f}")
print(f"{'='*72}")

# %% Cell 11: Test Evaluation
print("\n  Loading best model for TEST...")
bc = torch.load(f'{CKPT_DIR}/best.pth', map_location=device, weights_only=False)
model.load_state_dict(bc['model'])
torch.cuda.empty_cache()

test_loader_small = DataLoader(test_ds, batch_size=2, shuffle=False,
                               num_workers=0, pin_memory=True)
_, tr, tp, tf = evaluate(model, test_loader_small, max_gen=100, fast=False)

print(f"\n{'='*72}")
print(f"  TEST RESULTS (epoch {bc.get('epoch','?')})")
print(f"{'='*72}")
print(f"  ROUGE-1:  {tr['rouge1']:.4f}")
print(f"  ROUGE-2:  {tr['rouge2']:.4f}")
print(f"  ROUGE-L:  {tr['rougeL']:.4f}")
print(f"  Target 30%+: {'ACHIEVED' if tr['rougeL']>=0.30 else 'NOT MET'} ({tr['rougeL']*100:.1f}%)")

for i in range(min(3, len(tp))):
    print(f"\n  [{i+1}] REF: {tf[i][:250]}")
    print(f"  [{i+1}] GEN: {tp[i][:250]}")

# %% Cell 12: Sample Reports
print('\n' + '='*72)
print('  SAMPLE GENERATED REPORTS')
print('='*72)
for i in range(min(5, len(test_ds))):
    img, labels = test_ds[i]
    ref_ids = labels.clone()
    ref_ids[ref_ids == -100] = tokenizer.pad_token_id
    gt_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    gen_text = model.generate_report(img.unsqueeze(0).to(device), tokenizer, fast=False)
    if isinstance(gen_text, list):
        gen_text = gen_text[0]
    print(f"\n--- Sample {i+1} ---")
    print(f"Ground Truth: {gt_text[:300]}")
    print(f"Generated:    {gen_text[:300]}")

# %% Cell 13: Training Curves
import matplotlib.pyplot as plt
eps = [r['epoch'] for r in hist]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(eps, [r['train_loss'] for r in hist], 'b-o', label='Train', markersize=4)
ax1.plot(eps, [r['val_loss'] for r in hist], 'r-o', label='Val', markersize=4)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Report Generation Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(eps, [r['rouge1'] for r in hist], 'g-o', label='ROUGE-1', markersize=4)
ax2.plot(eps, [r['rouge2'] for r in hist], 'm-o', label='ROUGE-2', markersize=4)
ax2.plot(eps, [r['rougeL'] for r in hist], 'b-o', label='ROUGE-L', markersize=4)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score'); ax2.set_title('ROUGE Scores')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/training_curves.png', dpi=150)
plt.show()
print(f"\nSaved to: {CKPT_DIR}")
