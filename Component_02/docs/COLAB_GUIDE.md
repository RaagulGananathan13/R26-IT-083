# Colab L4 — Exact Steps, Zero Wasted Compute Units

Read this once end to end before you touch Colab. Every rule here exists because
breaking it costs CU.

---

## The single most important decision

**Do the slow, boring part on your laptop. Do only the GPU part on Colab.**

Your `signals_cache/` is **17,221 separate files, 4 GB**. Uploading that to Drive
takes hours, and Colab reading 17,221 small files *from Drive* is so slow it would
dominate training time — you would pay GPU rates to wait on network I/O.

Instead you run the packer locally (CPU only, ~12 min, already running), which turns
all 17,221 files into **3 big files, 2.1 GB total**. You upload those. Colab copies
them to local disk in about a minute and never touches Drive again.

| Approach | Upload | Colab I/O | GPU time wasted |
|---|---|---|---|
| Upload `signals_cache/` (17,221 files) | 3–6 hours | very slow, per-epoch | **30–60 min** |
| **Upload the 3 packed files (recommended)** | 25–45 min | one 1-min copy | **~0 min** |

---

## Step 1 — On your laptop (already done for you)

```bash
python -X utf8 Component_02/train/train_gpu.py --pack
```

This writes into `Component_02/data/`:

| File | Size |
|---|---|
| `train_X.npy` | 1.66 GB |
| `val_X.npy` | 205 MB |
| `test_X.npy` | 205 MB |
| `train_Y.npy`, `val_Y.npy`, `test_Y.npy` | ~1 MB total |
| `train_ids.npy`, `val_ids.npy`, `test_ids.npy` | tiny |
| `train.done`, `val.done`, `test.done` | tiny (verification markers) |

The band-pass filter and normalisation are **already applied**. Colab does no
preprocessing at all.

---

## Step 2 — Google Drive structure

You said you made a folder called `Component_02` in your Drive. Use **exactly** this
layout — the notebook expects it:

```
MyDrive/
└── Component_02/                        ← your existing folder
    ├── data/                            ← 2.1 GB, the packed arrays
    │   ├── train_X.npy
    │   ├── train_Y.npy
    │   ├── train_ids.npy
    │   ├── train.done
    │   ├── val_X.npy
    │   ├── val_Y.npy
    │   ├── val_ids.npy
    │   ├── val.done
    │   ├── test_X.npy
    │   ├── test_Y.npy
    │   ├── test_ids.npy
    │   └── test.done
    │
    ├── src/                             ← ~90 KB, the code
    │   ├── __init__.py
    │   ├── models.py
    │   ├── preprocess.py
    │   ├── quality.py
    │   ├── calibration.py
    │   ├── conformal.py
    │   ├── report.py
    │   ├── verify.py
    │   ├── xai.py
    │   └── pipeline.py
    │
    ├── train/                           ← ~60 KB
    │   ├── preflight.py
    │   ├── train_gpu.py
    │   ├── fit_calibration.py
    │   └── Component02_Colab.ipynb      ← open THIS in Colab
    │
    ├── audit/
    │   └── 08_verify_fixes.py           ← optional, for the final check
    │
    └── csv/                             ← ~3 MB, needed by fit_calibration.py
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── norm_stats.json
```

### What NOT to upload

| Do not upload | Why |
|---|---|
| `_archive/data/signals_cache/` | 4 GB / 17,221 files — already inside the packed arrays |
| `_archive/data/raw_signals/` | 20 GB of `.dat`/`.hea` — not needed |
| `_archive/checkpoints_report_gen/` | 1.3 GB of a component that was deleted |
| `_archive/checkpoints*/` | the old model; you are training a new one |
| `__pycache__/` | regenerated automatically |
| `Component_02/audit/results/` | generated on your laptop |

The `csv/` folder is a copy of four small files from `_archive/data/`. The notebook
puts them where the scripts expect them.

**Upload tip:** zip `data/` before uploading (`train_X.npy` etc. compress poorly, but
one 2.1 GB upload is far more reliable than twelve). Drive's web uploader resumes
badly; the desktop Google Drive app is more reliable for a file this size.

---

## Step 3 — In Colab

Open `train/Component02_Colab.ipynb` from Drive. Run the cells in order.

### Rule 1 — Preflight before anything else

```
!python Component_02/train/preflight.py --batch 128
```

30 seconds. It checks: GPU attached, VRAM free, packed data present and *not corrupt*,
disk space, and runs a **real forward+backward pass** at your chosen batch size to
confirm it fits. It prints an estimated minutes-per-epoch.

**If preflight says FAIL, do not run the training cell.** That is the whole point.

### Rule 2 — Attach the GPU *before* you start, not after

Runtime → Change runtime type → **L4 GPU**. Changing it later restarts the session and
you lose the copied data.

### Rule 3 — Copy data to local disk, never train off Drive

The notebook does this. `/content/` is fast local SSD; `/content/drive/` is network.

### Rule 4 — One run. No sweeps.

```
!python Component_02/train/train_gpu.py --epochs 40 --batch 128 --max-minutes 60
```

The hyper-parameters are already chosen. A sweep on this task buys you noise and costs
you your month's units.

### Rule 5 — If it disconnects, resume. Do not restart.

```
!python Component_02/train/train_gpu.py --resume
```

A checkpoint is written every epoch with optimizer, scheduler, EMA and RNG state. A
disconnect at epoch 30 costs you one epoch, not thirty.

---

## What the script does to protect your compute units

| Protection | What it prevents |
|---|---|
| `--resume` | a Colab disconnect costing the whole run |
| automatic OOM recovery | the run dying at minute 38; batch halves and the epoch restarts |
| `--max-minutes 60` | an overrunning job being killed mid-write |
| atomic checkpoint writes | a corrupt `best_model.pt` after a hard disconnect |
| NaN/divergence guard | 35 more epochs of a run that already broke |
| missing-file guard in the packer | training on a partial dataset (the archive's E-4 bug) |
| per-worker RNG seeding | augmentation silently becoming a no-op after epoch 1 |
| `preflight.py` | every setup mistake above, for 30 seconds instead of an hour |

---

## Expected timings on L4

| Step | Time |
|---|---|
| Mount Drive + copy 2.1 GB to local | 1–2 min |
| Preflight | 30 s |
| Training, 40 epochs @ batch 128 | 30–45 min |
| Calibration + conformal fit | 1–2 min |
| Verification suite | 2 min |
| **Total** | **~50 min, one session** |

If preflight estimates more than ~1.5 min/epoch, something is wrong — most likely the
data is still on Drive rather than local disk. Stop and check.

---

## If you hit OOM anyway

The script halves the batch and continues, but you lose those minutes. To avoid it
entirely, run preflight with the batch you intend to use — it does a real
forward+backward and tells you the peak VRAM. On a 24 GB L4, batch 128 has large
headroom; batch 256 also fits and is ~15% faster if preflight confirms it.

If you are given a **T4** instead of an L4 (16 GB, no bf16), preflight will say so.
Use `--batch 64`. It will take roughly twice as long.

---

## Step 4 — After training

```
!python Component_02/train/fit_calibration.py \
    --model resnet_se --ckpt Component_02/checkpoints/best_model.pt --filter
!python Component_02/audit/08_verify_fixes.py | tail -40
```

Then download three small files back to your laptop:

* `checkpoints/best_model.pt` (~10 MB)
* `checkpoints/calibrator.json`
* `checkpoints/conformal_triage.json`

Drop them into `Component_02/checkpoints/` locally and run:

```bash
ECG_MODEL=resnet_se ECG_FILTER=1 \
ECG_CKPT=Component_02/checkpoints/best_model.pt \
python -X utf8 Component_02/app/app.py
```

---

## Honest expectation

Target: **0.9297 → 0.935–0.945** macro-AUROC. Published PTB-XL results cluster at
0.92–0.94. If you land at 0.938, that is a real, defensible improvement. If you land at
0.931, say so — the conformal contribution does not depend on beating the baseline, and
a student who reports an honest null result is doing better science than one who
tunes on the test set until the number moves.

**You do not need this run to present.** Everything in `RESEARCH_CONTRIBUTION.md`
already works on the existing checkpoint. The retrain is an improvement, not a
dependency.
