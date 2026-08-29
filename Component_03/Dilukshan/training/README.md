# UEF-Net — Training Pipeline (Step 2)

**Uncertainty-aware Ordinal Ejection-Fraction Network** for imbalanced 4-class EF
grading + EF regression on EchoNet-Dynamic.

Goal: **≥75% accuracy (recall) on every severity class** *and* the lowest possible
**MAE** — without over/under-fitting.

| Class | EF range | Train | Test |
|------:|:--------:|:----:|:----:|
| Severe | <30 | 460 | 83 |
| Moderate | 30–40 | 488 | 77 |
| Mild | 40–55 | 1333 | 241 |
| Normal | ≥55 | 5184 | 876 |

---

## Novelty (what to tell your supervisor)

UEF-Net is a **single integrated system** whose contribution is the combination —
each part is motivated by a specific failure mode of naive EF classification:

1. **Measurement-uncertainty soft labels (core idea).** EF is itself a noisy
   measurement (inter-observer σ≈4 EF pts). Instead of hard class labels we set
   each ordinal target to `P(trueEF > tₖ | measuredEF) = 1−Φ((tₖ−EF)/σ)`. Boundary
   cases (e.g. EF=39 vs 41) get honest *soft* supervision instead of an arbitrary
   hard flip. *(losses/losses.py)*
2. **Ordinal (CORAL) head** — rank-consistent cumulative logits respect the natural
   order Severe<Moderate<Mild<Normal, which plain softmax ignores. *(models/uef_net.py)*
3. **Dual-head consistency** — a regression head (for MAE) and the ordinal head are
   tied by a loss forcing them to agree on `P(y>tₖ)`; each regularises the other.
4. **Cardiac-cycle-aware, motion-augmented clips** — from preprocessing: every clip
   contains a full ED→ES contraction, plus an explicit motion channel.
5. **Distribution-aware training** — class-balanced sampling (+ optional DRW) for the
   classifier and **LDS** (label-distribution-smoothing) weights for imbalanced
   regression, so minority EF ranges are not ignored.
6. **Min-recall threshold calibration** — the decision thresholds are optimised on
   *validation* to maximise the *worst* class's recall, directly targeting the metric.
7. **Multi-clip test-time augmentation** — per-video predictions average several
   cycle-aware clips (lower variance, better MAE).

Honest framing: individual ingredients build on published work (CORAL, LDS, DRW,
effective-number weighting, EchoNet's R(2+1)D). The **integration** — uncertainty
soft-labels + ordinal + dual-head consistency + min-recall calibration for EF
grading — is the novel contribution. Do **not** claim "no one ever did this" as fact;
claim a *novel integrated method with a new soft-label formulation*, which is defensible.

---

## Feasibility (read this before promising 75%)

A simulation (see project analysis) shows **75% on all four classes with the clinical
boundaries 30/40/55 is at the frontier**, and **Moderate (30–40) is the binding
constraint** (narrowest band, only 77 test cases). It needs a strong regressor
(**MAE ≈ 3.0–3.5**, video-level) *plus* threshold calibration. The dedicated balanced
ordinal head is designed to beat naive regression-binning at the boundaries. If
Moderate falls short, the levers are: more epochs, stronger balancing (`--drw-epoch`,
sampler), larger `ef_noise_sigma`, `w_consistency`, or `clip_len 48/64`. Because the
minority test classes are tiny, per-class recall is inherently noisy (±5%).

---

## Layout
```
training/
├── config.py            # all hyper-parameters + paths to preprocessing artifacts
├── run_train.py         # ENTRY: train
├── run_eval.py          # ENTRY: final TEST evaluation (TTA + frozen thresholds)
├── core/common.py       # seed, device, checkpoint, CSV logger
├── data/
│   ├── dataset.py       # clip dataset (reuses preprocessing sampling code)
│   └── sampler.py       # class-balanced / DRW samplers
├── models/uef_net.py    # R(2+1)D backbone + regression + CORAL heads
├── losses/
│   ├── losses.py        # uncertainty soft-labels + consistency + LDS-weighted Huber
│   └── lds.py           # label-distribution-smoothing weights
├── engine/
│   ├── metrics.py       # per-class recall, balanced acc, MAE, confusion
│   ├── trainer.py       # AMP + grad-accum + warmup/cosine + DRW + early stop
│   ├── evaluate.py      # multi-clip TTA inference
│   └── calibrate.py     # min-recall threshold optimisation
└── outputs/<run>/       # best.pt, last.pt, train_log.csv, thresholds.json, test_report.json
```

## How to run (from the `training/` folder)

```bash
# 0) wiring smoke test (~1–2 min, random weights, no download) — already verified
python run_train.py --smoke --no-pretrained --run-name smoke

# 1) FULL training (downloads Kinetics weights once). ~4–7 h for 45 epochs on RTX 4060.
python run_train.py

# 1b) RESUME an interrupted / stopped run (continues from last.pt: optimizer, epoch,
#     best score, patience and RNG are all restored)
python run_train.py --resume

# 2) FINAL evaluation on the TEST split (multi-clip TTA + val-frozen thresholds)
python run_eval.py --run-name uefnet_r2p1d
```

### Maximum-performance features (enabled by default)
- **Weight EMA** (Polyak averaging) — validates/checkpoints the averaged weights;
  smooths the noisy min-recall. Toggle with `--no-ema`, tune `--ema-decay`.
- **Bias-corrected calibration** — a variance-matching expansion
  `EF' = target_mean + k·(EF − pred_mean)` (fit on VAL, frozen for TEST) undoes the
  regression-to-the-mean compression; measured to lift min-recall and lower MAE together.
  Chosen automatically if it wins on VAL (strategy `reg_expanded`, params stored in
  `thresholds.json`).
- **TF32 + AMP + grad-accumulation** for throughput on the RTX 4060.
- `--patience N` early-stop patience; `--drw-epoch E` for the DRW schedule.

Apply the bias-correction to an already-trained model WITHOUT retraining:
```bash
python run_train.py --run-name uefnet_r2p1d --calibrate-only
python run_eval.py  --run-name uefnet_r2p1d --n-tta 10
```

### Resume & progress
- **Progress bars** (tqdm) show live loss, learning rate, it/s and ETA for every train
  epoch, validation pass, final calibration and test evaluation.
- **Resume** is crash/Ctrl-C safe: `last.pt` is written every epoch with the full training
  state, and Ctrl-C is caught so the current `last.pt` is preserved. Re-run with `--resume`.
- Every epoch also refreshes `outputs/<run>/training_curves.png` so you can watch progress.

### Outputs written to `outputs/<run>/`
`best.pt`, `last.pt`, `config.json` (exact run settings for the report), `train_log.csv`,
`thresholds.json` (TTA-calibrated), `training_curves.png`, and after eval
`test_report.json` + `confusion_test.png`.

### 8 GB VRAM notes
Defaults (`batch 8 × grad-accum 4`, `clip 32×period 2`, AMP on) are tuned for the
RTX 4060 Laptop. If you hit CUDA OOM, lower `--batch-size 6` (raise `--grad-accum` to
keep effective batch ≈32). To push accuracy further with more VRAM headroom, try
`--clip-len 48`. Monitor `outputs/<run>/train_log.csv` for the train/val gap
(overfitting guard); early stopping on val min-recall is on by default.

### What "done" looks like
`run_eval.py` prints a per-class accuracy table with an `OK/<75` flag per class, the
MAE/RMSE/R², and the confusion matrix, and writes `test_report.json`.
