python ecg_fetch.py --tier 1 --workers 32
# Component_04 — Clean-Room PTB-XL Rebuild

## Context

Component_02 ships a working PTB-XL 5-superclass ECG classifier, but the audit in
[AUDIT_FINDINGS.md](Component_02/AUDIT_FINDINGS.md) and this session's re-derivation exposed problems
that cannot be patched in place:

1. **Two different models are reported as one system.** The published metrics (macro-AUROC 0.9297,
   macro-F1 0.7172) come from `_archive/checkpoints_ecg_only/best_model.pt` (1.02M params, epoch 7),
   but the conformal/calibration results come from `Component_02/checkpoints/best_model.pt`
   (1.59M params, epoch 13). Their test logits correlate only r=0.880.
2. **A real preprocessing bug.** `norm_stats.json` was computed on *unfiltered* mV
   (`_archive/training/prepare_data.py:210-245`, 1,000-record sample) but is applied *after*
   band-pass filtering ([train_gpu.py:118-121](Component_02/train/train_gpu.py#L118-L121)). The
   network is normalised by statistics that do not describe its own input distribution.
3. **Per-class results are weak where it matters.** HYP F1 0.512, MI F1 0.708. HYP per-class
   *accuracy* (0.9041) is actually **below** the always-negative baseline (0.9229).
4. **Single seed.** Only `seed0` exists, so reported bootstrap CIs capture test-set sampling noise
   only — not training variance. Improvements of ~0.005 are unfalsifiable.

Component_04 is a clean-room rebuild that fixes all four, maximises per-class performance honestly,
and produces the panel-facing contribution/novelty/gap documentation for Progress Presentation 2.

**Decisions taken (confirmed with user):** ECG rebuild (not the group's ACS/MIMIC C4 spec) ·
maximise raw performance and report honestly, no selective-prediction framing · train on Colab GPU
at 500 Hz · keep the existing 17,221-record subset and document the comparability caveat.

### Honest target statement — read this before building

A 0.75 F1 floor on **every** class is not attainable at full coverage. Published PTB-XL SOTA for HYP
is ~0.55–0.65 F1. The realistic outcome of this plan:

| Class | C2 now | C4 expected | F1 ≥ 0.75? |
|---|---|---|---|
| NORM | 0.869 | 0.88–0.89 | yes |
| MI | 0.708 | 0.73–0.76 | borderline |
| STTC | 0.765 | 0.78–0.80 | yes |
| CD | 0.775 | 0.79–0.81 | yes |
| HYP | 0.512 | 0.56–0.63 | **no** |
| macro | 0.726 | 0.76–0.79 | — |

What **is** deliverable for all five classes at full coverage: **accuracy ≥ 0.75** (already met),
**balanced accuracy ≥ 0.75** (already met), and **recall ≥ 0.75** via the constrained threshold
policy in `thresholds.py`. The writeup states the F1 position plainly, with bootstrap CIs and the
label-quality evidence that explains HYP's ceiling. Do not fabricate a 0.75 HYP F1.

---

## Folder structure

```
Component_04/
├── README.md                    # what it is, how to run, headline results
├── RESEARCH_CONTRIBUTION.md     # panel answers: gap, novelty, contribution
├── PRESENTATION_NOTES.md        # slide-by-slide script + expected questions
├── requirements.txt
├── configs/
│   ├── base.yaml                # shared: data paths, preprocessing, eval
│   ├── seresnet.yaml            # arch A
│   ├── inception.yaml           # arch B
│   └── xresnet.yaml             # arch C
├── src/
│   ├── __init__.py
│   ├── paths.py                 # COPY VERBATIM from Component_02/src/paths.py
│   ├── config.py                # NEW: dataclass config, YAML load, hash for provenance
│   ├── preprocess.py            # REBUILT (see below)
│   ├── dataset.py               # NEW: packing + PackedECG Dataset + loaders
│   ├── models.py                # REBUILT: 3 architectures
│   ├── losses.py                # NEW: AsymmetricLoss + FocalLoss + BCE
│   ├── metrics.py               # NEW: complete per-class metric suite
│   ├── thresholds.py            # NEW: F1-optimal + recall-constrained policies
│   ├── ensemble.py              # NEW: multi-seed/arch averaging + TTA
│   └── calibration.py           # COPY from Component_02/src/calibration.py
├── scripts/
│   ├── 00_env_check.py
│   ├── 01_dataset_analysis.py   # the "deep dataset check"
│   ├── 02_build_norm_stats.py   # THE preprocessing fix
│   ├── 03_pack.py
│   ├── 04_train.py
│   ├── 05_predict.py            # logits -> artifacts/
│   ├── 06_evaluate.py           # the full per-class report
│   └── 07_ablation_report.py    # generates all result tables
├── notebooks/
│   └── Component04_Colab.ipynb
├── artifacts/                   # gitignored: checkpoints, packed npy, logits
└── results/                     # committed: tables (.md/.json), figures (.png)
```

---

## Phase 1 — Preprocessing rebuilt from scratch

`src/preprocess.py` keeps the **correct** parts of
[Component_02/src/preprocess.py](Component_02/src/preprocess.py) — `resample_to` (rate-based, never
resample-to-length), `bandpass` (Butterworth HP-3 @0.5 Hz, LP-4 @40 Hz, `iirnotch` Q=30 @50 Hz, all
zero-phase `filtfilt`), and `center_or_pad`. Reuse these signatures directly; they already fix
audit E-5.

Seven defects to fix, each with a ledger entry in `results/preprocessing_fixes.md`:

| # | Defect | Fix |
|---|---|---|
| P1 | Norm stats computed pre-filter, applied post-filter | Recompute per-lead mean/std **after** the exact filter chain |
| P2 | Stats from 1,000-record sample (`random_state=42`) | Stream all 13,801 train records (Welford, float64 accumulator) |
| P3 | `per_record` median coupled to `do_filter` (train) but always on (serve) | Decouple into `PreprocessConfig`; one JSON drives train + serve |
| P4 | `pack()` skips `resample_to`/`center_or_pad`, assumes (5000,12) | Route packing through the same `prepare()` as serving; assert shape/fs |
| P5 | `age == 300` sentinel on 255 records | Clip to 90 + emit `age_capped` flag; report cleaned stats (59.46/16.95 vs stored 62.54/32.45) |
| P6 | Height/weight constant-imputed, undocumented | Restore NaN via `height_missing`/`weight_missing`; document in dataset analysis |
| P7 | Global z-score only | Add per-lead robust (median/IQR) scaling as an **ablation arm, default OFF** |

**P7 default-OFF is deliberate and must be justified in the writeup:** the audit showed a ×10 gain
error drives HYP to 0.998, so per-record amplitude rescaling destroys the voltage evidence HYP
depends on. Ablate it, report the number, keep it off.

**Fast path for re-normalisation (saves 30–45 min).** Existing packed arrays store
`(f − median − μ_old)/σ_old`. Since median removal is per-record and precedes normalisation, the
median-removed filtered signal is exactly recoverable as `z = X·σ_old + μ_old`. So new statistics
can be computed, and new packs produced, from `Component_02/data/*_X.npy` by a per-lead affine —
no re-filtering. `02_build_norm_stats.py` implements **both** paths (`--source packed` and
`--source cache`, the latter reading `_archive/data/signals_cache/{ecg_id}.npy`) and asserts they
agree to float16 tolerance. Default to `cache` for the committed artifact; `packed` is the shortcut.

Augmentation (`augment()`): keep the existing five (Gaussian noise σ=0.05 p=0.5; amplitude scale
0.9–1.1 p=0.5 — **keep narrow, same reason as P7**; circular shift ±250 p=0.3; lead dropout 1–2
leads p=0.25; sinusoidal baseline wander p=0.2). Add two, config-gated: **random time masking**
(zero a 0.2–0.8 s window, p=0.3) and **multi-label mixup** (α=0.2, p=0.5, targets mixed by the same
λ). Both are ablation arms.

**Output of this phase:** `Component_04/artifacts/norm_stats_v4.json` + `preprocess_config.json`
(hashed, embedded in every checkpoint for provenance).

---

## Phase 2 — Dataset deep analysis

`scripts/01_dataset_analysis.py` reads `Component_02/csv/*.csv` and the packed arrays. Emits
`results/dataset_analysis.md` + `.json`:

- Split integrity: patient-disjointness, `strat_fold` distribution (train 1–8 / val 9 / test 10),
  ECG-ID overlap. Expect 0 leakage — re-verify, don't assume.
- Per-class prevalence per split and the drift between them
  (train `[.408,.176,.265,.276,.087]` vs test `[.413,.157,.267,.282,.077]`).
- Label cardinality (1×14,251 / 2×2,413 / 3×487 / 4×70), **0 NORM+abnormal rows** — the structural
  constraint the model must respect.
- Co-occurrence matrix between the 5 classes.
- Signal integrity: NaN/Inf, flat leads, amplitude p50/p99/max, records >10 mV.
- Demographics: the age-300 sentinel, imputation constants, missingness rates.
- **The `likelihood == 100` filter**: 17,221 of 21,799 (21% dropped). Quantify what this does to
  comparability with Strodthoff et al. State it as a limitation; do not silently benchmark against
  published numbers.
- Per-class positive counts in test (NORM 707, MI 268, STTC 456, CD 483, **HYP 132**) — HYP's 132
  positives are the direct cause of its ±0.07 F1 confidence interval.

---

## Phase 3 — Models and training

`src/models.py` — three architectures for a diverse ensemble. Reuse `SEBlock`,
`SEResidualBlock`, `MultiKernelStem`, `AttentionPool` from
[Component_02/src/models.py](Component_02/src/models.py) (they are already parametric and correct;
fix the wrong `~2.4M params` docstring at `models.py:168` — the real trainable count is 1,584,326):

- **A `SEResNet1D`** — C2's backbone, deepened to 5 stages with stochastic depth.
- **B `InceptionTime1D`** — parallel multi-scale kernels; different inductive bias, strong on PTB-XL.
- **C `XResNet1D`** — the Strodthoff benchmark leader's structure.

`src/losses.py` — **AsymmetricLoss** (Ridnik et al. 2021: `γ+=0`, `γ−=4`, `clip=0.05`) as the
primary loss, purpose-built for multi-label imbalance; `FocalLoss` (γ=2) and plain `BCE` as
ablation arms. **Hard rule carried over from audit C-6: never combine class weighting with
`WeightedRandomSampler`.** The loss/sampler choice is XOR-enforced in config validation and the run
aborts if both are set.

Training (`scripts/04_train.py`), adapted from
[train_gpu.py](Component_02/train/train_gpu.py) — keep its genuinely good machinery: atomic
checkpoint save, `--resume`, OOM auto-halving, NaN guard, EMA (decay 0.999), `--max-minutes` budget.
Two changes that matter:

- **Fix the hardcoded `DATA = ROOT/_archive/data`** (`train_gpu.py:59-60`) which bypasses
  `paths.py`. All paths go through `src/paths.py` + config.
- **Select on macro-F1 at val-tuned thresholds, not macro-AUROC.** C2 selected on AUROC, which does
  not optimise the metric being reported. This alone is worth ~0.005–0.01 macro-F1.

Recipe: AdamW (lr 3e-3, wd 1e-2), OneCycleLR (`pct_start=0.25`, `div_factor=20`), batch 128,
bf16 AMP, grad clip 1.0, 50 epochs, label smoothing 0.02.

Runs: **3 architectures × 3 seeds = 9**, ~40 min each on an L4. If time is short, seeds 0–2 of
arch A alone still produce a valid variance-aware result; the ensemble rows fill in as runs land.

---

## Phase 4 — Inference-time gains

`src/thresholds.py` — two policies, both fitted **on val fold 9 only**:
- `f1_optimal`: per-class threshold maximising F1 (C2 already does this; keep it).
- `recall_constrained`: maximise precision subject to **recall ≥ 0.75** per class. This is what
  delivers the "every class ≥75% recall" line honestly.

Report the val-fitted vs test-oracle optimism gap for both (C2's was +0.0106; expect similar).

`src/ensemble.py`:
- **Multi-seed + multi-arch logit averaging** — the single most reliable gain (+0.01–0.02 macro-F1).
- **TTA**: average logits over circular shifts of {−250, 0, +250} samples. Cheap, ~+0.003.
- **NORM mutual exclusion**: ground truth has 0 NORM+abnormal rows; suppress NORM when any
  abnormal class fires. Measured this session on C2: exact-match **0.6213 → 0.6435 (+2.2 pts)** for
  a 1.3-pt NORM F1 cost. Free, and it makes predictions structurally valid.

---

## Phase 5 — Evaluation protocol

`src/metrics.py` + `scripts/06_evaluate.py` produce, **per class**, at both threshold policies:
accuracy, balanced accuracy, precision, recall, specificity, F1, AUROC, AUPRC, MCC, and the full
confusion count. Plus macro / micro / weighted aggregates, and set-level exact-match, Hamming
accuracy, Jaccard, empty-prediction count, NORM+abnormal count.

Two variance sources reported **separately** — this is the core methodological fix:
- **Test-set sampling variance**: bootstrap 2,000 resamples → 95% CI per metric per class.
- **Training variance**: mean ± std across seeds.

Mandatory context columns that make the numbers honest:
- The **always-negative baseline accuracy** next to every per-class accuracy (this is what exposes
  HYP: 0.9041 model vs 0.9229 trivial).
- The **always-{NORM} baseline** (0.4132) next to exact-match.

`scripts/07_ablation_report.py` emits the ablation table: each intervention (norm-stats fix,
loss choice, sampler, mixup, time masking, robust scaling, architecture, ensemble size, TTA,
threshold policy, NORM exclusion) with its Δmacro-F1, ΔHYP-F1, and whether the delta exceeds the
seed std. **Interventions inside seed noise are reported as null results, not wins.**

---

## Phase 6 — Panel documentation

`RESEARCH_CONTRIBUTION.md` must not collide with what
[CONTRIBUTION_FINAL.md](Component_02/CONTRIBUTION_FINAL.md) already stakes (subgroup-conditional
conformal validity + Mondrian calibration). Component_04's claim is separate:

**Research gap.** PTB-XL superclass performance is published as a single macro number from a single
seed. That macro hides a 36-point per-class F1 spread (NORM 0.87 → HYP 0.51). No published PTB-XL
work separates training variance from test-set sampling variance, so reported gains of ~0.005 are
unfalsifiable. And the reference preprocessing normalises with statistics computed on a different
signal distribution than the network actually receives.

**Contribution.**
1. **C4-1 — Preprocessing correctness.** Identify and fix the pre-filter/post-filter normalisation
   mismatch; measure its isolated effect by ablation. A reproducibility contribution with a number
   attached, not an assertion.
2. **C4-2 — Variance-aware per-class benchmarking.** Every result as mean ± std over ≥3 seeds *and*
   bootstrap CI over the test set, reported per class, with trivial baselines alongside. Shows
   which published-style "improvements" survive noise.
3. **C4-3 — Rare-class recovery ablation.** Which interventions actually move HYP and MI, and which
   do not. Negative results included.
4. **C4-4 — A documented performance ceiling for HYP** on this data, with the CI and the
   label-quality analysis (132 test positives; 21% likelihood filter) that explains it.

`PRESENTATION_NOTES.md`: slide-by-slide script, the before/after table, and prepared answers to the
questions this invites — *"why is HYP still below 75?"*, *"is this better than Component_02 or just
noise?"*, *"why not compare to Strodthoff?"*, *"what did you personally contribute?"*

---

## Execution order (deadline-aware)

Run in this order so a partial night still yields a complete presentation:

1. `00_env_check.py` → `01_dataset_analysis.py` → `02_build_norm_stats.py` → `03_pack.py` (local
   CPU, ~45 min total). **`results/dataset_analysis.md` and the preprocessing-fix ledger are
   presentable on their own.**
2. `06_evaluate.py --logits Component_02/checkpoints/test_logits_seed0.npy` — runs the new
   evaluation protocol against the *existing* checkpoint. **This gives a complete, honest results
   section before any new training finishes**, and becomes the baseline row.
3. Upload packed arrays (~2.1 GB) to Drive → run `Component04_Colab.ipynb`.
4. As each run lands, `05_predict.py` → re-run `06_evaluate.py` and `07_ablation_report.py`.
   Tables regenerate; nothing is hand-edited.
5. Write `README.md` / `RESEARCH_CONTRIBUTION.md` / `PRESENTATION_NOTES.md` from the generated
   tables.

Every script: explicit CLI, fail-loud input validation (shape, dtype, fs, NaN, split sizes), a
deterministic `results/` artifact, and a provenance header recording config hash + git state.

---

## Verification

- `00_env_check.py` passes on this machine (CPU) and on Colab (CUDA + bf16).
- **Preprocessing equivalence:** `02_build_norm_stats.py --source packed` and `--source cache`
  agree on per-lead mean/std within float16 tolerance. Assert in-script.
- **No leakage regression:** `01_dataset_analysis.py` re-verifies 0 patient overlap across splits
  and split sizes exactly 13,801 / 1,709 / 1,711.
- **Train/serve identity:** a unit check that `prepare()` applied to a raw WFDB record equals the
  corresponding row of the packed array to within float16 tolerance. This is the check whose
  absence allowed the C2 mismatch.
- **Metric correctness:** `06_evaluate.py` reproduces this session's independently computed C2
  numbers exactly when pointed at `Component_02/checkpoints/test_logits_seed0.npy` — macro-AUROC
  0.9320, macro-F1 0.7257, exact-match 0.6388, HYP F1 0.5118. If it does not, the metric code is
  wrong.
- **Threshold honesty:** val-fitted thresholds only; the test-oracle gap is reported, never used.
- **Ablation validity:** every claimed improvement exceeds the across-seed std, or it is labelled a
  null result.
- End-to-end: `04_train.py --epochs 2 --max-minutes 5` completes and writes a loadable checkpoint
  before committing to the full Colab run.
