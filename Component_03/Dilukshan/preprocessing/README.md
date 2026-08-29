# EchoNet-Dynamic preprocessing pipeline

Reproducible preprocessing for left-ventricular ejection-fraction (EF)
regression and four-class severity classification.

| Class | EF range |
|---:|:---|
| 0 — severe | `EF < 30` |
| 1 — moderate | `30 <= EF < 40` |
| 2 — mild | `40 <= EF < 55` |
| 3 — normal | `EF >= 55` |

No preprocessing or model can guarantee a target accuracy. Stage 5 is a
fail-closed data release gate; model performance must be established on the
held-out test set with confidence intervals.

## Mandatory regeneration after the contour correction

The official EchoNet-Dynamic mask construction excludes the first tracing pair,
which is the LV long axis, before joining the two chamber walls. The current
geometry implementation follows that convention and records
`echonet_dynamic_no_long_axis_v1` in `keyframes.csv`.

Any unversioned `keyframes.csv` and manifest created by an older revision are
stale. Before final training, regenerate stages 2 and 1, which preserves the
existing decoded-cache metadata, and then run the exhaustive release gate:

```powershell
python run_preprocessing.py --only 2,1,5
```

Stage 5 fails if it sees stale keyframe geometry. Do not begin final training
until it reports `PASS` or `PASS_WITH_WARNINGS` and every warning has been
reviewed.

## Temporal protocol

- Training may use ED/ES-aware clip sampling. With the aligned final defaults,
  32 frames at stride 2 cover 62 native-frame intervals. Both keyframes are
  range-contained whenever their separation is at most 62. Longer transitions
  retain the fixed stride and are explicitly reported as uncontainable.
- Validation and test sampling is label-free: it must not receive ED/ES ground
  truth. Deterministic views are placed uniformly over all valid clip starts.
- Videos shorter than the requested cadence are sampled monotonically from the
  first through last frame. Sampling never wraps from the end back to frame 0.

## Stages

```text
stage0_audit.py          video existence/metadata audit
stage2_keyframes.py      versioned ED/ES extraction from LV tracings
stage1_labels.py         labels, imbalance weights, cache-metadata preservation
stage3_norm_stats.py     train-only EF and pixel statistics
stage4_cache_clips.py    atomic decoded-video caching and manifest overlay
stage5_verify.py         exhaustive cache, sampler, geometry, and visual QA gate
```

Run everything from this directory:

```powershell
python run_preprocessing.py
```

Resume a complete-cache run without re-decoding valid files:

```powershell
python run_preprocessing.py --only 4 --resume
python run_preprocessing.py --only 5
```

A limited smoke test preserves metadata for rows outside its scope and never
overwrites global normalization statistics:

```powershell
python run_preprocessing.py --limit 50
```

Stage 4 supports `--compress`, `--max-frames`, `--denoise`, and `--workers`.
New cache paths are stored relative to the preprocessing directory for
portability. Existing absolute manifest paths remain readable.

## Outputs

- `artifacts/manifest.csv`: labels, split, keyframes, weights, portable cache
  path, verified frame count, and cache status.
- `artifacts/norm_stats.json`: train-only EF and pixel normalization values.
- `cache/videos/*.{npy,npz}`: atomic `uint8 (T,112,112)` decoded-video caches.
- `artifacts/verification_report.json`: exhaustive release-gate evidence.
- `artifacts/viz/*.png`: safe-named clip, motion, and real ED/ES contour QA.
