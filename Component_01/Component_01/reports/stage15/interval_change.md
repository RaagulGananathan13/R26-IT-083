# Stage 15 — Acquisition-Induced False Interval Change

Pairs of consecutive studies from the same patient, ordered by **true**
`StudyDate`/`StudyTime`. Restricted to pairs where the radiologist recorded
**no change**, so any movement the model reports is spurious by construction.

## Radiologist: cardiomegaly unchanged

n = 1666 pairs from 692 patients

| Transition | n | False worsening | False improvement | Asymmetry [95% CI] | Sig |
|---|---|---|---|---|---|
| AP->AP | 1035 | 3.0% | 3.8% | -0.77 [-2.07, +0.49] | no |
| PA->PA | 245 | 4.1% | 6.1% | -2.04 [-5.58, +1.28] | no |
| PA->AP | 207 | 13.5% | 1.9% | +11.59 [+6.37, +16.92] | **YES** |
| AP->PA | 179 | 3.9% | 9.5% | -5.59 [-11.17, +0.00] | no |

### Four arms

| Arm | n | Asymmetry [95% CI] | Significant |
|---|---|---|---|
| A - same projection (control) | 1280 | -1.02 [-2.23, +0.18] | no |
| B - shuffled temporal order (null) | 386 | +1.55 [-2.22, +5.37] | no |
| C - PA->AP, true order (finding) | 207 | +11.59 [+6.37, +16.92] | **YES** |
| D - C + per-projection thresholds (the fix) | 207 | +8.21 [+2.96, +13.62] | **YES** |

### Tests

| Comparison | Difference [95% CI] | p |
|---|---|---|
| Finding vs same-projection control | +12.61 [+7.36, +18.11] | 0.00000 |
| Finding vs shuffled-order null | +10.05 [+3.65, +16.59] | 0.00150 |
| Threshold fix vs uncorrected | -3.40 [-5.94, -1.40] | 0.00150 |

## Radiologist: all 8 findings unchanged (stricter)

n = 397 pairs from 271 patients

| Transition | n | False worsening | False improvement | Asymmetry [95% CI] | Sig |
|---|---|---|---|---|---|
| AP->AP | 184 | 3.3% | 1.1% | +2.17 [-0.55, +5.52] | no |
| PA->PA | 107 | 3.7% | 2.8% | +0.93 [-2.97, +5.17] | no |
| PA->AP | 55 | 10.9% | 0.0% | +10.91 [+3.64, +20.00] | **YES** |
| AP->PA | 51 | 2.0% | 11.8% | -9.80 [-20.00, +0.00] | no |

### Four arms

| Arm | n | Asymmetry [95% CI] | Significant |
|---|---|---|---|
| A - same projection (control) | 291 | +1.72 [-0.66, +4.29] | no |
| B - shuffled temporal order (null) | 106 | -2.83 [-9.35, +3.00] | no |
| C - PA->AP, true order (finding) | 55 | +10.91 [+3.64, +20.00] | **YES** |
| D - C + per-projection thresholds (the fix) | 55 | +9.09 [+1.82, +16.36] | **YES** |

### Tests

| Comparison | Difference [95% CI] | p |
|---|---|---|
| Finding vs same-projection control | +9.12 [+1.45, +18.02] | 0.01850 |
| Finding vs shuffled-order null | +13.62 [+3.84, +24.35] | 0.00500 |
| Threshold fix vs uncorrected | -1.81 [-5.45, +0.00] | 0.72500 |
