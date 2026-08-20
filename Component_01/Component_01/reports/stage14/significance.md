# Stage 14 — Paired Significance Testing

All tests are McNemar (mid-p) on paired predictions over the same test set,
with Holm-Bonferroni correction within each family. `p_holm` is the value to quote.

## Family 1 — Classifier vs an always-negative baseline

| Pathology | Acc A | Acc B | b | c | Diff % [95% CI] | p (mid-p) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 83.21 | 49.58 | 2197 | 609 | +33.63 [+31.65, +35.61] | <0.0002 | <0.0002 | **YES** |
| Edema | 85.18 | 77.38 | 810 | 442 | +7.79 [+6.34, +9.24] | <0.0002 | <0.0002 | **YES** |
| Pleural_Effusion | 86.13 | 68.93 | 1191 | 379 | +17.20 [+15.63, +18.77] | <0.0002 | <0.0002 | **YES** |
| Atelectasis | 70.58 | 73.38 | 995 | 1127 | -2.79 [-4.71, -0.89] | 0.0042 | 0.0045 | **YES** |
| Consolidation | 89.31 | 94.32 | 103 | 340 | -5.02 [-5.88, -4.16] | <0.0002 | <0.0002 | **YES** |
| Lung_Opacity | 70.33 | 76.09 | 696 | 968 | -5.76 [-7.45, -4.08] | <0.0002 | <0.0002 | **YES** |
| Pneumonia | 89.39 | 91.89 | 119 | 237 | -2.50 [-3.28, -1.72] | <0.0002 | <0.0002 | **YES** |
| Pneumothorax | 95.28 | 96.27 | 95 | 142 | -0.99 [-1.63, -0.36] | 0.0022 | 0.0045 | **YES** |

## Family 2 — Classifier vs report generator

| Pathology | Acc A | Acc B | b | c | Diff % [95% CI] | p (mid-p) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 83.21 | 80.39 | 317 | 184 | +2.82 [+1.89, +3.74] | <0.0002 | <0.0002 | **YES** |
| Edema | 85.18 | 79.33 | 501 | 225 | +5.84 [+4.74, +6.95] | <0.0002 | <0.0002 | **YES** |
| Pleural_Effusion | 86.13 | 79.99 | 506 | 216 | +6.14 [+5.04, +7.24] | <0.0002 | <0.0002 | **YES** |
| Atelectasis | 70.58 | 75.03 | 578 | 788 | -4.45 [-5.98, -2.92] | <0.0002 | <0.0002 | **YES** |
| Consolidation | 89.31 | 92.44 | 138 | 286 | -3.13 [-3.98, -2.28] | <0.0002 | <0.0002 | **YES** |
| Lung_Opacity | 70.33 | 74.59 | 491 | 692 | -4.26 [-5.68, -2.83] | <0.0002 | <0.0002 | **YES** |
| Pneumonia | 89.39 | 88.88 | 213 | 189 | +0.51 [-0.32, +1.34] | 0.2318 | 0.2318 | no |
| Pneumothorax | 95.28 | 95.89 | 61 | 90 | -0.61 [-1.12, -0.10] | 0.0184 | 0.0367 | **YES** |

## Family 3 — Global vs per-projection thresholds (Contribution 1)

| Pathology | Acc A | Acc B | b | c | Diff % [95% CI] | p (mid-p) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 83.21 | 83.19 | 24 | 23 | +0.02 [-0.26, +0.31] | 0.8854 | 0.8854 | no |

> A NON-significant result supports Contribution 1: per-projection thresholds change the disparity without changing overall accuracy.

## Stage 13 deferral — paired bootstrap on the difference of gaps

McNemar does not apply here: a deferral policy never changes a prediction, only
which cases are answered, so two policies are identical on every case they share.

| Quantity | Value |
|---|---|
| Statistic | abs(gap_global) - abs(gap_conditional), percentage points |
| Observed | **5.8286** |
| 95% bootstrap CI | [3.4559, 6.7908] |
| p (two-sided) | **0.00040** |
| Replicates favouring conditional | 99.98% |
| Coverage (conditional / global) | 85.83% / 85.83% |
| Bootstrap replicates | 10000 |
