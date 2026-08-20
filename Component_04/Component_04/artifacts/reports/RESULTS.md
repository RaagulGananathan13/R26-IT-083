# Component 04 — Results
Primary disclosure horizon **H = 24h** after ED arrival. All figures are on the held-out **test** fold, which is patient-disjoint from train and validation and was evaluated once.

## 1. Stage 1 — ACS detection
| AUROC | AUPRC | Sensitivity | Specificity | NPV | PPV | Balanced acc. |
|---|---|---|---|---|---|---|
| 0.9560 | 0.6921 | 0.9135 | 0.8628 | 0.9941 | 0.2844 | 0.8882 |

## 2. Stage 2 — subtype classification (ground-truth ACS)
| class | recall | precision | f1 | support | meets 75% target |
|---|---|---|---|---|---|
| UA | 0.8000 | 0.7719 | 0.7857 | 110 | YES |
| NSTEMI | 0.7888 | 0.8945 | 0.8383 | 516 | YES |
| STEMI | 0.7372 | 0.5206 | 0.6103 | 137 | NO |

Macro-F1 **0.7448**, balanced accuracy **77.53%**, minimum per-class recall **73.72%**.

## 3. End-to-end four-class decision
Composition: **joint**.

| class | recall | precision | f1 | support |
|---|---|---|---|---|
| No_ACS | 0.8969 | 0.9904 | 0.9414 | 12,786 |
| UA | 0.7727 | 0.0842 | 0.1519 | 110 |
| NSTEMI | 0.7112 | 0.4611 | 0.5595 | 516 |
| STEMI | 0.5839 | 0.4848 | 0.5298 | 137 |

Macro-F1 **0.5456**, balanced accuracy **74.12%**.

### Confidence intervals (patient-level cluster bootstrap, 1000 resamples)
| class | recall | 95% CI | F1 | F1 95% CI |
|---|---|---|---|---|
| No_ACS | 0.8969 | [0.892, 0.902] | 0.9414 | [0.938, 0.944] |
| UA | 0.7741 | [0.697, 0.849] | 0.1521 | [0.124, 0.181] |
| NSTEMI | 0.7114 | [0.668, 0.750] | 0.5591 | [0.528, 0.593] |
| STEMI | 0.5841 | [0.500, 0.661] | 0.5287 | [0.458, 0.594] |

## 4. Progressive Horizon Modelling
What the model can know, and when.

| horizon (h) | S1 AUROC | S1 AUPRC | S2 macro-F1 | S2 min recall | E2E macro-F1 |
|---|---|---|---|---|---|
| 0.0000 | 0.8763 | 0.4138 | 0.5662 | 0.3727 | 0.3263 |
| 6.0000 | 0.9121 | 0.5172 | 0.6581 | 0.5818 | 0.4196 |
| 24.0000 | 0.9560 | 0.6921 | 0.7448 | 0.7372 | 0.5456 |

## 5. Full ED population
Performance outside the Intended Use Population, where ACS prevalence is roughly half that of the screening cohort.

| class | recall | precision | f1 | support |
|---|---|---|---|---|
| No_ACS | 0.9593 | 0.9946 | 0.9766 | 29,645 |
| UA | 0.7207 | 0.0969 | 0.1708 | 111 |
| NSTEMI | 0.6865 | 0.4456 | 0.5404 | 555 |
| STEMI | 0.5816 | 0.4556 | 0.5109 | 141 |