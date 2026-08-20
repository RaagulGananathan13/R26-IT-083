# Component 04 — Ablation Studies

All runs share the same patient-disjoint split, the same fixed model capacity and seed 42; only the stated factor varies.

## A. Modality ablation

| configuration | n_features | S1 AUROC | S1 AUPRC | S2 macro-F1 | dS1 AUPRC | dS2 macro-F1 |
|---|---|---|---|---|---|---|
| ALL modalities | 242 | 0.9562 | 0.7179 | 0.7593 | 0.0000 | 0.0000 |
| - text | 178 | 0.9503 | 0.6930 | 0.7428 | -0.0249 | -0.0165 |
| - ecg | 190 | 0.9525 | 0.7044 | 0.7399 | -0.0135 | -0.0194 |
| - vitals | 204 | 0.9545 | 0.7076 | 0.7494 | -0.0103 | -0.0099 |
| - labs | 216 | 0.9530 | 0.6859 | 0.7419 | -0.0320 | -0.0174 |
| - demographics | 226 | 0.9561 | 0.7140 | 0.7773 | -0.0040 | 0.0180 |
| - medications | 226 | 0.9546 | 0.7130 | 0.7652 | -0.0049 | 0.0059 |
| - history | 226 | 0.9555 | 0.7141 | 0.7610 | -0.0038 | 0.0017 |
| - interaction | 228 | 0.9556 | 0.7174 | 0.7637 | -0.0005 | 0.0044 |
| only text | 64 | 0.8260 | 0.3335 | 0.5425 | 0.0000 | 0.0000 |
| only ecg | 52 | 0.8390 | 0.3283 | 0.5491 | 0.0000 | 0.0000 |
| only vitals | 38 | 0.6983 | 0.1096 | 0.4868 | 0.0000 | 0.0000 |
| only labs | 26 | 0.8652 | 0.5579 | 0.6016 | 0.0000 | 0.0000 |
| only demographics | 16 | 0.6721 | 0.1056 | 0.3682 | 0.0000 | 0.0000 |
| only medications | 16 | 0.6337 | 0.0909 | 0.3841 | 0.0000 | 0.0000 |
| only history | 16 | 0.5345 | 0.0661 | 0.1552 | 0.0000 | 0.0000 |
| only interaction | 14 | 0.9244 | 0.5962 | 0.6602 | 0.0000 | 0.0000 |

## C. Splitting protocol

| protocol | S1 AUROC | S1 AUPRC | patients in both folds | contaminated test rows |
|---|---|---|---|---|
| random stratified (original) | 0.9538 | 0.6791 | 5,804 | 7,627 |
| patient-level grouped (ours) | 0.9558 | 0.7194 | 0 | 0 |

## D. Evaluation population

| population | test n | prevalence | S1 AUROC | S1 AUPRC |
|---|---|---|---|---|
| Intended Use Population | 13,549 | 0.0563 | 0.9562 | 0.7179 |
| full ED | 30,452 | 0.0265 | 0.9713 | 0.6946 |

## B. Referral-Diagnosis Masking

| configuration | S1 AUROC | S1 AUPRC | S2 macro-F1 | S2 STEMI recall |
|---|---|---|---|---|
| RDM ON (masked, default) | 0.9558 | 0.7194 | 0.7688 | 0.7007 |
| RDM OFF (leaky text) | 0.9562 | 0.7181 | 0.7633 | 0.6934 |
