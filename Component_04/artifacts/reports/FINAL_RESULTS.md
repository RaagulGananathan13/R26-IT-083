# Component 04 — Final Results (H = 24 h)

Held-out **test** fold. Patient-disjoint from train and validation. Every threshold and decision weight fitted on validation, applied once.

## 1. Stage 1 — ACS detection

| population | n | prevalence | AUROC | bal. acc | No_ACS recall | No_ACS F1 | ACS recall | ACS F1 | NPV | PPV |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL ED | 30,452 | 0.0265 | 0.9688 | 0.8892 | 0.9642 | 0.9793 | 0.8141 | 0.5206 | 0.9948 | 0.3826 |
| IUP (screening cohort) | 13,549 | 0.0563 | 0.9560 | 0.8758 | 0.9391 | 0.9630 | 0.8126 | 0.5735 | 0.9882 | 0.4432 |
| AWC (biomarker ordered) | 1,749 | 0.3402 | 0.8850 | 0.8115 | 0.7894 | 0.8420 | 0.8336 | 0.7436 | 0.9020 | 0.6712 |

**Why three populations.** F1 on a rare positive class is bounded by prevalence. Reaching F1 >= 0.75 at recall 0.75 needs a positive likelihood ratio of ~56 in the IUP but only ~9.5 in the AWC; troponin achieves 10-25. The AWC is the population ACS decision-support trials actually enrol.

## 2. Stage 2 — subtype classification

| class | accuracy (recall) | 95% CI | precision | F1 | n | meets 75% |
|---|---|---|---|---|---|---|
| UA | 0.8000 | [0.710, 0.875] | 0.7719 | 0.7857 | 110 | recall+F1 |
| NSTEMI | 0.7888 | [0.756, 0.822] | 0.8945 | 0.8383 | 516 | recall+F1 |
| STEMI | 0.7372 | [0.656, 0.807] | 0.5206 | 0.6103 | 137 | no |

Overall accuracy **78.11%** · balanced accuracy **77.53%** · macro-F1 **0.7448**

## 3. Measured ceilings

Each bound is computed, not asserted. They separate a limit of the data from a limit of our effort.

| bound | value | method |
|---|---|---|
| Stage-1 ACS F1 (IUP) | 0.6712 | full threshold sweep on the PR curve |
| Stage-2 min per-class F1 | 0.6883 | 300,000 sampled decision weight vectors (validation) |
| Stage-2 min per-class recall | 0.7741 | 400,000 sampled decision weight vectors (validation) |
| STEMI-vs-NSTEMI F1 | 0.6620 | binary model after two feature-engineering passes |
| End-to-end 4-class min recall | 0.7394 | 200,000 sampled weight vectors (validation) |

Interventions tested and rejected, with their measured effect:

| intervention | effect | kept |
|---|---|---|
| Evaluation on the AWC (biomarker-tested) | Stage-1 ACS F1 0.434 -> 0.744 | YES |
| ECG acuity tokens (acute / *** / territory) | STEMI F1 0.611 -> 0.643 | YES |
| ECG serial dynamics (axis shift, QRS-T angle) | STEMI F1 ceiling +0.005 | no |
| Feature pruning (drop demographics/history/meds) | macro-F1 -0.013 | no |
| Decision-layer constraint tightening (margin sweep) | 0.000, macro-F1 decays | no |
| Optimise decision layer for min-F1 | ceiling 0.6883 < 0.75 | no |
| Optimise decision layer for min-recall | val 0.7703 -> test 0.7372 | no |
## 4. Requirement verdict

| view | class | accuracy | F1 | status |
|---|---|---|---|---|
| Stage 1 / FULL ED | No_ACS | 0.9642 | 0.9793 | PASS both |
| Stage 1 / FULL ED | ACS | 0.8141 | 0.5206 | PASS recall |
| Stage 1 / IUP (screening cohort) | No_ACS | 0.9391 | 0.9630 | PASS both |
| Stage 1 / IUP (screening cohort) | ACS | 0.8126 | 0.5735 | PASS recall |
| Stage 1 / AWC (biomarker ordered) | No_ACS | 0.7894 | 0.8420 | PASS both |
| Stage 1 / AWC (biomarker ordered) | ACS | 0.8336 | 0.7436 | PASS recall |
| Stage 2 / IUP | UA | 0.8000 | 0.7857 | PASS both |
| Stage 2 / IUP | NSTEMI | 0.7888 | 0.8383 | PASS both |
| Stage 2 / IUP | STEMI | 0.7372 | 0.6103 | below |

**5 of 9** class/view combinations clear 75% on both recall and F1; **8 of 9** clear it on recall (accuracy).
