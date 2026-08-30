# Component-wise architecture ablation

**Question.** `ECGResNetSE` adds three things to the plain 1-D ResNet: squeeze-excitation,
a multi-kernel stem, and attention pooling. Do they earn their 566k extra parameters?

**Answer.** No. Two of the three do nothing measurable, and the third actively costs
accuracy — almost all of it on hypertrophy, the one class whose evidence is amplitude.

---

## Protocol

Three seeds per architecture (0, 1, 2), trained with the same script, the same packed
data, the same 40-epoch OneCycle schedule. The only difference between runs is the
network and the seed.

Scored once on the untouched test fold (fold 10, n = 1,711). Because both systems in
each comparison see the *same* records, the bootstrap resamples record indices
identically for both and takes the paired difference — which removes the variance
contributed by which records happen to be in the fold, since that is shared and
therefore not evidence either way.

Reproduce with `train/compare_architectures.py`.

---

## The three architectures

| Name | Components | Parameters |
|---|---|---|
| `resnet` | — | 1,018,501 |
| `resnet_se_no_se` | multi-kernel stem + attention pooling | 1,536,358 |
| `resnet_se` | multi-kernel stem + attention pooling + **squeeze-excitation** | 1,584,326 |

`resnet_se_no_se` differs from `resnet_se` by exactly the four SE blocks (47,968
parameters). Nothing else changes.

---

## Results — 3-seed ensembles

| Architecture | macro-AUROC | macro-AUPRC | HYP AUROC |
|---|---|---|---|
| `resnet` | 0.9440 | 0.8233 | 0.9248 |
| `resnet_se_no_se` | **0.9446** | 0.8187 | **0.9253** |
| `resnet_se` | 0.9404 | 0.8142 | 0.9106 |

### Per-seed spread

| Architecture | macro-AUROC |
|---|---|
| `resnet` | 0.9392 ± 0.0007 |
| `resnet_se_no_se` | 0.9374 ± 0.0017 |
| `resnet_se` | 0.9312 ± 0.0020 |

---

## Pairwise tests

Paired bootstrap, 10,000 resamples, identical record indices.

| Comparison | Δ macro-AUROC | 95 % CI | p | Verdict |
|---|---|---|---|---|
| `no_se` − `resnet` | +0.0006 | [−0.0028, +0.0038] | **0.7410** | not significant |
| `no_se` − `resnet_se` | +0.0042 | [+0.0014, +0.0070] | **0.0040** | **significant** |
| `resnet` − `resnet_se` | +0.0036 | [+0.0003, +0.0070] | **0.0316** | **significant** |

**Reading the three together:**

1. Adding the stem and attention pooling to the plain network changes nothing
   (p = 0.74). Those two components are 518k parameters that buy no measurable accuracy.
2. Adding squeeze-excitation on top of them **loses** 0.0042 AUROC (p = 0.004).
3. The net effect of all three is therefore a loss (p = 0.032), and squeeze-excitation
   accounts for the whole of it.

---

## Where the loss lives

Per-class AUROC, `resnet_se_no_se` against `resnet_se`:

| Class | no_SE | resnet_se | Δ |
|---|---|---|---|
| NORM | 0.9656 | 0.9661 | −0.0005 |
| MI | 0.9536 | 0.9534 | +0.0001 |
| STTC | 0.9434 | 0.9423 | +0.0011 |
| CD | 0.9350 | 0.9295 | +0.0055 |
| **HYP** | **0.9253** | **0.9106** | **+0.0147** |

Hypertrophy is three times the next-largest effect, and the remaining classes are flat.

**A mechanism, not a coincidence.** This component's own training notes state the
reason augmentation avoids wide amplitude scaling: *"voltage IS the HYP evidence."*
Left-ventricular hypertrophy is diagnosed from QRS amplitude — Sokolow-Lyon and Cornell
criteria are both voltage sums. Squeeze-excitation recalibrates channels by learned
importance, which is an operation on relative amplitude across leads. It is therefore
acting directly on the signal hypertrophy depends on, and suppressing it.

That the effect is confined to HYP, and that removing SE alone recovers it entirely
(+0.0147 against +0.0141 for removing all three), is what makes this an explanation
rather than a correlation.

---

## What this changes

`resnet_se` is the shipped model. On this evidence it should not be: the plain
`resnet` matches it everywhere and beats it on hypertrophy, with 566k fewer parameters
and a shorter forward pass.

Not acted on yet, deliberately. The shipped calibration, the conformal thresholds and
the integrated backend are all fitted to `resnet_se`, and swapping the served
architecture means refitting all three and re-verifying every published figure. That is
a decision to take with the numbers in hand, not a change to make quietly.

**Still open.** The three remaining ablations — `resnet_se_no_stem`, `resnet_se_no_attn`
and `resnet_se_plain` — would separate the stem from the attention pooling. The result
above says their *combined* effect is nil, but not whether one helps and the other
hurts by the same amount. Roughly 30 minutes each at three seeds.
