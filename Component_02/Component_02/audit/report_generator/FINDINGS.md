# Neural report generation, and what the corpus will support

**Question.** Component 02 writes its report from a template. Can a trained
generator do better, and how would anyone know?

**Answer.** It beats every trivial baseline and preserves findings well. But the
metric almost everyone would report — BLEU — is nearly uninformative here, and
establishing that took more work than training the model.

---

## The model

Flan-T5-base (248M), fine-tuned on PTB-XL reports conditioned on the
classifier's five superclass labels plus age and sex. It never sees the
waveform, which is what makes its output checkable against the findings.

| | |
|---|---|
| Precision | bf16 (A100) |
| Epochs | 10 configured, **early stopped at 9** |
| Best epoch | **6**, validation loss 0.6106 |
| Training time | ~12 min |
| Test fold | 1,622 reports |

Validation bottomed at epoch 6 and rose for three epochs while training loss
kept falling — 0.4362 to 0.3897. `best.pt` holds epoch 6; taking the last epoch
would have shipped a measurably worse model.

---

## Why BLEU is nearly useless on this corpus

Two models that do not exist were scored on the held-out fold first:

| | BLEU-4 | ROUGE-L |
|---|---|---|
| Always emit `"Sinus rhythm normal ECG."` | 0.2549 | 0.4054 |
| Five-string lookup by class, no model | 0.2423 | 0.3554 |
| **Flan-T5** | **0.2965** | **0.4583** |

The trained model clears the lookup by +0.054 BLEU and +0.103 ROUGE-L. That is
a real gain, and it is small because 7,540 unique strings cover 16,801 records
and the ten commonest account for a third of them. **Any BLEU reported from this
corpus without those floors beside it is not interpretable.**

---

## Finding preservation

The clinical property: does the generated text assert exactly the findings the
classifier passed in?

| | |
|---|---|
| Exact set match | **73.92 %** |
| No invented finding | **80.89 %** |
| No dropped finding | **92.48 %** |

| Class | Recall | Precision | F1 | Support | Reference ceiling |
|---|---|---|---|---|---|
| NORM | 1.0000 | 0.8284 | 0.9061 | 700 | 80.4 % |
| MI | 0.7975 | 0.9847 | **0.8813** | 242 | 77.4 % |
| STTC | 0.9210 | 0.7064 | 0.7996 | 405 | 75.3 % |
| CD | 0.9891 | 0.9978 | **0.9934** | 457 | 79.8 % |
| HYP | 0.5963 | 0.9155 | 0.7222 | 109 | 54.9 % |

**MI precision 0.9847** is the number a clinician cares about most: when it says
infarction, it is almost always right.

**The model exceeds the reference ceiling on every class.** The ceiling is the
share of *human* reports that state the labelled finding. The generator learned
`label -> phrase` from the records that do state it and applies that rule
uniformly, so it is **more internally consistent than the cardiologists' own
reports**. That is a stronger result than matching the references would have
been, and it is not what we expected to find.

---

## The vocabularies were the hard part, and were twice wrong

Finding preservation depends entirely on deciding when a piece of text asserts a
finding. That list does two jobs at once — it sets recall by deciding what
counts as an assertion, and precision by deciding what counts as an invention —
so widening it is a trade, not an improvement.

**First attempt, too narrow.** HYP matched only `hypertroph`. PTB-XL's
hypertrophy superclass also covers atrial overload and enlargement, and the
corpus words LVH as `strain`, `LVH` or `voltages are high`. Measured HYP recall
came out **0.3303** and was reported as a data limit. It was a matcher bug.

**Second attempt, too wide.** Adding every plausible synonym raised HYP recall
to 0.6972 and collapsed its precision from 1.0000 to **0.2559**. `strain` alone
appears 1,498 times in the corpus and only 24 % of those are hypertrophy — it
also describes right-heart strain and rate-related change.

**Third attempt, measured.** Each candidate term was applied to the *reference*
reports, where the true label is known, and kept only if it raised F1 against
that label. Model-free, so the vocabulary cannot be tuned to flatter the model.

| Class | F1 of the matcher | Terms kept | Rejected |
|---|---|---|---|
| NORM | 0.8495 | `normal tracing` | `no pathology`, `regular ecg` |
| MI | 0.7791 | `qs complex` | `myocardial damage`, `q wave`, `scar`, `necrosis` |
| STTC | 0.6935 | `t abnormal`, `st depression`, `t negative` | `st elevation`, `qrs(t) abnormal`, 4 more |
| CD | 0.8292 | `hemiblock`, `intraventricular block`/`delay`, `lafb`, `lpfb` | `intraventricular conduction`, `p-widening`, `aberrant`, `delay` |
| HYP | 0.5323 | `lvh`, `atrial enlargement`, `atrial overload`, `rvh` | **`strain`**, `voltages are high`, `amplitude criteria`, 4 more |

Reproduce the selection with `notebooks/train_report_generator.ipynb`; the
chosen lists are in the ceiling and scoring cells.

---

## What the corpus is, underneath

| | |
|---|---|
| Reports | 17,216, of which **7,540 unique** |
| Median length | 11 words |
| Machine-translated from German | ~81 % carry artefacts |
| Never translated at all | **2.4 % Swedish**, 0.5 % German |

The Swedish is not evenly spread: 5.0 % of hypertrophy records against 0.6 % of
normal ones. Left in, it teaches the model to emit Swedish for exactly the
findings that matter. It is filtered before training.

Artefacts corrected before training: `sine rhythm` → `sinus rhythm` (25.5 % of
records), `position type` → `axis` (33.2 %, a calque of *Lagetyp*), `ekg` →
`ECG` (16.6 %), and a stray `N.NN unconfirmed` confidence field (30.7 %).

---

## What this changes

Nothing yet, deliberately. The template generator still ships. This model is
trained and measured, and the component's `verify.py` already describes where it
would go:

> *"It is the gate, not the generator: any future natural-language layer must
> pass through `verify_paraphrase()` or it does not ship. If verification fails,
> the deterministic template text is emitted instead — degraded fluency, never
> degraded safety."*

Routing this generator through that gate is the next step, and the numbers above
are what it would be gated on.

---

## Reproduce

```
notebooks/train_report_generator.ipynb        (Colab, ~12 min on an A100)
checkpoints/report_generator/metrics.json     full history and both vocabularies
checkpoints/report_generator/test_generations.csv   all 1,622 outputs
```

Weights are gitignored. PTB-XL is CC-BY 4.0, so the corpus itself is
redistributable — cite Wagner et al., *Scientific Data* 7:154, 2020.
