# Component 02 — Deep Technical Audit
## XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting System

**Audited:** 4 August 2026
**Subject:** `_archive/` (the entire existing implementation)

> **STATUS: all findings below are now closed in `Component_02/`.**
> Run `python -X utf8 Component_02/audit/08_verify_fixes.py` — 26/26 checks pass.
> See [README.md](README.md) for the fixed system and
> [RESEARCH_CONTRIBUTION.md](RESEARCH_CONTRIBUTION.md) for what was built on top of it.
>
> | Finding | Fix | Evidence |
> |---|---|---|
> | C-1 flatline → "MI 0.691" | `src/quality.py` refuses uninterpretable records | 8 adversarial inputs, all handled |
> | C-2 fusion-model leakage | model withdrawn; ECG-only model is the deliverable | `06_leakage_audit.py` |
> | C-3 Tier-3 hallucination | BioBART removed; `src/verify.py` gates all text | 200/200 verified |
> | C-4 NORM + abnormality | structurally impossible in `src/report.py` | 0 contradictions |
> | C-5 Grad-CAM race | per-model lock + per-call hooks in `src/xai.py` | 4 concurrent calls match |
> | C-6 miscalibration | temperature scaling in `src/calibration.py` | ECE 0.183 → 0.018 |
> | E-1 plaintext password | moved to env vars | `prepare_data.py` |
> | E-2 repo at home dir | RP-Venu is now its own repo | `.gitignore` |
> | E-3/5/6/7/8/9/10 | see `app/app.py` and `src/` module headers | `08_verify_fixes.py` |
**Method:** every headline number was independently re-derived from the shipped
checkpoints and CSVs. Nothing in this document is quoted from the project's own
`test_results.json` or `README.md` without verification.

Reproduce with:

```
python -X utf8 Component_02/audit/01_data_audit.py      # split / label / cache integrity
python -X utf8 Component_02/audit/02_model_audit.py     # re-evaluate the deployed classifier
python -X utf8 Component_02/audit/03_report_audit.py    # the three report tiers
python -X utf8 Component_02/audit/04_runtime_audit.py   # live Flask pipeline
python -X utf8 Component_02/audit/05_xai_audit.py       # XAI faithfulness (new evidence)
python -X utf8 Component_02/audit/06_leakage_audit.py   # fusion-model leakage proof
```

Raw output: `Component_02/audit/results/`

---

## 1. Verdict

| Area | Status |
|---|---|
| Data pipeline & splits | **Sound.** Official PTB-XL folds, zero patient leakage, cache intact. |
| Deployed classifier (ECG-only ResNet) | **Real and reproducible.** macro-AUROC 0.9297, macro-F1 0.7172. |
| Reported classifier numbers | **Honest.** Reproduced to 4 decimal places, delta 0.0000. |
| Multi-modal fusion model | **Invalid.** Target leakage; strictly worse than the ECG-only model once removed. |
| Tier 3 "BioBART smoother" | **Non-functional.** Identity function 80% of the time; hallucinated a diagnosis in 42 records. |
| "Cardiac Risk Reporting" | **Not implemented.** No risk score, no interval, no rhythm, no HR anywhere. |
| XAI component | **Never evaluated.** Zero faithfulness evidence existed before this audit. |
| Clinical safety guards | **Absent.** Flatline → "MI". µV file → "HYP 100%". No disclaimer in the UI. |

The classification core is genuinely competent work. The layers built on top of it —
fusion, Tier 3, and the "risk reporting" deliverable — do not currently survive
scrutiny. Everything below is fixable inside a final-year timeline, and the
priority ordering in §8 is written for that.

---

## 2. What actually exists and what it scores

### 2.1 Data

| | Value |
|---|---|
| Official PTB-XL v1.0.3 | 21,799 records / 18,869 patients |
| README claims | 21,837 records |
| **Actually used** | **17,221 records / 15,174 patients** |
| Train / Val / Test | 13,801 / 1,709 / 1,711 |
| Split rule | `strat_fold` 1–8 / 9 / 10 — the official PTB-XL protocol |
| Patient overlap between splits | **0** (verified) |
| Corrupt / zero-filled / NaN signals | **0** of 17,221 (verified byte-by-byte) |

The 21% shortfall comes from the labelling rule: only SCP codes with
`likelihood == 100` were kept. See §4.1 — this is a methodological problem, not a
data-integrity one.

### 2.2 The deployed model — independently re-measured

`_archive/checkpoints_ecg_only/best_model.pt`, 1,018,501 parameters, best epoch 7.

| Class | Prevalence | AUROC | **AUPRC** | F1 @ 0.5 | F1 @ tuned thr | 95% CI on F1 |
|---|---|---|---|---|---|---|
| NORM | 41.3% | 0.9571 | 0.9310 | 0.8728 | 0.8728 | 0.855 – 0.890 |
| MI | 15.7% | 0.9397 | 0.7745 | 0.6851 | 0.6860 | 0.641 – 0.729 |
| STTC | 26.7% | 0.9321 | 0.8246 | 0.7215 | 0.7750 | 0.746 – 0.804 |
| CD | 28.2% | 0.9146 | 0.8614 | 0.7643 | 0.7586 | 0.728 – 0.788 |
| HYP | **7.7%** | 0.9050 | **0.5405** | 0.3992 | 0.4938 | **0.424 – 0.557** |
| **Macro** | | **0.9297** | **0.7864** | 0.6886 | **0.7172** | 0.697 – 0.737 |

Every published figure reproduced exactly (delta `+0.0000`). The reported numbers
are honest — that is worth saying plainly.

Three things the current write-up omits:

- **AUPRC is the correct headline metric** for imbalanced multi-label data and is
  much less flattering: 0.786 macro, and **0.54 for HYP** — barely above the 0.077
  prevalence baseline in a useful sense. AUROC 0.905 for HYP reads as strong and
  is misleading.
- **Confidence intervals.** HYP's F1 interval spans 0.42–0.56 on 132 positives. A
  point estimate of "0.49" is not defensible without it.
- **Threshold optimism is small and correctly handled.** Thresholds were fit on
  validation, not test. Refitting on test would give 0.7278 (+0.0106). The
  methodology here is clean.

### 2.3 Context vs. the literature

Strodthoff et al. (2021) report macro-AUROC ≈ 0.92–0.93 for the PTB-XL
5-superclass task. **0.9297 sits inside that band** — but on a filtered subset
(§4.1) that removed the ambiguous cases, so it is not directly comparable. The
thesis must say this explicitly or a reviewer will find it.

---

## 3. Critical defects

### C-1 Flatline and noise produce confident diagnoses

Measured on the deployed model:

| Input | NORM | MI | STTC | CD | HYP | Reported to user |
|---|---|---|---|---|---|---|
| **All-zero (disconnected leads)** | 0.029 | **0.691** | **0.602** | 0.425 | 0.551 | **"consistent with myocardial infarction"** |
| Pure Gaussian noise | 0.036 | **0.629** | 0.462 | **0.657** | 0.171 | MI + conduction disturbance |
| Real ECG in µV (×1000) | 0.000 | 0.000 | **1.000** | 0.000 | **1.000** | STTC + HYP, 100% confidence |
| Real ECG inverted | 0.042 | **0.818** | 0.239 | **0.755** | 0.542 | MI + CD |
| Real ECG (correct) | **0.802** | 0.149 | 0.220 | 0.255 | 0.106 | NORM |

There is no signal-quality gate anywhere in `app.py`. A flatlined lead set, a
unit mismatch, or a polarity error each yields an urgent-sounding report with no
warning. This is the single most dangerous defect in the system.

### C-2 Multi-modal fusion model is target-leaked

`_archive/training/train.py` feeds a ClinicalBERT embedding of **`report_en`** —
the cardiologist's own report — into the classifier. The 5 labels were derived
from the SCP codes attached to that same report.

Ablation on the identical test set (`06_leakage_audit.py`):

| Configuration | macro-AUROC | macro-F1 |
|---|---|---|
| A — full (signal + demographics + **report text**) | **0.9567** | **0.7733** |
| B — report text zeroed (signal + demographics) | 0.9072 | 0.6271 |
| C — signal zeroed (demographics + report text) | 0.8904 | 0.5535 |
| D — **report text alone** | **0.8872** | 0.5127 |

Read row D: with **no ECG at all**, just the report embedding, the model reaches
0.887 macro-AUROC. And row B is the honest capability of that architecture —
**0.9072, which is worse than the deployed ECG-only ResNet at 0.9297.**

So "Multi-Modal Fusion for maximum accuracy" (NOVELTY 2 in the documentation) is
backwards: fusion helps only because it is reading the answer, and once the
answer is removed it underperforms the simpler model. The 0.9567 / 0.7733 figures
in `checkpoints/test_results.json` cannot appear in the thesis as an ECG result.

*Mitigating fact: `app.py` deploys the clean ECG-only model, and the README
quotes the clean numbers. The leaked model is not in the serving path.*

### C-3 Tier 3 "BioBART smoother" does not work — and hallucinated a diagnosis

Measured across all 1,711 shipped audit records:

| Behaviour | Count | Share |
|---|---|---|
| Byte-identical to the Tier 2 template | 404 | 23.6% |
| Template with leading characters **clipped off** (corruption) | 974 | 56.9% |
| Genuinely rewritten | 333 | 19.5% |
| Records where Tier 3 **dropped** a clinical concept present in Tier 2 | 103 | 6.0% |

And the finding that breaks the central safety claim:

> **In 42 records Tier 3 replaced "ECG shows predominantly normal features" with
> "Graphic atrial fibrillation."**

Atrial fibrillation is not one of the 5 classes. The model cannot detect it. It
was invented by the decoder — the exact failure mode the three-tier architecture
was designed to make impossible.

**Root cause.** `train_report_gen.py` trains BioBART by injecting CNN features via
`encoder_outputs=`, which **bypasses BART's own text encoder entirely** (that
encoder is explicitly frozen and never used). The cross-attention layers are then
fine-tuned to consume CNN feature distributions. `smooth_report()` in `app.py`
later calls `bart.generate(input_ids=...)`, which routes text **through that
never-trained encoder** and asks decoder cross-attention to interpret an input
distribution it has never seen. Tier 3 is running the model off-distribution by
construction; the garbling is the predictable result, not a tokenisation quirk.

The `_fix_smoothed_output()` guard added later does catch the "Graphic" case by
falling back to the template — but it is a prefix-matching heuristic plus a
hard-coded typo blacklist (`"ST-seal" → "ST-segment"`), not a semantic guarantee.
Live re-test with the current guard: **4 of 5 template inputs collapse back to
the template verbatim.** Tier 3 contributes nothing and costs 2.55 s per request.

### C-4 The report contradicts itself in 5.8% of cases

`report_templates.py` emits one sentence per class above threshold, with no
mutual-exclusion rule between NORM and the four abnormal classes. On the test set
**99 of 1,711 records (5.8%)** fire NORM *and* an abnormality:

| Combination | Count |
|---|---|
| NORM + CD | 55 |
| NORM + STTC | 38 |
| NORM + MI | 5 |
| NORM + HYP | 1 |

Real output for ECG 38:

> "ECG shows predominantly normal features. Minor non-specific findings are
> present but do not meet criteria for a defined abnormality. **Conduction delay
> is detected**, consistent with bundle branch block or intraventricular
> conduction disturbance. **Cardiology referral is recommended.**"

The nurse is told the ECG is normal and that it requires cardiology referral, in
one paragraph. Note the **ground truth has zero such rows** — the labelling rule
explicitly zeroes NORM when any pathology is present. The model uses 5 independent
sigmoids that cannot express that constraint, and no post-hoc guard was added.

### C-5 Grad-CAM race condition under concurrent requests

`app.gradcam` is a single module-level object that stores `.activations` and
`.gradients` as instance state. Flask's server is threaded by default. Four
concurrent calls were run against the live object: **3 of 4 returned heatmaps
that differ from the sequential result.** Two clinicians using the demo at once
get each other's explanations — silently, with no error raised.

### C-6 Probabilities are badly miscalibrated but shown as "% confidence"

| Class | Brier | ECE | mean predicted p | actual prevalence | over-prediction |
|---|---|---|---|---|---|
| NORM | 0.099 | 0.130 | 0.447 | 0.413 | 1.08× |
| MI | 0.093 | 0.156 | 0.298 | 0.157 | **1.90×** |
| STTC | 0.147 | 0.214 | 0.461 | 0.267 | **1.73×** |
| CD | 0.124 | 0.176 | 0.403 | 0.282 | 1.43× |
| HYP | 0.124 | **0.242** | 0.320 | 0.077 | **4.14×** |

Cause: `WeightedRandomSampler` oversampling **and** focal-loss `alpha` class
weights were applied simultaneously — the imbalance correction is applied twice,
so sigmoid outputs are not posterior probabilities. The UI prints them as
"probability %" next to a threshold, which invites exactly the misreading a
clinician would make. HYP saying "32%" when the true rate is 7.7% is a 4× error.

---

## 4. Methodological problems

### 4.1 The `likelihood == 100` label filter is not standard practice

`PTB_XL_Dataset_Labeling_Guide.txt` states this threshold means "**both
cardiologists fully agreed**". That is not what the field means in PTB-XL. The
`scp_codes` likelihood is a single annotator's per-statement confidence, and a
value of `0.0` means *likelihood information was not recorded* — not "0%
confident". Records were annotated by one cardiologist; `validated_by_human` marks
the separately validated subset (12,849 of your 17,221).

Consequences:
- 4,578 records (21%) were dropped — disproportionately the ambiguous ones.
- The task is made **easier** than the published benchmark.
- Results are **not comparable** to Strodthoff et al. or any PTB-XL paper.

This does not invalidate the model. It does mean the comparison table in the
thesis needs a stated caveat, or a re-run on the standard label set.

### 4.2 Age sentinel not cleaned

PTB-XL anonymises age > 89 as `300`. **255 records** carry `age == 300`, uncleaned.
This inflates the stored normalisation statistics:

| | mean | std |
|---|---|---|
| `norm_stats.json` (used in training) | 62.54 | **32.45** |
| Cleaned (excluding 300) | 59.46 | **16.95** |

The demographic branch of the fusion model therefore normalised age against a
standard deviation that is roughly double the true one, compressing all real age
variation into ±0.9σ. Affects the fusion model only.

### 4.3 Height/weight imputation is undocumented

67.7% of heights and 56.2% of weights are missing and appear to have been filled
with sex-conditional constants (height: 174.0 ×6,529 and 160.0 ×5,724; weight:
77.0 ×5,620 and 62.0 ×4,359). The `*_missing` flags are supplied, which is the
right pattern — but the imputation is nowhere described in the documentation.

### 4.4 The hallucination metric is circular

`hallucination_audit()` in `evaluate_hybrid.py` checks whether the report mentions
a class the classifier did not detect. Tier 2 reports are *generated from* the
classifier output, so the metric is ~0 by construction. It measures template
correctness, not hallucination. A real metric compares generated findings against
**ground truth**, and by that measure Tier 2 "introduces" myocardial infarction in
256 records — those are simply the model's false positives, which is the number
that actually matters.

### 4.5 ROUGE numbers are not meaningful as reported

65.5% of reference reports contain "normal" and roughly 41% are literally
`"sinus rhythm normal ekg"`. The legacy tier's ROUGE-L of 0.4337 is dominated by
memorising that one 4-token string. 5.2% of references are untranslated
Swedish/German fragments that no English model can match. ROUGE must be reported
stratified by NORM vs abnormal, or dropped.

### 4.6 The XAI component has never been evaluated

The thesis title leads with "XAI". The repository contains **no faithfulness
metric, no sanity check, no deletion/insertion curve, no comparison to a random
baseline** for either Grad-CAM or Integrated Gradients. `05_xai_audit.py` supplies
the first such evidence — see §7.

Two implementation issues found by inspection:
- `compute_lead_saliency()` takes `.abs()` before summing over time, discarding
  attribution sign. A lead that argues *against* the diagnosis is displayed as
  "important" to the clinician.
- Integrated Gradients ships with `steps=30`. IG's completeness axiom gives a
  direct error measure; see §7 for the measured value.

---

## 5. Engineering and security defects

| ID | Issue | Location |
|---|---|---|
| **E-1** | **PhysioNet credentials in plaintext** — `USERNAME = "<redacted>"`, `PASSWORD = "<redacted>"` (a teammate's account) — **the real values were committed in an earlier revision of this file and must be treated as compromised; rotate the account** | `training/prepare_data.py:26-27` |
| **E-2** | **Git repo root is `C:\Users\Venushan`** — the whole home directory, remote `work-sheet.git`, no `.gitignore`, 113 untracked entries including `.ssh/`. One `git add -A` publishes SSH keys and the password above. The research project itself is **not under version control at all**. | repo layout |
| **E-3** | No `secure_filename()` on uploads — `os.path.join(tmpdir, dat_file.filename)` accepts `../` path traversal | `app.py:539-540` |
| **E-4** | Silent zero-fill on download failure: `except Exception: np.save(npy_path, np.zeros(...))` — an all-zero ECG enters training as a real record. *No records were actually affected (verified 0/17,221), but the trap is still armed for any re-run.* | `training/prepare_data.py:111` |
| **E-5** | `/predict` resamples **any** length to 5000 with no `fs` or duration check — a 30 s strip is compressed 3×, a 2.5 s strip stretched 4×, heart rate silently distorted | `app.py:547-549` |
| **E-6** | No lead-**order** validation (lead *count* is checked). A single I↔II swap changed the diagnosis in **4 of 25** records tested (16%), with no warning | `app.py:545-546` |
| **E-7** | Legacy free-generation tier — documented as "ablation only, CAN hallucinate" — is executed on **every** request and returned as `"report"` in the JSON | `app.py:439-443` |
| **E-8** | **No clinical disclaimer anywhere in the UI** (486 lines of `index.html`, zero matches for "research use", "not a medical device", "disclaimer") | `templates/index.html` |
| **E-9** | `app.run(host="0.0.0.0")` — binds all interfaces, no auth, no upload size limit | `app.py:560` |
| **E-10** | Model architectures are duplicated verbatim in 5 files (`app.py`, `predict.py`, `evaluate_hybrid.py`, `train_ecg_only.py`, `train_report_gen.py`). Any change must be made 5× or inference silently diverges from training | project-wide |
| **E-11** | `checkpoints_report_gen/` holds **1.3 GB** of checkpoints for a component that is a no-op (§C-3) | repo size |
| **E-12** | `evaluation_results/` is empty — `evaluate_hybrid.py` has no committed successful run | `evaluation_results/` |

**Performance (measured, CPU):** cold start 26.3 s, 1,567 MB RSS, **11.77 s per
ECG**. Breakdown: matplotlib plot 5.22 s, Tier-3 smoother 2.55 s, IG 1.83 s,
legacy generation 1.49 s, Grad-CAM 0.06 s, **classifier 0.02 s**. 99.8% of the
latency is in components that add little or nothing. Deleting Tier 3 and the
legacy tier alone would take this to ~7.7 s.

---

## 6. What "Cardiac Risk Reporting" is missing

Scanned all 1,711 generated reports for the content a triage/nurse report needs:

| Element | Present in |
|---|---|
| Heart rate (bpm) | **0 / 1711** |
| Rhythm classification (AF, flutter, tachy/brady) | **0 / 1711** |
| PR / QRS / QT intervals | **0 / 1711** |
| QRS axis | **0 / 1711** |
| Risk score or triage priority | **0 / 1711** |
| Infarct localisation (anterior / inferior / lateral) | **0 / 1711** |
| Model uncertainty statement | **0 / 1711** |
| Signal-quality statement | **0 / 1711** |
| Patient age / sex | 192 / 1711 (incidental word matches only) |

The template engine can produce **63 distinct reports** across 1,711 patients
(theoretical ceiling 242). The cardiologists produced **1,055 distinct reports**
for the same patients. 587 patients receive one identical NORM paragraph.

The component is named *"...and Cardiac Risk Reporting System"*. As built it is a
5-class classifier with a lookup table. **No risk stratification exists in the
codebase.** This is the largest gap between the stated deliverable and the
implementation, and it is the one an examiner will ask about first.

---

## 7. XAI faithfulness — new evidence

No XAI evaluation existed anywhere in the repository. These are the first
faithfulness measurements for this system (`05_xai_audit.py`, n = 40 test records,
target = highest-probability class). **The results are good — this section is
thesis material, not a defect list.**

### 7.1 Deletion / insertion test — Grad-CAM is faithful

Mask the top-k% of time points ranked by Grad-CAM and watch the target
probability fall; compare against masking the same number of randomly chosen
points.

| Masked | Deletion: Grad-CAM | Deletion: random | Insertion: Grad-CAM | Insertion: random |
|---|---|---|---|---|
| 0% | 0.757 | 0.757 | 0.357 | 0.357 |
| 10% | **0.606** | 0.741 | **0.527** | 0.390 |
| 20% | **0.492** | 0.721 | **0.592** | 0.421 |
| 30% | **0.417** | 0.690 | **0.630** | 0.473 |
| 50% | 0.374 | 0.593 | 0.677 | 0.602 |
| 90% | 0.359 | 0.385 | 0.740 | 0.744 |

- **Deletion AUC 0.3933 vs random 0.5368** (lower is better) → **faithful**
- **Insertion AUC 0.5751 vs random 0.5039** (higher is better) → **faithful**

Removing the top 10% of Grad-CAM-ranked time points costs 0.151 of probability;
removing 10% at random costs 0.016 — a **9× difference**. The heatmap is reading
the model, not decorating the plot.

### 7.2 Model-randomisation sanity check (Adebayo et al., NeurIPS 2018)

Re-initialise the classifier head with random weights and recompute the heatmap.
If the explanation barely changes, it is an edge detector rather than an
explanation.

**Spearman(trained, randomised) = 0.163** → **passes.** The heatmap genuinely
depends on the learned classifier.

### 7.3 Integrated Gradients is numerically sound

| Check | Result |
|---|---|
| Spearman(30 steps, 200 steps) | **0.999** |
| Top-1 lead agreement, 30 vs 200 steps | **19 / 20** |
| Spearman(zero baseline, mean baseline) | **0.999** |
| Completeness error @ 30 / 100 / 300 / 1000 steps | **1.3% / 0.2% / 0.1% / 0.0%** |

The shipped `steps=30` is adequate; the lead ranking is stable to step count and
to the baseline choice. *(An earlier pass in this audit reported a 35.6%
completeness error — that mean was inflated by records where F(x) ≈ F(baseline)
makes a relative error ill-conditioned. The corrected figure is 1.3%.)*

### 7.4 The one real XAI defect

`compute_lead_saliency()` (`app.py:300`) applies `.abs()` before summing over
time, discarding attribution sign. A lead whose evidence points **away** from the
diagnosis is displayed to the clinician as "important". Fix: report signed
attribution, or show magnitude and direction separately.

### 7.5 What is still missing for the thesis

- No alignment check against clinical landmarks (do MI heatmaps concentrate on
  ST segments? do CD heatmaps concentrate on QRS?). This is the check a
  cardiologist reviewer will want, and it is the natural next experiment.
- No comparison against a second attribution method as a control.
- No inter-method agreement between the temporal (Grad-CAM) and spatial (IG)
  explanations.

---

## 8. Priority queue

### P0 — before the system is demonstrated to anyone

1. **Signal-quality gate.** Reject flatline, saturation, implausible amplitude, and
   out-of-range sampling rate *before* classification. Fixes C-1.
2. **Unit/gain sanity check** on upload (median QRS amplitude must land in a
   physiological band); reject or auto-scale µV files. Fixes C-1.
3. **NORM mutual exclusion** in `report_templates.py` — if any abnormal class
   fires, suppress the NORM sentence and say so. Fixes C-4. ~10 lines.
4. **Clinical disclaimer** in the UI and in every generated report. Fixes E-8.
5. **Rotate the PhysioNet password**, remove it from source, move to `.env`.
   Fixes E-1.
6. **Get the project into its own git repo** with a `.gitignore` — today a single
   `git add -A` at `C:\Users\Venushan` publishes `.ssh/` and that password.
   Fixes E-2.

### P1 — required for the thesis to hold up

7. **Withdraw or re-run the fusion model** without `report_en`. Present row B
   (0.9072) or delete the claim. Fixes C-2.
8. **Delete Tier 3, or replace it** with a rule-based realiser / an LLM constrained
   by a verification pass that checks every Tier-2 finding survives. Do not defend
   the current implementation. Fixes C-3.
9. **Fix the Grad-CAM race** — construct the CAM object per request, or lock it.
   Fixes C-5.
10. **Calibrate.** Temperature or isotonic regression fit on validation; then the
    UI can honestly say "probability". Fixes C-6.
11. **Report AUPRC and bootstrap CIs** alongside AUROC/F1 everywhere.
12. **State the `likelihood == 100` filter** and its effect on comparability, or
    re-run on the standard label set. Fixes 4.1.

### P2 — closes the gap to the stated deliverable

13. **Build the actual risk reporting layer**: HR, PR/QRS/QT, axis, rhythm class,
    a defensible triage tier, and infarct localisation from the lead attributions.
    This is the missing component, not a refinement of it.
14. **Evaluate the XAI properly** — deletion/insertion curves, the randomisation
    sanity check, and R-peak alignment. §7 is the starting point.
15. **Consolidate the 5 duplicated model definitions** into one module. Fixes E-10.
16. **Remove the legacy free-generation path** from the request cycle. Fixes E-7.
17. Fix the age-300 sentinel and document the imputation. Fixes 4.2, 4.3.
18. Correct the README: 17,221 records, not 21,837.

---

## 9. What is genuinely good

Worth defending in the viva, because it is real:

- **The split protocol is correct.** Official `strat_fold`, patient-disjoint,
  verified zero leakage. Many student projects fail here; this one does not.
- **Thresholds were tuned on validation, not test.** Measured optimism is only
  +0.0106. This is textbook-correct and most projects get it wrong.
- **Every reported classifier number reproduces exactly.** No inflation, no
  cherry-picking. The numbers can be trusted.
- **Data integrity is perfect** — 17,221/17,221 signals present, correct shape,
  zero NaN, zero corrupt.
- **0.9297 macro-AUROC is competitive** with the published PTB-XL benchmark band.
- **The classifier-first / template-second idea is sound** and genuinely does
  eliminate hallucination *at Tier 2*. The idea is good; only the Tier 3
  implementation on top of it is broken.
- **BatchNorm is stable at batch size 1** — verified max deviation 1.19e-07,
  zero decision flips between the training batch size and the serving path.
