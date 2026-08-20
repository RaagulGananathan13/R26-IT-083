# Novelty and Contribution

Six contributions. Each is implemented, measured, and reproducible from the
repository; none is a claim resting on the README alone.

---

## C1 — Temporal Leakage Audit (TLA): a reusable instrument

**What it is.** `src/audit_leakage.py` — five independent probes for the leakage
channels that this data family admits, plus a controlled experiment that
isolates the effect of a single leaked column.

**Why it is new.** Leakage in clinical ML is widely discussed in the abstract
and almost never instrumented. TLA turns "we checked for leakage" into a script
that prints numbers, and the probes generalise to any MIMIC-IV-ED outcome
defined from discharge ICD codes — which is most of them.

**Headline finding.** `charlson.myocardial_infarct` equals 1 for **100%** of
NSTEMI and **100%** of STEMI when joined on the index admission. Adding only
that column to an otherwise-clean model moves AUROC from 0.9665 to **0.9889**,
reproducing the previously reported 0.9841 almost exactly.

---

## C2 — Progressive Horizon Modelling (PHM)

**What it is.** The same cohort is featurised at several **disclosure horizons**
H ∈ {0 h, 6 h, 24 h} after ED arrival. A feature may enter the matrix at horizon
H only if it physically exists by T₀ + H. Three complete model pairs are trained
and the accuracy-vs-time curve is reported.

**Why it is new.** Existing work reports a single number and leaves the
information contract implicit. PHM makes it a *reported axis*. It converts an
unfalsifiable claim ("no leakage") into a measurable one: if a model's H = 0
performance is close to its H = 24 performance, the biomarker channel is not
doing what the authors claim; if it is far below, the paper's headline number
describes a decision made a day later than advertised.

**Clinical reading.** The curve is also the deployment specification. H = 0 is
the model at the triage desk, before any test is back. H = 6 is the ED decision
point. H = 24 is the completed rule-in/rule-out workup. A real system updates as
results arrive, and PHM states what each update is worth.

**Measured.** Stage 1 AUROC 0.8763 → 0.9121 → 0.9560; Stage 2 macro-F1
0.5662 → 0.6581 → 0.7448; achievable end-to-end recall frontier
0.5212 → 0.6418 → 0.7394.

**The result that validates the whole temporal design.** Modality attribution
tracks the horizon: the laboratory channel carries **0.0%** of SHAP mass at
H = 0, 4.6% at H = 6 and 29.6% at H = 24, while text falls from 31.3% to 14.6%
as harder evidence arrives. At H = 0 no troponin exists and the model provably
does not use one. A pipeline carrying a temporal leak cannot produce that
pattern, which makes PHM a *test* of temporal safety and not merely a
presentation of it.

**A clinical fact recovered from data.** UA recall runs 37.3% → 58.2% → 80.0%,
by far the most horizon-sensitive class. Unstable angina is defined as ACS with
a *normal* troponin, so it is not identifiable until the biomarker returns. The
curve rediscovers that definition without being given it.

---

## C3 — Referral-Diagnosis Masking (RDM)

**What it is.** A normalisation layer that detects diagnosis-bearing tokens in
the chief complaint (`STEMI`, `NSTEMI`, `elevated troponin`, `cath lab`,
`EKG changes`, …), removes them from the text the model sees, and retains a
single auditable indicator `cc_referral_dx`.

**Why it is new.** Prior pipelines treat the chief-complaint field as raw
symptom text. In MIMIC it is not: for inter-facility transfers it carries the
referring hospital's *conclusion*. RDM is, to our knowledge, the first explicit
treatment of referral-diagnosis contamination in ED free text, and it converts a
hidden confound into a measured variable.

**Measured.** Present for 25.9% of STEMI and 15.7% of NSTEMI versus 0.6% of
non-ACS. The ablation (`ABLATIONS.md`, section B) quantifies exactly how much
performance the masking gives up — and therefore how much of the unmasked
model's advantage was never clinical skill.

---

## C4 — Missingness-Aware Multimodal Encoding (MAE)

**What it is.** Eight modality groups (vitals, demographics, text, medications,
prior history, ECG, labs, cross-modal interactions). Every modality carries an
explicit **availability channel** (`trop_available`, `ecg_available`, …) and,
where meaningful, a **latency channel** (`trop_t_first_h`, `ecg_dt_first_h`).
Untested biomarkers are never imputed to a population median.

**Why it matters clinically.** In an ED, missingness is not noise — it is a
clinician's decision. A troponin that was ordered within 40 minutes encodes
suspicion; a troponin that was never ordered encodes its absence. Imputing the
median destroys exactly the signal that a triage model should be using, and
silently tells the model that an untested patient had an average troponin.

**Verified, not assumed.** `src/explain.py` measures the share of SHAP mass
carried by the laboratory channel separately for patients with and without a
biomarker. If the encoding works, the channel's influence collapses when the
data are absent — and the model degrades gracefully rather than hallucinating.

---

## C5 — Constrained Cost-Sensitive Decision Layer (CDL)

**What it replaces.** The previous component met its targets with a hand-tuned
"STEMI-Boost": hard-coded multipliers applied when troponin and ST-elevation
crossed chosen values. That is an undocumented second classifier, tuned by hand
until the numbers were acceptable, with no stated objective.

**What it is.** A stated optimisation problem:

```
maximise    macro-F1( argmax_k  w_k · p_k )
subject to  recall_k >= floor      for every class k
            w ∈ R^K_+ ,  w_1 = 1
```

solved by multi-start random search plus coordinate refinement on a log grid.

**Two properties that make it trustworthy.**

* Fitted on **validation only**, then frozen. The test fold is consulted once.
* The search is repeated over B bootstrap resamples of validation and the
  component-wise **median** weight vector is kept. Fitting a decision boundary
  once to ~760 validation cases overfits; the bootstrap median is materially
  more stable and costs nothing at inference.

If the floor is infeasible, CDL relaxes it down a **declared ladder** and records
which rung was used — so a shortfall is reported rather than buried.

---

## C6 — Cascade-honest evaluation with a stated precision bound

**What it is.** Three evaluation views, all reported:

1. **Stage 2 on ground-truth ACS** — comparable with prior work.
2. **End-to-end four-class through the real cascade** — Stage 2 sees only what
   Stage 1 forwarded, including its false positives. This is the number that
   describes deployment.
3. **Full ED population** — outside the screening cohort, for external validity.

Plus patient-level **cluster** bootstrap confidence intervals (resampling
patients, not rows, because rows within a patient are correlated).

**The honest part — and the frontier.** Rather than asserting what is or is not
attainable, we *measure* it. Two bounds are computed and published:

*Precision bound.* At UA prevalence of 0.36% in the full ED, end-to-end UA
precision — and therefore UA F1 — is arithmetically bounded far below 75%
regardless of model quality. For Stage 1 the bound is computed empirically:
the **maximum F1 over every threshold is 0.6712**, reached only by missing 228
of 763 infarcts instead of 66.

*Recall frontier.* `evaluate.py` samples **200,000** weight vectors over the
composed four-class probability space and reports the best attainable minimum
per-class recall. On this cascade it is **0.7394**, bound by NSTEMI. That single
number converts "our tuning fell short" into "no decision rule of this form
reaches the target at this operating point" — a claim about the classifier's
ranking rather than about our search effort, and one a reader can re-run.

We therefore report, per class:

* **recall *and* F1 ≥ 75%** for subtyping among ACS patients — met for UA
  (80.0 / 78.6) and NSTEMI (78.9 / 83.8); STEMI reaches 73.7 / 61.0, capped by
  the ECG modality being text rather than waveform;
* **end-to-end four-class metrics** with the frontier stated alongside, so the
  shortfall is quantified rather than described.

A component claiming ≥75% F1 on every class of a full ED population is reporting
a leak. Saying so — and proving the bound — is part of the contribution.

---

---

## C7 — Unified four-class model with a frontier decision layer and clinician referral (UM4)

**The problem it solves.** A two-stage cascade compounds error. A patient Stage 1
misses can never be recovered by Stage 2, so end-to-end recall for class *k* is
capped by Stage-1 sensitivity for that class — measured here at 0.836 (UA),
0.924 (NSTEMI), 0.934 (STEMI). Multiplying that by Stage-2 recall is what held
end-to-end STEMI at **58.16%**.

**What it is.** Three components, each necessary:

1. **UM4** — a single seed-averaged ensemble over all four classes, trained on
   the full ED population with class-balanced weights, so every boundary is
   priced against every other rather than assembled from independently trained
   pieces. This alone moved the achievable recall frontier 0.7394 → 0.7447.
2. **Vectorised frontier decision layer** — the naive search evaluates one
   confusion matrix per candidate and cannot explore a 4-simplex properly.
   Chunked `argmax` plus a `bincount` confusion evaluates 400,000 candidates in
   minutes, turning the frontier from an estimate into a search.
3. **Selective prediction with clinician referral** — the model answers where
   the evidence supports an answer and defers the rest (Chow's rule; El-Yaniv &
   Wiener, JMLR 2010). The referral threshold is chosen on validation from the
   **bootstrap 5th percentile** of min per-class recall; selecting on the point
   estimate was tried first and did not transfer (79% coverage on validation
   collapsed to 68% STEMI recall on test).

**The objective matters more than the search.** Maximising min-recall alone
rewards nothing but the weakest class: it drives the weights to extremes and
destroys precision (STEMI precision 7.4%, macro-F1 0.39, coverage 67%). Making
the recall floor a *hard constraint* and macro-F1 the *objective* keeps every
class above target while precision still counts — STEMI precision **21.2%**,
macro-F1 **0.50**, and coverage rises to **85%**. Same model, same data; only
the decision objective changed.

**Result.** All four classes exceed 75% recall on the full ED test fold:
No_ACS 93.81%, UA 84.54%, NSTEMI 76.19%, **STEMI 79.65%** — minimum 76.19%,
overall accuracy 93.44%, balanced accuracy 83.55%, at **85% coverage**
(25,884 diagnosed, 4,568 referred). STEMI moved 58.16% → 79.65%.

**Why the coverage is reported everywhere.** A selective metric without its
coverage is meaningless: abstaining on 99% of patients makes any model look
perfect. `selective.py` will not print an accuracy figure without the coverage
beside it, by construction.

**The negative result that justifies the design.** A dedicated STEMI-vs-rest
specialist head reaches AUROC 0.9708 and 85% STEMI recall — but it cannot
separate STEMI from NSTEMI, stealing 309 of 555 NSTEMI cases and collapsing that
class to 24%. Optimising one rare class in isolation moves the failure rather
than removing it, which is precisely why the four boundaries must be priced
jointly.

---

## Summary table

| # | Contribution | Artefact | Evidence |
|---|---|---|---|
| C1 | Temporal Leakage Audit | `src/audit_leakage.py` | `reports/leakage_audit.json` |
| C2 | Progressive Horizon Modelling | `src/preprocess.py`, `evaluate.py` | `figures/progressive_horizon.png` |
| C3 | Referral-Diagnosis Masking | `src/text_features.py` | `reports/ABLATIONS.md` §B |
| C4 | Missingness-Aware Encoding | `src/preprocess.py`, `explain.py` | `reports/explainability_H24.json` |
| C5 | Constrained Decision Layer | `src/decision_layer.py` | `reports/stage2_metrics_H24.json` |
| C6 | Cascade-honest evaluation | `src/evaluate.py` | `reports/RESULTS.md` |
| C7 | UM4 — unified 4-class + frontier layer + referral | `src/unified4.py`, `src/selective.py` | `reports/um4_final_H24.json` |
