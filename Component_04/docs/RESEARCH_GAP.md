# Research Gap

## 1. The clinical problem

Acute Coronary Syndrome (ACS) is a time-critical emergency. Of patients arriving
at an Emergency Department with chest pain, only a small minority are having an
ACS event, yet missing one is among the most consequential errors in emergency
medicine. Triage must therefore be **highly sensitive** (do not send an infarct
home) while remaining **specific enough** to be usable (do not activate the cath
lab for every case of reflux).

The three ACS subtypes require different responses on different clocks:

| Subtype | Defining evidence | Response |
|---|---|---|
| Unstable Angina (UA) | ischaemic symptoms, **normal** troponin | admit, serial troponin, risk stratify |
| NSTEMI | **raised** troponin, no persistent ST elevation | early invasive strategy, typically < 24 h |
| STEMI | persistent **ST elevation** or new LBBB | immediate reperfusion, door-to-balloon < 90 min |

So subtyping is not a cosmetic refinement of the binary call. It selects the
treatment pathway.

## 2. What the literature has already done

Machine-learning ACS triage on MIMIC-IV-ED is a well-populated area. Reported
figures are consistently strong — AUROC in the 0.95–0.99 band is routine, and
the preceding version of this component reported **AUROC 0.9841**.

Three things are near-universal in that literature:

1. **Single-point evaluation.** One AUROC, on one split, with no statement of
   what information was available at what time.
2. **Binary framing.** ACS vs no-ACS. Subtyping, when present, is evaluated only
   on ground-truth ACS patients, which is not the population the deployed model
   would see.
3. **No temporal contract.** Papers state which *tables* were joined, not which
   *timestamps* were admissible.

## 3. The gap

> **Published ACS models on MIMIC-IV-ED are evaluated under protocols that admit
> information unavailable at the moment of the decision they claim to support,
> and the field has no standard instrument for detecting this.**

This is not a hypothetical concern. Auditing the preceding version of this
component (`src/audit_leakage.py`, reproducible end to end) found five distinct
channels, all of which appear in published pipelines built on these tables:

### L1 — Same-admission comorbidity leak

The Charlson comorbidity table is derived from an admission's own ICD codes.
The ACS label is derived from the same codes. Joining Charlson on the **index**
`hadm_id` therefore copies the label into the feature matrix:

```
P(charlson.myocardial_infarct = 1 | NSTEMI) = 1.0000
P(charlson.myocardial_infarct = 1 | STEMI)  = 1.0000
```

That single column reaches AUROC 0.9200 on its own. Adding nothing but this
column to an otherwise-clean 221-feature model moves it from AUROC 0.9665 to
**0.9889** — which is within noise of the 0.9841 previously reported.

### L2 — Patient-level leak

31.1% of subjects have more than one ED stay (one has 199). Under the random
stratified split used previously, **57.3% of test rows belong to a patient the
model already trained on**.

### L3 — Laboratory look-ahead

Troponin was pulled with the bound `charttime >= intime` on a `hadm_id` join,
i.e. the entire inpatient stay. Median draw time is **21.8 h after ED arrival**;
47% land beyond 24 h; the maximum is **148 days**. The model was reading the
inpatient troponin peak that *defines* NSTEMI.

### L4 — ECG look-ahead

ECG findings were aggregated with `groupby(subject_id).max()` and no time bound.
Only **4.6%** of the resulting (stay, ECG) pairs fall within a plausible triage
window; 36% come from after the encounter and 57% from more than a day before.
The ST-elevation flag frequently originates from the ECG recorded *during* the
infarction being predicted.

### L5 — Referral-diagnosis leak in free text

MIMIC chief complaints for inter-facility transfers carry the referring
hospital's diagnosis verbatim: `"STEMI, Transfer"` (175 records),
`"Elevated troponin, Transfer"`, `"NSTEMI"`. It is present for **25.9% of STEMI**
and **15.7% of NSTEMI** cases versus 0.6% of non-ACS. A triage model must not be
credited for reading a diagnosis that arrived with the patient.

## 4. Why the gap persists

Each leak individually looks like a reasonable engineering choice — join the
comorbidity table, take the max troponin, use all the ECGs, keep the text. None
is detectable from a metrics table. All of them inflate results in the same
direction, so nothing looks anomalous. And because the field reports a single
number rather than an information-availability contract, there is no place in a
standard results section where the problem would surface.

## 5. What this component contributes

1. A **reusable leakage-audit instrument** for this data family, with five
   independent probes plus a controlled leaky-vs-safe experiment.
2. A **temporally-safe feature construction** in which every feature carries an
   explicit availability timestamp relative to ED arrival.
3. **Progressive Horizon Modelling** — instead of one number, the accuracy-vs-time
   curve, which makes the information contract a reported quantity.
4. A **constrained decision layer** that replaces hand-tuned probability boosts
   with a stated optimisation problem under a per-class recall floor.
5. **Honest reporting of what is and is not achievable** at 0.36% UA prevalence,
   including the arithmetic bound on end-to-end precision.

See `NOVELTY_AND_CONTRIBUTION.md`.
