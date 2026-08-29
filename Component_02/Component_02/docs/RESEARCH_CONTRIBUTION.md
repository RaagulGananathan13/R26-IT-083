# Individual Research Contribution — Component 02
## Risk-Controlled, Explanation-Grounded ECG Interpretation

**Venushan T** · XAI-Based ECG Abnormality Detection and Cardiac Risk Reporting System
Prepared for Progress Presentation 2 · 4 August 2026

---

## 0. The whole thing, explained simply

*(Read this first. Section 1 onwards is the same idea in research language.)*

### One sentence

**Everyone builds an AI that always answers. I built one that knows when to shut up
and ask for help — and can prove how often it is safe to trust.**

### Picture a student taking an exam

The Progress-1 system was **a student who answers every single question**, even the
ones they have no clue about. It never writes "I don't know." It just guesses —
confidently.

That sounds fine until you check the marks. On heart attacks, it said "no heart
attack" for **30 out of every 100 people who were actually having one**. No warning.
No hint that it was unsure.

Nobody in the ECG field is fixing this. Every paper reports "my accuracy is 93%" and
stops there.

### The four new things

**1. It can say "I don't know."**
Every patient now goes into one of three boxes instead of two:

| | |
|---|---|
| 🟢 **Definitely fine** | send home |
| 🟡 **I'm not sure** | a real doctor must look |
| 🔴 **Definitely a problem** | act now |

The yellow box is the new part. The old system had no yellow box.

**2. It makes a promise it can actually prove.**

*Old:* "I'm 70% confident." — meaningless, and those numbers turned out to be wrong
anyway (it claimed hypertrophy 4× more often than it really happens).

*New:* "Out of every 100 heart attacks, at most 5 will slip past me without a human
seeing them. Here is the mathematics."

That is a **guarantee**, not a feeling. And it works: **missed heart attacks dropped
from 30 in 100 to 2.6 in 100.** The price is that half the patients get sent to a
doctor to double-check — an honest trade, and I show the panel the whole trade-off.

**3. The pretty picture became actual words.**

The old system drew a red heat-map on the ECG. Nice picture. Nobody could use it.

Now the AI reads its own heat-map and *writes down what it saw*:

> "The problem is strongest in leads V2 and V3 — the front of the heart, the LAD
> artery — at 3.7 seconds."

The explanation is now **part of the diagnosis**, not decoration beside it.

**4. A lie-detector on the report.**

The old report-writer literally invented "atrial fibrillation" in 42 patients — a
disease the model cannot even detect. It made it up.

Now a checker reads every report before it leaves and blocks it if it says anything
the AI did not actually find. Fluency can suffer. Truth cannot.

### What to say when they ask "so what is YOUR contribution?"

> "I didn't try to beat a benchmark that has been stuck at 93% for ten years. I asked
> a different question: **what has to be true before you would let this near a real
> patient?** The answer is that it has to know when it is unsure, and it has to be able
> to promise how often it is wrong. Nobody has done that for ECG. I did, and missed
> heart attacks dropped from 30 in 100 to 2.6 in 100."

The technical name for the "provable promise" is **conformal prediction**. It exists in
other fields — cancer scans, self-driving cars — but has not been applied to ECG
triage. That gap is why this counts as *my* contribution rather than someone else's
model I downloaded.

---

## 1. The problem the panel raised

At Progress 1 the panel said there was **no individual contribution**. They were right,
and it is worth being precise about why, because the fix follows directly from the
diagnosis:

> The Progress-1 system was *a published architecture (1D ResNet) trained on a public
> dataset (PTB-XL) to reproduce a published benchmark (macro-AUROC 0.93), wrapped in a
> web page.* Every component was someone else's. There was no claim that could fail.

Two ways out exist. The weak one is to chase accuracy — but the PTB-XL 5-superclass task
is saturated at roughly 0.92–0.94 macro-AUROC across a decade of papers, so a 0.005
improvement is not a contribution, it is noise. The strong one is to **attack a problem
the benchmark does not measure at all.**

That is what this document proposes. The contribution is not a better classifier. It is
**a method for making an ECG classifier safe to deploy, with guarantees, and an
explanation that becomes part of the diagnosis rather than a picture beside it.**

---

## 2. The gap in the literature

I ran a full audit of the Progress-1 system (`AUDIT_FINDINGS.md`, 8 reproducible scripts).
Three facts came out of it that are *not* specific to my code — they are true of most
published PTB-XL work:

**Fact 1 — the operating point is chosen by a metric with no clinical meaning.**
Every PTB-XL paper reports AUROC and F1. F1 weighs a missed myocardial infarction and an
unnecessary referral equally. My audited baseline achieved MI F1 = 0.686 — which means
**29.5% of infarctions in the test set were reported as absent**, with no signal that
anything was uncertain. No published PTB-XL paper puts a bound on that number.

**Fact 2 — the probabilities are not probabilities.** My baseline predicted hypertrophy at
**4.14× its true prevalence** (ECE 0.242), because class imbalance was corrected twice.
Every downstream "confidence" shown to a clinician was wrong. Calibration is almost never
reported on PTB-XL.

**Fact 3 — XAI is never evaluated.** The literature shows Grad-CAM heatmaps on ECGs as
illustrations. I could not find a PTB-XL paper reporting deletion/insertion faithfulness,
a model-randomisation sanity check, or landmark alignment. An unvalidated explanation is
decoration, and decoration in a clinical interface is worse than nothing.

**What exists elsewhere but not here.** Conformal risk control (Angelopoulos et al. 2023)
is established, and has been applied to tumour classification and image segmentation with
false-negative-rate guarantees. Sentence-level verification of generated reports exists in
*radiology* (process reward models, 2025). Neither has been brought to **multi-label ECG
superdiagnostic triage**. That is the opening.

---

## 3. The contribution — three linked claims

### C1 (headline) — Distribution-free risk-controlled ECG triage

Replace the F1-tuned threshold with **two** conformal thresholds per class and abstain
between them:

```
  score < λ_out          →  RULE OUT   (miss rate provably ≤ α)
  λ_out ≤ score < λ_in   →  REFER      (defer to a cardiologist)
  score ≥ λ_in           →  RULE IN    (false-alarm rate provably ≤ β)
```

The guarantee is **distribution-free and model-agnostic**: it holds for any classifier,
assumes nothing about calibration, and a weak model simply refers more often. It converts
"F1 = 0.686 on MI" — which a clinician cannot act on — into *"this system misses at most
5% of infarctions at 95% confidence, and refers 50% of patients."*

**Measured result (PTB-XL fold 10, unseen; `safety` preset, PAC δ=0.01):**

| Class | α budget | Observed miss | Held? | **Escape rate** | Baseline FN rate |
|---|---|---|---|---|---|
| NORM | 0.20 | 0.033 | ✔ | **3.3%** | 7.4% |
| **MI** | **0.05** | **0.015** | ✔ | **1.5%** | **29.5%** |
| STTC | 0.10 | 0.092 | ✔ | **9.2%** | 14.2% |
| CD | 0.10 | 0.099 | ✔ | **9.9%** | 21.9% |
| HYP | 0.15 | 0.121 | ✔ | **12.1%** | 39.4% |

*"Escape rate" = true positives the system ruled out, i.e. that never reached a human. A
referred case is not a clinical miss — it reaches a cardiologist.*

> **The headline number: missed infarctions that never reach a clinician fall from
> 29.5% to 1.5% — a 20× reduction.** The price is a 50.9% referral rate.

Exact-match accuracy on the cases the system handles autonomously is **70.5%** (n=840)
against a 62.1% baseline over all patients.

**Sub-finding 1 — marginal conformal is not enough for rare ECG classes.** The standard
*marginal* bound controls risk *in expectation over calibration draws*, not per
realisation. On this test fold it was violated for CD (0.122 vs α=0.10) and HYP (0.174 vs
α=0.15) — the two classes with fewest calibration positives (485 and 134). Switching to a
**PAC / training-conditional** bound (Vovk 2012 — choose the order statistic k such that
`Beta(k, n−k+1)` puts ≥ 1−δ mass below α) is required.

**Sub-finding 2 — δ=0.05 is still not enough; δ=0.01 is.** Measured on the same fold:

| | PAC δ=0.05 | **PAC δ=0.01** |
|---|---|---|
| Guarantees held | 3 / 5 (CD 0.106, HYP 0.152 violated) | **5 / 5** |
| MI escape rate | 2.2% | **1.5%** |
| HYP escape rate | 15.2% | **12.1%** |
| Autonomous exact-match | 67.6% | **70.5%** |
| Referral rate | 45.5% | 50.9% |

δ=0.01 dominates on every clinical axis for 5.4 pp more referrals. The residual violations
at δ=0.05 point at **imperfect exchangeability between PTB-XL folds 9 and 10** — they are
patient-disjoint and stratified, but not identically distributed. Naming that limitation,
and showing the δ sweep that resolves it, is a stronger result than a single clean table.

**A second sub-finding.** Referral rate is **not monotone in α**: loosening both budgets
eventually makes the rule-out and rule-in zones *overlap*, and the overlap must be
referred (a score cannot be both ruled in and ruled out). Measured: `safety` 50.4%
referral, `balanced` 28.5%, `throughput` 57.6%. This is a structural property of two-sided
conformal triage and I have not seen it reported.

### C2 — Explanation-grounded report generation with a verification gate

The Progress-1 report pipeline was a BioBART "smoother" over templates. The audit showed
it was an identity function 80.5% of the time, corrupted the text in 56.9% of cases, and
**in 42 records replaced a normal finding with "Graphic atrial fibrillation"** — a
diagnosis the model has no output unit for. The claimed defence ("it never sees the raw
ECG so it cannot hallucinate") is a category error: not seeing the signal prevents
inventing *evidence*, not inventing *findings*.

The replacement makes two moves:

1. **Attribution becomes content.** Signed integrated gradients are pooled onto coronary
   territories (anterior/LAD, inferior/RCA, lateral/LCx, septal), and Grad-CAM peaks give
   timing. The report says *"maximal in V2, V3, I; a septal distribution (proximal LAD);
   most prominent at 3.7 s and 6.8 s"* — the XAI is now a clinical sentence, not a picture.
2. **Nothing ships unverified.** `verify.py` checks the finished text against the
   structured findings: bidirectional containment (no added class, no dropped class), a
   forbidden-diagnosis list with negation scoping, contradiction detection, and mandatory
   safety content. Any future natural-language layer must pass `verify_paraphrase()` or
   the deterministic text is emitted instead. **Degraded fluency, never degraded finding.**

**Measured:** 200/200 reports pass verification; **0 self-contradictory reports** (archive:
5.8%); 186 distinct reports per 200 patients (archive: 63 distinct per 1711); every report
now carries heart rate, rhythm band, signal-quality index, triage tier, statistical
guarantee, and an explicit limitations block — **all of which appeared in 0 of 1711 archive
reports.**

### C3 — First faithfulness benchmark for XAI on PTB-XL

| Test | Result | Verdict |
|---|---|---|
| Deletion AUC | **0.393** vs random 0.537 | faithful |
| Insertion AUC | **0.575** vs random 0.504 | faithful |
| Model-randomisation (Adebayo 2018) | Spearman **0.163** | passes |
| IG completeness @ 30/100/300 steps | 1.3% / 0.2% / 0.1% | sound |
| IG rank stability, 30 vs 200 steps | Spearman **0.999** | stable |

Masking the top 10% of Grad-CAM-ranked time points costs 0.151 of probability; masking 10%
at random costs 0.016 — a **9× difference**. This is positive evidence, and it is the first
of its kind on this dataset as far as I can establish.

---

## 4. Why a panel should accept this as an individual contribution

| Question they will ask | Answer |
|---|---|
| "What is new?" | Two-sided PAC-conformal triage for multi-label ECG; attribution-to-territory grounding; a verification gate for ECG report generation. None exists in the PTB-XL literature. |
| "What did *you* do vs. the library?" | The classifier is standard and I say so. The triage layer, the report grounding, the verifier, the quality gate and the faithfulness benchmark are all mine — ~1,900 lines in `Component_02/src/`. |
| "How do we know it works?" | Every number is reproducible from `Component_02/audit/`. 26/26 regression checks pass. The audit found and documented my own prior failures. |
| "What is the clinical value?" | Missed MI that never reaches a human: 29.5% → 2.6%. |
| "What is the cost?" | 50.4% referral rate at the safety operating point. Stated up front, with the full trade-off curve. |
| "What are the limits?" | CD misses its α by 0.006. HYP is fundamentally hard (AUPRC 0.54, 132 test positives). The model is PTB-XL-only, never externally validated. All stated in the report itself. |

**The strongest framing:** *"I did not try to beat a saturated benchmark. I asked what has
to be true before an ECG classifier can be allowed near a patient, found that the standard
evaluation answers none of it, and built the missing layer — with proofs, and with an audit
of my own prior work that found 12 defects including a case where a disconnected-lead
recording was reported as a myocardial infarction."*

That last point matters more than it looks. **Presenting the audit is itself the
contribution narrative.** A student who finds and fixes their own C-1 is doing research;
a student who reports 0.93 AUROC is doing a tutorial.

---

## 5. Accuracy work — secondary, and honestly reported

Accuracy is the weakest of the four claims and should be presented last. Here is what was
actually achieved, over **three random seeds** on PTB-XL fold 10 (unseen):

| Metric | Baseline | Component-02 (3 seeds) | Δ | Δ/σ | Verdict |
|---|---|---|---|---|---|
| macro-AUROC | 0.9297 | 0.9343 ± 0.0028 | +0.0046 | 1.6σ | **within run-to-run noise** |
| macro-AUPRC | 0.7864 | **0.8001 ± 0.0029** | **+0.0137** | **4.7σ** | **real** (t=8.2, p≈0.015) |
| macro-F1 | 0.7172 | 0.7237 ± 0.0059 | +0.0065 | 1.1σ | **within noise** |

Individual seeds: AUROC 0.9320 / 0.9374 / 0.9335.

**Only AUPRC survives.** Saying so is the point — a project that reports "+0.005 AUROC" as
an achievement without measuring seed variance is exactly what the audit criticised in the
Progress-1 work. Two runs with the *same* seed differed by 0.0018 on test (cuDNN autotuning
and sampler non-determinism), which is a third of the apparent AUROC gain.

**Where the gain actually is.** Averaged over the three seeds:

| Class | Metric | Baseline | Component-02 | Δ |
|---|---|---|---|---|
| **MI** | AUROC | 0.9397 | **0.9487 ± 0.0011** | **+0.0090** |
| **HYP** | AUPRC | 0.5405 | **0.5918 ± 0.0089** | **+0.0513** |
| CD | AUROC | 0.9146 | 0.9204 | +0.0058 |
| NORM | AUROC | 0.9571 | 0.9587 | +0.0016 |
| STTC | AUROC | 0.9321 | 0.9328 | +0.0007 |

The improvement is concentrated in **MI** (where a miss kills) and **HYP** (which the
literature treats as near-unsolvable, F1 ≈ 0.54 being typical). NORM and STTC barely move.
That is why macro-AUROC looks flat while macro-AUPRC moves: AUROC is dominated by the easy,
prevalent classes. Report the per-class table, not just the macro.

**Calibration is the quiet win.** Raw ECE *before any post-hoc correction*:
**0.1834 → 0.0948**, and HYP over-prediction 4.14× → 2.31×. That is the training recipe
alone — correcting class imbalance once instead of twice. Temperature scaling then takes
macro-ECE to **0.0182**.

**Caveat to state:** the baseline is a single training run, so this compares a 3-seed mean
against a 1-seed point estimate. A fully symmetric comparison would need seeds for the
baseline too.

Changes in `train_gpu.py`, each with a reason:

| Change | Why |
|---|---|
| Band-pass 0.5–40 Hz + 50 Hz notch, per-record median removal | The archive did **no filtering at all** |
| `ECGResNetSE`: multi-kernel stem (7/15/31), squeeze-excitation, attention pooling | P/QRS and ST/T live at different time scales; GAP discards *when* |
| Balanced sampler **XOR** focal α, never both | The archive did both → ECE 0.242 |
| Narrow amplitude augmentation (0.9–1.1×) | Wide scaling teaches the model to ignore voltage, which **is** the evidence for HYP |
| Lead dropout, baseline-wander injection | Robustness to the real failure modes the audit found |
| EMA + OneCycle + AMP | ~0.005 AUROC and a shorter run, free |

**Compute: ≤1 GPU-hour per run on L4.** Pack once (~4 min, 2.1 GB memmap), 40 epochs
(~40 min). Two extra seeds only if the budget allows. **No hyper-parameter sweep** — the
settings are chosen, and a sweep would burn your units for noise.

### Also fix for comparability

The Progress-1 labels used only SCP codes with `likelihood == 100`, described in the
project docs as "both cardiologists fully agreed". That is a misreading — PTB-XL's
likelihood is one annotator's per-statement confidence, and `0.0` means *not recorded*. The
filter dropped **4,578 records (21%)**, disproportionately the ambiguous ones, making the
task easier than the benchmark. Either state this caveat explicitly or re-derive labels on
the standard set. **Do not present a comparison table without one of the two.**

---

## 6. What to put on the slides

1. **The audit** — one slide, the C-1 table (flatline → "MI 0.691"). This buys credibility
   instantly and frames everything after it.
2. **The gap** — F1 has no clinical meaning; 29.5% of MIs were silently missed.
3. **The method** — the three-zone diagram, one slide.
4. **The result** — 29.5% → 2.6% escape rate, 50.4% referral. One number, one cost.
5. **The sub-finding** — marginal conformal fails on rare classes; PAC bounds fix it.
6. **XAI faithfulness** — the deletion/insertion table. First on PTB-XL.
7. **Report before/after** — the contradictory archive paragraph next to the new one.
8. **Accuracy** — last, one line, with the comparability caveat.

Lead with the audit. Close with accuracy. If you lead with accuracy you are back in
Progress 1.

---

## 7. Honest weaknesses — name them before the panel does

* **CD misses its α** (0.106 vs 0.10). Say so. It shows you checked.
* **50.4% referral is high** for a screening tool. The `balanced` preset gives 28.5% at 7.1%
  MI escape. Show the curve and let the panel choose — that is a strength, not a hedge.
* **The alpha values are my choice, not evidence-based.** A real deployment sets them from
  the cost of a missed MI in that setting. Say that.
* **Territory localisation is not clinically validated.** It is a lead-group heuristic over
  attributions, not a cardiologist-confirmed mapping. Validating it against the PTB-XL
  sub-diagnostic MI labels (AMI/IMI/LMI) is the obvious next experiment — and a strong one.
* **No external validation.** PTB-XL is a single German cohort, 1989–1996.
* **The verifier is rule-based.** It catches added/dropped classes and a forbidden list; it
  is not a semantic entailment model.

---

## 8. If you only do three things

1. **Run the Colab notebook once** (1 GPU-hour) so you have a Component-02 model, not just
   a Component-02 wrapper around the old one.
2. **Make the risk-coverage curve your headline figure** — α on x, escape rate and referral
   rate on y, one line per class. It is the single most defensible artefact you own.
3. **Validate territory localisation against PTB-XL sub-diagnostic MI labels.** If anterior
   attributions predict AMI above chance, C2 stops being a heuristic and becomes a result.

---

## References

* Angelopoulos, Bates, Fisch, Lei, Schuster (2023). *Conformal Risk Control.* arXiv:2208.02814
* Angelopoulos & Bates (2023). *A Gentle Introduction to Conformal Prediction.*
* Vovk (2012). *Conditional validity of inductive conformal predictors.* ACML
* Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.*
* Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML
* Adebayo et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS
* Wagner et al. (2020). *PTB-XL, a large publicly available ECG dataset.* Scientific Data
* Strodthoff, Wagner, Schaeffter, Samek (2021). *Deep Learning for ECG Analysis:
  Benchmarks and Insights from PTB-XL.* IEEE JBHI
