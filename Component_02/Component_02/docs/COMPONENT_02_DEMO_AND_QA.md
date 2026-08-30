# Component 02 — 10-minute demo and Q&A preparation

**Venushan T** · ECG Abnormality Detection and Cardiac Risk Reporting

Companion to `COMPONENT_02_COMPLETE_GUIDE.md`, which carries the full technical
detail and every number.

---

## Part 0 — before you walk in

### The spine of the talk

Everything you say should serve one sentence:

> **A guarantee that holds on average is not a guarantee for your patient.**

Everything before it earns the right to say it. Everything after it is evidence.
If you find yourself saying something that does not serve that sentence, cut it.

### Two sentences to memorise word for word

Everything else can be improvised. These two cannot.

**The contribution claim (you say this at 5:00):**
> A conformal ECG system can satisfy its advertised guarantee exactly — and still
> be unsafe for identifiable groups of patients.

**The close (you say this at 9:15):**
> Conformal prediction for ECG is not mine — it is published. What is mine is the
> measurement that the guarantee it advertises is not the guarantee a clinician
> receives, on three independent axes, with the fix and the cost of the fix both
> reported.

### Setup checklist — do this the night before

- [ ] `python -X utf8 backend/server.py` starts and reports `browseEnabled: true`
- [ ] Frontend loads at `http://localhost:5173`
- [ ] **Pick your demo record in advance and test it.** Do not browse live.
- [ ] Have a **screenshot or screen recording** of the working output as a fallback.
      Something always breaks in front of a panel.
- [ ] Know that `audit/08_verify_fixes.py` stops on `ecg_id 9` — those files were
      never included in the handover. Do not run it live.
- [ ] Have `analysis/results/02_operating_point.txt` and
      `audit/results/07_conformal_eval.txt` open in tabs, in case a number is challenged.
- [ ] Slides: **7, at most 8.** One idea each.

### Slide budget

| # | Slide | The one thing it carries |
|---|---|---|
| 1 | Title + the 29.5% number | The problem is not accuracy, it is silence about uncertainty |
| 2 | Pipeline diagram, quality gate highlighted | The gate comes *before* the model |
| 3 | Three-zone number line | Five independent decisions, not one choice |
| 4 | Escape-rate table | 29.5% → 1.5%, at 50.9% referral |
| 5 | Subgroup violation table | Two cells survive Holm correction |
| 6 | Electrode + out-of-scope, one slide | Same failure, two more axes |
| 7 | Report before/after | The generator was measured and removed |
| 8 | The close, one sentence on an otherwise empty slide | Stop talking |

---

## Part 1 — the 10-minute runsheet

Times are **cumulative**. If you are past the mark at a checkpoint, cut from the
block that follows — **never** from the close.

---

### 0:00 – 1:00 · Open

**On screen:** title slide, or nothing.

**Say:**

> A 12-lead ECG has to be read within ten minutes of a patient arriving, and that
> reading decides whether someone goes to the cath lab. Machine ECG interpretation
> already exists in every hospital. The gap is not accuracy — it is that these
> systems never say *"I don't know."*

**Land the number:**

> The baseline I inherited missed **29.5% of infarctions** and reported them as
> normal, with nothing on the screen suggesting anything was uncertain. That is
> what I set out to fix.

**Do not:** introduce yourself at length, list the five classes yet, or apologise
for anything.

---

### 1:00 – 2:00 · The architecture, fast

**On screen:** the pipeline diagram.

**Say:**

> A 1-D residual network over 12 leads × 5,000 samples — ten seconds at 500 Hz.
> Two design choices worth naming: a **multi-kernel stem**, because a P wave lasts
> 80 milliseconds and a T wave 200, and one kernel width cannot see both; and
> **attention pooling**, because average pooling throws away *when* something
> happened, and an infarct is a local event. 1.59 million parameters, about 20
> milliseconds per trace.

**Then point at the gate — this is the part that matters:**

> The quality gate sits **before** the classifier. A trace that fails quality
> control produces **no probability at all**. That ordering is deliberate: an
> uninterpretable ECG must never be able to return a reassuring number. The old
> system, given an all-zero recording from disconnected electrodes, returned
> *"consistent with myocardial infarction"* at 0.69.

**Do not:** walk the layer list. One minute buys the stem, the pooling and the
gate. Nothing else.

---

### 2:00 – 3:30 · How a prediction is actually made

**On screen:** the three-zone number line.

**Correct the assumption before they make it:**

> This is not a five-way choice. There is **no softmax and no argmax** anywhere in
> the decision path — five independent sigmoids, and each class is decided against
> **its own two thresholds**. A patient can have three findings, one, or none.
> That is the right shape: a real heart can have both a conduction disturbance and
> hypertrophy, and forcing the model to choose would be clinically wrong.

**The three zones:**

> Below `λ_out`, the class is **ruled out**, with a provable miss rate at most α.
> Above `λ_in`, it is **ruled in**, with a false-alarm rate at most β. In between,
> the system **does not decide**, and says so.

**Where the thresholds come from:**

> These are not chosen by hand and not chosen by maximising F1. They are order
> statistics of the calibration scores — sort the scores of everyone who genuinely
> has the condition, and take the m-th smallest. That is why the guarantee holds
> for **any** model. A weak model does not break the promise; it just refers more
> patients.

**The line that sells it:**

> MI carries α = 0.05 because a missed infarction kills. NORM carries α = 0.20
> because missing NORM only costs an unnecessary review. Those budgets are
> **clinical policy**, and I put them in the open so they can be argued with
> rather than hidden inside a threshold.

---

### 3:30 – 5:00 · Results

**On screen:** escape-rate table.

**Headline first:**

> Every class clears 0.75 accuracy and 0.75 recall on a test fold used once.
> Macro-AUROC 0.9343 ± 0.0028 across three seeds.

**Then the number that actually matters:**

> A referred case is **not** a clinical miss — it reached a human. The only true
> miss is a positive that was ruled out and therefore never reached anyone.
> **Missed infarctions fall from 29.5% to 1.5%.**

**Say the price in the same breath — do not let them ask for it:**

> The cost is that **50.9% of patients** have at least one class deferred to a
> cardiologist. That trade-off curve, not a single accuracy figure, is the
> deliverable.

**Pre-empt the F1 question here rather than defending it later:**

> You will notice hypertrophy F1 is 0.41. F1 weights a missed infarction and an
> unnecessary review equally, and no cardiology pathway does that — the ESC 0/1
> hour algorithm and HEART are both governed by sensitivity and NPV. Hypertrophy
> NPV is **0.981**. That is the number that decides whether you can rule it out.

---

### 5:00 – 7:30 · The contribution — the centre of the talk

Give this the most time. If you are running late, you have already cut from
earlier blocks.

**The claim, said once, slowly:**

> A conformal ECG system can satisfy its advertised guarantee exactly — and still
> be unsafe for identifiable groups of patients.

#### Axis 1 · the patient (~60 s)

**On screen:** subgroup violation table.

> Every class passes overall. Inside subgroups, two violations survive Holm
> correction over 23 comparisons and a 2,000-draw bootstrap that refits the
> threshold each draw. **Conduction disturbance in under-50s misses 33.3% against
> a promised 10%** — adjusted p of 5 times ten to the minus six. The confidence
> interval lies entirely above the bound.

> The overall CD figure is 9.9% — comfortably inside the promised 10%. It gives no
> hint that a third of young patients with conduction disturbance are missed.

**Add the clinical weight — this is what makes a cardiologist react:**

> Conduction disturbance in the young is not benign. Under 50 it raises Brugada,
> ARVC and inherited conduction disease — causes of sudden cardiac death in young
> adults. That is the group where a miss is least acceptable, and it is where the
> system fails worst.

**Mention the negative result — it is a credibility marker:**

> Seven other apparent violations did **not** survive multiple-testing correction,
> and I report them as noise. Saying so is part of the result.

#### Axis 2 · the recording (~40 s)

> A swapped cable produces a perfectly clean signal. The quality gate accepted
> **587 of 600** reversed traces at a mean quality index of 1.000. But RA/LA
> reversal flips at least one label in 86.8% of records and **voids nine
> guarantees** — under it the system fails to recognise 99.6% of normal ECGs while
> still printing its promise.

> The detector is a stated physiology rule, not a classifier: aVR is negative and
> lead I positive in essentially every normal heart, so positive aVR with inverted
> lead I is the signature. 70% sensitivity, 4.5% false positives.

#### Axis 3 · the label space (~40 s)

> There is no sixth output unit. Five sigmoids can each say "not me", but nothing
> can say "none of the above, and here is what it is instead."

> Of 114 atrial-fibrillation records in the test fold, **113 carried a statistical
> guarantee. None of those guarantees concerned atrial fibrillation. Two were
> reported as a normal ECG.**

**Then the line that widens it beyond your project:**

> Every five-superclass PTB-XL paper inherits this, because the benchmark scores
> only the five classes it defined.

#### Close the block by unifying

> Marginal validity holds across the population, across all recordings, and across
> the label space. Conditional validity fails within a subgroup, within a
> mis-acquired recording, and when the disease is not in the space. **One
> principle, three independent demonstrations.**

---

### 7:30 – 8:30 · The fixes, and the report model

**On screen:** report before/after.

**Fixes, one line each:**

> Mondrian calibration takes cells satisfying the bound from 14 of 23 to 22 of 23.
> Electrode reversal and out-of-scope rhythm both **withdraw the guarantee while
> keeping the diagnosis**. Each fix is partial, and I report the residual: the
> scope check catches 60 of 114 AF records, the electrode detector restored only
> 1 of 9 voided guarantees, and Mondrian makes one cell infeasible outright.

**The report model — frame it as a finding, not a deletion:**

> The previous system's language contribution was a BioBART smoother — a
> sequence-to-sequence transformer paraphrasing a template. Audited on all 1,711
> test records, it was byte-identical to its input 23.6% of the time and identical
> but truncated a further 56.9% — an identity function plus a truncation bug. It
> dropped a clinical finding in 103 records. And it invented **atrial
> fibrillation**, a class the model has no output unit for, in **41**.

> All 41 are the *same* substitution: "ECG shows predominantly normal features"
> becomes "Graphic atrial fibrillation." One template opening, one deterministic
> failure mode.

> It was removed on evidence and replaced with a generator where every sentence is
> emitted from a Finding object carrying its own evidence, behind a verifier that
> any future language model must also pass.

---

### 8:30 – 9:15 · Limitations, volunteered

**Say these before anyone asks.**

> PTB-XL only, no external validation. Five superclasses — atrial fibrillation is
> not detected, and its absence from a report is not evidence of its absence.
> Intervals and axis are not measured. Territory localisation is a lead-group
> heuristic, not clinically validated. And every fix is partial, with each
> partiality measured rather than estimated.

**Why you volunteer them:** a panel that *finds* a limitation you hid discounts
everything else you said. A panel that watches you name it reads the rest as
measured.

---

### 9:15 – 10:00 · Close

> Conformal prediction for ECG is not mine — it is published. What is mine is the
> measurement that the guarantee it advertises is not the guarantee a clinician
> receives, on three independent axes, with the fix and the cost of the fix both
> reported.

**Then stop talking.** No summary slide. No re-listing of contributions. No
filling the silence. The last sentence is the one they carry into questions.

---

## Part 2 — the live demo (if you show one)

**Have it loaded but do not plan to use it.** If a panellist asks, one record
end-to-end is worth two minutes of description. If nobody asks, do not offer — it
eats the clock and something always breaks.

### The 90-second version

Pick **one MI record** you have tested. Walk it in four beats:

**1 · The gate passed, and here is what it measured (15 s)**

> Signal quality index 1.00, heart rate 61.6, all twelve leads usable. If any of
> that had failed, we would not be looking at a probability right now.

**2 · Five decisions, not one (30 s)**

> Three classes ruled out, each with a stated miss-rate bound. One referred —
> conduction disturbance sits between the thresholds, so the system explicitly
> refuses to decide it. And MI ruled in.

Point at the referred class specifically. **That is the demo.** Anyone can show a
correct prediction; showing a *refusal* is the thing your component does that
others do not.

**3 · Why it said that (30 s)**

> Leads V1, V2 and V5, concentrating in the septal territory — proximal LAD. The
> Grad-CAM peaks at 1.5 and 3.8 seconds. Note the peak times and the curve are
> computed by different code paths; that they agree is a cross-check.

**4 · What it promises, and what it will not promise (15 s)**

> "Misses at most 5% of infarctions, at 99% confidence." And if I had flagged this
> recording as electrode-reversed, that sentence would be replaced by GUARANTEES
> SUSPENDED — the diagnosis stays, the promise goes.

### The one demo that beats all the others

If you have time to prepare only one thing, prepare **a refused record**.

Feed it a flatline or a corrupted file and show that it returns **no probability
at all** — just the refusal and the reason. Then say:

> The system I audited returned "consistent with myocardial infarction" at 0.69 on
> this input.

That single contrast makes the entire design argument in fifteen seconds.

---

## Part 3 — Q&A preparation

Answers are written to be **said**, not read. Each opens with the shortest true
answer, then the evidence if they push.

---

### Tier 1 — you will almost certainly get these

**Q: "Isn't your accuracy just average? 0.86 is not impressive."**

> Accuracy is not the deliverable — the risk–coverage curve is. At 49% autonomous
> handling I miss 1.5% of infarctions. Ask any published PTB-XL model what its miss
> rate is at a stated coverage and it cannot tell you, because it never abstains.
> The comparison is not accuracy against accuracy; it is a number with a bound
> against a number without one.

---

**Q: "Why is your F1 so low on hypertrophy?"**

> Because F1 is the wrong metric for a rule-out system, and hypertrophy is the
> hardest class in this dataset for four measurable reasons.

> F1 weights a missed infarction and an unnecessary review equally. No cardiology
> pathway does that. Hypertrophy F1 is 0.41; hypertrophy **NPV is 0.981**.

> And the four reasons are measured, not asserted: 7.7% prevalence with only 132
> positives in the test fold; 63.8% of hypertrophy records also carry ST/T change
> and none are normal; the diagnosis rests on QRS **amplitude**, which any
> amplitude normalisation destroys — which is why my augmentation is capped at
> 0.9 to 1.1×, not 0.8 to 1.2×; and the ECG criteria for LVH have known low
> sensitivity against echocardiography, so the label itself is a proxy.

> The published ceiling for hypertrophy F1 on PTB-XL is about 0.54. My AUPRC
> improved from 0.5405 to 0.5842 over the audited baseline — a real gain on the
> hardest class.

---

**Q: "Is conformal prediction your contribution?"**

**Say no immediately. Hesitating here looks like you were hoping they would not ask.**

> No. Conformal prediction is published — Vovk, Angelopoulos and Bates are cited,
> and it has already been applied to ECG in 2025. Mondrian calibration is Vovk,
> 2003.

> What is mine is the measurement that the guarantee is not valid conditionally —
> on the patient, on the recording, and on the label space — with the fix and the
> cost of the fix both quantified. No prior work I found asks that question of an
> ECG system.

---

**Q: "Why not just detect atrial fibrillation? It is the most common arrhythmia."**

> Because the model has no output unit for it, and I am not permitted to name a
> class I never trained on. Adding it would mean retraining on a different label
> space, which is a different project.

> What I did instead is answer a narrower and more honest question: *is this rhythm
> outside the region where my guarantee was calibrated?* R-R irregularity gives
> AUROC 0.912 for that, and the pipeline already detects R-peaks for heart rate,
> so the feature costs nothing. At a 5% false-positive budget it catches 60 of the
> 114 AF records and withholds the guarantee on them. The diagnosis is still
> reported. Only the false promise is removed.

> It is partial and I say so: 53 of the 114 are still missed and still receive a
> guarantee.

---

**Q: "Show me it working."**

Use the refused-record demo above. It is stronger than a correct prediction.

---

### Tier 2 — the sharper ones

**Q: "You ship `resnet_se`, but your own ablation says it is the worst of the
three. Why?"**

**This is the hardest question you will get. Do not bluff it.**

> That is correct, and it is my own measurement. Squeeze-excitation costs 0.0042
> macro-AUROC at p = 0.004, and almost all of it lands on hypertrophy — the one
> class whose evidence is amplitude, which is exactly what an SE block re-weights.

> The shipped bundle predates the ablation. The weights are dated 18 August; the
> architecture comparison 26 to 27 August. The calibrator and the conformal
> thresholds are fitted **per model** — swapping the backbone means refitting the
> entire safety layer and re-verifying every guarantee on the test fold.

> The correct next step is refitting onto `resnet_se_no_se`, and I state that. What
> I did not do is quietly re-label the shipped model to match the better result.

---

**Q: "Your guarantee held on the test fold. How do I know it holds on my hospital's
data?"**

> You do not, and the guarantee does not claim that. Conformal validity is
> **marginal over an exchangeable calibration distribution** — it holds for data
> that looks like the calibration data.

> That is precisely why this project exists. My whole contribution is measuring
> what happens when the exchangeability assumption breaks: a different age
> subgroup, a mis-wired recording, a disease outside the label space. All three
> break it, and all three are ways your hospital's data could differ from mine.

> The honest answer is: recalibrate on your own population. The method transfers;
> the thresholds do not.

---

**Q: "You removed the report model. Isn't that a reduction in scope?"**

> The opposite. The contribution was never the generator — it was discovering that
> a report generator in a clinical system needs a **gate**, and building the gate.

> The seq2seq layer was measured against its own safety claim. That claim was "it
> cannot hallucinate because it never sees the raw ECG", and it confuses two
> things: not seeing the signal prevents inventing *evidence*, it does nothing to
> stop a paraphraser adding, dropping or negating a *finding*. It dropped findings
> in 103 records and fabricated atrial fibrillation in 41.

> What ships now emits heart rate, rhythm, signal quality, the conformal zone, the
> guarantee, the territory and the triage tier — every one of which the old
> pipeline emitted in **zero** of 1,711 reports, despite being called a "Cardiac
> Risk Reporting System."

---

**Q: "Can I see the report model run?"**

> No — and I should be straightforward about why. Its checkpoint did not come
> across into this component; only its frozen output on all 1,711 test records
> did. The audit is fully reproducible, the model is not re-runnable here.

> The stronger point survives anyway: the free-text tier scores ROUGE-L 0.587 on
> normal records and 0.403 on abnormal ones. It scores best exactly where there is
> nothing to say. That is why ROUGE was abandoned as the report metric.

---

**Q: "50.9% referral is very high. Is this usable?"**

> It is high, and it is a policy choice rather than a fixed property. Three presets
> ship — safety, balanced and throughput — and the whole risk–coverage curve is
> reported rather than one point. You pick the operating point your clinical
> setting can staff.

> There is also a property worth knowing: the referral rate is **not monotone** in
> α. Once the rule-out and rule-in regions overlap, the overlap must be referred,
> because a score cannot be simultaneously ruled in and ruled out. So very loose
> budgets can *increase* referrals. That is a property of two-sided conformal
> triage, not a bug, and I state it.

---

**Q: "How do I know your test fold is really untouched?"**

> Three things. The splits are the official PTB-XL fold protocol — 1 to 8 train,
> 9 validation, 10 test — not my own partition. Patient overlap between all three
> pairs is zero, verified, which matters because 9.9% of patients have more than
> one recording. And every threshold, temperature and conformal bound is fitted on
> fold 9 only; fold 10 is scored once.

> The audit script that checks this is `analysis/01_dataset_deep_audit.py` and it
> runs in seconds.

---

**Q: "Your results are not comparable to published PTB-XL benchmarks."**

> Correct, and I say so rather than being caught on it. I kept only SCP codes with
> likelihood equal to 100 — only diagnoses the cardiologist was certain about —
> which drops 21% of the dataset. That makes the task cleaner and my numbers not
> directly comparable. I report the number; I do not claim the comparison.

---

### Tier 3 — the ones that mean they are interested

**Q: "Why calibrate at all if the conformal layer is distribution-free?"**

> Good question — the conformal guarantee genuinely does not need it. Temperature
> scaling is monotone, so it cannot reorder patients and the bounds are
> mathematically unchanged.

> It is there for the **human**. The old system printed raw sigmoid outputs to a
> clinician as "probability %" while hypertrophy was over-predicted 4.14 times.
> The guarantee was fine; the number on the screen was a lie. Calibration fixes
> the number a person reads, not the maths.

---

**Q: "Why PAC rather than the standard marginal conformal bound?"**

> Because the marginal bound controls the miss rate *in expectation over repeated
> calibration draws*, and I only get one calibration draw.

> I measured it: the marginal version was violated on this single test realisation
> for conduction disturbance, 0.122 against a promised 0.10, and hypertrophy,
> 0.174 against 0.15 — the two classes with fewest calibration positives. The
> training-conditional PAC bound at delta 0.01 holds all five. The cost is
> referrals, 45.5% to 50.9%, and I report that.

---

**Q: "Your two models disagree. Which one is right?"**

> Neither, individually — and that is the finding. They disagree on at least one
> class in **58.9%** of records, and reach opposite conclusions, one ruling a class
> in while the other rules it out, in **10.5%**.

> The serving rule is that a class is ruled out only if **every** model rules it
> out, so the merged rule-out set is the intersection and the merged miss rate is
> bounded by the tightest single-model guarantee. It misses fewer true positives
> than either model on every class, because they do not miss the same cases. The
> price is referrals.

> The uncomfortable part is what it says about single-model deployment: a
> clinician sees one of those two answers, with a guarantee attached, and no
> indication the other exists.

---

**Q: "Is the Grad-CAM faithful? How do you know it is not just a pretty picture?"**

> Honest answer: I treat it as a check on *where the model looked*, never as
> localisation evidence. The peak times and the attention curve are computed by
> different code paths, so their agreement is a cross-check, and I state it so it
> is checkable.

> The territory mapping is a **lead-group heuristic**, not a clinically validated
> localiser, and I label it that way in the report itself.

> Component 01 in this project measured Grad-CAM repeatability on chest films at
> SSIM 0.12 — the literature is clear that saliency maps are the easiest thing in
> a system like this to over-read.

---

**Q: "What would you do next, with another six months?"**

> Three things, in order. **External validation** on a non-German cohort, because
> conformal validity is exactly the property most likely to break under
> distribution shift and I have no evidence about that. **Refit the safety layer
> onto `resnet_se_no_se`**, which the ablation says is the better backbone.
> And **extend the label space** so that atrial fibrillation is a class rather than
> a scope check — the scope check is a mitigation, not a solution.

---

### Tier 4 — hostile or off-target, and how to stay calm

**Q: "This is not really novel, it is an application of existing methods."**

> Application of an existing method to a new domain, on its own, would be fair
> criticism. What I am claiming is a **measurement result**: that the guarantee
> these methods advertise is not the guarantee the clinician receives, demonstrated
> on three independent axes with significance testing. That measurement did not
> exist before, and it changes how the method should be deployed. The fix I apply
> is prior work; the evidence that a fix is *needed* is not.

---

**Q: "Could this be used clinically?"**

> No, and nothing in the system claims otherwise. It is a research prototype, not
> a medical device, with no clinical validation and no external validation. Every
> output carries a disclaimer, and the report itself refuses language like
> "confirmed diagnosis" through the verifier's overclaim check.

> What I would say is that the *mechanism* — declining to answer where the evidence
> is weak — is the property such a system would need before that conversation
> could start.

---

**Q: (a question you genuinely do not know the answer to)**

> I do not know. I have not measured that. What I can tell you is [the nearest
> thing you did measure], and measuring [their question] would take [the concrete
> approach].

**Never invent a number.** A panel forgives "I have not measured that." It does
not forgive a figure that turns out to be wrong, and it will remember it for every
other number you gave.

---

## Part 4 — delivery notes

**One number per sentence.** "29.5% to 1.5%" lands. "Macro-AUROC 0.9343 with
AUPRC 0.8001 at accuracy 0.864" is noise — put those on the slide and say one of
them out loud.

**Never say "unfortunately" about your own limitation.** Say *"measured"*, *"and
the cost of that is"*, *"which I report"*. A limitation delivered as an apology
reads as a flaw; the same limitation delivered as a measurement reads as rigour.

**When challenged, do not immediately concede and do not immediately defend.**
Say what you measured. If they are right, say "that is a fair point and I have not
measured it" and move on. Both of those are stronger than arguing.

**Numbers you must not get wrong:**

| Figure | Correct value | Common slip |
|---|---|---|
| Hallucinated AF records | **41** | `docs/AUDIT_FINDINGS.md` says 42 — the data says 41 |
| Referral rate, shipped | **50.9%** | 44.9% is the *baseline* model at δ = 0.05 |
| Autonomous handling | **49.1%** | 55.1% is the baseline |
| Test fold size | **1,711** | 1,709 is validation |
| MI escape rate | **1.5%** | — |
| Records used / official | **17,221 / 21,799** | — |
| Shipped model | **`resnet_se`, seed 0, δ = 0.01** | `07_conformal_eval.txt` is the *baseline* |

**Before the panel, do one consistency pass** over `docs/`. Three numbers were
found drifting from the artefacts that produced them; there may be more. Every
figure you say should be reproducible from a named script in front of them.

---

*Full technical detail: `COMPONENT_02_COMPLETE_GUIDE.md`*
