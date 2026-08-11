# What Is My Novelty?

**Raagul Gananathan · IT22130020**
*Component: Cardiomegaly Detection with XAI and Automatic Report Generation*

> This document answers one question honestly: **what did I independently contribute?**
> Every claim here is backed by a measured number from my 4,722-image test set.
> Things I did *not* invent are listed first, so nothing is accidentally over-claimed.

---

## Part 1 — What is NOT my novelty

I have to be honest about this first, because claiming any of it would be wrong.

| Component | Why it is not mine |
|---|---|
| ConvNeXt-Base classifier | someone else's architecture (Meta, 2022) |
| BioBART report decoder | someone else's model (2022) |
| Grad-CAM heatmaps | published 2017 |
| CheXpert labelling | Stanford, 2019 |
| **The AP/PA performance gap existing** | already published (medRxiv 2026) |
| **Removing prior-study references** | CXR-PRO already did this |
| My accuracy numbers | good, but not better than published work |

**Assembling existing models — however carefully — is not a research contribution.**
That is exactly what my panel told me at Progress 1, and they were right.

---

## Part 2 — What IS my novelty

Three things. All measured, all reproducible, all mine.

---

# ⭐ Contribution 1
## I proved the fairness measurement this field uses is broken

### Explained simply

Chest X-rays come in two kinds:

🧍 **PA** — the patient can **stand up**. Clear picture.
🛏️ **AP** — the patient is **too sick to stand**, so the X-ray is taken in bed. Blurrier,
the heart looks bigger, the shoulder blades cover the lungs.

Researchers want to check whether an AI is **fair** to both groups. Everyone uses the same
test, called **TPR Disparity**. Think of it as **a pass mark on an exam** — you set a
score, and anyone above it is called "positive".

### What a published paper did

A peer-reviewed paper — **Pereira et al., MIDL 2023** — tried to improve that fairness
score. They:

- **retrained their entire AI** using a complicated adversarial method
- used a lot of computing power
- and **their AI got worse at diagnosing** as a result (−0.91 AUC)

Their fairness score improved by **46.7%**.

### What I did

**I retrained nothing.** I simply used **a different pass mark for bed X-rays than for
standing X-rays**. It took seconds and cost nothing.

**My fairness score improved by 73.3%.**

| | Pereira et al. (MIDL 2023) | **Me** |
|---|---|---|
| Fairness score improvement | 46.7% | **73.3%** |
| Accuracy lost | −0.91 AUC | **0.00** |
| Retraining needed | **Yes** | **No** |
| Compute used | GPU training | **none** |

### The part that matters most

I then measured whether my AI had actually gotten better at telling sick patients from
healthy ones. Here is the result, printed to ten decimal places:

```
one pass mark for everyone     : 0.8554320277
separate pass mark per group   : 0.8554320277
difference                     : 0.0000000000
```

**The AI did not change. Not by one part in a trillion.**

The fairness score moved 73%. The AI was exactly the same AI, making exactly the same
judgements, in exactly the same order.

> ### 🔑 The finding
> **The test everyone uses does not measure fairness. It measures where you drew the line.**
>
> A model can look 73% "fairer" while being provably, bit-for-bit, identical.

### Why this counts as a contribution

- I **beat a peer-reviewed paper on its own chosen measurement**, for free
- I proved **mathematically** that the improvement was meaningless
- I searched the literature specifically for this and **did not find it published** for
  chest X-ray projection bias

**This is mine.**

---

# ⭐ Contribution 2
## I proved this problem cannot be fixed by building a better AI

### Explained simply

My AI really is worse on bed X-rays. This is not a measurement trick — it is real:

| | AI score |
|---|---|
| 🧍 standing (PA) | **0.8864** |
| 🛏️ bed (AP) | **0.8224** |
| **gap** | **0.0639** |

And this matters enormously, because **bed X-rays come from the sickest patients.**
The AI is weakest exactly where it matters most.

So I tried **three completely different ways** to fix it.

| Attempt | In simple words | Result |
|---|---|---|
| **1. Change the pass mark** | grade the two groups differently | ❌ **provably impossible** — proved it cannot work, to 1e-12 |
| **2. Blindfold the AI** | stop it seeing which X-ray type it is | ❌ AI got **much worse** (−0.0789), gap barely moved |
| **3. Two specialists** | give it separate rules for each type | ❌ **no help at all** (+0.0003) |

Attempt 2 is worth explaining. I blindfolded the AI *completely* — it became literally
unable to tell a bed X-ray from a standing one (score exactly **0.5000**, pure coin-flip,
**better blindfolding than the published paper achieved**). And the gap still barely
moved. So it isn't that the AI is "cheating" by noticing the X-ray type.

I also found that pushing harder makes things worse in a specific way: at high strength
the gap *reverses* while the AI's accuracy falls to **0.4650 — worse than random
guessing.** Equality achieved by breaking the model for everybody.

### 🔑 The finding

> **The gap is not the AI's fault. Bed X-rays genuinely contain less usable information.**
>
> The shoulder blades sit over the lungs. The heart is magnified because the patient is
> closer to the machine. Lying down makes fluid spread out differently. Portable machines
> produce lower-quality images.
>
> **You cannot invent information the camera never captured.**

### Why this matters

It tells the field to **stop trying to fix this with clever algorithms** and address it
**at the X-ray machine**, or by flagging bed X-rays for extra human review instead of
pretending the AI is equally reliable on them.

Nobody had tested all three intervention types on this problem.

---

# ⭐ Contribution 3
## I built the fix that Contribution 2 said was the only one left

Look again at what Contribution 2 concluded: *flag bed X-rays for extra human review
instead of pretending the AI is equally reliable on them.*

Contribution 3 is me actually doing that — and measuring it.

### Explained simply

Right now my AI answers **every** X-ray, even the ones where it is basically flipping
a coin. It never says "I'm not sure."

So I let it say that.

If the answer is **0.95** → confident yes. If it's **0.03** → confident no. But if it's
**0.41** when the cut-off is **0.409**? That's a coin flip wearing a lab coat. **Send that
one to a doctor.**

That part is not new — it's called selective prediction, and other people have done it.

### 🔑 The new part

**Bed X-rays and standing X-rays should not get the same amount of caution.**

> **Think of a student marking exam papers.**
>
> Some papers have messy handwriting, some are neat. The student makes more mistakes on
> the messy ones — that's just true, it's not their fault.
>
> A bad rule: *"set aside 20% of all papers for the teacher."*
> A good rule: **"set aside lots of the messy ones and almost none of the neat ones."**
>
> Same amount of teacher time. Very different result.

My AI does the second one. And I never told it to — I let it work out the right amounts
from the validation data, and it decided on its own to **refer 23% of bed X-rays and only
0.3% of standing ones.**

### What I did

I tested **four** versions, because three of them exist only to try to prove the fourth
one is nothing special:

| | What it does | Why it's there |
|---|---|---|
| **A** | answer everything | the starting point |
| **B** | skip cases **at random** | 🪤 the trap — proves the confidence score is real |
| **C** | skip the unsure ones, **same rate everywhere** | the fair fight |
| **D** | skip the unsure ones, **more on bed X-rays** | mine |

### 🔑 The finding

| | Accuracy | Bed-vs-standing gap |
|---|---|---|
| A · answer everything | 83.2% | **6.68** |
| B · random skipping | 83.2% | — |
| C · skip evenly | **89.0%** | **6.28** ← *barely moved* |
| D · skip more on bed X-rays | 88.0% | **−0.62** ← *gone* |

Three things fell out of this:

**1 · Random skipping does nothing.** 83.2% → 83.2%. So the improvement in C and D is
really coming from the AI knowing when it's unsure — not from just throwing away hard cases.

**2 · Skipping evenly does NOT fix the unfairness.** 6.68 → 6.28. Almost nothing! Being
more careful *everywhere* helps both groups equally and leaves the gap exactly where it was.

**3 · Skipping unevenly fixes it.** 6.68 → −0.62. Gone.

### Why this counts as a contribution

It's the **same lesson as Contribution 1, proved a second time with a different mechanism**:

> Contribution 1 — the **cut-off** should depend on how the X-ray was taken.
> Contribution 3 — so should **how much you trust it**.
>
> Both work. Everything that tried to fix the *AI itself* failed.

And it fixes the gap by making the AI **more careful on bed X-rays** — not by making it
worse on standing ones. That matters: in Contribution 2, the clever method "fixed"
fairness by breaking the good group. This levels **up**.

### ⚠️ The honest catch — say this out loud

The AI now refuses about **1 in 7 cases.** So:

> ❌ *"My model gets 88%."* — that's cheating.
> ✅ **"88% accuracy on the 81% of cases it chooses to answer, referring 19% to a radiologist."**

If you don't say the second half, someone will catch you. Said properly, it isn't a
weakness at all — **it's exactly how real hospital triage software works.**

### And one idea I killed

I first tried something that sounded smarter: check whether the **classifier** and the
**report** agree with each other, and refer the cases where they argue.

It lost to the dumb version. Plain confidence scored **86.64%**; my clever idea scored
**85.57%**. So I threw it away.

That's now the **fifth** hypothesis of mine that my own experiments have destroyed.

---

# ⭐ The five ideas I killed myself

This is the section I am most proud of, and the one most students cannot show.

**Five times I had a clever idea. Five times I built the experiment that could prove it
was rubbish. Five times it was rubbish. And I wrote all five down instead of hiding them.**

### Why a "control" is the whole point

> Imagine you take a magic pill and your headache goes away.
>
> Did the pill work? **You have no idea.** Headaches go away on their own.
>
> To actually know, you need a second person who takes a **fake** pill. If they get better
> too, your magic pill is just a sugar pill with good marketing.
>
> That fake pill is called a **control**. Every one of my five ideas died because I built
> the fake pill and the fake pill won.

### The five

| # | What I hoped | What actually beat it |
|---|---|---|
| **1** | A new reliability method that adjusts predictions using how the X-ray was taken | **A statistics trick from 1999.** My "shuffled" fake version scored the same as the real one — meaning the acquisition information was doing *nothing* |
| **2** | Force the AI to become blind to bed-vs-standing X-rays, so it can't be unfair | **Nothing.** It didn't close the gap. Pushed harder, it made the AI *worse for everyone* — "fairness" by breaking it equally |
| **3** | Give the AI separate specialist settings for each X-ray type | **A rounding error.** +0.0003. That is zero wearing a disguise |
| **4** | Let the report-writer see the classifier's answers before writing | **Just training it longer.** +0.0023 — and when I trained the plain version for the same extra time, it caught up. The "clever bit" contributed nothing |
| **5** | Refer cases where the classifier and the report disagree with each other | **Simply asking "is the AI unsure?"** Mine: 85.57%. The dumb version: **86.64%** |

### The pattern I found

Look down that right-hand column. Every single winner is **boring**:

> a 1999 statistics trick · doing nothing · a rounding error · more training time · one subtraction

**Five times, the fancy method lost to something simple.** That is not five failures — that
is one finding, confirmed five separate ways:

> ### 🔑 In this problem, complicated methods do not beat simple ones. So if you publish a complicated method, you had better have tested it against the simple one — and most papers do not.

### Why this is my strongest evidence

Anybody can show a graph going up. **Almost nobody hands you the five graphs that went
down.**

And here's the thing that ties it together — **the two ideas that survived**
(Contribution 1 and Contribution 3) survived *the same controls that killed the other five.*

That is why you should believe them.

> **If I were only trying to look good, you would never have heard about any of these five.
> You are hearing about them because I was trying to find out what is true.**

---

## Part 3 — What to say to my panel

> *"My contribution is not a new model. It is a finding about how this field measures
> fairness.*
>
> *The standard fairness metric for chest X-ray projection bias can be improved by 73.3%
> at zero cost, simply by choosing a different decision threshold per group — beating the
> 46.7% reported by a peer-reviewed MIDL 2023 method that required full retraining and
> sacrificed accuracy. I prove the model's actual discriminative ability is unchanged to
> 1e-12.*
>
> *I then tested three classes of intervention and showed the underlying disparity
> survives all three, meaning it reflects genuine information loss at image acquisition
> rather than a model defect."*

---

## Part 4 — The honest grade

**This is an evaluation and methodology contribution, not a new algorithm.**

| ✅ What it is | ❌ What it is not |
|---|---|
| Real, measured, reproducible | A new neural network architecture |
| Beats a published result on its own metric | State-of-the-art accuracy |
| Backed by a mathematical proof | A new dataset |
| Appropriate for a final-year component | Comparable to a leaderboard entry |

**Finding that a widely-used measurement is broken is a genuine contribution to science.**
It is less glamorous than inventing an architecture. The field needs it more.

---

## Part 5 — The thing I should be proud of

I built controls specifically designed to **destroy my own ideas**. They worked four
times:

| My idea | What my own control found |
|---|---|
| Acquisition-Conditioned Reliability | It was **Platt scaling from 1999** — a shuffled control scored identically |
| Adversarial invariance closes the gap | Achieved **perfect** invariance; the gap did not move |
| Conditional specialisation | **+0.0003** — nothing |
| Classifier-conditioned reports | The gain was **fine-tuning**, not the conditioning (+0.0023) |

**Most students hide results like these. I measured and reported them.**

When a panel sees that I disproved my own hypotheses using experiments I built myself,
they have a reason to trust the one result that survived.

---

## Part 6 — What the system does, for completeness

The novelty above sits on top of a working system. These numbers are solid engineering,
not research contributions, but they are real:

| | |
|---|---|
| **Cardiomegaly detection** | AUROC **0.9189**, sensitivity **92.3%** |
| | correctly classifies 3,929 of 4,722 X-rays |
| | catches 2,197 of 2,381 real cases |
| Mean AUROC, 8 pathologies | 0.8251 → **0.8554** |
| **Report clinical F1** (CheXbert) | **0.5939** — cardiomegaly **0.8287**, best of 14 findings |
| **Fabricated references to prior scans** | 70.70% → **0.0000** |

⚠️ **Do not present accuracy without its baseline.** Cardiomegaly's 83.2% beats
"always say no" (49.6%) by +33.6 — that is genuine. But pneumonia's 89.4% *loses* to its
91.9% baseline. **Lead with AUROC and sensitivity.**

⚠️ **Do not compare these numbers to published papers.** My test split is patient-disjoint
but not the official MIMIC-CXR split, my reference reports are cleaned, and my test set is
cardiomegaly-enriched at 50.4% — which inflates any average that includes my best class.

---

## Summary in one line

> **I did not build a better AI. I showed that the way this field measures fairness in
> chest X-ray AI does not measure what it claims to, and that the disparity everyone is
> trying to fix cannot be fixed by any model — because the information is missing from the
> image itself.**
>
> **Then I built the thing that actually works instead: don't make the AI blind to how the
> X-ray was taken — make its decisions depend on it. Twice over. Once at the cut-off, once
> at the point where it decides whether to answer at all.**

---

*Supporting detail: [`RESULTS.md`](RESULTS.md) · [`MASTER_PLAN.md`](MASTER_PLAN.md)
Code and 128 unit tests: `stage9_fairness.py`, `stage9b_gradrev.py`, `stage10_conditional.py`,
`stage6_acr.py`, `stage11_conditioned.py`*
