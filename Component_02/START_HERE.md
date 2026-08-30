# Start Here — A Plain-English Guide

**Read this first.** No maths background needed. Everything technical is
explained the first time it appears.

Component 02 · ECG Abnormality Detection & Cardiac Risk Reporting
Part of the Explainable AI System for Cardiovascular Disease Detection

---

## What this is, in one paragraph

A doctor sticks 10 electrodes on a patient's chest and limbs. A machine records
the heart's electrical activity for 10 seconds and prints 12 squiggly lines —
that's an **ECG** (electrocardiogram). Reading it takes a trained cardiologist.
This system reads it automatically: it says what's wrong, shows *which part of
the signal* made it think so, writes a clinical report, and — the important part
— **tells you how often it is wrong, with a mathematical proof.**

---

## Part 1 · The problem we're solving

### Heart disease is the world's biggest killer, and ECGs are how you catch it

An ECG is cheap, fast, and available in almost every clinic. The problem is that
**reading one properly requires a cardiologist**, and there aren't enough of
them — especially outside big cities.

So people build AI to read ECGs. Hundreds of papers do this. They all report a
number like *"our accuracy is 93%."*

### But "93% accurate" doesn't tell a doctor what they need to know

Imagine a smoke alarm that is "93% accurate." Would you sleep under it?

You'd want to know something different: **when it stays silent, how often is
there actually a fire?** That's a completely different question, and "93%
accurate" does not answer it.

Same with ECGs. A doctor doesn't want "93% accurate." They want:

> *"When this system tells me the patient is fine, how often is it wrong?"*

Almost no ECG AI answers that question. **That's the gap this project works in.**

---

## Part 2 · What the system actually does

An ECG goes in. Seven things happen, in order.

### Step 1 · Quality check — *"Is this even a real ECG?"*

Before anything else, the system inspects the signal. It refuses to continue if:

- all the lines are flat (the electrodes fell off)
- it's just noise (the patient was moving)
- the voltage is impossible (the machine's settings were wrong)
- the recording is too short or too long

**Why this matters:** the previous version of this system was given a completely
flat signal — a disconnected patient — and confidently reported *"myocardial
infarction"* (a heart attack). It now says **"REFUSED — every lead is flat"**
instead.

> A system that says "I can't read this" is safer than one that guesses.

### Step 2 · Clean up the signal

Remove slow drift (from the patient breathing) and electrical hum from the mains
power. Standard signal processing, nothing clever.

### Step 3 · Classify — *"What's wrong?"*

A neural network sorts the ECG into five categories:

| Code | Means | In plain English |
|---|---|---|
| **NORM** | Normal | Nothing wrong |
| **MI** | Myocardial Infarction | Heart attack (or scar from an old one) |
| **STTC** | ST/T Change | Possible reduced blood supply to the heart |
| **CD** | Conduction Disturbance | The heart's electrical wiring is faulty |
| **HYP** | Hypertrophy | The heart muscle wall has thickened |

A patient can have more than one at once.

### Step 4 · Fix the confidence numbers

Neural networks are **overconfident** — they say "90% sure" when they're right
only 60% of the time. Ours was especially bad at hypertrophy: it claimed that
diagnosis **4× more often than it actually occurs**.

A correction called **temperature scaling** fixes this. Afterwards, when the
system says 30%, it really means 30%.

### Step 5 · Decide — *the important part*

Most AI has two answers: **yes** or **no**. This system has **three**:

| | Meaning |
|---|---|
| 🟢 **RULE OUT** | Confidently absent. Safe to move on. |
| 🟡 **REFER** | *I'm not sure.* A human must look at this. |
| 🔴 **RULE IN** | Confidently present. Act on it. |

**The yellow box is the point.** Most systems have no way to say "I don't know" —
they're forced to guess. This one is allowed to admit uncertainty.

And each decision comes with a **promise backed by mathematics**:

> *"When I rule out a heart attack, I am wrong at most 5 times in 100."*

That's not a feeling. It's a proof. The technique is called **conformal
prediction**.

### Step 6 · Explain — *"Why did you say that?"*

Two explanations are produced:

- **When in the 10 seconds** the system was looking (shown as red shading on the
  ECG plot)
- **Which of the 12 leads** mattered, and whether each one *supported* or
  *argued against* the diagnosis

These get turned into words the report can use. For example:

> *"Evidence is strongest in leads V1, V2 and V5 — a septal distribution
> (proximal LAD artery)."*

Translation: *the damage is in the wall between the two sides of the heart, fed
by a specific artery.* A cardiologist can act on that.

> **A real example:** for test record 271, the system said *"septal"*. The
> cardiologist's own written report for that same patient said *"anteroseptal
> myocardial infarction."* The AI found the same location, independently.

### Step 7 · Write the report — and check it

The system writes a clinical report. Then **a second program reads that report
and blocks it if anything is wrong.**

Why? The previous version had an AI write the reports freely, and in **42
patients it invented "atrial fibrillation"** — a condition this system cannot
even detect. It made it up.

Now every sentence must trace back to something the classifier actually found.
Anything added, dropped, or invented is rejected.

---

## Part 3 · The research contribution

*This is the part the panel cares about. It's one idea.*

### Meet the lifeguard

A lifeguard promises: *"Out of every 100 swimmers who get into trouble, I'll miss
at most 10."*

End of summer, you check the records. He missed **9.9 out of 100**.
Promise kept. Everyone claps.

Then someone splits the numbers by age:

- Among **adults**: missed 4 in 100. Excellent.
- Among **children**: missed **33 in 100**.

He kept his promise *on average* — by being brilliant with adults and bad with
children. **The average hid the problem completely.**

### That's exactly what we found

Our system's promises hold overall. Every single class passes:

| | Promised | Actually delivered | |
|---|---|---|---|
| Heart attack | miss ≤ 5% | 1.5% | ✅ |
| Conduction problem | miss ≤ 10% | 9.9% | ✅ |
| Normal | miss ≤ 20% | 19.0% | ✅ |

Then we split by patient age and sex — **something nobody had done before for
ECG AI** — and found:

> ### Conduction problems in patients under 50
> ### Promised: miss at most 10%. **Actually missed 33.3%.**

The overall number, 9.9%, gave **no hint** of it.

### Is that just bad luck with a small group?

That's the first question anyone asks. It was tested three separate ways:

| Test | Result |
|---|---|
| Confidence interval | 23.2% – 45.3% — the **whole range** is above the 10% promise |
| Statistical test | p = 0.0000002, still significant after correcting for testing 23 groups |
| Re-ran the whole calibration 2,000 times | The promise broke in **100% of them** |

It's not luck. It's structural.

**And we found the same thing for "normal" in patients over 70** — promised ≤20%
missed, delivered 33.0%.

**Seven other groups looked suspicious but did not survive proper statistical
testing. We report those as noise.** Saying that out loud is part of doing this
honestly.

### Why a doctor cares about *this* group specifically

A conduction problem in someone **under 50** is not a minor finding. It can mean
**Brugada syndrome** or **ARVC** — inherited conditions that cause **sudden
cardiac death in young, otherwise healthy people**.

That is the group where missing it matters most. And it's where the system was
worst.

### The fix

Instead of one promise for everybody, make **a separate promise for each group**.
(The technical name is *Mondrian calibration*.)

| | Groups where the promise held |
|---|---|
| One promise for everyone (standard practice) | 14 out of 23 |
| **A promise per group** | **22 out of 23** |

### And the catch — a second finding

To promise something to a group, you need enough patients *from that group* to
check against. For ST/T changes in under-50s we only had 42. The maths refused to
make any promise at all.

> **The groups that most need protection are the ones with the least data to
> protect them with.**

---

## Part 3b · The biggest problem of all — a disease the system cannot see

### The system only knows five things

NORM, MI, STTC, CD, HYP. That's it. There is **no box** for atrial fibrillation —
the most common serious heart rhythm problem in the world, and a condition
associated with roughly one stroke in four.

So what happens when a patient with atrial fibrillation walks in?

You might hope the system says *"I don't recognise this."* It cannot. It has no
way to say that. It spreads the evidence across the five boxes it does have,
picks an answer, and **prints its statistical promise next to it.**

### What we measured on our own test data

**114 patients in the test set had atrial fibrillation or flutter documented by a
cardiologist.**

| | |
|---|---|
| Given a report with a statistical guarantee attached | **113 of 114** |
| Guarantees that were about atrial fibrillation | **0** |
| **Told their ECG was normal** | **2** |

Read that again. The system promised *"I miss at most 5% of heart attacks"* to
114 people whose actual problem was a rhythm it cannot even name. **Two of them
were told they were fine** — with four and three guarantees printed underneath.

And **14.3% of the whole dataset** carries some condition the five boxes cannot
express.

**Every paper using these five PTB-XL classes has this problem. None of them
mention it**, because the benchmark only scores the five boxes it invented.

### The fix — using physiology, not more AI

Atrial fibrillation has a defining feature: **the heartbeats are irregularly
irregular.** No pattern at all in the gaps between beats.

The system already finds the heartbeats to calculate the pulse, so measuring how
irregular they are is **free**. As a pure separator it is strong — a score of
**0.912** — but we deliberately tuned it to raise few false alarms, and that
choice means it only catches about half of them.

Now, when the beats are irregularly irregular, the system says:

> *"This rhythm looks like it's outside the five things I know. I have no output
> for atrial fibrillation and cannot assess it. **My guarantees do not apply
> here.** An arrhythmia has NOT been ruled out."*

It still gives its five-class opinion. It just stops making a promise it cannot
keep.

**60 of the 114 AF patients now get that warning instead of a false guarantee.**

### Being honest about the limits

- It **misses 53 of the 114** — including ECG 15796, the very case that was
  reported as normal. That one still slips through.
- It only catches **irregular** rhythms. A pacemaker, a fast but steady abnormal
  rhythm, or Brugada syndrome all keep a regular beat — those remain invisible.

We report both numbers. A fix that only half works is still worth having, as long
as you say which half.

---

## Part 4 · The numbers, explained

Tested on 1,711 patients the system had never seen.

| Class | Accuracy | Recall | NPV |
|---|---|---|---|
| NORM | 88.3% | 79.6% | 86.8% |
| MI | 88.4% | 83.6% | **96.7%** |
| STTC | 86.8% | 80.3% | 92.6% |
| CD | 86.9% | 80.5% | 92.1% |
| HYP | 81.7% | 81.1% | **98.1%** |

**What these words mean:**

- **Accuracy** — how often the system is right overall
- **Recall** (also called *sensitivity*) — of all the patients who really have the
  condition, what fraction did we catch? **This is the safety number.**
- **NPV** (negative predictive value) — when we say "you don't have it," how often
  are we right? **This is the rule-out number.**

Every class is above **75% accuracy** and above **75% recall**. NPV averages
**93.3%**.

### "Why is the F1 score low for hypertrophy?"

You'll see a number called **F1** in most papers. It's low for hypertrophy (0.41)
and someone will ask about it.

**F1 treats a missed heart attack and an unnecessary double-check as equally
bad.** No doctor thinks that. Sending a healthy person for one extra test costs
20 minutes. Sending a heart attack home can kill them.

Real cardiology guidelines are built on **recall and NPV**, never F1. Our
hypertrophy F1 is 0.41 — but its **NPV is 98.1%**. When it says you don't have
hypertrophy, it's right 98 times out of 100.

**Hypertrophy is also genuinely the hardest class**, for reasons we measured:

- Only **1,468 examples** in the whole dataset (8.5%), just **132** in the test set
- **63.8% of hypertrophy cases also have ST/T changes**, 35.7% also have conduction
  problems — it almost never appears alone
- It's diagnosed by how *tall* the waves are, and standard signal processing
  flattens exactly that information
- Even the labels are imperfect — ECG is not the gold standard for hypertrophy;
  ultrasound is

The best published results get about 0.54. Ours is in that range. **Nobody has
reached 0.75, and a claim that we had would not survive checking.**

---

## Part 5 · How to run it

You need Python 3.10+ and Node 18+. No GPU required.

```bash
cd Component_02
pip install -r requirements.txt
```

**Terminal 1** — the brain:
```bash
python -X utf8 backend/server.py
```

**Terminal 2** — the screen:
```bash
cd frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:5173**

> Getting `can't open file '...Component_02\Component_02\...'`?
> You're already inside the folder — drop the `Component_02/` from the command.

### Try these three things

1. **Click a heart attack case** → see the red/yellow/green boxes, the report, and
   the highlighted ECG
2. **Click a borderline case** → watch it land in 🟡 REFER and say *"a cardiologist
   must review this"*
3. **Upload a broken file** → watch it get **REFUSED** instead of diagnosed

---

## Part 6 · Words you'll hear

| Term | Plain meaning |
|---|---|
| **ECG / EKG** | The 12-line recording of the heart's electrical activity |
| **Lead** | One of the 12 viewpoints. Each "sees" the heart from a different angle |
| **PTB-XL** | The public dataset used — 17,221 real ECGs from German hospitals |
| **Conformal prediction** | A way to attach a *provable* error rate to a prediction |
| **Calibration** | Fixing overconfident percentages so they mean what they say |
| **Recall / Sensitivity** | Of the people who have it, how many did we catch |
| **NPV** | When we say "you don't have it," how often we're right |
| **Grad-CAM** | The technique that highlights *when* in the signal the AI was looking |
| **Marginal vs conditional** | "True on average" vs "true for this specific patient" |

---

## Part 7 · What this system cannot do

Say these before anyone asks. Every one is true.

- **It only knows 5 conditions.** It cannot detect **atrial fibrillation** — the
  most common heart rhythm disorder in the world — or any arrhythmia. If it
  doesn't mention them, that means nothing. It now *warns* you when the rhythm
  looks irregular (catching 60 of 114 such cases), but a steady out-of-scope
  rhythm still passes silently.
- **It has only ever seen German patients** from 1989–1996. Nobody knows if it
  works on Sri Lankan patients. That is the next study.
- **It doesn't measure intervals** (PR, QRS, QT) or the heart's electrical axis —
  things a cardiologist reads routinely.
- **No cardiologist has reviewed its reports.** Every claim here is statistical.
- **The artery localisation is an educated guess** based on which leads mattered.
  It has not been clinically validated.
- **It is not a medical device.** It is decision support. A qualified clinician
  must review every single report.

---

## Where to go next

| File | What's in it |
|---|---|
| `README.md` | How to run it, and the API for connecting it to other systems |
| `docs/PANEL_ANSWERS.md` | The full research write-up, with anticipated questions |
| `docs/AUDIT_FINDINGS.md` | The 12 problems found in the previous version |
| `docs/COLAB_GUIDE.md` | How to retrain the model on a GPU |

---

## The one sentence

> **My system passes its own safety guarantee on every class. I'm going to show
> you why that isn't good enough — and that the patients it fails are the ones
> who can least afford it.**
