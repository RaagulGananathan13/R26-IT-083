# The Individual Contribution — Final

**Venushan T** · Progress Presentation 2
*Supersedes §3 of RESEARCH_CONTRIBUTION.md, which over-claimed novelty.*

---

## 1. What I checked first: is it actually new?

Before claiming anything, the prior art:

| Prior work | What they did | Overlap with me |
|---|---|---|
| *Annals of Noninvasive Electrocardiology*, 2025 | Split, class-conditional, and learn-then-test conformal prediction on PTB-XL (21,799 ECGs) → high-risk / low-risk / **uncertain** | **Large.** Conformal triage on PTB-XL is theirs. |
| Strodthoff et al., *Comput. Biol. Med.*, 2024 | Attribution maps + PTB-XL hierarchical labels to separate anterior vs inferior MI | **Large.** XAI→MI-subtype is theirs. |
| Angelopoulos et al., 2023 | Conformal Risk Control | Method I use |
| *Pitfalls of Conformal Predictions for Medical Image Classification*, 2025 | Exchangeability breaks under shift — **in imaging** | Adjacent, not ECG |

So "conformal prediction for ECG" is **not** my contribution, and I will say so on the slide. What none of them tested is the question below.

---

## 2. The contribution

> ### A conformal ECG system can satisfy its advertised guarantee exactly — and still be unsafe for identifiable groups of patients.

Conformal validity is **marginal**: it holds *on average over the whole test
distribution*. A cardiologist never treats the average. They treat a 42-year-old
with palpitations, or a 78-year-old woman with atypical chest pain. Nothing in
the marginal guarantee stops the misses from concentrating in one of those groups.

Nobody has tested whether ECG conformal guarantees hold **conditionally on the
patient**. I did.

---

## 3. The result

Thresholds fitted marginally on fold 9 (the standard approach, and the one the
2025 paper uses), then the realised miss rate measured **inside each subgroup**
of the unseen fold 10. Miss rate = true positives that were ruled out.

| Class | Promised α | **Overall** | Male | Female | <50 | 50–69 | ≥70 |
|---|---|---|---|---|---|---|---|
| NORM | 0.20 | **0.190 ✓** | 0.158 | 0.225 | 0.103 | 0.228 | **0.330** |
| MI | 0.05 | **0.015 ✓** | 0.020 | 0.008 | 0.000 | 0.011 | 0.019 |
| STTC | 0.10 | **0.092 ✓** | 0.103 | 0.083 | 0.128 | 0.080 | 0.093 |
| CD | 0.10 | **0.099 ✓** | 0.085 | 0.117 | **0.333** | 0.099 | 0.042 |
| HYP | 0.15 | **0.121 ✓** | 0.182 | 0.061 | 0.444¹ | 0.159 | 0.066 |

¹ n = 9 positives — not reliable, shown for completeness.

**Every class passes marginally. Nine class–subgroup cells violate the bound.**

The two that matter, both with adequate samples:

> **Conduction disturbance, patients under 50: promised ≤ 10% missed, delivered
> 33.3%** (n = 66 positives). **3.3× the advertised bound.**
>
> **Normal ECG, patients 70+: promised ≤ 20%, delivered 33.0%** (n = 103).

A hospital deploying this would have been told "we miss at most 10% of conduction
disturbances." For their under-50 patients that statement is false by a factor of
three — and **the overall number, 9.9%, gives no hint of it.**

### The fix works

Mondrian (group-conditional) conformal calibration — a separate threshold per
subgroup (Vovk et al., 2003):

| | Cells satisfying the bound |
|---|---|
| Marginal calibration (standard) | 14 / 23 |
| **Mondrian (group-conditional)** | **22 / 23 (96%)** |

### And it is not free — a second finding

Group-conditional calibration needs enough positives *per group*. Below roughly
1/α calibration positives the PAC bound is infeasible and the group simply cannot
be certified: **STTC in under-50s had 42 calibration positives and returned
λ = −∞ — the system can never rule out ST/T change in a young patient.**

That is the honest engineering trade: **conditional validity costs calibration
data, and the groups that most need it are the ones with least of it.**

---

## 4. Why a cardiologist cares

- **Conduction disturbance in the young is not benign.** In a patient under 50 it
  raises Brugada, ARVC, and inherited conduction disease — the causes of sudden
  cardiac death in young adults. This is precisely the group where a missed
  conduction abnormality is least acceptable, and precisely where the system
  fails worst.
- **"Normal" in the elderly is the highest-volume decision in any ED.** A 33%
  miss rate on ruling in NORM for over-70s means the system defers or mislabels
  a third of the patients it is supposed to clear.
- **Subgroup safety is a regulatory requirement, not a nicety.** The FDA has
  authorised 1,018 AI-enabled devices (104 cardiovascular) as of March 2025, and
  subgroup performance reporting is central to that review. A guarantee that
  holds only marginally would not survive it.

---

## 5. What is honestly mine

| Claim | Status |
|---|---|
| Conformal prediction for ECG | **Not mine** — 2025 prior art, cited |
| XAI → MI subtype localisation | **Not mine** — Strodthoff 2024, cited |
| **Conditional (subgroup) validity of ECG conformal guarantees** | **Mine.** No prior work. |
| **Demonstration that marginal validity hides 3× subgroup violations** | **Mine.** Measured, n reported. |
| **Mondrian calibration as the remedy, with its data cost quantified** | **Mine** (method is Vovk 2003; the ECG application and the cost analysis are mine) |
| Accuracy | **Within the published band.** Not a claim. |

---

## 6. The accuracy question, answered honestly

PTB-XL 5-superclass has been **0.92–0.94 macro-AUROC for six years**. My model is
**0.9343 ± 0.0028** over 3 seeds — inside that band. Only macro-AUPRC
(**+0.0137**, 4.7σ, p≈0.015) clears seed noise; AUROC (+0.0046) and F1 (+0.0065)
do not.

Papers above 0.94 use foundation-model pretraining on 100k+ ECGs. That is not
reachable with one L4 GPU, and I will not pretend otherwise. **Accuracy is not my
contribution and I present it in one line.**

---

## 7. A null result I am also reporting

I tested whether the signal-quality gate removes a *label-dependent* subset
(which would void exchangeability). On PTB-XL: **it does not.** Signal quality is
uniform (SQI = 1.00 across all classes), and injecting realistic corruption
shifted class prevalence by at most **0.005**.

The concern is real in principle but **not measurable on a clean benchmark**.
Reporting a null result rather than quietly dropping it is part of the work.

---

## 8. Slides

1. **The audit** — flatline → "MI 0.691". Buys credibility in ten seconds.
2. **What exists** — conformal ECG (2025), XAI→subtype (2024). Name them.
3. **The gap** — marginal ≠ conditional. One sentence.
4. **The table** — every class passes overall; CD in under-50s misses 33%.
5. **The fix** — Mondrian, 14/23 → 22/23.
6. **The cost** — STTC under-50 cannot be certified at all (λ = −∞).
7. **Clinical framing** — conduction disease in the young; sudden cardiac death.
8. **Accuracy** — one line, honest, last.

**Opening line:** *"My system passes its own safety guarantee on every class. I'm
going to show you why that isn't good enough."*

---

## 9. Reproduce

```bash
python -X utf8 Component_02/audit/10_conditional_validity.py
```
→ `audit/results/10_conditional_validity.{txt,json}`

---

## 10. Limitations to state before the panel does

- **Single dataset, single centre.** PTB-XL is one German cohort, 1989–96. The
  external-validity question (does this hold in another hospital?) is the
  obvious next study and needs data I do not have.
- **Small subgroup samples.** HYP in under-50s has 9 positives; that row is
  indicative only. CD<50 (n=66) and NORM≥70 (n=103) carry the argument.
- **Single seed for the conformal analysis.** Thresholds come from seed 0.
- **Subgroups are age and sex only** — the variables PTB-XL provides. Ethnicity,
  device, and comorbidity are not available and may matter more.
- **No clinician has reviewed the outputs.** Every claim here is statistical.

---

## References

1. *Conformal prediction for AMI risk on PTB-XL.* Ann Noninvasive Electrocardiol, 2025. doi:10.1111/anec.70099
2. Strodthoff N. et al. *Explaining deep learning for ECG analysis: building blocks for auditing and knowledge discovery.* Comput Biol Med, 2024.
3. Angelopoulos A., Bates S., Fisch A., Lei L., Schuster T. *Conformal Risk Control.* arXiv:2208.02814, 2023.
4. Vovk V., Lindsay D., Nouretdinov I., Gammerman A. *Mondrian confidence machine.* 2003.
5. Vovk V. *Conditional validity of inductive conformal predictors.* ACML, 2012.
6. *Pitfalls of Conformal Predictions for Medical Image Classification.* arXiv:2506.18162, 2025.
7. Wagner P. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Sci Data 7:154, 2020.
8. Strodthoff N. et al. *Deep learning for ECG analysis: benchmarks and insights from PTB-XL.* IEEE JBHI, 2021.
9. Guo C. et al. *On calibration of modern neural networks.* ICML, 2017.
10. Adebayo J. et al. *Sanity checks for saliency maps.* NeurIPS, 2018.
