# Demo set — R26-IT-083

Studies each component predicts correctly, ready to drag into the console.

> ⚠️ **Credentialed data.** Components 01–03 draw on MIMIC-CXR, PTB-XL and
> EchoNet-Dynamic, all governed by data use agreements. This folder is
> excluded from git and must not be copied into an archive you send
> anywhere. Only Component 04's PDFs are synthetic and freely shareable.

## What this is

Curated studies from each component's held-out test split, selected because the component predicts them correctly and stands behind the result. Chosen so a demonstration does not fail for reasons unrelated to the work.

## What this is not

A performance sample. These cases were selected BY the models' own correctness, so their hit rate here is 100 % by construction and means nothing. The unselected test-set figures are in `true_performance`.

---

## `01_chest_xray`

*Unselected test-set performance: Cardiomegaly AUROC 0.9189, sensitivity 92.3 %, specificity 74.0 % on the full n=4,722 test set.*

| File | Ground truth | Predicted | Verdict |
|---|---|---|---|
| `PA_cardiomegaly_01.png` | Cardiomegaly | Cardiomegaly present (p=0.800) | actionable |
| `PA_cardiomegaly_02.png` | Cardiomegaly | Cardiomegaly present (p=0.916) | actionable |
| `PA_normal_01.png` | No cardiomegaly | No cardiomegaly (p=0.001) | actionable |
| `PA_normal_02.png` | No cardiomegaly | No cardiomegaly (p=0.099) | actionable |
| `AP_cardiomegaly_01.png` | Cardiomegaly | Cardiomegaly present (p=0.925) | caution |
| `AP_cardiomegaly_02.png` | Cardiomegaly | Cardiomegaly present (p=0.892) | caution |
| `AP_normal_01.png` | No cardiomegaly | No cardiomegaly (p=0.004) | caution |
| `AP_normal_02.png` | No cardiomegaly | No cardiomegaly (p=0.009) | caution |

**How to use.** Upload the file, then set Projection to the value in its name (PA or AP). The projection selects the operating point, so it changes the result.

---

## `02_ecg`

*Unselected test-set performance: Macro accuracy 0.864, macro recall 0.810 on the untouched test fold 10 (n=1,711).*

| File | Ground truth | Predicted | Verdict |
|---|---|---|---|
| `NORM_12886_hr.dat  +  NORM_12886_hr.hea` | NORM | normal ECG | actionable |
| `NORM_19324_hr.dat  +  NORM_19324_hr.hea` | NORM | normal ECG | actionable |
| `MI_18363_hr.dat  +  MI_18363_hr.hea` | MI | myocardial infarction, conduction disturbance | actionable |
| `MI_12870_hr.dat  +  MI_12870_hr.hea` | MI | myocardial infarction | actionable |
| `STTC_03454_hr.dat  +  STTC_03454_hr.hea` | STTC | ST/T change, conduction disturbance, ventricular hypertrophy | actionable |
| `STTC_18422_hr.dat  +  STTC_18422_hr.hea` | STTC | ST/T change, conduction disturbance | actionable |
| `CD_10950_hr.dat  +  CD_10950_hr.hea` | CD | myocardial infarction, conduction disturbance | actionable |
| `CD_21148_hr.dat  +  CD_21148_hr.hea` | CD | conduction disturbance | actionable |
| `HYP_13899_hr.dat  +  HYP_13899_hr.hea` | HYP | ST/T change, ventricular hypertrophy | actionable |
| `HYP_12252_hr.dat  +  HYP_12252_hr.hea` | HYP | ST/T change, ventricular hypertrophy | actionable |

**How to use.** Upload BOTH files; they must share a base name.

---

## `03_echocardiogram`

*Unselected test-set performance: MAE 3.979 EF points, 73.0 % overall accuracy, min per-class recall 0.723 on the untouched test split (n=1,277).*

| File | Ground truth | Predicted | Verdict |
|---|---|---|---|
| `severe_01_0X9A03DC1334986F3.npy` | Severe(<30) (true EF 10.2 %) | EF 19.6 % -- Severe(<30) | actionable |
| `severe_02_0X5FF8D238E3A43BA9.npy` | Severe(<30) (true EF 17.9 %) | EF 16.5 % -- Severe(<30) | actionable |
| `moderate_01_0X7EEA66DBE251854B.npy` | Moderate(30-40) (true EF 35.2 %) | EF 39.8 % -- Moderate(30-40) | deferred |
| `moderate_02_0X7CEA8E5FA8F3FB3F.npy` | Moderate(30-40) (true EF 39.4 %) | EF 44.9 % -- Moderate(30-40) | deferred |
| `mild_01_0X6D1D29802905D6E0.npy` | Mild(40-55) (true EF 45.4 %) | EF 45.2 % -- Mild(40-55) | actionable |
| `mild_02_0X33C3D95C20A0D931.npy` | Mild(40-55) (true EF 45.2 %) | EF 48.5 % -- Mild(40-55) | actionable |
| `normal_01_0X6FCE7C69AE34FBF7.npy` | Normal(>=55) (true EF 66.8 %) | EF 63.9 % -- Normal(>=55) | actionable |
| `normal_02_0X55AE4F5E25A9609F.npy` | Normal(>=55) (true EF 67.5 %) | EF 65.9 % -- Normal(>=55) | actionable |

- **`moderate_01_0X7EEA66DBE251854B.npy`** — Graded correctly but DEFERRED. Moderate spans only 30-40 EF, so every case in this class sits near a boundary and the conformal interval straddles it. Expect a referral, not a call -- this is the component behaving as designed.
- **`moderate_02_0X7CEA8E5FA8F3FB3F.npy`** — Graded correctly but DEFERRED. Moderate spans only 30-40 EF, so every case in this class sits near a boundary and the conformal interval straddles it. Expect a referral, not a call -- this is the component behaving as designed.

**How to use.** Upload directly; .npy is a cached clip array.

---

## `04_ed_triage`

*Unselected test-set performance: Stage-1 AUROC 0.9560 with NPV 99.41 %; Stage-2 subtyping macro-F1 0.7448 on the patient-disjoint test fold.*

| File | Ground truth | Predicted | Verdict |
|---|---|---|---|
| `sample_01_stemi.pdf` | STEMI | STEMI | actionable |
| `sample_02_nstemi.pdf` | NSTEMI | NSTEMI | deferred |
| `sample_03_unstable_angina.pdf` | deferral (UA is the hardest class) | No_ACS | deferred |
| `sample_04_non_cardiac.pdf` | No_ACS | No_ACS | caution |
| `sample_05_sparse.pdf` | deferral (sparse record) | STEMI | deferred |

**How to use.** Upload on the Triage console's PDF tab, or load it from the sample list there.

---

## Regenerating

```bash
cd backend
python scripts/build_demo_set.py
```

Selection re-runs against the live models, so a study that stops being
predicted correctly drops out rather than silently going stale.
