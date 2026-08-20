# R26-IT-083

**Explainable AI System for Cardiovascular Disease Detection and Diagnosis**

Four independently developed research components behind one service and one
clinical console.

> ⚕️ Research prototype. **Not a medical device**, not clinically validated, and
> not for diagnosis or treatment decisions. Every output requires review by a
> qualified clinician.

---

## The components

| # | Owner | Modality | Answers | Dataset |
|---|---|---|---|---|
| **01** | Raagul Gananathan (IT22130020) | Chest radiograph | Cardiomegaly + 7 co-pathologies, Grad-CAM, draft report | MIMIC-CXR |
| **02** | Venushan T | 12-lead ECG | 5 superclasses with conformal rule-in / rule-out triage | PTB-XL |
| **03** | Dilukshan Viyapury (IT22219534) | Echocardiogram | Ejection fraction + 4-class severity grade | EchoNet-Dynamic + CAMUS |
| **04** | Abishnan J (IT22140234) | ED triage record | ACS detection + UA / NSTEMI / STEMI subtyping | MIMIC-IV-ED |

Each keeps its own weights, its own frozen decision rule and its own published
figures. The integration reproduces them unchanged and adds no probabilities of
its own.

---

## What ties them together

The four answer different clinical questions and share no findings. What they
*do* share is that **each was built around a mechanism that declines to commit
when its own evidence is weak**:

| Component | Its honesty mechanism |
|---|---|
| 01 | Per-projection operating points and selective deferral (AP AUROC 0.8224 vs PA 0.8864) |
| 02 | Quality gate → refusal before any probability exists; conformal zones; guarantee withdrawal on electrode reversal or out-of-scope rhythm |
| 03 | Split-conformal EF interval, learned aleatoric σ, inter-clip epistemic disagreement |
| 04 | Declared feature-availability horizon, constrained decision layer, clinician referral |

The backend normalises those four vocabularies into one `actionability`
verdict — `actionable` / `caution` / `deferred` / `withheld` / `unavailable` —
so a caller applies one rule across all modalities: **do not act on a result
that is not actionable.** The console makes that verdict control the visual
hierarchy rather than sit in a corner badge.

---

## Layout

```
R26-IT-083/
├── backend/          unified FastAPI service (package `cvxai`)
├── frontend/         Next.js clinical console
├── Component_01/     chest radiograph — model + training code
├── Component_02/     ECG — model + training + audit suite
├── Component_03/     echocardiogram — model + training code
└── Component_04/     ED triage — model + training + leakage audit
```

All four components live in this repository. There are no submodules and no
external repository links, so a plain clone gives you the whole system:

```bash
git clone https://github.com/RaagulGananathan13/R26-IT-083.git
```

---

## Running it

Two terminals. The backend must be up first.

```powershell
# terminal 1
cd backend
python -m pip install -r requirements.txt
python run.py --warm                      # http://127.0.0.1:8000

# terminal 2
cd frontend
npm install
npm run dev                               # http://localhost:3000
```

Use `python -m pip`, not bare `pip`: a default Windows Python install puts
`python.exe` on PATH but not the `Scripts\` folder that holds `pip.exe`.

`backend/README.md` and `frontend/README.md` carry the detail.

---

## Data and weights are NOT in this repository

Three of the four datasets are credentialed. MIMIC-CXR, MIMIC-IV-ED and PTB-XL
are governed by PhysioNet data use agreements; EchoNet-Dynamic and CAMUS carry
their own terms. **No images, waveforms, videos, reports or derived datasets are
distributed here, and neither are model weights**, which are derived from
credentialed data and are equally non-redistributable.

The working tree is roughly 41 GB; this repository is a few megabytes of source.
`.gitignore` is the boundary — read it before adding anything, and re-check
`git status` before a commit that touches a component directory.

To reproduce any component you need your own credentialed access and the
component's own preprocessing pipeline.

---

## Licence

Code: MIT. Data: not licensed here, and not included. See each component's
README for its own terms.
