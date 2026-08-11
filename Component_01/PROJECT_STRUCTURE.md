# Project Structure — Component 01

A complete file-by-file map of `Component_01/Component_01/`, what each thing is
for, and what is safe to leave out.

**Nothing in this document has been deleted or modified.** It is a survey only.

- **Total size:** 4.7 GB
- **Actually needed to run the demo:** ~1.7 GB
- **Actually needed if you strip the optional extras:** ~1.7 GB of which 1.6 GB is two model files

---

## ⚠️ First — the portability question, answered properly

> *"If I send only the `Component_01/Component_01` folder, will it work?"*

## ✅ RESOLVED — the folder is now self-contained

`cardio_test.csv` has been copied to `review_cases/cardio_test.csv` (verified
**byte-identical**, SHA-256 `780ec059…`, 4,786 reports) and `backend/config.py` now points
at the local copy, falling back to the out-of-tree original only if the copy is missing.

Verified after the change: ground truth indexes **4,786 reports**, and a sample of 120
demo images from `review_cases/` matched **120 / 120**.

**Nothing in `data/` was moved or modified — that copy is still there and untouched.**

> ### ⚠️ One catch if you transfer by **git**
>
> `review_cases/.gitignore` contains `*` — it excludes the entire folder, **including the
> new CSV**. That is deliberate protection: it stops MIMIC-CXR data being pushed to GitHub,
> which would breach the PhysioNet DUA.
>
> | Transfer method | Ground Truth works? |
> |---|---|
> | **ZIP / USB / direct copy** | ✅ yes — the file travels |
> | **git clone / GitHub** | ❌ no — the file is excluded by design |
>
> **Do not remove that `.gitignore` to "fix" this.** If you need ground truth on a machine
> that got the code via git, copy the CSV across separately — and only to someone
> PhysioNet-credentialed.

---

### The original analysis (for reference)

I checked every path in the backend that points outside the folder. There were exactly two:

| Path in `backend/config.py` | Points to | Size | What breaks without it |
|---|---|---|---|
| `ORIGINAL_TEST_CSV` | ~~`../data/.../cardio_test.csv`~~ → **now `review_cases/cardio_test.csv`** | **4.5 MB** | ✅ **fixed** |
| `TEST_IMAGE_DIR` | `../data/output/cardio_image_384/` | 4.9 GB | ✅ **Nothing** — see below |

### The good news about the 4.9 GB one

`TEST_IMAGE_DIR` is only read by `get_test_sample()` in `inference.py`. That method is
**not connected to any API endpoint** — the server exposes only `/predict`, `/health` and
`/thresholds`. It is unreachable code. **You do not need the 4.9 GB image directory.**

### What actually happens if your friend runs it as-is

| Feature | Works? |
|---|---|
| Upload an X-ray → cardiomegaly prediction | ✅ |
| 7 co-pathologies | ✅ |
| Grad-CAM heatmap | ✅ |
| AI-generated report | ✅ |
| Raw output view | ✅ |
| AP/PA projection selector + reliability notice | ✅ |
| ⚠️ Uncertain → refer to radiologist | ✅ |
| **Ground Truth toggle** | ✅ (CSV now inside the folder — see above) |

It will **not crash** even if the CSV is absent. The fallback is deliberate — the service
prints `!! original dataset not found ... Ground Truth will be unavailable` and carries on.

> ⚠️ **Licensing note.** That CSV contains MIMIC-CXR report text, which is
> PhysioNet-credentialed and under a Data Use Agreement. **Do not send it to anyone who
> is not credentialed.** If your friend is not, delete it from `review_cases/` before
> sending — everything except Ground Truth still works.

---

## The tree

```
Component_01/Component_01/
│
├── 📄 Documentation (5 files)
├── 🐍 Python modules (11 files)
├── 📓 Notebooks (13 files)
├── backend/          the API server
├── frontend/         the React UI
├── checkpoints/      trained models  ........... 4.3 GB
├── training_manifest/ the data splits ........... 23 MB
├── reports/          measured outputs ........... 3.4 MB
├── review_cases/     demo X-rays ................ 222 MB
├── stage1_clean/     intermediate ............... 54 MB
├── stage3_labels/    intermediate ............... 56 MB
├── reports_stage2/   one config file
└── __pycache__/      junk
```

---

## 1 · Documentation

| File | What it is | Keep? |
|---|---|---|
| `README.md` | Main project document — what it does, how it works, results | ✅ essential |
| `RESULTS.md` | Every measured number with confidence intervals and references | ✅ essential |
| `MASTER_PLAN.md` | The research plan, all stages, reference papers | ✅ essential |
| `NOVELTY_EXPLANATION.md` | The three contributions + five falsified ideas, in plain language | ✅ essential |
| `PANEL_ANSWERS.md` | Scripted answers to likely panel questions | ✅ essential |
| `PROJECT_STRUCTURE.md` | This file | ✅ |

---

## 2 · Python modules — the live system

| File | What it does | Needed to run? |
|---|---|---|
| `cxr_transforms.py` | Image preprocessing — per-image z-score, **not** ImageNet normalisation | ✅ **YES** — imported by the backend |
| `stage11_conditioned.py` | Report generator architecture + `build_prompt()` | ✅ **YES** — imported at inference |
| `build_review.py` | Extracts the 8 findings from report text | ⚠️ analysis only |
| `stage13_deferral.py` | ⭐ Contribution 3 — the deferral analysis | ⚠️ analysis only |
| `chexpert_fusion.py` | Stage 3 label fusion (text-adjudicated) | ❌ already run |
| `extract_review_cases.py` | Built `review_cases/` | ❌ already run |
| `extract_cardiomegaly_cases.py` | Built the cardiomegaly folders | ❌ already run |

### Python modules — failed experiments 🔴

**These are not broken code. They are working code for hypotheses that turned out to be
wrong.** They are the evidence behind §9 of RESULTS.md. **Do not delete them** — they are
the strongest proof of independent work in the project.

| File | Hypothesis | Verdict |
|---|---|---|
| `stage6_acr.py` | Acquisition-conditioned reliability | 🔴 **Falsified** — equalled Platt scaling (1999) |
| `stage9b_gradrev.py` | Adversarial invariance closes the AP/PA gap | 🔴 **Falsified** — gap unchanged; model collapsed when pushed |
| `stage10_conditional.py` | FiLM conditional specialisation | 🔴 **Falsified** — +0.0003 |
| `stage9_fairness.py` | ⭐ Per-projection operating points | ✅ **SURVIVED** — this is Contribution 1 |

---

## 3 · Notebooks

Colab notebooks, one per stage. **None are needed to run the demo** — they produced the
checkpoints and the cached results. Keep them all as evidence of the work.

### Pipeline that produced the shipped system ✅

| Notebook | What it did |
|---|---|
| `Stage1_Report_Target_Cleaning.ipynb` | Removed prior-study references from training **targets** — hallucination 70.70% → 0.0000 |
| `Stage2_Image_Transforms.ipynb` | Chose per-image z-score (measured 4.4× better than ImageNet) |
| `Stage4_Report_Generator.ipynb` | First report model — contains the `inputs_embeds` encoder fix |
| `Stage4B_Decoding_Ablation.ipynb` | 6 decoding strategies; greedy beat beam-4 on 5 of 7 metrics |
| `Stage5_Classifier_Training.ipynb` | **The shipped classifier** — mean AUROC 0.8554 |
| `Stage11_Conditioned_Report.ipynb` | **The shipped report generator** — clinical F1 0.5937 |
| `Stage12_CheXbert_Evaluation.ipynb` | Independent validation — agreed with my extractor to 0.002 |
| `Stage9A_Operating_Point_Fairness.ipynb` | ⭐ **Contribution 1** — 73.3% disparity reduction at zero cost |

### Notebooks for failed experiments 🔴

| Notebook | Verdict |
|---|---|
| `Stage6_Acquisition_Conditioned_Reliability.ipynb` | 🔴 falsified — Platt scaling won |
| `Stage6B_Validation.ipynb` | the control run that killed Stage 6 |
| `Stage9B_Lambda_Calibration.ipynb` | tuning for Stage 9B |
| `Stage9B_Gradient_Reversal.ipynb` | 🔴 falsified — invariance didn't close the gap |
| `Stage10A_Feature_Probe.ipynb` | 🔴 falsified — +0.0003 |

> **These five are not waste.** They are four of the five entries in your "ideas I killed
> myself" table. A panel that asks *"how do we know you didn't just get lucky?"* is
> answered by these notebooks.

---

## 4 · `backend/` — the API server (156 KB) ✅ **all essential**

| File | Purpose |
|---|---|
| `main.py` | FastAPI app. Endpoints: `/predict`, `/health`, `/thresholds` |
| `config.py` | All paths, model stats, generation settings |
| `thresholds.json` | Per-class **and per-projection** decision thresholds (Contribution 1) |
| `requirements.txt` | Python dependencies |
| `models/classifier.py` | ConvNeXt-Base architecture, exactly matching Stage 5 |
| `models/report_generator.py` | Vision encoder + projection + BioBART |
| `services/inference.py` | Loads both models, runs prediction |
| `services/gradcam.py` | Grad-CAM heatmaps |
| `services/thresholds.py` | ⭐ Contribution 1 in the live system |
| `services/deferral.py` | ⭐ Contribution 3 in the live system |

---

## 5 · `frontend/` — the React UI (65 MB)

| Path | Purpose | Keep? |
|---|---|---|
| `src/App.jsx` | Main app, upload + projection selector | ✅ |
| `src/components/UploadZone.jsx` | Drag-and-drop upload | ✅ |
| `src/components/ResultsPanel.jsx` | Diagnosis card + findings | ✅ |
| `src/components/GradCamViewer.jsx` | Heatmap display | ✅ |
| `src/components/ReportViewer.jsx` | AI report / raw / ground truth tabs | ✅ |
| `src/components/ReliabilityNotice.jsx` | AP/PA reliability (Contribution 1) | ✅ |
| `src/components/DeferralNotice.jsx` | ⚠️ Refer to radiologist (Contribution 3) | ✅ |
| `src/components/Header.jsx`, `App.css`, `main.jsx` | Layout and styling | ✅ |
| `package.json`, `vite.config.js`, `index.html` | Build config | ✅ |
| **`node_modules/`** (~65 MB, 2,485 files) | Downloaded libraries | ❌ **do not send** — `npm install` regenerates it |
| **`dist/`** | Compiled build output | ❌ **do not send** — `npm run build` regenerates it |

---

## 6 · `checkpoints/` — 4.3 GB ⚠️ **the size problem**

| File | Size | Status |
|---|---|---|
| `stage5/best.pt` | **673 MB** | ✅ **REQUIRED** — the shipped classifier |
| `stage11/best.pt` | **975 MB** | ✅ **REQUIRED** — the shipped report generator |
| `stage4/best.pt` | **2.7 GB** | ⚠️ **Superseded** — the old report model, kept only as a fallback |
| `stage5/thresholds.json` | 4 KB | ✅ small |

> **`stage4/best.pt` is 2.7 GB and is not used.** The backend loads Stage 11 and only
> falls back to Stage 4 if Stage 11 is missing. It is larger than Stage 11 because it
> stores optimizer state that Stage 11 does not.
>
> **Leaving it out cuts your transfer from 4.7 GB to 2.0 GB** and changes nothing about
> how the demo behaves. Keep a copy somewhere as provenance for the Stage 4 → Stage 11
> comparison, but it does not need to travel.

---

## 7 · `training_manifest/` — 23 MB

| File | Size | Purpose |
|---|---|---|
| `manifest_test.csv` | 2.4 MB | ✅ **REQUIRED** — loaded at server startup |
| `manifest_val.csv` | 2.3 MB | ✅ needed to re-run Stage 13 |
| `manifest_train.csv` | 19 MB | ❌ training only |
| `manifest_config.json` | small | how the splits were built |

---

## 8 · `reports/` — 3.4 MB ✅ **small and valuable**

| Path | Purpose |
|---|---|
| `stage6/cache/probs_test.npy` | 4,722 × 8 cached predictions — **the entire basis of Stage 13** |
| `stage6/cache/probs_val.npy` | 4,474 × 8 — the calibration set |
| `stage12/reports_stage11_test.txt` | All 4,722 generated reports |
| `stage12/references_test.txt` | The matching radiologist reports |
| `stage12/MANUAL_REVIEW.md` | Side-by-side comparison for manual reading |
| `stage13/deferral_policy.json` | ✅ **REQUIRED** — the frozen deferral cut-offs |
| `stage13/deferral.png` | The chart for your slides |
| `stage13/table.md`, `summary.json` | Full four-arm results |

> Keep this whole folder. It is tiny and it is what makes your results reproducible
> without a GPU.

---

## 9 · `review_cases/` — 222 MB, 2,027 images

Real X-rays sorted by how the system performed, for demos and manual inspection.

| Folder | What's in it |
|---|---|
| `cardiomegaly_present/` | ✅ correctly identified cardiomegaly |
| `cardiomegaly_absent/` | ✅ correctly ruled it out |
| `cardiomegaly_missed/` | ❌ false negatives |
| `cardiomegaly_false_positive/` | ❌ false alarms |
| `perfect_normal/`, `perfect_with_findings/` | all 8 findings correct |
| `worst/` | worst-performing cases |

**Keep for the demo** (you need images to upload), but you only need a handful. Trimming
each folder to ~20 images takes this from 222 MB to under 10 MB.

> ⚠️ These are MIMIC-CXR images — same DUA restriction as the CSV. Only share with
> credentialed people.

---

## 10 · Intermediate data — 110 MB ❌ **not needed**

| Folder | Size | What |
|---|---|---|
| `stage1_clean/` | 54 MB | Cleaned reports — already folded into `training_manifest/` |
| `stage3_labels/` | 56 MB | Fused labels — already folded into `training_manifest/` |
| `reports_stage2/` | 4 KB | One transform-config JSON — tiny, keep it |

These are **superseded intermediates**. Nothing reads them any more. Keep them locally for
provenance; don't send them.

---

## 11 · 🗑️ Junk — safe to ignore

| Path | What |
|---|---|
| `__pycache__/` | Python bytecode cache — regenerates automatically |
| `backend/__pycache__/` | Same |

Harmless. They regenerate on every run.

---

## 12 · ❓ Files that don't belong to this component

These are **not code** and are unrelated to Component 01's pipeline:

| File | Note |
|---|---|
| `IT22130020_Raagul G_Project Proposal Report.docx` / `.pdf` | Another student's proposal |
| `R26-IT-087_IT22281296_Thishoharini.V.pdf` | Another student's document |
| `Proposal Template.docx` | Blank template |

Group admin documents that ended up in the code folder. **Harmless, but they don't belong
in a code repository** — move them to a `docs/` folder or out of the project. Flagging
only; I have not touched them.

---

## Summary — what to actually send

### Minimum working demo — **~1.7 GB**

```
backend/                       (minus __pycache__)
frontend/                      (minus node_modules/ and dist/)
checkpoints/stage5/best.pt
checkpoints/stage11/best.pt
checkpoints/stage5/thresholds.json
training_manifest/manifest_test.csv
reports/stage13/deferral_policy.json
cxr_transforms.py
stage11_conditioned.py
run_backend.bat
review_cases/                  (trim to ~20 images per folder)
+ cardio_test.csv              (4.5 MB, if credentialed — for Ground Truth)
```

### Add for full reproducibility — **+15 MB**

```
reports/                       (all of it)
training_manifest/manifest_val.csv
stage13_deferral.py
build_review.py
all *.md documentation
all *.ipynb notebooks
all stage*.py modules          (including the failed ones)
```

### Leave behind — **~2.9 GB**

```
checkpoints/stage4/best.pt     2.7 GB — superseded
stage1_clean/                   54 MB — superseded
stage3_labels/                  56 MB — superseded
training_manifest/manifest_train.csv  19 MB
frontend/node_modules/          65 MB — npm install regenerates
frontend/dist/                        — npm run build regenerates
__pycache__/                          — regenerates
```

---

## Setup for whoever receives it

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
cd ..
python -m uvicorn backend.main:app --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

**On startup the backend prints which model it loaded and whether Ground Truth is
available.** Read those lines — they tell you immediately if a file is missing.

---

## One thing worth saying out loud

**Roughly 70% of this project by file count is failed experiments and superseded
intermediates.** That is not untidiness — it is the audit trail. The four falsified
notebooks and their three Python modules are what let you answer *"how do we know you
didn't just get lucky?"* with evidence instead of assurance.

**Do not delete them to make the folder look cleaner.**
