# `src/` — module map

25 modules in four role-based packages.

Imports stay flat (`from config import CFG`) rather than relative
(`from ..core.config import CFG`). Each module opens with a bootstrap that puts
the four package directories on `sys.path`, so any module can be run directly:

```python
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]
```

That keeps `python models/train_stage2.py 24` working with no install step and
no `-m` invocation, which matters for a pipeline meant to be run stage by stage.

Run order is enforced by `run_all.py`; the groupings below are structural.

---

## Entry points

| Module | Purpose |
|---|---|
| `run_all.py` | the whole pipeline, one command. `--from`, `--only`, `--skip`, `--force`, `--all-horizons` |
| `predict.py` | single-patient inference with explanation. `--demo`, `--json`, `--stay-id` |

## Core infrastructure

| Module | Purpose |
|---|---|
| `core/config.py` | paths, seeding, YAML loader, label maps. Single source of truth |
| `core/utils.py` | metrics, cluster bootstrap, plots, `BinBudget` (GPU OOM backoff) |
| `core/progress.py` | self-refreshing progress bars with live VRAM readout |
| `core/study_store.py` | SQLite-backed Optuna studies — makes every search resumable |

## Data — acquisition and feature construction

| Module | Purpose |
|---|---|
| `data/preprocess.py` | **C2/C4** temporally-safe multimodal features at H ∈ {0,6,24}h |
| `data/text_features.py` | **C3** normalisation, Referral-Diagnosis Masking, lexicon, TF-IDF→SVD |
| `data/split.py` | patient-level grouped iterative stratification (zero subject overlap) |
| `data/dataset.py` | split-aware assembly; fits the vectoriser on train only |
| `data/ecg_fetch.py` | targeted MIMIC-IV-ECG waveform download (2.4 GB, not 144 GB) |
| `data/ecg_waveform.py` | ST-segment measurement from raw signal, ESC/AHA criteria |
| `data/labs_fetch.py` | expanded cardiac biomarkers from the PhysioNet file server |

## Models — training and decision

| Module | Purpose |
|---|---|
| `models/train_stage1.py` | ACS detection: XGB+LGBM, Optuna, isotonic calibration |
| `models/train_stage2.py` | subtyping: grouped 5-fold CV, Optuna, decision layer |
| `models/unified4.py` | **C7** unified four-class model + frontier decision layer |
| `models/decision_layer.py` | **C5** constrained cost-sensitive weights, fitted on validation |
| `models/selective.py` | selective prediction with clinician referral (Chow's rule) |
| `models/recalibrate.py` | refit the decision layer without retraining the models |
| `models/inference.py` | unified predictor loading every frozen artefact |

## Analysis — evaluation and explanation

| Module | Purpose |
|---|---|
| `analysis/audit_leakage.py` | **C1** five leakage probes + controlled leaky-vs-safe experiment |
| `analysis/evaluate.py` | **C6** cascade-honest evaluation, recall frontier, bootstrap CIs |
| `analysis/explain.py` | SHAP feature / modality / token attribution |
| `analysis/ablations.py` | modality, RDM, split protocol, cohort |
| `analysis/final_report.py` | consolidated results across all evaluation populations |

---

## Dependency order

```
config, utils, progress, study_store      (no internal deps)
        |
text_features -> preprocess -> split -> dataset
        |
train_stage1, train_stage2 -> decision_layer
        |
inference -> evaluate, unified4, selective, explain, ablations, final_report
        |
predict, recalibrate
```

`audit_leakage.py` depends only on the raw tables plus `preprocess` output, so it
can run any time after preprocessing.

## Conventions

- Anything *learned* (TF-IDF vocabulary, SVD basis, calibrator, decision weights)
  is fitted on **train** or **validation** only, never on test.
- The test fold is scored once per experiment.
- Every stage writes JSON to `artifacts/reports/` so results are inspectable
  without rerunning anything.
