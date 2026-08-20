# Unified Backend — R26-IT-083

**Explainable AI System for Cardiovascular Disease Detection and Diagnosis**

One FastAPI service over all four research components. Single process, single
port, single response contract.

| Component | Owner | Modality | Answers |
|---|---|---|---|
| **01** | Raagul Gananathan (IT22130020) | Chest radiograph | Cardiomegaly + 7 co-pathologies, Grad-CAM, draft report |
| **02** | Venushan T | 12-lead ECG | 5 superclasses with conformal rule-in / rule-out triage |
| **03** | Dilukshan Viyapury (IT22219534) | Echocardiogram | Ejection fraction + 4-class severity grade |
| **04** | Abishnan J (IT22140234) | ED triage record | ACS detection + UA / NSTEMI / STEMI subtyping |

> ⚕️ Research prototype. **Not a medical device**, not clinically validated.
> Every output requires review by a qualified clinician.

---

## Run it

```bash
cd backend
python -m pip install -r requirements.txt
python run.py
```

Then open **<http://127.0.0.1:8000/docs>**.

```bash
python run.py --warm      # load all four components before serving
python run.py --reload    # auto-reload during development
```

> Use `python -m pip`, not bare `pip`. A default Windows Python install puts
> `python.exe` on PATH but not the `Scripts\` directory that holds `pip.exe`,
> so `pip` alone reports *"The term 'pip' is not recognized"*. The module form
> works regardless, and it guarantees the packages land in the same interpreter
> that will run the service.

Nothing needs configuring. The component roots are discovered from the
repository layout; `.env.example` documents every override.

### Reading the startup log

Everything the service emits goes through one formatter, including the
components' own `print()` output, which is captured and prefixed with `|`:

```
16:00:44 INFO [-] cvxai:               cvxai 1.0.0  |  project R26-IT-083
16:00:45 INFO [-] cvxai:               device: cuda:NVIDIA GeForce RTX 4060 Laptop GPU
16:00:47 INFO [-] cvxai.adapters.cxr:  | [classifier] Stage 5 epoch 12 | val 0.8483 | EMA weights applied: 346
16:00:54 INFO [-] cvxai.adapters.cxr:  loaded … in 9.3s
16:00:55 INFO [7835760624e5] cvxai:    GET /api/v1/cxr/analyze -> 200 (3912 ms)
```

`[-]` is the request id, absent outside a request. Health probes and browser
asset requests are not logged, so the clinical traffic stays visible.

One line looks like a problem and is not:

- `[report_gen] epoch 0 | metric None | EMA applied: 0` — how Component 01's
  Stage 11 checkpoint was saved. Its metadata carries no epoch counter or EMA
  shadow weights; the weights themselves load in full.

If you see `[service] !! original dataset not found … cardio_test.csv`, the
credentialed MIMIC-CXR data is not visible at `<Component_01>/data`. Only
`ground_truth_report` is affected — prediction, Grad-CAM and report generation
are unaffected — and `GET /api/v1/components` reports it under `notes`. See
[Validating Component 01 against real radiographs](#validating-component-01-against-real-radiographs)
for the junction that resolves it.

Anything that genuinely stops a component serving appears as `unavailable` in
`/api/v1/health` with the reason in `detail` — never as a log line you have to
notice.

---

## The integration argument

Four components that merely share a port are four components, not a system.
What makes these four a system is a property they already had independently:

**each was built around a mechanism that declines to commit when its own
evidence is weak.**

| Component | Its honesty mechanism |
|---|---|
| 01 | Per-projection operating points and selective deferral — AP films measure AUROC 0.8224 against 0.8864 on PA, so the AP deferral margin is 0.2247 against PA's 0.0029 |
| 02 | Quality gate → refusal before any probability exists; conformal rule-in / rule-out zones; guarantee withdrawal on suspected electrode reversal or out-of-scope rhythm; report verification gate |
| 03 | Split-conformal EF interval, learned aleatoric σ, inter-clip epistemic disagreement |
| 04 | Declared feature-availability horizon, constrained decision layer, clinician referral below a validation-chosen confidence |

These mechanisms are expressed in four incompatible vocabularies. The service
normalises them into one `reliability` block carrying a single verdict:

| `actionability` | Meaning |
|---|---|
| `actionable` | The component stands behind this result |
| `caution` | The result stands, but measured reliability is reduced |
| `deferred` | The component declines to commit; refer to a clinician |
| `withheld` | Output suppressed after a quality or verification failure |
| `unavailable` | The component could not run |

A caller applies **one** rule across all four modalities — *do not act on a
result that is not actionable* — without knowing anything about projections,
conformal zones or disclosure horizons.

The component-native payload is never rewritten. It is returned verbatim under
`raw`, so every published figure stays checkable against the component that
produced it.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service index |
| `GET` | `/api/v1/health` | Per-component readiness, with reasons |
| `GET` | `/api/v1/components` | Registry with model cards |
| `GET` | `/api/v1/components/{id}` | One component: metrics and limitations |
| `GET` | `/api/v1/cohorts` | Dataset provenance and **measured** cohort overlap |
| `POST` | `/api/v1/components/{id}/warm` | Load weights ahead of first use |
| `POST` | `/api/v1/cxr/analyze` | **01** — `file`, optional `view` (AP/PA) |
| `POST` | `/api/v1/ecg/analyze` | **02** — `dat_file` + `hea_file` |
| `POST` | `/api/v1/echo/analyze` | **03** — `file` (video or cached `.npy`) |
| `POST` | `/api/v1/triage/analyze` | **04** — JSON triage record |
| `POST` | `/api/v1/assessment` | All supplied modalities for one patient |

### Examples

```bash
# Chest radiograph, bedside film
curl -F "file=@study.png" -F "view=AP" \
     http://127.0.0.1:8000/api/v1/cxr/analyze

# 12-lead ECG
curl -F "dat_file=@00039_hr.dat" -F "hea_file=@00039_hr.hea" \
     http://127.0.0.1:8000/api/v1/ecg/analyze

# Echocardiogram
curl -F "file=@study.avi" http://127.0.0.1:8000/api/v1/echo/analyze

# ED triage record — every field optional
curl -H "Content-Type: application/json" -d '{
  "age": 61, "sex": "M", "heartrate": 108, "sbp": 92,
  "chief_complaint": "Crushing chest pain radiating to left arm",
  "troponin": [1.2, 6.8], "troponin_hours": [0.8, 3.5],
  "ecg": {"st_elevation": true, "acute": true}
}' http://127.0.0.1:8000/api/v1/triage/analyze

# Multi-modal
curl -F "patient_id=DEMO-001" -F "cxr_file=@study.png" -F "cxr_view=PA" \
     -F "echo_file=@study.avi" -F 'triage_json={"age":61,"chief_complaint":"chest pain"}' \
     http://127.0.0.1:8000/api/v1/assessment
```

### Response shape

```jsonc
{
  "component": "cxr",
  "status": "ok",
  "headline": "Cardiomegaly present (p=0.741)",
  "findings": [
    { "name": "Cardiomegaly", "present": true,
      "probability": 0.741, "threshold": 0.409 }
  ],
  "reliability": {
    "actionability": "caution",
    "level": "reduced",
    "reasons": ["AP (bedside) film. Measured AUROC on AP films is 0.8224 versus 0.8864 on PA…"],
    "guarantees": ["Selective deferral fitted on validation and frozen: 85.8 % of studies answered…"],
    "guarantees_void": false,
    "coverage": 0.858
  },
  "explanation": { "gradcam_png_base64": "…", "gradcam_caveat": "…" },
  "narrative": "FINDINGS: The heart is mildly enlarged…",
  "model": { "metrics": {…}, "limitations": [...], "decision_rule": "threshold=0.4090 from the AP operating point" },
  "raw": { /* the component's own payload, unmodified */ },
  "elapsed_ms": 1240,
  "request_id": "3f8ac6c6c660"
}
```

**A component declining to answer is a `200`, not an error.** A refused ECG, a
deferred radiograph and a referred triage case are the components doing their
job; turning them into HTTP errors would push callers into retry loops around a
safety mechanism. Errors are reserved for a genuine service failure and carry
`{"error", "message", "detail"}`.

---

## The multi-modal endpoint

`POST /api/v1/assessment` runs whichever modalities were supplied and answers
two questions the individual endpoints cannot:

1. **Can any of this be acted on?** — by reducing the per-component verdicts to
   their worst case, because a chain of evidence is no stronger than its
   weakest link.
2. **Do the modalities agree?** — as traceable observations, each naming the
   values it rests on:

   > `[discordance] cxr+echo` — An enlarged cardiac silhouette with preserved
   > ejection fraction. Cardiomegaly on a radiograph is a geometric finding and
   > does not require reduced systolic function; on an AP film it may also
   > reflect projection magnification.
   > **basis:** Component 01 cardiomegaly p=0.741 at threshold 0.409;
   > Component 03 EF 72.8 % graded Normal(≥55).

> **This is an aggregation, not a fusion model.** No joint model was trained
> across the four modalities and no combined performance is claimed. Every
> clinical number belongs to exactly one component and is reproduced unchanged.
> This is stated in the response itself, under `method_note`.

### Why fusion is not merely undone but impossible — measured

That claim is load-bearing, so it is measured rather than asserted.
`GET /api/v1/cohorts` reports the figures, regenerated by
`python scripts/measure_cohort_overlap.py`:

| Pair | Linkable | Shared patients |
|---|---|---|
| **01 + 04** (both MIMIC-IV derived, shared `subject_id`) | **yes** | **19,979** — 81.6 % of the radiograph cohort |
| 01 + 02, 01 + 03, 02 + 03, 02 + 04, 03 + 04 | no | 0, by construction |

Component 02 is PTB-XL (Physikalisch-Technische Bundesanstalt, Germany,
1989–96) and Component 03 is EchoNet-Dynamic (Stanford) plus CAMUS (France).
Different institutions, countries and decades, with no identifier that could
link them to each other or to MIMIC. A four-modality cohort therefore cannot be
constructed from these datasets at all — the overlap is zero by construction,
not merely unmeasured.

**One pair is genuinely linkable.** 19,979 patients appear in both Component
01's and Component 04's cohorts, so a paired radiograph + ED-triage study is
feasible future work. It is not done here, and patient-level overlap alone
would not be enough: a valid study also needs temporal linkage — the radiograph
must fall inside the ED stay window — which has not been established.

A modality that is absent, unavailable or failing appears in `skipped` rather
than failing the request — a broken echo loop must not cost the radiologist
their chest film.

---

## Architecture

```
backend/
├── run.py                   development entrypoint
├── requirements.txt
├── .env.example
└── cvxai/
    ├── main.py              app factory, middleware, error handlers
    ├── settings.py          env-driven config, component-root discovery
    ├── core/
    │   ├── sandbox.py       ★ module-namespace isolation
    │   ├── registry.py      adapter registry, lazy loading, health
    │   ├── errors.py        error taxonomy
    │   └── logging.py       request-id correlation
    ├── schemas/             the shared contract (common, triage, assessment)
    ├── adapters/            one per component: cxr, ecg, echo, triage
    ├── services/
    │   └── assessment.py    cross-modal aggregation
    └── api/v1/              routing
```

Adapters own three things and nothing else: whether the component *can* serve,
loading it once, and translating one study into the envelope. **No component
science is reimplemented.** Thresholds, calibration maps, conformal bounds and
decision rules are read from each component's own frozen artefacts and applied
by the component's own code.

### `core/sandbox.py` — why it exists

The four components were written independently and never intended to share a
process. Their top-level module names collide: `config`, `core`, `data` and
`models` are each claimed by two components. Measured on this repository:

```python
sys.path.insert(0, C3_TRAINING); from config import CFG   # Component 03's
sys.path.insert(0, C4_SRC);      from config import CFG   # STILL Component 03's
```

Component 04 receives Component 03's configuration object, and the first
symptom is a wrong number rather than an exception. Import order would silently
decide which component works.

Each component therefore runs inside a `ModuleSandbox`. Entering it installs
that component's `sys.path` entries, environment and finders; leaving it lifts
the component's own modules out of `sys.modules` into a private store.
Ownership is decided by file location, so shared packages (`torch`, `numpy`,
`transformers`) are imported once and stay global rather than being duplicated
four times. `verify_owns()` asserts afterwards that a name resolved to the
right component, turning the silent failure into a loud one.

**Concurrency.** `sys.modules` is process-global, so activation is serialised
by one re-entrant lock: inference is one-at-a-time across the service. That is
a deliberate correctness-over-throughput choice — the components carry their
own thread-affinity constraints too — and it is why the service runs with a
single worker. Route handlers are declared `def` so FastAPI runs them in its
threadpool and `/health` stays responsive while a model is busy.

### Component-specific obstacles solved

- **01** — its package is literally named `backend`, the same as this folder.
  The sandbox prepends the component root and `verify_owns("backend")` asserts
  the resolution, so a shadowing accident is a clear error, never a wrong
  number.
- **02** — every path carries a ` (1)` suffix from a zip extraction
  (`src (1)/pipeline (1).py`) while the code says `from .models import …`.
  `SuffixTolerantFinder` maps clean module paths onto whichever spelling exists
  on disk. Its asset resolver also expects clean names, so the small serving
  assets are staged under clean names in `backend/.cache/`. **Nothing in the
  component tree is renamed or written to**, and the same code keeps working if
  those names are ever cleaned up.
- **03** — ships *no* serving path: `run_eval.py` and `run_ensemble.py` work
  over the cached manifest, not over one uploaded study. The single-study path
  is reproduced step for step from `engine/evaluate.py::run_inference`, using
  the component's own sampling and motion routines so there is no train/serve
  skew.
- **04** — UM4's operating points are stated as a *coverage*, which is a
  population quantity; a single patient has no population. The coverage is
  converted once at load into an absolute cut-off on the top-two margin: the
  `(1 − coverage)` quantile over the component's persisted **validation**
  scores. Validation, not test.

---

## Maintenance scripts

All four live in `scripts/` and are safe to re-run.

```bash
# Measure the cohort overlap that backs the no-fusion claim (seconds, CPU)
python scripts/measure_cohort_overlap.py

# Fit and persist Component 03's ENSEMBLE-level decision rule (~15 min, GPU)
python scripts/freeze_echo_ensemble_calibration.py

# Score the chest-radiograph endpoint against real labelled MIMIC-CXR studies
python scripts/validate_cxr_endpoint.py --n 200

# Strip Component 02's " (1)" zip suffixes (dry run by default, reversible)
python scripts/normalize_component02_paths.py
python scripts/normalize_component02_paths.py --apply
```

### `normalize_component02_paths.py` — already applied

Component 02 arrived as a zip that de-duplicated every name: the package was
`src (1)/` holding `models (1).py`, while the code inside said
`from .models import …`. It could not import itself, find its own assets, or
run its audit suite; its own README documents the clean names throughout.

**11,660 paths have been renamed**, including the component root
(`Component_02 (1)` → `Component_02`), and Component 02 now runs standalone
again — `backend/server.py` starts, resolves its checkpoints, calibrator and
conformal thresholds, and reports `browseEnabled: true`.

The script is dry-run by default, refuses any rename that would overwrite an
existing path, skips `node_modules`, and writes a timestamped manifest to
`.cache/` so `--undo` restores the previous state exactly.

> One pre-existing data gap is unrelated to the rename and remains: the bundled
> PTB-XL signals are partial — **3,001 complete `.dat`/`.hea` pairs out of
> 21,799 records**. Uploading works fully; browsing the built-in test set works
> only for the records present, and Component 02's `audit/08_verify_fixes.py`
> stops on `ecg_id 9`, whose files were never included in the handover.

### `freeze_echo_ensemble_calibration.py`

Component 03's published headline (MAE 3.979, min-class recall 0.723) comes
from a decision rule fitted on the **ensemble's** validation predictions, which
`run_ensemble.py` refits per invocation and never writes out. Only per-member
`thresholds.json` files are persisted.

Running this script reproduces that calibration step exactly — using the
component's own `run_predictions` and `calibrate` — and stores the result in
`.cache/echo/`. The adapter prefers it when present. Either way,
`model.decision_rule` states which rule is actually in force, and says so
explicitly when it is the member-level fallback.

---

## Testing

```bash
python -m pytest                      # 50 tests
python -m pytest -m "not integration" # 33, no weights loaded, ~2 s
python -m pytest -m integration       # 17, real weights, ~25 s
```

Integration tests skip themselves when a component's assets are absent, so the
suite stays green on a checkout without checkpoints.

### Validating Component 01 against real radiographs

Components 02, 03 and 04 all ship a study the test-suite can score against a
known label. Component 01's images are credentialed MIMIC-CXR data held outside
the repository, so the endpoint could only be exercised with synthetic noise —
which proves the pipeline runs but nothing about whether it is right.

`scripts/validate_cxr_endpoint.py` closes that gap. It draws a stratified
sample from `manifest_test.csv`, posts each real image through the live
serving path, and compares the served decision with the radiologist-adjudicated
label, reporting Wilson intervals against Component 01's published figures
(accuracy 83.2 %, sensitivity 92.3 %, specificity 74.0 % at n = 4,722).

It needs the dataset visible at `<Component_01>/data/output/cardio_image_384`.
On this machine that is a directory junction to `C:\Users\dviya\Desktop\data`:

```powershell
New-Item -ItemType Junction -Path ..\Component_01\data -Target C:\Users\dviya\Desktop\data
```

The junction also restores Component 01's original-report index, so
`ground_truth_report` is populated for bundled test images instead of null.

**Result on 200 stratified real studies** (seed 20260818, 128 AP / 72 PA):

| Metric | Served | 95 % Wilson CI | Published (n = 4,722) | |
|---|---|---|---|---|
| Accuracy | 0.7900 | [0.728, 0.841] | 0.832 | consistent |
| Sensitivity | 0.8800 | [0.802, 0.930] | 0.923 | consistent |
| Specificity | 0.7000 | [0.604, 0.781] | 0.740 | consistent |

Confusion TP 88 / FP 30 / FN 12 / TN 70; 14.0 % deferred, against the policy's
85.8 % target coverage. **AP accuracy 0.766 against PA 0.833** — the serving
path reproduces the acquisition gap that is Component 01's research
contribution, on real films, rather than merely running without error.

Two further tests guard mistakes that were made and caught during this
integration, and both are the kind that produce plausible wrong numbers rather
than errors:

- `test_conformal_interval_is_clinically_plausible` — Component 03's `q_hat`
  was calibrated against the **inter-clip spread alone**. Feeding it the
  combined aleatoric-plus-epistemic σ instead is arithmetically valid and
  clinically meaningless: it widened a 95 % interval from ≈ ±7 EF points to
  ±37.
- `test_projection_selects_the_operating_point` — asserts the AP threshold
  exceeds PA, the same direction as the clinical CTR convention.

---

## Notes for the team

**The per-component servers are superseded.** Component 01's FastAPI app and
Component 02's Flask app are no longer run; this service replaces both. Their
source is deliberately left in place because the unified backend **imports the
model code that lives beside them** — deleting `Component_01/…/backend/` would
delete the classifier, report generator and Grad-CAM definitions this service
depends on. Both still run standalone if you want them to.

**Components 03 and 04 gain an API they never had.** Both were research code
with no serving surface at all.

**What this work changed outside `backend/`.** Three things, all reversible and
all recorded:

| Change | Why | Reverse |
|---|---|---|
| 11,660 paths in Component 02 renamed, root `Component_02 (1)` → `Component_02` | zip-extraction suffixes stopped the component importing itself | `scripts/normalize_component02_paths.py --undo <manifest>` |
| Junction `Component_01/data` → `C:\Users\dviya\Desktop\data` | reconnects Component 01 to its own dataset and original-report index | `rmdir Component_01\data` (removes the link, not the data) |
| Two stale `__pycache__` directories deleted in Component 02 | they shadowed the renamed modules | regenerated on next import |

No source file inside any component was edited.

**Performance, measured on this machine** (RTX 4060 Laptop, CUDA):

| Step | Time |
|---|---|
| Component 01 first load | ≈ 29 s (ConvNeXt + BioBART) |
| Chest radiograph inference | ≈ 3.9 s |
| ECG inference (with XAI) | ≈ 3.8 s |
| Echocardiogram, 3 seeds × 10 clips | ≈ 2.5 s |
| ED triage | ≈ 1.5 s |
| Four-modality assessment | ≈ 19 s cold, ≈ 12 s warm |

Use `--warm` or `POST /components/{id}/warm` before a live demonstration; the
first request otherwise pays the full model-loading cost.

---

## Licence and data

Code: MIT. **Data is not.** MIMIC-CXR, MIMIC-IV-ED and PTB-XL are credentialed
under PhysioNet data use agreements; EchoNet-Dynamic and CAMUS carry their own
terms. No images, waveforms, videos, reports or derived datasets are
distributed here, and model weights derived from credentialed data are not
redistributable either.
