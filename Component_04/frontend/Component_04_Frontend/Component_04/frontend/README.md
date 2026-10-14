# Component 04 — ACS Clinical Review Frontend

React 19 + Vite 6 user interface for the **Temporally-Safe Explainable Acute Coronary Syndrome Triage** research component. Inspired by the dense clinical-worklist approach of Component 02, with ACS-specific inputs and output sections.

> **Safety:** Research prototype only. Not a medical device, diagnosis or clinical decision tool. The four synthetic examples contain **invented probabilities**, not outputs from trained models; they must not be interpreted as model evaluation results.

## Features

- Responsive clinical review workspace with a patient input form (vitals, clinical text, two troponin values and interpreted ECG flags).
- Four clearly labelled **synthetic** scenarios (STEMI, NSTEMI, UA, No ACS), each with example probability bars and explanatory panels.
- Input edits invalidate the displayed example so outdated illustrative results cannot be mistaken for a fresh inference.
- Optional HTTP adapter connection with request/error/loading states; **no live prediction** when an adapter URL is missing.
- Feature, modality and complaint-token sections appear **only if those fields are returned** by the adapter; absent explanations are explicitly labelled unavailable.

## Start locally

From `Component_04/frontend`:

```bash
npm install
npm run dev
```

Open `http://localhost:5174`.

To verify production compilation:

```bash
npm run build
npm run preview
```

## Connect the research Python model

**Important:** The existing `Component_04/src/predict.py` is a **CLI**, not a FastAPI/Flask server. A separate HTTP adapter is necessary for real predictions. Do not point the frontend at an unrelated backend from Components 01–03.

1. Write a server-side adapter which safely calls `src/models/inference.py` / `ACSPredictor` and validates inputs, feature-availability horizon, auth and clinical data handling.
2. Serve `POST /predict`, receiving a JSON object and returning a JSON response with `prediction`, `p_acs`, and `probabilities`.
3. Optional keys: `risk_level`, `recommended_action`, `feature_importance`, `modality_attribution`, and `text_attribution`. Only return attributions that were actually calculated.
4. Copy `.env.example` to `.env.local`; set `VITE_ACS_API_URL=http://localhost:<port>` as appropriate, and restart Vite.
5. Configure the server's CORS for the development frontend origin when necessary.

### Expected JSON response example (shape only)

```json
{
  "prediction": "NSTEMI",
  "p_acs": 0.86,
  "probabilities": { "No_ACS": 0.14, "UA": 0.04, "NSTEMI": 0.75, "STEMI": 0.07 },
  "horizon_h": 24
}
```

This example is a **contract illustration**, not an actual inference. The current CLI `explain_row` returns probabilities and token annotations, **not SHAP feature contributions**; showing SHAP in live mode requires calculating and returning genuine SHAP arrays from the explainability pipeline.

## Related files

- `../README.md` — research method, evaluation, leakage audit, limitations.
- `../src/predict.py` — existing CLI single-patient prediction and input mapping.
- `../src/models/inference.py` — trained cascade predictor.
- `../src/models/unified4.py` — separately evaluated unified 4-class model; not assumed to be the same as the CLI's cascade.
- `../src/analysis/explain.py` — SHAP / modality / token explainability calculations.

## Privacy and deployment

Never commit patient data, private API credentials, the original MIMIC datasets or browser-accessible tokens. `VITE_` variables are bundled into client JavaScript and **are not suitable for secrets**. The frontend alone does not provide a secure patient-care deployment. Additional verification, authorization, security testing, provenance checks and external clinical validation are required before any real-world use.
