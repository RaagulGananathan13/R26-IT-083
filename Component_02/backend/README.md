# Backend — Component-02 JSON API

Serves data only. Rendering is the React client's job (`../frontend`).
All model code is imported from `../src` — nothing is redefined here.

## Run

**From inside the `Component_02/` folder:**

```bash
cd Component_02
pip install -r requirements.txt
python -X utf8 backend/server.py        # http://127.0.0.1:5000
```

From the parent folder instead, use the full path:

```bash
python -X utf8 Component_02/backend/server.py
```

> `can't open file '...Component_02\Component_02\backend\server.py'` means you
> are already inside `Component_02/` and doubled the prefix. Drop it.

The server locates its own files from `__file__`, so either working directory
works — only the path you type has to match where you are.

## Defaults

| Variable | Default | Note |
|---|---|---|
| `ECG_CKPT` | `Component_02/checkpoints/best_model.pt` | the **retrained** model |
| `ECG_MODEL` | `resnet_se` | |
| `ECG_FILTER` | `1` | band-pass on, matching how it was trained |
| `HOST` | `127.0.0.1` | `0.0.0.0` is opt-in |
| `PORT` | `5000` | |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | |

> The old server defaulted to the **archive** checkpoint while the shipped
> calibrator belonged to the retrained model. That pairing runs fine and produces
> invalid probabilities and void guarantees.

## Startup provenance enforcement

`calibrator.json` and `conformal_triage.json` record the model they were fitted
for. If they do not match the model being loaded, **the server refuses to start**
and prints the command that fixes it:

```
REFUSING TO START — the safety layer does not match the model:
    - calibrator was fitted for model=resnet_se filter=True,
      but this server runs model=resnet filter=False
```

Override with `ECG_ALLOW_MISMATCH=1` only if you know why you want it.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | model, calibrator/conformal status, δ, class counts, thresholds |
| GET | `/api/patients/<class>?q=` | test-fold records for a class |
| POST | `/api/analyze/<ecg_id>?theme=` | full analysis + ground truth |
| POST | `/api/demo?theme=` | random test record |
| POST | `/api/predict?theme=` | upload `.dat` + `.hea` |

`theme=dark` renders the ECG plot on a dark background so it matches the UI.

## Request path

```
quality gate → preprocess → classify → calibrate → conformal triage
             → XAI → grounded report → verify → JSON
```

The quality gate runs **before** the classifier, so a refused record never
produces a probability. Uploads are `secure_filename`-sanitised, capped at 25 MB,
lead-order validated (and reordered when possible), and rejected if the sampling
rate implies an unsupported duration.

## Supersedes

`Component_02/app/` was the server-rendered version (Flask + Jinja). It still
works, but this API + React client replaces it. Nothing in `_archive/` is used
at runtime except the dataset and its CSVs.
