"""
server.py — Component-02 JSON API.

Replaces the server-rendered Flask app in _archive/app.py (and supersedes
Component_02/app/app.py). This process serves DATA ONLY; the React client in
../frontend renders it.

Design rules carried over from the audit:
  * every architecture comes from src.models — nothing is redefined here (E-10)
  * the quality gate runs before the classifier; bad records are REFUSED (C-1)
  * XAI state is per-request (C-5)
  * uploads are sanitised and size-capped (E-3)
  * binds 127.0.0.1 unless HOST is set explicitly (E-9)

NEW — startup provenance enforcement:
  The old app defaulted to the archive checkpoint while the shipped calibrator
  belonged to the retrained model. That pairing silently produces invalid
  probabilities and void guarantees. This server now defaults to the retrained
  model AND refuses to start if the calibrator / conformal thresholds were fitted
  for something else. Set ECG_ALLOW_MISMATCH=1 only if you know why you want it.

Run:
    python -X utf8 Component_02/backend/server.py
    HOST=0.0.0.0 PORT=5000 python -X utf8 Component_02/backend/server.py
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
ROOT = os.path.dirname(COMP)
sys.path.insert(0, COMP)

from src import paths, signals                                 # noqa: E402
from src.models import CLASS_NAMES, LEAD_NAMES, SAMPLING_RATE   # noqa: E402
from src.pipeline import ECGPipeline                            # noqa: E402
from src.report import CLASS_FULL, DISCLAIMER                   # noqa: E402

# Assets are resolved by src.paths so this folder can be handed to someone else
# without the rest of the repository. See Component_02/README.md.
NORM_STATS = paths.require("norm_stats.json")
TEST_CSV = paths.find("test.csv")
CKPT_DIR = os.path.join(COMP, "checkpoints")

MAX_UPLOAD_MB = 25

# ── defaults now point at the RETRAINED model (was the archive model) ──────
CKPT = os.environ.get("ECG_CKPT", os.path.join(CKPT_DIR, "best_model.pt"))
MODEL = os.environ.get("ECG_MODEL", "resnet_se")
FILTER = os.environ.get("ECG_FILTER", "1") == "1"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")}})


# ══════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════
def _startup():
    print("=" * 70)
    print("  Component 02 — ECG API")
    print("=" * 70)
    if not os.path.exists(CKPT):
        raise SystemExit(
            f"Checkpoint not found: {CKPT}\n"
            f"Train one (train/train_gpu.py) or set ECG_CKPT to an existing file.")

    pipe = ECGPipeline.from_checkpoint(
        ckpt_path=CKPT,
        norm_stats_path=NORM_STATS,
        model_name=MODEL,
        calibrator_path=os.path.join(CKPT_DIR, "calibrator.json"),
        triage_path=os.path.join(CKPT_DIR, "conformal_triage.json"),
        do_filter=FILTER)

    print(f"  model      : {MODEL}   filter={FILTER}")
    print(f"  checkpoint : {os.path.relpath(CKPT, ROOT)}")
    print(f"  calibrator : {'loaded' if pipe.calibrator else 'MISSING'}")
    print(f"  conformal  : {'loaded' if pipe.triage else 'MISSING'}")

    # Provenance: a calibrator is only valid for the model it was fitted on.
    problems = []
    for label, obj in (("calibrator", pipe.calibrator), ("conformal", pipe.triage)):
        if obj is None:
            problems.append(f"{label} is missing — run train/fit_calibration.py")
            continue
        meta = getattr(obj, "fitted_for", {}) or {}
        if not meta:
            continue
        if (meta.get("model"), bool(meta.get("filter", False))) != (MODEL, FILTER):
            problems.append(
                f"{label} was fitted for model={meta.get('model')} "
                f"filter={meta.get('filter')}, but this server runs model={MODEL} "
                f"filter={FILTER}")
    if problems:
        msg = "\n".join(f"    - {p}" for p in problems)
        if os.environ.get("ECG_ALLOW_MISMATCH") == "1":
            print(f"  WARNING (ECG_ALLOW_MISMATCH=1):\n{msg}")
        else:
            raise SystemExit(
                f"\n  REFUSING TO START — the safety layer does not match the model:\n"
                f"{msg}\n\n  Fix with:\n"
                f"    python Component_02/train/fit_calibration.py --model {MODEL} "
                f"--ckpt {CKPT}{' --filter' if FILTER else ''} --from-logits --delta 0.01\n"
                f"  Override (not recommended): ECG_ALLOW_MISMATCH=1\n")

    if pipe.triage is not None:
        d = getattr(pipe.triage, "delta", None)
        print(f"  guarantee  : PAC delta={d}" if d else "  guarantee  : marginal")
    return pipe


PIPE = _startup()

TEST_DF = pd.read_csv(TEST_CSV) if TEST_CSV else pd.DataFrame()
BROWSE_ENABLED = bool(len(TEST_DF)) and signals.available()
FILE_BY_ID = (dict(zip(TEST_DF.ecg_id, TEST_DF.filename_hr))
              if "filename_hr" in TEST_DF.columns else {})
CLASS_INDEX = {}
for c in CLASS_NAMES:
    col = f"label_{c}"
    if col in TEST_DF.columns:
        sub = TEST_DF[TEST_DF[col] == 1][["ecg_id", "report_en", "age", "sex"]].copy()
        sub["sex"] = sub["sex"].map({0: "M", 1: "F"})
        sub["age"] = sub["age"].apply(lambda a: None if pd.isna(a) or a >= 300 else int(a))
        CLASS_INDEX[c] = sub.replace({np.nan: None}).to_dict("records")
    else:
        CLASS_INDEX[c] = []
print(f"  norm stats : {os.path.relpath(NORM_STATS, ROOT)}")
print(f"  test set   : {len(TEST_DF)} records"
      + ("" if TEST_CSV else "  (test.csv not found)"))
print(f"  signals    : {signals.source_description()}")
if not BROWSE_ENABLED:
    print("  NOTE: browsing the built-in test set is DISABLED. Add either")
    print("        data/raw_signals/ (1.9 GB) or data/signals_cache/ (3.9 GB).")
    print("        Uploading .dat/.hea files works regardless.")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════
def render_ecg(signal_mv: np.ndarray, cam, r_peaks, dark: bool = False) -> str:
    """Render on standard ECG paper: 25 mm/s, 10 mm/mV, 3x4 + rhythm strip.

    Cardiologists read against the grid, not against the curve. Small square =
    0.04 s x 0.1 mV, large square = 0.20 s x 0.5 mV, with the conventional
    calibration pulse. Anything else looks like a plot, not an ECG.
    """
    sig = signal_mv.T if signal_mv.shape[0] > signal_mv.shape[1] else signal_mv
    n = sig.shape[1]
    dur = n / float(SAMPLING_RATE)

    paper = "#fffafa" if not dark else "#12151c"
    fine = "#f7d9d9" if not dark else "#33232b"
    bold = "#eaacac" if not dark else "#523440"
    trace = "#141414" if not dark else "#e8eef7"
    label = "#333" if not dark else "#c9d4e3"

    COLS, ROWS = 4, 3
    seg = dur / COLS                      # 2.5 s per column, clinical standard
    cam_up = (np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(cam)), cam)
              if cam is not None and len(cam) else None)

    fig = plt.figure(figsize=(16, 9.5), facecolor=paper)
    ax = fig.add_axes([0.03, 0.075, 0.95, 0.895])
    ax.set_facecolor(paper)

    # ── grid: x in seconds, y in millivolts ──────────────────────────────
    total_x, row_h = dur, 3.0             # 3 mV of vertical space per row
    total_y = row_h * (ROWS + 1)          # +1 for the rhythm strip
    ax.set_xlim(0, total_x); ax.set_ylim(0, total_y)
    for x in np.arange(0, total_x + 1e-9, 0.04):
        ax.axvline(x, color=fine, lw=0.35, zorder=0)
    for x in np.arange(0, total_x + 1e-9, 0.20):
        ax.axvline(x, color=bold, lw=0.75, zorder=0)
    for y in np.arange(0, total_y + 1e-9, 0.1):
        ax.axhline(y, color=fine, lw=0.35, zorder=0)
    for y in np.arange(0, total_y + 1e-9, 0.5):
        ax.axhline(y, color=bold, lw=0.75, zorder=0)

    def draw(lead_idx, x0, x1, baseline, name, clip=True):
        i0, i1 = int(x0 * SAMPLING_RATE), int(x1 * SAMPLING_RATE)
        i1 = min(i1, n)
        t = np.arange(i0, i1) / float(SAMPLING_RATE)
        y = sig[lead_idx, i0:i1]
        if clip:
            y = np.clip(y, -1.4, 1.4)
        ax.plot(t, baseline + y, color=trace, lw=0.9, zorder=3,
                solid_joinstyle="round")
        ax.text(x0 + 0.05, baseline + 1.05, name, fontsize=10,
                fontweight="bold", color=label, zorder=4)

    # ── 3 x 4 standard layout ────────────────────────────────────────────
    order = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    for r, row in enumerate(order):
        base = total_y - (r + 0.5) * row_h
        for c, li in enumerate(row):
            x0 = c * seg
            draw(li, x0, x0 + seg, base, LEAD_NAMES[li])
            if c > 0:
                ax.plot([x0, x0], [base - 1.3, base + 1.3], color=bold,
                        lw=0.8, zorder=2)

    # ── rhythm strip: lead II, full duration ─────────────────────────────
    rb = total_y - (ROWS + 0.5) * row_h
    draw(1, 0, dur, rb, "II  (rhythm strip)", clip=True)

    # ── attention overlay ────────────────────────────────────────────────
    # Drawn as a thin band BENEATH each row, never over the trace. Painting a
    # wash across the waveform obscures the exact morphology a reader needs;
    # clinical annotation sits beside the signal, not on top of it.
    if cam_up is not None:
        band = 0.16
        for r, row in enumerate(order):
            base = total_y - (r + 0.5) * row_h
            for c, _ in enumerate(row):
                x0 = c * seg
                i0 = int(x0 * SAMPLING_RATE)
                i1 = min(int((x0 + seg) * SAMPLING_RATE), n)
                ax.imshow(cam_up[None, i0:i1], aspect="auto",
                          extent=[x0, x0 + seg, base - 1.45, base - 1.45 + band],
                          cmap="Reds", alpha=0.85, vmin=0, vmax=1,
                          interpolation="bilinear", zorder=4)
        ax.imshow(cam_up[None, :], aspect="auto",
                  extent=[0, dur, rb - 1.45, rb - 1.45 + band], cmap="Reds",
                  alpha=0.85, vmin=0, vmax=1, interpolation="bilinear", zorder=4)
        fig.text(0.035, 0.028, "model attention (band below each row)",
                 fontsize=8.5, color=label, family="monospace")

    # ── calibration pulse: 1 mV, 200 ms ──────────────────────────────────
    for r in range(ROWS):
        base = total_y - (r + 0.5) * row_h
        ax.plot([0.02, 0.02, 0.22, 0.22, 0.24],
                [base, base + 1.0, base + 1.0, base, base],
                color=trace, lw=1.1, zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.975, 0.028, "25 mm/s     10 mm/mV     0.5-40 Hz",
             ha="right", fontsize=8.5, color=label, family="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=125, facecolor=paper,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyse(signal_raw, fs, patient_id, dark=False, lead_names=None):
    t0 = time.time()
    res = PIPE.analyse(signal_raw, fs=fs, with_xai=True, lead_names=lead_names)
    payload = res.to_json()
    payload["patientId"] = patient_id
    payload["disclaimer"] = DISCLAIMER
    payload["classFullNames"] = CLASS_FULL
    q = res.quality
    payload["electrode"] = {
        "suspected": bool(q.electrode_suspect),
        "reversal": q.electrode_reversal,
        "confidence": round(float(q.electrode_confidence), 2),
        "reasons": list(q.electrode_reasons or []),
    }

    cam = None
    if res.explanations:
        first = next(iter(res.explanations.items()))
        payload["gradcamTarget"] = first[0]
        cam = first[1].cam
        e = first[1]
        payload["explanation"] = {
            "target": first[0],
            "territory": e.territory,
            "artery": e.territory_artery,
            "territoryScore": round(e.territory_score, 2),
            "topLeads": e.top_leads,
            "peaksSeconds": [round(x, 2) for x in e.cam_peaks_s],
        }
        payload["leads"] = sorted(
            [{"name": k, "signed": round(v, 1),
              "magnitude": round(e.lead_magnitude.get(k, 0.0), 1)}
             for k, v in e.lead_signed.items()],
            key=lambda d: -abs(d["signed"]))
    else:
        payload["leads"] = []

    if res.signal_mv is not None and res.signal_mv.size:
        payload["ecgImage"] = render_ecg(res.signal_mv, cam, res.r_peaks, dark)

    if res.probs_calibrated and res.zones:
        payload["classes"] = [{
            "name": c,
            "fullName": CLASS_FULL[c],
            "probability": round(res.probs_calibrated[c] * 100, 1),
            "probabilityRaw": round(res.probs_raw[c] * 100, 1),
            "zone": res.zones[c],
            "ruleOut": (round(PIPE.triage.thresholds[c].lambda_out * 100, 1)
                        if PIPE.triage else None),
            "ruleIn": (round(PIPE.triage.thresholds[c].lambda_in * 100, 1)
                       if PIPE.triage else None),
            "alpha": PIPE.triage.thresholds[c].alpha if PIPE.triage else None,
        } for c in CLASS_NAMES]
    else:
        payload["classes"] = []

    payload["elapsedMs"] = int((time.time() - t0) * 1000)
    return payload


# ══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    thr = ({c: {"ruleOut": round(t.lambda_out * 100, 1),
                "ruleIn": round(t.lambda_in * 100, 1),
                "alpha": t.alpha, "beta": t.beta,
                "nPos": t.n_pos_cal} for c, t in PIPE.triage.thresholds.items()}
           if PIPE.triage else {})
    return jsonify(
        status="ok",
        model=MODEL,
        filter=FILTER,
        checkpoint=os.path.basename(CKPT),
        calibrated=PIPE.calibrator is not None,
        conformal=PIPE.triage is not None,
        delta=getattr(PIPE.triage, "delta", None) if PIPE.triage else None,
        classes=CLASS_NAMES,
        classFullNames=CLASS_FULL,
        thresholds=thr,
        counts={c: len(CLASS_INDEX[c]) for c in CLASS_NAMES},
        nTest=len(TEST_DF),
        browseEnabled=BROWSE_ENABLED,
        disclaimer=DISCLAIMER,
    )


@app.get("/api/patients/<class_name>")
def patients(class_name):
    if not BROWSE_ENABLED:
        return jsonify(error="test-set browsing is disabled on this install "
                             "(signals_cache not present); use the upload panel",
                       patients=[], total=0), 200
    if class_name not in CLASS_INDEX:
        return jsonify(error=f"unknown class: {class_name}"), 400
    q = (request.args.get("q") or "").strip().lower()
    rows = CLASS_INDEX[class_name]
    if q:
        rows = [r for r in rows
                if q in str(r["ecg_id"]) or q in (r.get("report_en") or "").lower()]
    return jsonify(cls=class_name, total=len(rows), patients=rows[:500])


@app.post("/api/analyze/<int:ecg_id>")
def analyze_by_id(ecg_id):
    if not BROWSE_ENABLED:
        return jsonify(error="test-set browsing is disabled on this install "
                             "(signals_cache not present); use the upload panel"), 503
    row = TEST_DF[TEST_DF.ecg_id == ecg_id]
    if row.empty:
        return jsonify(error=f"ECG {ecg_id} is not in the test set"), 404
    try:
        sig = signals.load(ecg_id, FILE_BY_ID.get(ecg_id))
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 404
    dark = request.args.get("theme") == "dark"
    payload = analyse(sig, SAMPLING_RATE, ecg_id, dark)
    r = row.iloc[0]
    payload["referenceReport"] = str(r.get("report_en") or "")
    payload["groundTruth"] = [c for c in CLASS_NAMES if r[f"label_{c}"] == 1]
    age = r.get("age")
    payload["patient"] = {
        "age": None if pd.isna(age) or age >= 300 else int(age),
        "sex": {0: "M", 1: "F"}.get(r.get("sex")),
    }
    return jsonify(payload)


@app.post("/api/demo")
def demo():
    if not BROWSE_ENABLED:
        return jsonify(error="test-set browsing is disabled on this install "
                             "(signals_cache not present); use the upload panel"), 503
    return analyze_by_id(int(TEST_DF.sample(1).iloc[0]["ecg_id"]))


@app.post("/api/predict")
def predict():
    """Upload a WFDB .dat + .hea pair."""
    import wfdb

    if "dat_file" not in request.files or "hea_file" not in request.files:
        return jsonify(error="upload both a .dat and a .hea file"), 400
    dat_file, hea_file = request.files["dat_file"], request.files["hea_file"]
    dat_name = secure_filename(dat_file.filename or "")
    hea_name = secure_filename(hea_file.filename or "")
    if not dat_name.lower().endswith(".dat") or not hea_name.lower().endswith(".hea"):
        return jsonify(error="expected one .dat and one .hea file"), 400
    if os.path.splitext(dat_name)[0] != os.path.splitext(hea_name)[0]:
        return jsonify(error="the .dat and .hea files must share a base name"), 400

    with tempfile.TemporaryDirectory() as tmp:
        dat_file.save(os.path.join(tmp, dat_name))
        hea_file.save(os.path.join(tmp, hea_name))
        stem = os.path.splitext(hea_name)[0]
        try:
            rec = wfdb.rdrecord(os.path.join(tmp, stem))
        except Exception as e:
            return jsonify(error=f"could not read the WFDB record: {e}"), 400
        sig = np.asarray(rec.p_signal, dtype=np.float32)
        fs = int(rec.fs or SAMPLING_RATE)
        names = [str(s) for s in (rec.sig_name or [])]

    if sig.ndim != 2 or sig.shape[1] != 12:
        got = sig.shape[1] if sig.ndim == 2 else "?"
        return jsonify(error=f"expected 12 leads, got {got}"), 400

    warn = None
    if names:
        norm = [n.strip().upper() for n in names]
        expect = [n.upper() for n in LEAD_NAMES]
        if norm != expect:
            if sorted(norm) == sorted(expect):
                sig = sig[:, [norm.index(e) for e in expect]]
                warn = f"leads were reordered from {names} to the standard order"
            else:
                return jsonify(error=f"unrecognised lead set {names}"), 400

    dark = request.args.get("theme") == "dark"
    payload = analyse(sig, fs, os.path.splitext(hea_name)[0], dark)
    payload["uploaded"] = {"fs": fs, "samples": int(sig.shape[0]), "leads": names}
    if warn:
        payload.setdefault("quality", {}).setdefault("warnings", []).append(warn)
    return jsonify(payload)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"upload exceeds {MAX_UPLOAD_MB} MB"), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify(error=f"internal error: {e}"), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  {DISCLAIMER}\n")
    print(f"  API   http://{host}:{port}/api/health")
    print(f"  UI    http://localhost:5173  (run `npm run dev` in ../frontend)\n")
    app.run(host=host, port=port, debug=False, threaded=True)
