"""
fit_calibration.py — Fit the calibrator and the conformal triage layer on the
EXISTING baseline checkpoint. No GPU, no retraining, ~3 minutes on CPU.

This is the step that turns the audited baseline into the Component-02 system.
It produces the two artefacts the pipeline needs:

    Component_02/checkpoints/calibrator.json      (temperature scaling)
    Component_02/checkpoints/conformal_triage.json (rule-out / refer / rule-in)

and the evaluation tables for the thesis:

    Component_02/audit/results/07_conformal_eval.txt / .json

Discipline: the calibrator and the conformal thresholds are fitted on the
VALIDATION fold (strat_fold 9) ONLY, and verified on the TEST fold (fold 10).
Fitting either on test would be exactly the leak the audit warned about.

Usage:
    python -X utf8 Component_02/train/fit_calibration.py
    python -X utf8 Component_02/train/fit_calibration.py --model resnet_se \
        --ckpt Component_02/checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "Component_02"))

from src import preprocess as pp                                     # noqa: E402
from src.calibration import TemperatureCalibrator, calibration_report  # noqa: E402
from src.conformal import (ConformalTriage, RULE_IN, RULE_OUT, REFER,  # noqa: E402
                           risk_coverage_curve, PRESETS)
from src.models import CLASS_NAMES, build_model                       # noqa: E402

from src import paths as _paths  # noqa: E402
DATA = os.path.dirname(_paths.require("test.csv"))
CACHE = _paths.signals_cache() or ""
OUT_CKPT = os.path.join(ROOT, "Component_02", "checkpoints")
OUT_RES = os.path.join(ROOT, "Component_02", "audit", "results")

lines = []
def p(s=""):
    print(s, flush=True); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


PACK_DIR = os.environ.get("ECG_PACK_DIR",
                          os.path.join(ROOT, "Component_02", "data"))


def load_split(split, df, mean, std, do_filter):
    """Load a split's model inputs.

    Prefers the packed float16 memmap written by train_gpu.py --pack. Falls back
    to reading signals_cache per record. On Colab only the packed data is
    uploaded, so the fallback would fail — hence the explicit error message.
    """
    packed = os.path.join(PACK_DIR, f"{split}_X.npy")
    done = os.path.join(PACK_DIR, f"{split}.done")
    if os.path.exists(packed) and not os.environ.get("ECG_NO_PACKED"):
        # TRAIN/SERVE CONSISTENCY GUARD.
        # The packed arrays already have the band-pass baked in. Feeding filtered
        # data to a model trained on unfiltered data (or vice versa) silently
        # degrades everything downstream — it shifted macro-ECE from 0.183 to
        # 0.209 in testing before this check existed. Refuse rather than warn.
        if os.path.exists(done):
            with open(done) as f:
                meta = json.load(f)
            packed_filtered = bool(meta.get("filtered", True))
            if packed_filtered != bool(do_filter):
                raise SystemExit(
                    f"PREPROCESSING MISMATCH for '{split}'.\n"
                    f"  packed data was built with filter={packed_filtered}\n"
                    f"  you passed                     --filter={bool(do_filter)}\n"
                    f"A model must be served exactly the preprocessing it was "
                    f"trained on.\n"
                    f"  * new resnet_se model (trained on the packed data): add --filter\n"
                    f"  * old archive resnet model (trained unfiltered): run with\n"
                    f"    ECG_NO_PACKED=1 so it reads signals_cache instead.")
        X = np.asarray(np.load(packed, mmap_mode="r"), dtype=np.float32)
        Y = np.load(os.path.join(PACK_DIR, f"{split}_Y.npy")).astype(float)
        if len(X) != len(df):
            raise SystemExit(f"{split}: packed data has {len(X)} records but "
                             f"{split}.csv has {len(df)} — they are out of sync.")
        return X, Y

    if not os.path.isdir(CACHE):
        raise SystemExit(
            f"Neither packed data ({packed}) nor signals_cache ({CACHE}) is "
            f"available.\nOn Colab, pass --from-logits to reuse the logits that "
            f"train_gpu.py already saved in Component_02/checkpoints/.")

    X, Y = [], []
    for r in df.itertuples():
        s = np.load(os.path.join(CACHE, f"{int(r.ecg_id)}.npy")).astype(np.float32)
        if do_filter:
            s = pp.bandpass(s)
        s = pp.normalise(s, mean, std, per_record=do_filter)
        X.append(s.T)
        Y.append([getattr(r, f"label_{c}") for c in CLASS_NAMES])
    return np.stack(X), np.asarray(Y, dtype=float)


def find_saved_logits(seed: int):
    """train_gpu.py already dumped val/test logits — reuse them, no inference."""
    for v, t in ((f"val_logits_seed{seed}.npy", f"test_logits_seed{seed}.npy"),
                 ("val_logits.npy", "test_logits.npy")):
        for d in (OUT_CKPT, OUT_RES):
            pv, pt = os.path.join(d, v), os.path.join(d, t)
            if os.path.exists(pv) and os.path.exists(pt):
                return pv, pt
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT_CKPT, "best_model.pt"))
    ap.add_argument("--model", default="resnet_se", choices=["resnet", "resnet_se"])
    ap.add_argument("--filter", action="store_true",
                    help="apply the Component-02 band-pass. Only use this if the "
                         "checkpoint was TRAINED with filtering, otherwise it is a "
                         "train/serve mismatch.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="PAC confidence level; pass 0 for the weaker marginal bound")
    ap.add_argument("--reuse-logits", "--from-logits", dest="reuse_logits",
                    action="store_true",
                    help="skip inference entirely and reuse the logits train_gpu.py "
                         "saved in checkpoints/ (no model, no signal data needed)")
    ap.add_argument("--seed", type=int, default=0,
                    help="which seed's saved logits to reuse")
    ap.add_argument("--preset", default="safety",
                    choices=["safety", "balanced", "throughput"],
                    help="clinical operating point (see src/conformal.PRESETS)")
    args = ap.parse_args()
    if args.delta is not None and args.delta <= 0:
        args.delta = None

    os.makedirs(OUT_CKPT, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)

    hdr("0. SETUP")
    val = pd.read_csv(os.path.join(DATA, "val.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    Yv = val[[f"label_{c}" for c in CLASS_NAMES]].values.astype(float)
    Yt = test[[f"label_{c}" for c in CLASS_NAMES]].values.astype(float)

    lv_path, lt_path = find_saved_logits(args.seed)
    use_saved = args.reuse_logits and lv_path is not None

    p(f"  model      : {args.model}")
    p(f"  band-pass  : {'ON' if args.filter else 'OFF (matches the archive checkpoint)'}")
    p(f"  calibration split (fold 9) : {len(val)} records")
    p(f"  evaluation  split (fold 10): {len(test)} records")

    if use_saved:
        p(f"  logits     : reusing {os.path.basename(lv_path)} / "
          f"{os.path.basename(lt_path)} — no inference, no model load")
        Lv, Lt = np.load(lv_path), np.load(lt_path)
        # Prefer the labels saved alongside the logits; they are guaranteed to be
        # in the same row order as the logits.
        for name, arr in (("val_labels.npy", "Yv"), ("test_labels.npy", "Yt")):
            for d in (OUT_CKPT, OUT_RES):
                fp = os.path.join(d, name)
                if os.path.exists(fp):
                    y = np.load(fp).astype(float)
                    if arr == "Yv" and len(y) == len(Lv):
                        Yv = y
                    elif arr == "Yt" and len(y) == len(Lt):
                        Yt = y
                    break
        if len(Lv) != len(Yv) or len(Lt) != len(Yt):
            raise SystemExit(f"saved logits ({len(Lv)}/{len(Lt)}) do not match the "
                             f"label counts ({len(Yv)}/{len(Yt)})")
    else:
        if args.reuse_logits:
            p("  NOTE: --reuse-logits was given but no saved logits were found; "
              "running inference instead")
        p(f"  checkpoint : {args.ckpt}")
        if not os.path.exists(args.ckpt):
            raise SystemExit(f"checkpoint not found: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = build_model(args.model)
        model.load_state_dict(state["model_state"])
        model.eval()
        mean, std = pp.load_norm_stats(os.path.join(DATA, "norm_stats.json"))

        src = ("packed memmap" if os.path.exists(os.path.join(PACK_DIR, "val_X.npy"))
               else "signals_cache")
        p(f"\n  Running inference from {src} ...")
        Xv, Yv_p = load_split("val", val, mean, std, args.filter)
        Xt, Yt_p = load_split("test", test, mean, std, args.filter)
        Yv, Yt = Yv_p, Yt_p

        def logits(X):
            out = []
            for i in range(0, len(X), args.batch):
                with torch.no_grad():
                    out.append(model(torch.from_numpy(X[i:i + args.batch]).float()).numpy())
            return np.concatenate(out)

        Lv, Lt = logits(Xv), logits(Xt)
    Pv, Pt = 1 / (1 + np.exp(-Lv)), 1 / (1 + np.exp(-Lt))

    # ── 1. CALIBRATION ───────────────────────────────────────────────────
    hdr("1. CALIBRATION (temperature scaling, fitted on fold 9)")
    before = calibration_report(Pt, Yt)
    calib = TemperatureCalibrator(CLASS_NAMES).fit(Lv, Yv)
    Pt_cal = calib.predict_proba(Lt)
    Pv_cal = calib.predict_proba(Lv)
    after = calibration_report(Pt_cal, Yt)

    p(f"  {'class':<6} {'T':>7} {'bias':>7} | {'ECE before':>11} {'ECE after':>10} "
      f"{'over-pred before':>17} {'after':>7}")
    p("  " + "-" * 76)
    for k, c in enumerate(CLASS_NAMES):
        p(f"  {c:<6} {calib.temperature[k]:>7.3f} {calib.bias[k]:>7.3f} | "
          f"{before[c]['ece']:>11.4f} {after[c]['ece']:>10.4f} "
          f"{before[c]['over_prediction']:>16.2f}x {after[c]['over_prediction']:>6.2f}x")
    p("  " + "-" * 76)
    p(f"  {'MACRO':<6} {'':>7} {'':>7} | {before['macro_ece']:>11.4f} "
      f"{after['macro_ece']:>10.4f}")
    prov = {"model": args.model, "ckpt": os.path.basename(args.ckpt),
            "filter": bool(args.filter), "seed": args.seed}
    calib.save(os.path.join(OUT_CKPT, "calibrator.json"), fitted_for=prov)
    p(f"\n  saved -> checkpoints/calibrator.json")

    # ── 2. CONFORMAL TRIAGE ──────────────────────────────────────────────
    hdr("2. CONFORMAL RISK-CONTROLLED TRIAGE (fitted on fold 9)")
    preset = PRESETS[args.preset]
    p(f"  Operating point: '{args.preset}'")
    p(f"  Guarantee type: PAC / training-conditional at delta={args.delta}"
      if args.delta else "  Guarantee type: marginal (in expectation over calibration draws)")
    p("  Per-class miss-rate budget alpha and false-alarm budget beta:")
    for c in CLASS_NAMES:
        p(f"    {c:<6} alpha={preset['alpha'][c]:.2f}  beta={preset['beta'][c]:.2f}")

    triage = ConformalTriage(CLASS_NAMES, alpha=preset["alpha"], beta=preset["beta"],
                             delta=args.delta).fit(Pv_cal, Yv.astype(int))
    triage.save(os.path.join(OUT_CKPT, "conformal_triage.json"), fitted_for=prov)

    p(f"\n  {'class':<6} {'lambda_out':>11} {'lambda_in':>10} {'n_pos':>6} {'n_neg':>6} "
      f"{'feasible':>9}")
    p("  " + "-" * 58)
    for c in CLASS_NAMES:
        t = triage.thresholds[c]
        p(f"  {c:<6} {t.lambda_out:>11.4f} {t.lambda_in:>10.4f} {t.n_pos_cal:>6} "
          f"{t.n_neg_cal:>6} {str(t.guarantee_feasible):>9}")
        if t.note:
            p(f"         note: {t.note}")

    # ── 3. VERIFY GUARANTEES ON TEST ─────────────────────────────────────
    hdr("3. DO THE GUARANTEES HOLD ON THE UNSEEN TEST FOLD?")
    ev = triage.evaluate(Pt_cal, Yt.astype(int))
    p(f"  {'class':<6} {'alpha':>6} {'miss':>7} {'held':>6} | {'beta':>6} {'FA':>7} "
      f"{'held':>6} | {'ruleout%':>9} {'refer%':>7} {'rulein%':>8}")
    p("  " + "-" * 82)
    for c in CLASS_NAMES:
        e = ev[c]
        p(f"  {c:<6} {e['alpha']:>6.2f} {e['empirical_miss_rate']:>7.3f} "
          f"{str(e['guarantee_held']):>6} | {e['beta']:>6.2f} "
          f"{e['empirical_false_alarm']:>7.3f} {str(e['fa_guarantee_held']):>6} | "
          f"{e['rule_out_frac']*100:>8.1f}% {e['refer_frac']*100:>6.1f}% "
          f"{e['rule_in_frac']*100:>7.1f}%")
    p()
    p("  CLINICAL ESCAPE RATE — true positives that were RULED OUT (i.e. never")
    p("  reached a human at all). A referred case is not a clinical miss.")
    p(f"    {'class':<6} {'escape':>8} {'vs baseline FN rate':>21}")
    baseline_fn = {"NORM": 0.0736, "MI": 0.2948, "STTC": 0.1425, "CD": 0.2195, "HYP": 0.3939}
    for c in CLASS_NAMES:
        esc = ev[c].get("clinical_escape_rate", float("nan"))
        p(f"    {c:<6} {esc*100:>7.1f}% {baseline_fn[c]*100:>20.1f}%")
    p("  (baseline = the archive's F1-tuned single threshold, which had no")
    p("   referral option: every false negative escaped)")

    o = ev["_overall"]
    p()
    p(f"  Patients handled autonomously (no class deferred): "
      f"{o['autonomous_rate']*100:.1f}%")
    p(f"  Patients referred to a cardiologist              : "
      f"{o['patient_referral_rate']*100:.1f}%")
    if "autonomous_exact_match" in o:
        p(f"  Exact-match accuracy on the autonomous subset    : "
          f"{o['autonomous_exact_match']*100:.1f}%  (n={o['autonomous_n']})")
        p(f"  (baseline exact-match over ALL patients was 62.1%)")
        p(f"  Miss rate WITHIN the autonomous subset:")
        for c in CLASS_NAMES:
            if f"autonomous_miss_{c}" in o:
                p(f"    {c:<6} {o[f'autonomous_miss_{c}']*100:>6.1f}%  "
                  f"(n={o[f'autonomous_npos_{c}']} positives)")

    # ── 4. RISK-COVERAGE CURVE ───────────────────────────────────────────
    hdr("4. RISK-COVERAGE TRADE-OFF (the headline figure for the thesis)")
    curves = {}
    for k, c in enumerate(CLASS_NAMES):
        rows = risk_coverage_curve(Pv_cal, Yv.astype(int), Pt_cal, Yt.astype(int), k, c)
        curves[c] = rows
        p(f"\n  {c} — how much can we safely rule out at each miss-rate budget?")
        p(f"    {'alpha':>7} {'feasible':>9} {'lambda_out':>11} {'ruled out':>10} "
          f"{'observed miss':>14}")
        for r in rows:
            p(f"    {r['alpha']:>7.2f} {str(r['feasible']):>9} {r['lambda_out']:>11.4f} "
              f"{r['rule_out_frac']*100:>9.1f}% {r['empirical_miss']*100:>13.1f}%")

    # ── save ─────────────────────────────────────────────────────────────
    with open(os.path.join(OUT_RES, "07_conformal_eval.json"), "w") as f:
        json.dump({"calibration_before": before, "calibration_after": after,
                   "temperature": calib.temperature.tolist(),
                   "bias": calib.bias.tolist(),
                   "triage_eval": ev, "risk_coverage": curves}, f, indent=2, default=str)
    with open(os.path.join(OUT_RES, "07_conformal_eval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if not use_saved:                      # don't overwrite what we just read
        np.save(os.path.join(OUT_RES, "test_logits.npy"), Lt)
        np.save(os.path.join(OUT_RES, "val_logits.npy"), Lv)
        np.save(os.path.join(OUT_RES, "val_labels.npy"), Yv)
        np.save(os.path.join(OUT_RES, "test_labels.npy"), Yt)
    p(f"\nSaved -> {OUT_RES}")


if __name__ == "__main__":
    main()
