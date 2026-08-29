"""
preflight.py — Run this BEFORE training. It costs seconds and can save an hour
of compute units.

Every check below corresponds to a way a Colab run silently wastes CU:
    wrong paths          -> crash at minute 4, after you paid for the GPU
    missing packed data  -> falls back to slow per-file reads, 6x the time
    too-large batch      -> OOM at epoch 1
    no GPU attached      -> trains on CPU for 8 hours
    tiny disk            -> crash at the checkpoint write
    corrupt memmap       -> trains on garbage and you never find out

Usage:  python Component_02/train/preflight.py
        python Component_02/train/preflight.py --batch 128
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

PACK_DIR = os.path.join(COMP, "data")

CKPT_DIR = os.path.join(COMP, "checkpoints")

ok, warn, fail = [], [], []
def good(m): ok.append(m); print(f"  [ OK ] {m}")
def soft(m): warn.append(m); print(f"  [WARN] {m}")
def bad(m): fail.append(m); print(f"  [FAIL] {m}")
def hdr(t): print(f"\n{'='*72}\n  {t}\n{'='*72}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--model", default="resnet_se")
    args = ap.parse_args()

    # ── 1. Python / packages ─────────────────────────────────────────────
    hdr("1. ENVIRONMENT")
    print(f"  python {sys.version.split()[0]}")
    try:
        import torch
        print(f"  torch  {torch.__version__}")
    except ImportError:
        bad("torch is not installed"); return summarise()
    for mod in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            __import__(mod)
        except ImportError:
            bad(f"{mod} is not installed")
    if not fail:
        good("all required packages import")

    # ── 2. GPU ───────────────────────────────────────────────────────────
    hdr("2. GPU")
    if not torch.cuda.is_available():
        bad("no CUDA device. Runtime > Change runtime type > GPU. "
            "Training on CPU would take ~8 hours instead of ~40 minutes.")
    else:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        good(f"{name}, {total:.1f} GB total, {free:.1f} GB free")
        if free < 6:
            bad(f"only {free:.1f} GB free — something else is holding the GPU. "
                f"Runtime > Restart session.")
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
        if not torch.cuda.is_bf16_supported():
            soft("bf16 unavailable; will fall back to fp16 + GradScaler (fine, "
                 "marginally slower)")

    # ── 3. Data ──────────────────────────────────────────────────────────
    hdr("3. DATA")
    packed_ok = True
    expected = {"train": 13801, "val": 1709, "test": 1711}
    for split, n_exp in expected.items():
        xp = os.path.join(PACK_DIR, f"{split}_X.npy")
        yp = os.path.join(PACK_DIR, f"{split}_Y.npy")
        if not os.path.exists(xp):
            packed_ok = False
            soft(f"{split}_X.npy missing — will be built by --pack (~4 min)")
            continue
        try:
            X = np.load(xp, mmap_mode="r")
            Y = np.load(yp)
        except Exception as e:
            bad(f"{split} pack is unreadable: {e}"); packed_ok = False; continue
        if X.shape[1:] != (12, 5000):
            bad(f"{split}_X has shape {X.shape}, expected (N, 12, 5000)")
        elif len(X) != len(Y):
            bad(f"{split}: {len(X)} signals but {len(Y)} labels")
        elif len(X) != n_exp:
            soft(f"{split}: {len(X)} records (expected {n_exp}) — fine if you "
                 f"rebuilt the label set")
        else:
            # sample a few records: are they real, finite, non-constant?
            idx = np.linspace(0, len(X) - 1, 5).astype(int)
            sub = np.asarray(X[idx], dtype=np.float32)
            if not np.isfinite(sub).all():
                bad(f"{split}: sampled records contain NaN/Inf")
            elif float(sub.std()) < 1e-6:
                bad(f"{split}: sampled records are constant — the pack is corrupt")
            else:
                good(f"{split}: {len(X):,} records, "
                     f"{os.path.getsize(xp)/1e9:.2f} GB, std={sub.std():.3f}, "
                     f"positives/class={Y.sum(0).astype(int).tolist()}")

    if not packed_ok:
        # the packer needs the raw cache
        from src import paths, signals as _sig
        if _sig.available():
            good(f"signal source present: {_sig.source_description()} (packer can run)")
        else:
            bad("neither packed data nor a signal source found. "
                "Add data/raw_signals/ or data/signals_cache/ — see README.")
        for f in ("train.csv", "val.csv", "test.csv", "norm_stats.json"):
            if paths.find(f) is None:
                bad(f"{f} is missing (looked in csv/, data/, ../_archive/data/)")

    # ── 4. Disk ──────────────────────────────────────────────────────────
    hdr("4. DISK")
    free_gb = shutil.disk_usage(COMP).free / 1e9
    need = 0.5 if packed_ok else 2.6
    if free_gb < need:
        bad(f"{free_gb:.1f} GB free, need ~{need:.1f} GB")
    else:
        good(f"{free_gb:.1f} GB free (need ~{need:.1f} GB)")

    # ── 5. Model + a real forward/backward at the chosen batch ───────────
    hdr("5. MODEL & BATCH SIZE (real forward + backward)")
    if fail:
        print("  skipped — fix the failures above first")
        return summarise()

    from src.models import build_model, FocalLoss             # noqa: E402
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    good(f"{args.model}: {n_par:,} parameters")

    amp_dtype = (torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    crit = FocalLoss(gamma=2.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    bs = args.batch
    while bs >= 8:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 12, 5000, device=dev)
            y = torch.randint(0, 2, (bs, 5), device=dev).float()
            t0 = time.time()
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=amp_dtype,
                                    enabled=dev.type == "cuda"):
                    loss = crit(model(x), y)
                loss.backward()
                opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            peak = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else 0
            good(f"batch={bs} runs: {dt*1000:.0f} ms/step, peak VRAM {peak:.2f} GB")
            if bs != args.batch:
                soft(f"requested batch {args.batch} did not fit; use --batch {bs}")
            # ── time / CU estimate ──
            steps = int(np.ceil(13801 / bs))
            mins = steps * dt / 60
            print(f"\n  Estimated {mins:.1f} min/epoch, "
                  f"{mins*40:.0f} min for 40 epochs (+ ~2 min validation overhead)")
            if mins * 40 > 120:
                soft(f"40 epochs would take {mins*40:.0f} min. Reduce --epochs "
                     f"or check that the GPU is actually being used.")
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs //= 2
            soft(f"OOM — retrying at batch {bs}")
        except Exception as e:
            bad(f"forward/backward failed: {type(e).__name__}: {e}")
            break
    else:
        bad("could not run even batch=8 — the GPU is too small or already busy")

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 6. Write permissions ─────────────────────────────────────────────
    hdr("6. OUTPUT PATHS")
    for d in (CKPT_DIR, PACK_DIR):
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".preflight_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            good(f"writable: {d}")
        except Exception as e:
            bad(f"cannot write to {d}: {e}")

    return summarise()


def summarise():
    hdr("PREFLIGHT SUMMARY")
    print(f"  {len(ok)} passed, {len(warn)} warnings, {len(fail)} failures")
    for m in warn:
        print(f"    WARN  {m}")
    for m in fail:
        print(f"    FAIL  {m}")
    if fail:
        print("\n  DO NOT START TRAINING. Fix the failures above first —\n"
              "  every one of them burns compute units for nothing.")
        return 1
    print("\n  Clear to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
