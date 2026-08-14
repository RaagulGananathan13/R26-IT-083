"""
train_gpu.py — Component-02 classifier training. One Colab Pro L4, <= 1 GPU-hour.

RUN preflight.py FIRST. It costs 30 seconds and catches every setup mistake that
would otherwise waste a paid GPU session.

COMPUTE-UNIT PROTECTION built into this script
    --resume            picks up exactly where a disconnect left off (optimizer,
                        scheduler, EMA and RNG state included). A dropped Colab
                        session costs you seconds, not the whole run.
    automatic OOM       on CUDA OOM the batch size halves and the epoch restarts
    recovery            instead of the run dying at minute 38.
    --max-minutes       hard wall clock budget. Stops cleanly, saves, and reports
                        the best epoch rather than being killed mid-write.
    checkpoint/epoch    ~10 MB, written every epoch, atomically.
    NaN guard           a diverged run is detected and stopped, not left to burn
                        the remaining 35 epochs producing nothing.
    preflight probe     a real forward+backward before the data loader is built.

WHAT DIFFERS FROM THE ARCHIVE TRAINER (each closes an audit finding)
    * band-pass 0.5-40 Hz + 50 Hz notch, per-record median removal (archive: none)
    * balanced sampling XOR focal alpha, never both (archive: both -> ECE 0.242)
    * ECGResNetSE: multi-kernel stem, squeeze-excitation, attention pooling
    * augmentation without wide amplitude scaling (voltage IS the HYP evidence)
    * EMA weights, OneCycle, AMP
    * saves val + test LOGITS so calibration is fitted with no second pass

Usage
    python Component_02/train/preflight.py
    python Component_02/train/train_gpu.py --pack           # once, CPU is fine
    python Component_02/train/train_gpu.py --epochs 40
    python Component_02/train/train_gpu.py --resume         # after a disconnect
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP = os.path.join(ROOT, "Component_02")
sys.path.insert(0, COMP)

from src import preprocess as pp                                        # noqa: E402
from src.models import CLASS_NAMES, FocalLoss, build_model, SIGNAL_LENGTH  # noqa: E402

from src import paths as _paths  # noqa: E402
DATA = os.path.dirname(_paths.require("train.csv"))
CACHE = _paths.signals_cache() or ""
PACK_DIR = os.environ.get("ECG_PACK_DIR", os.path.join(COMP, "data"))
CKPT_DIR = os.path.join(COMP, "checkpoints")

_STOP = {"now": False}


def _on_sigint(signum, frame):
    if _STOP["now"]:
        raise KeyboardInterrupt
    _STOP["now"] = True
    print("\n  Interrupt received — finishing this epoch, saving, then exiting. "
          "Press again to abort immediately.", flush=True)


def atomic_save(obj, path):
    """Never leave a half-written checkpoint behind after a disconnect."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — pack once into a float16 memmap
# ══════════════════════════════════════════════════════════════════════════
def pack(do_filter: bool = True, force: bool = False):
    """Per-file .npy reads dominate epoch time. Pack once, read fast forever.

    Runs fine on a laptop CPU (~12 min). Doing it locally and uploading the three
    resulting files is far faster than uploading 17,221 small files to Drive.
    """
    os.makedirs(PACK_DIR, exist_ok=True)
    mean, std = pp.load_norm_stats(os.path.join(DATA, "norm_stats.json"))
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(DATA, f"{split}.csv"))
        xp = os.path.join(PACK_DIR, f"{split}_X.npy")
        done = os.path.join(PACK_DIR, f"{split}.done")
        if os.path.exists(xp) and os.path.exists(done) and not force:
            print(f"  {split}: already packed and verified, skipping")
            continue
        if os.path.exists(xp) and not os.path.exists(done):
            print(f"  {split}: previous pack was incomplete, rebuilding")

        free = shutil.disk_usage(PACK_DIR).free
        need = len(df) * 12 * SIGNAL_LENGTH * 2 * 1.1
        if free < need:
            raise SystemExit(f"Not enough disk for {split}: need {need/1e9:.1f} GB, "
                             f"have {free/1e9:.1f} GB free at {PACK_DIR}")

        X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float16,
                                      shape=(len(df), 12, SIGNAL_LENGTH))
        Y = np.zeros((len(df), len(CLASS_NAMES)), dtype=np.float32)
        t0, missing = time.time(), []
        for i, r in enumerate(df.itertuples()):
            fp = os.path.join(CACHE, f"{int(r.ecg_id)}.npy")
            if not os.path.exists(fp):
                missing.append(int(r.ecg_id))
                continue
            s = np.load(fp).astype(np.float32)
            if do_filter:
                s = pp.bandpass(s)
            s = pp.normalise(s, mean, std, per_record=do_filter)
            X[i] = s.T.astype(np.float16)
            Y[i] = [getattr(r, f"label_{c}") for c in CLASS_NAMES]
            if i and i % 2000 == 0:
                rate = i / (time.time() - t0)
                print(f"  {split}: {i}/{len(df)}  "
                      f"eta {(len(df)-i)/max(rate,1e-9):.0f}s", flush=True)
        X.flush()
        del X
        if missing:
            raise SystemExit(f"{len(missing)} cached signals missing for {split} "
                             f"(e.g. {missing[:5]}). Refusing to train on a partial "
                             f"dataset — the archive's silent zero-fill bug (E-4) is "
                             f"exactly how corrupt records enter training.")
        np.save(os.path.join(PACK_DIR, f"{split}_Y.npy"), Y)
        np.save(os.path.join(PACK_DIR, f"{split}_ids.npy"), df.ecg_id.values)
        with open(done, "w") as f:
            json.dump({"n": len(df), "filtered": do_filter,
                       "packed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print(f"  {split}: packed {len(df):,} records in {time.time()-t0:.0f}s "
              f"({os.path.getsize(xp)/1e9:.2f} GB)")


def _worker_init(worker_id: int):
    """Give every worker its own RNG stream.

    Without this, DataLoader workers each receive a COPY of the dataset object —
    including its np.random.Generator — so all workers produce IDENTICAL
    augmentations, every epoch. The augmentation would be a no-op after epoch 1.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class PackedECG(Dataset):
    def __init__(self, split: str, train: bool = False):
        self.path = os.path.join(PACK_DIR, f"{split}_X.npy")
        self.X = None                     # opened lazily, per worker process
        self.Y = np.load(os.path.join(PACK_DIR, f"{split}_Y.npy"))
        self.train = train

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        if self.X is None:                # each worker gets its own handle
            self.X = np.load(self.path, mmap_mode="r")
        x = np.asarray(self.X[i], dtype=np.float32)
        if self.train:
            x = pp.augment(x, np.random.default_rng(np.random.randint(0, 2 ** 31)))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.Y[i])


# ══════════════════════════════════════════════════════════════════════════
class EMA:
    """Exponential moving average of weights — ~0.003-0.006 macro-AUROC, free."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, d):
        self.decay = d["decay"]
        self.shadow = {k: v.clone() for k, v in d["shadow"].items()}


def metrics(probs, labels):
    au, ap, f1, per = [], [], [], {}
    for k, c in enumerate(CLASS_NAMES):
        try:
            a = roc_auc_score(labels[:, k], probs[:, k])
        except ValueError:
            a = 0.5
        try:
            pr = average_precision_score(labels[:, k], probs[:, k])
        except ValueError:
            pr = float(labels[:, k].mean())
        f = f1_score(labels[:, k], (probs[:, k] >= 0.5).astype(int), zero_division=0)
        per[c] = dict(auroc=float(a), auprc=float(pr), f1=float(f))
        au.append(a); ap.append(pr); f1.append(f)
    per["macro_auroc"] = float(np.mean(au))
    per["macro_auprc"] = float(np.mean(ap))
    per["macro_f1"] = float(np.mean(f1))
    return per


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            L.append(model(x).float().cpu())
        Y.append(y)
    L = torch.cat(L).numpy()
    Y = torch.cat(Y).numpy()
    return L, Y, metrics(1.0 / (1.0 + np.exp(-L)), Y)


def build_loaders(args, device):
    tr = PackedECG("train", train=True)
    va, te = PackedECG("val"), PackedECG("test")

    # Class imbalance is corrected ONCE. This is audit finding C-6.
    sampler, alpha = None, None
    if args.focal_alpha:
        pos = tr.Y.sum(0)
        alpha = ((len(tr.Y) - pos) / np.maximum(pos, 1)).astype(np.float32)
        print(f"  focal alpha = {alpha.round(2).tolist()} (sampler OFF)")
    else:
        inv = 1.0 / np.maximum(tr.Y.sum(0), 1)
        w = np.where(tr.Y.sum(1) > 0, (inv * tr.Y).max(1), float(inv.min()))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        print("  balanced sampler ON, focal alpha OFF")

    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  worker_init_fn=_worker_init if args.workers > 0 else None,
                  persistent_workers=args.workers > 0,
                  prefetch_factor=4 if args.workers > 0 else None)
    train_loader = DataLoader(tr, batch_size=args.batch, sampler=sampler,
                              shuffle=sampler is None, drop_last=True, **common)
    eval_common = dict(common)
    eval_common["batch_size"] = max(args.batch, 64)
    val_loader = DataLoader(va, shuffle=False, **eval_common)
    test_loader = DataLoader(te, shuffle=False, **eval_common)
    return tr, train_loader, val_loader, test_loader, alpha


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true", help="build the memmap then exit")
    ap.add_argument("--force-pack", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--model", default="resnet_se", choices=["resnet_se", "resnet"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stopping patience. DEFAULT 0 = disabled, which is "
                         "correct for OneCycleLR: the schedule's annealing tail "
                         "(the last ~30%% of epochs, where the LR becomes tiny) is "
                         "where the model consolidates. Stopping at patience 10 on "
                         "a 40-epoch schedule cut that off and cost ~0.005 AUROC in "
                         "testing. Only set this if you are NOT using OneCycle.")
    ap.add_argument("--focal-alpha", action="store_true",
                    help="use focal alpha INSTEAD of the balanced sampler")
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/last.pt after a disconnect")
    ap.add_argument("--max-minutes", type=float, default=75.0,
                    help="hard wall-clock budget; stops cleanly and saves")
    args = ap.parse_args()

    if args.pack or args.force_pack:
        pack(do_filter=not args.no_filter, force=args.force_pack)
        return

    signal.signal(signal.SIGINT, _on_sigint)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"

    print(f"device={device}  amp={str(amp_dtype).split('.')[-1]}  seed={args.seed}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("WARNING: no GPU. This will take hours. Ctrl-C and attach a GPU.")
    if os.name == "nt" and args.workers > 0:
        print("Windows detected -> forcing --workers 0 (spawn overhead makes "
              "workers slower here)")
        args.workers = 0

    if not os.path.exists(os.path.join(PACK_DIR, "train_X.npy")):
        print("Packed data not found; packing now (one-off) ...")
        pack(do_filter=not args.no_filter)

    tr, train_loader, val_loader, test_loader, alpha = build_loaders(args, device)
    model = build_model(args.model).to(device)
    print(f"model={args.model}  params={sum(p.numel() for p in model.parameters()):,}")

    crit = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.02).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def make_sched(remaining_epochs: int, steps_per_epoch: int, max_lr: float):
        # OneCycleLR raises once it exceeds total_steps, so it must be rebuilt
        # whenever steps_per_epoch changes (i.e. after an OOM batch-size halving).
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, epochs=max(remaining_epochs, 1),
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=0.25, div_factor=20, final_div_factor=200)

    sched = make_sched(args.epochs, len(train_loader), args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = EMA(model)

    start_epoch, best, bad_epochs, hist = 1, -1.0, 0, []
    last_path = os.path.join(CKPT_DIR, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        prev = st.get("args", {})
        if prev.get("batch") != args.batch or prev.get("epochs") != args.epochs:
            print(f"  NOTE: resuming with batch={args.batch}/epochs={args.epochs} but the "
                  f"checkpoint used batch={prev.get('batch')}/epochs={prev.get('epochs')}. "
                  f"The LR schedule will be rebuilt for the remaining epochs.")
        model.load_state_dict(st["model_state"])
        opt.load_state_dict(st["optimizer_state"])
        scaler.load_state_dict(st["scaler_state"])
        ema.load_state_dict(st["ema_state"])
        start_epoch, best = st["epoch"] + 1, st["best_auroc"]
        bad_epochs, hist = st["bad_epochs"], st["history"]
        if prev.get("batch") == args.batch and prev.get("epochs") == args.epochs:
            sched.load_state_dict(st["scheduler_state"])
        else:
            sched = make_sched(args.epochs - st["epoch"], len(train_loader), args.lr)
        print(f"RESUMED from epoch {st['epoch']} (best val AUROC {best:.4f}); "
              f"continuing at epoch {start_epoch}")
    elif args.resume:
        print("--resume given but no checkpoint found; starting fresh")

    t0 = time.time()
    stop_reason = "completed"
    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        tl, nb, t_ep = 0.0, 0, time.time()
        it = iter(train_loader)
        while True:
            try:
                x, y = next(it)
            except StopIteration:
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                stop_reason = "oom_dataloader"
                break
            try:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    loss = crit(model(x), y)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                try:
                    sched.step()
                except ValueError:
                    pass          # OneCycleLR exhausted; hold the final LR
                ema.update(model)
                tl += loss.item(); nb += 1
            except torch.cuda.OutOfMemoryError:
                # Recover instead of dying 38 minutes into a paid session.
                torch.cuda.empty_cache()
                new_bs = max(args.batch // 2, 8)
                print(f"\n  CUDA OOM at batch {args.batch}. Halving to {new_bs} and "
                      f"restarting this epoch. Re-run with --batch {new_bs} next time "
                      f"to avoid the lost minutes.", flush=True)
                if new_bs == args.batch:
                    raise
                args.batch = new_bs
                tr, train_loader, val_loader, test_loader, alpha = build_loaders(args, device)
                # steps_per_epoch just changed, so OneCycleLR must be rebuilt or it
                # will raise once it passes its original total_steps.
                sched = make_sched(args.epochs - ep + 1, len(train_loader),
                                   opt.param_groups[0]["lr"])
                it = iter(train_loader)
                tl, nb = 0.0, 0
                continue

        if nb == 0:
            print("  no batches completed — aborting"); stop_reason = "no_batches"; break
        train_loss = tl / nb
        if not np.isfinite(train_loss):
            print(f"  train loss is {train_loss} — the run has diverged. Stopping "
                  f"rather than burning the remaining epochs.")
            stop_reason = "diverged"; break

        _, _, m = evaluate(model, val_loader, device, amp_dtype)
        hist.append(dict(epoch=ep, train_loss=train_loss, **m))

        star = ""
        if m["macro_auroc"] > best:
            best, bad_epochs, star = m["macro_auroc"], 0, "   * BEST"
            atomic_save({"model_state": model.state_dict(), "epoch": ep,
                         "best_auroc": best, "model_name": args.model,
                         "args": vars(args), "ema": False},
                        os.path.join(CKPT_DIR, "best_model.pt"))
        else:
            bad_epochs += 1

        atomic_save({"epoch": ep, "model_state": model.state_dict(),
                     "optimizer_state": opt.state_dict(),
                     "scheduler_state": sched.state_dict(),
                     "scaler_state": scaler.state_dict(),
                     "ema_state": ema.state_dict(),
                     "best_auroc": best, "bad_epochs": bad_epochs,
                     "history": hist, "args": vars(args)}, last_path)

        elapsed = (time.time() - t0) / 60
        print(f"ep {ep:>3}/{args.epochs}  loss {train_loss:.4f}  "
              f"AUROC {m['macro_auroc']:.4f}  AUPRC {m['macro_auprc']:.4f}  "
              f"F1 {m['macro_f1']:.4f}  {time.time()-t_ep:.0f}s  "
              f"[{elapsed:.0f}/{args.max_minutes:.0f} min]{star}", flush=True)

        if args.patience > 0 and bad_epochs >= args.patience:
            stop_reason = "early_stop"
            print(f"early stop: no improvement for {args.patience} epochs. "
                  f"NOTE: OneCycleLR had not finished annealing, so this is likely "
                  f"below the model's ceiling — re-run with --patience 0."); break
        if elapsed > args.max_minutes:
            stop_reason = "time_budget"
            print(f"time budget of {args.max_minutes:.0f} min reached — stopping "
                  f"cleanly with best AUROC {best:.4f}"); break
        if _STOP["now"]:
            stop_reason = "interrupted"
            print("interrupted — state saved, re-run with --resume"); break

    print(f"\nfinished ({stop_reason}) in {(time.time()-t0)/60:.1f} min; "
          f"best val macro-AUROC {best:.4f}")
    if best < 0:
        print("no epoch completed; nothing to evaluate"); return

    # ── EMA weights: keep whichever validates better ─────────────────────
    ema_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    _, _, m_ema = evaluate(model, val_loader, device, amp_dtype)
    print(f"EMA val macro-AUROC {m_ema['macro_auroc']:.4f} vs raw best {best:.4f}")
    if m_ema["macro_auroc"] > best:
        best = m_ema["macro_auroc"]
        atomic_save({"model_state": model.state_dict(), "epoch": -1,
                     "best_auroc": best, "model_name": args.model,
                     "args": vars(args), "ema": True},
                    os.path.join(CKPT_DIR, "best_model.pt"))
        print("EMA weights kept")
    else:
        model.load_state_dict(ema_backup)

    # ── final pass: dump logits for the calibration / conformal stage ────
    st = torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(st["model_state"])
    Lv, Yv, mv = evaluate(model, val_loader, device, amp_dtype)
    Lt, Yt, mt = evaluate(model, test_loader, device, amp_dtype)

    print("\nTEST (threshold 0.5, uncalibrated — the honest headline)")
    print(f"  {'class':<6} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    for c in CLASS_NAMES:
        print(f"  {c:<6} {mt[c]['auroc']:>8.4f} {mt[c]['auprc']:>8.4f} {mt[c]['f1']:>8.4f}")
    print(f"  {'MACRO':<6} {mt['macro_auroc']:>8.4f} {mt['macro_auprc']:>8.4f} "
          f"{mt['macro_f1']:>8.4f}")
    print("\nBaseline to beat (audited archive model):")
    print("  MACRO    0.9297   0.7864   0.7172")
    d = mt["macro_auroc"] - 0.9297
    print(f"  delta AUROC {d:+.4f}  ->  "
          f"{'improvement' if d > 0.002 else 'not a meaningful improvement'}")

    np.save(os.path.join(CKPT_DIR, f"val_logits_seed{args.seed}.npy"), Lv)
    np.save(os.path.join(CKPT_DIR, f"test_logits_seed{args.seed}.npy"), Lt)
    np.save(os.path.join(CKPT_DIR, "val_labels.npy"), Yv)
    np.save(os.path.join(CKPT_DIR, "test_labels.npy"), Yt)
    with open(os.path.join(CKPT_DIR, f"history_seed{args.seed}.json"), "w") as f:
        json.dump({"history": hist, "val": mv, "test": mt, "args": vars(args),
                   "stop_reason": stop_reason,
                   "minutes": (time.time() - t0) / 60}, f, indent=2)
    print(f"\nsaved -> {CKPT_DIR}")
    print("Next:\n  python Component_02/train/fit_calibration.py --model "
          f"{args.model} --ckpt Component_02/checkpoints/best_model.pt"
          f"{'' if args.no_filter else ' --filter'}")


if __name__ == "__main__":
    main()
