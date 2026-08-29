"""UEF-Net training loop."""
from __future__ import annotations
import math, time, json, dataclasses
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import CFG
from core.common import (seed_everything, get_device, save_checkpoint,
                         load_checkpoint, CSVLogger, count_params,
                         capture_rng_state, restore_rng_state)
from core.plots import plot_training_curves
from data.dataset import EchoClipDataset
from data.sampler import build_sampler
from models.uef_net import UEFNet
from models.ema import ModelEMA
from losses.losses import UEFLoss
from engine.evaluate import run_inference
from engine.calibrate import calibrate
from engine.metrics import regression_metrics, classification_metrics


def _stratified_tune_calibration_split(labels, fraction: float, seed: int):
    """Deterministically reserve part of VAL for one-time calibration."""
    labels = np.asarray(labels, dtype=np.int64)
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    tune, calibration = [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise ValueError(f"class {class_id} has too few VAL cases for tune/calibration split")
        rng = np.random.default_rng(int(seed) + 104729 * (int(class_id) + 1))
        indices = indices.copy()
        rng.shuffle(indices)
        n_cal = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
        calibration.extend(indices[:n_cal].tolist())
        tune.extend(indices[n_cal:].tolist())
    return sorted(tune), sorted(calibration)


class Trainer:
    def __init__(self, cfg=CFG, smoke: bool = False, resume: bool = False):
        self.cfg = cfg
        self.smoke = smoke
        self.resume = resume
        cfg.ensure_dirs()
        seed_everything(cfg.seed)
        self.device = get_device(cfg.device)

        self.train_ds = EchoClipDataset("TRAIN", cfg, train=True)
        self.val_ds = EchoClipDataset("VAL", cfg, train=False, n_views=1)
        self.ef_mean, self.ef_std = self.train_ds.ef_mean, self.train_ds.ef_std

        if getattr(cfg, "model_version", "uefnet_v1") == "uefnet_v1":
            # Historical runs used all of VAL for both purposes.
            self.val_tune_indices = list(range(len(self.val_ds)))
            self.val_calibration_indices = list(range(len(self.val_ds)))
        else:
            self.val_tune_indices, self.val_calibration_indices = \
                _stratified_tune_calibration_split(
                    self.val_ds.ef_class,
                    float(getattr(cfg, "calibration_fraction", 0.30)),
                    int(cfg.seed),
                )
            split_path = cfg.OUT_DIR / "validation_partition.json"
            partition = {
                "schema_version": 1,
                "seed": int(cfg.seed),
                "calibration_fraction": float(getattr(cfg, "calibration_fraction", 0.30)),
                "tune_files": [self.val_ds.files[i] for i in self.val_tune_indices],
                "calibration_files": [self.val_ds.files[i] for i in self.val_calibration_indices],
                "tune_class_counts": np.bincount(
                    self.val_ds.ef_class[self.val_tune_indices], minlength=cfg.n_classes
                ).astype(int).tolist(),
                "calibration_class_counts": np.bincount(
                    self.val_ds.ef_class[self.val_calibration_indices], minlength=cfg.n_classes
                ).astype(int).tolist(),
            }
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(partition, f, indent=2)
                f.write("\n")

        self.model = UEFNet(cfg).to(self.device)
        self.ema = ModelEMA(self.model, cfg.ema_decay) if cfg.use_ema else None
        print(f"[trainer] UEF-Net params: {count_params(self.model):.1f}M | "
              f"pretrained_loaded={self.model.pretrained_loaded} | device={self.device} | "
              f"ema={'on' if self.ema else 'off'}")

        counts = np.bincount(self.train_ds.ef_class, minlength=cfg.n_classes)
        self.criterion = UEFLoss(
            cfg, self.ef_mean, self.ef_std, class_counts=counts).to(self.device)
        groups = self.model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
        self.base_lrs = [g["lr"] for g in groups]
        self.optimizer = torch.optim.AdamW(groups)
        self.scaler = GradScaler("cuda", enabled=(cfg.amp and self.device.type == "cuda"))
        self.logger = CSVLogger(cfg.LOG_CSV)

        self.val_loader = DataLoader(
            Subset(self.val_ds, self.val_tune_indices),
            batch_size=max(2, cfg.batch_size), shuffle=False,
            num_workers=cfg.num_workers, pin_memory=(self.device.type == "cuda"))
        self._train_loader = None

    def _eval_module(self):
        """The network used for validation / checkpointing (EMA if enabled)."""
        return self.ema.module if self.ema is not None else self.model

    # -------------------------------------------------------- data / schedule
    def _get_train_loader(self, epoch: int):
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(epoch)
        # (re)build only when the sampler regime changes (DRW boundary)
        if self._train_loader is None or epoch == self.cfg.drw_epoch:
            sampler = build_sampler(self.train_ds, self.cfg, epoch)
            # persistent_workers=False on purpose: workers are torn down at the end
            # of each train epoch so they do NOT coexist with the validation loader's
            # workers (that overlap = 2x processes = out-of-RAM on Windows spawn).
            self._train_loader = DataLoader(
                self.train_ds, batch_size=self.cfg.batch_size, sampler=sampler,
                num_workers=self.cfg.num_workers, pin_memory=(self.device.type == "cuda"),
                drop_last=True, persistent_workers=False)
        return self._train_loader

    def _set_lr(self, epoch: int):
        c = self.cfg
        if epoch < c.warmup_epochs:
            factor = (epoch + 1) / max(1, c.warmup_epochs)
        else:
            progress = (epoch - c.warmup_epochs) / max(1, c.epochs - c.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * factor
        return factor

    # ------------------------------------------------------------- one epoch
    def train_one_epoch(self, loader, epoch: int):
        self.model.train()
        self._set_lr(epoch)
        use_amp = self.cfg.amp and self.device.type == "cuda"
        accum = max(1, self.cfg.grad_accum)
        self.optimizer.zero_grad(set_to_none=True)
        meters = {"loss": 0.0, "reg": 0.0, "ord": 0.0, "con": 0.0,
                  "class": 0.0, "nll": 0.0, "rank": 0.0, "heads": 0.0}
        n = 0
        t0 = time.time()

        max_iters = 6 if self.smoke else len(loader)
        lr_now = self.optimizer.param_groups[0]["lr"]
        pbar = tqdm(loader, total=max_iters, desc=f"epoch {epoch:02d} [train]",
                    leave=False, dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            if it >= max_iters:
                break
            video = batch["video"].to(self.device, non_blocking=True)
            targets = {k: batch[k].to(self.device, non_blocking=True)
                       for k in ("ef", "ef_z", "ef_class", "lds_w")}
            with autocast("cuda", enabled=use_amp):
                ef_z, ord_logits, aux = self.model(video)
                loss, parts = self.criterion(
                    ef_z, ord_logits, targets, aux=aux, epoch=epoch)
                loss = loss / accum
            self.scaler.scale(loss).backward()

            if (it + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            for k in meters:
                meters[k] += parts[k]
            n += 1
            if it % 10 == 0:
                pbar.set_postfix(loss=f"{parts['loss']:.3f}", reg=f"{parts['reg']:.3f}",
                                 ord=f"{parts['ord']:.3f}", lr=f"{lr_now:.1e}")
        pbar.close()

        # Do not silently discard a final incomplete accumulation group.
        remainder = n % accum
        if remainder:
            self.scaler.unscale_(self.optimizer)
            correction = accum / remainder
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)

        for k in meters:
            meters[k] /= max(1, n)
        meters["sec"] = round(time.time() - t0, 1)
        return meters

    # ------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(self, epoch: int):
        mb = 3 if self.smoke else 0
        desc = None if self.smoke else f"epoch {epoch:02d} [val]"
        preds = run_inference(self._eval_module(), self.val_loader, self.cfg, self.device,
                              self.ef_mean, self.ef_std, max_batches=mb, desc=desc)
        reg = regression_metrics(preds["ef_true"], preds["ef_pred"])
        if getattr(self.cfg, "model_version", "uefnet_v1") != "uefnet_v1":
            # Model selection must not refit thresholds/expansion every epoch.
            # Use one fixed, uncalibrated decision rule; calibrate once after freeze.
            y_pred = preds.get("class_pred", preds["ord_pred"])
            cls = classification_metrics(preds["y_true"], y_pred, self.cfg.n_classes)
            strategy = "class_argmax" if "class_pred" in preds else "ordinal_rank"
            return reg, {"best_strategy": strategy, "best": cls,
                         "strategies": {strategy: cls}}
        cal = calibrate(preds["ef_pred"], preds["y_true"], preds["ord_pred"], self.cfg,
                        ef_true=preds["ef_true"])
        # the reported MAE reflects the winning strategy (bias-corrected if it won)
        reg["mae"] = cal["best"].get("mae", reg["mae"])
        return reg, cal

    # ---------------------------------------------- final TTA calibration
    @torch.no_grad()
    def finalize_calibration(self):
        """
        Recalibrate decision thresholds on VAL using the SAME multi-clip TTA
        protocol used at test time, so the frozen thresholds are not biased by
        the cheaper single-view validation used during training.
        """
        from core.common import load_checkpoint
        c = self.cfg
        if not c.CKPT_BEST.exists():
            return
        # best.pt holds the EMA weights; load them into self.model for inference
        load_checkpoint(c.CKPT_BEST, self.model, map_location=self.device)
        val_tta = EchoClipDataset("VAL", c, train=False, n_views=c.n_tta_clips)
        # The frozen decision rule (thresholds for the tiny Severe class especially)
        # generalizes far better when fit on ALL of VAL rather than the 30%
        # calibration sub-split (~30 vs ~100 Severe cases).  Model selection is
        # already complete and TEST is still untouched, so this stays honest.
        if getattr(c, "calibrate_on_full_val", False):
            cal_indices = list(range(len(val_tta)))
            partition = "full_val"
        else:
            cal_indices = self.val_calibration_indices
            partition = "calibration"
        loader = DataLoader(Subset(val_tta, cal_indices),
                            batch_size=max(2, c.batch_size // 2), shuffle=False,
                            num_workers=c.num_workers, pin_memory=(self.device.type == "cuda"))
        preds = run_inference(self.model, loader, c, self.device, self.ef_mean, self.ef_std,
                              desc="final TTA calibration")
        cal = calibrate(
            preds["ef_pred"], preds["y_true"], preds["ord_pred"], c,
            ef_true=preds["ef_true"], ord_dist=preds.get("ord_dist"),
            class_dist=preds.get("class_dist"), pred_std=preds.get("ef_pred_std"))
        best = cal["best"]
        mae = best.get("mae", regression_metrics(preds["ef_true"], preds["ef_pred"])["mae"])
        payload = dict(schema_version=2, strategy=cal["best_strategy"],
                       clinical_primary=cal["clinical_primary"],
                       thresholds=best.get("thresholds"),
                       ef_calibration=cal.get("ef_calibration"),
                       val_min_recall=best["min_class_recall"], val_mae=mae,
                       protocol=f"label_free_TTA_{c.n_tta_clips}clip",
                       validation_partition=partition,
                       ef_mean=self.ef_mean, ef_std=self.ef_std,
                       calibration=cal)
        with open(c.THRESH_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"[trainer] FINAL TTA-calibrated ({cal['best_strategy']}): "
              f"thr={best.get('thresholds')} affine={cal.get('ef_calibration')} | "
              f"val min-recall {best['min_class_recall']:.3f} | val MAE {mae:.3f} | per-class "
              f"{';'.join(f'{r:.2f}' for r in best['per_class_recall'])}")
        return cal

    # -------------------------------------------------------- config snapshot
    def _dump_config(self):
        cfg_d = {}
        for f in dataclasses.fields(self.cfg):
            v = getattr(self.cfg, f.name)
            cfg_d[f.name] = str(v) if hasattr(v, "__fspath__") else v
        cfg_d["pretrained_loaded"] = self.model.pretrained_loaded
        cfg_d["params_M"] = round(count_params(self.model), 2)
        json.dump(cfg_d, open(self.cfg.OUT_DIR / "config.json", "w"), indent=2, default=str)

    # ---------------------------------------------------------- resume state
    def _try_resume(self):
        """Load last.pt and restore full training state. Returns start_epoch."""
        c = self.cfg
        if not (self.resume and c.CKPT_LAST.exists()):
            return 0, (-1.0, -1e9), 0
        ckpt = load_checkpoint(c.CKPT_LAST, self.model, self.optimizer, self.scaler,
                               map_location=self.device)
        extra = ckpt.get("extra", {}) or {}
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        bs = extra.get("best_score", [-1.0, -1e9])
        best_score = (float(bs[0]), float(bs[1]))
        patience = int(extra.get("patience", 0))
        if self.ema is not None and extra.get("ema") is not None:
            self.ema.load_state_dict(extra["ema"])
        restore_rng_state(extra.get("rng", {}))
        print(f"[trainer] RESUMED from {c.CKPT_LAST.name}: start_epoch={start_epoch} "
              f"best_min_recall={best_score[0]:.3f} patience={patience}")
        return start_epoch, best_score, patience

    # ------------------------------------------------------------------ fit
    def fit(self):
        c = self.cfg
        config_path = self.cfg.OUT_DIR / "config.json"
        if not (self.resume and config_path.exists()):
            self._dump_config()
        start_epoch, best_score, patience = self._try_resume()
        epochs = 1 if self.smoke else c.epochs
        best_epoch = start_epoch - 1
        interrupted = False

        try:
            for epoch in range(start_epoch, epochs):
                loader = self._get_train_loader(epoch)
                tr = self.train_one_epoch(loader, epoch)
                reg, cal = self.validate(epoch)
                best = cal["best"]
                mn = best["min_class_recall"]; ba = best["balanced_acc"]; mae = reg["mae"]

                row = dict(epoch=epoch, lr=round(self.optimizer.param_groups[0]["lr"], 8),
                           train_loss=round(tr["loss"], 4), train_reg=round(tr["reg"], 4),
                           train_ord=round(tr["ord"], 4), train_con=round(tr["con"], 4),
                           train_class=round(tr["class"], 4),
                           train_nll=round(tr["nll"], 4), train_rank=round(tr["rank"], 4),
                           val_mae=round(mae, 3), val_min_recall=round(mn, 4),
                           val_balanced_acc=round(ba, 4), val_strategy=cal["best_strategy"],
                           per_class=";".join(f"{r:.2f}" for r in best["per_class_recall"]),
                           sec=tr["sec"])
                self.logger.log(row)
                print(f"[epoch {epoch}] val MAE {mae:.3f} | min-recall {mn:.3f} | "
                      f"bal-acc {ba:.3f} | strat {cal['best_strategy']} | "
                      f"per-class {row['per_class']} | {tr['sec']}s")

                score = (mn, -mae)
                is_best = score > best_score or (self.smoke and epoch == start_epoch)
                if is_best:
                    best_score = score
                    best_epoch = epoch
                    patience = 0
                else:
                    patience += 1

                # last.pt carries full resume state (best_score, patience, RNG, EMA)
                save_checkpoint(c.CKPT_LAST, self.model, self.optimizer, self.scaler,
                                epoch=epoch, best_metric=best_score[0],
                                extra=dict(best_score=list(best_score), patience=patience,
                                           rng=capture_rng_state(),
                                           ema=(self.ema.state_dict() if self.ema else None)))
                if is_best:
                    # checkpoint the EVAL weights (EMA if enabled) as best.pt
                    save_checkpoint(c.CKPT_BEST, self._eval_module(),
                                    epoch=epoch, best_metric=mn,
                                    extra=dict(calibration=cal, regression=reg))
                    json.dump(dict(strategy=cal["best_strategy"], thresholds=best.get("thresholds"),
                                   expansion=best.get("expansion"),
                                   val_min_recall=mn, val_mae=mae, epoch=epoch,
                                   ef_mean=self.ef_mean, ef_std=self.ef_std),
                              open(c.THRESH_JSON, "w"), indent=2)
                    print(f"  * new best (min-recall {mn:.3f}, MAE {mae:.3f}) -> {c.CKPT_BEST.name}")

                if not self.smoke:
                    plot_training_curves(c.LOG_CSV, c.OUT_DIR / "training_curves.png")

                if patience >= c.early_stop_patience and not self.smoke:
                    print(f"[trainer] early stop at epoch {epoch} "
                          f"(no min-recall gain for {patience} epochs)")
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("\n[trainer] interrupted — last.pt is saved; resume with --resume.")

        print(f"[trainer] done. best epoch {best_epoch} | best val min-recall "
              f"(single-view) = {best_score[0]:.3f} | MAE = {-best_score[1]:.3f}")
        if not self.smoke and not interrupted:
            print("[trainer] running final TTA calibration on VAL ...")
            self.finalize_calibration()
        print(f"[trainer] artifacts in {c.OUT_DIR}")
        return best_score
