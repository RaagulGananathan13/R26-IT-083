"""
================================================================================
 UEF-Net :: Training Configuration
================================================================================
Uncertainty-aware Ordinal Ejection-Fraction Network for imbalanced EF grading.

Consumes the preprocessing artifacts (sibling `preprocessing/` folder):
  * manifest.csv    (labels, ED/ES frames, imbalance weights, cache paths)
  * norm_stats.json (ef_mean/std, pixel_mean/std)
  * cache/videos/*.npy (decoded uint8 clips)

Targets:
  * regression : minimise MAE on EF
  * 4-class    : >=75% accuracy (recall) on EVERY severity class

Boundaries (MUST match preprocessing/config.py):
    Severe <30 | Moderate 30-40 | Mild 40-55 | Normal >=55
================================================================================
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import json
import math
import os
import re
from typing import Any, Mapping


@dataclass
class TrainConfig:
    # ------------------------------------------------------------- identity
    run_name: str = "uefnet_r2p1d"
    seed: int = 1337

    # ----------------------------------------------------------------- paths
    TRAIN_DIR: Path = Path(__file__).resolve().parent
    ROOT: Path = Path(__file__).resolve().parent.parent
    # preprocessing artifacts
    PREP_DIR: Path = field(init=False)
    MANIFEST: Path = field(init=False)
    NORM_JSON: Path = field(init=False)
    CACHE_DIR: Path = field(init=False)

    # ------------------------------------------------------------ label maps
    EF_THRESHOLDS: tuple = (30.0, 40.0, 55.0)
    CLASS_NAMES: tuple = ("Severe(<30)", "Moderate(30-40)", "Mild(40-55)", "Normal(>=55)")

    # --------------------------------------------------------- clip / input
    clip_len: int = 32              # frames per clip (8GB-friendly with r2p1d)
    sampling_period: int = 2        # stride -> spans 64 native frames (~1.4 cycles)
    frame_size: int = 112
    in_channels: int = 2            # grayscale + motion(tempdiff)
    motion_mode: str = "tempdiff"   # "none" | "tempdiff" | "flow"

    # --------------------------------------------------------- augmentation
    aug_pad: int = 12               # pad then random-crop back to frame_size
    aug_intensity_jitter: float = 0.1
    aug_time_jitter: bool = True    # random cycle-aware window (train)
    cycle_aware_probability: float = 0.5  # otherwise sample label-free full-video

    # -------------------------------------------------------------- model
    backbone: str = "r2plus1d_18"
    pretrained: bool = True         # Kinetics-400 (downloads once); graceful fallback
    dropout: float = 0.3
    feat_dim: int = 512

    # ------------------------------------------------------ model version
    # "uefnet_v1" = original (CoralHead only). "uefnet_v2" = ordered-cutpoint
    # ordinal head + auxiliary softmax class head + heteroscedastic uncertainty
    # + pairwise rank loss + DRW loss re-weighting + honest tune/calibration split.
    model_version: str = "uefnet_v1"
    use_class_head: bool = True         # v2 auxiliary softmax classification head
    predict_uncertainty: bool = True    # v2 heteroscedastic (log-var) regression head
    aux_channel_init_scale: float = 0.05

    # --------------------------------------------------- loss / uncertainty
    ef_noise_sigma: float = 4.0     # clinical EF measurement noise -> soft ordinal labels
    w_reg: float = 1.0              # regression loss weight
    w_ord: float = 1.0              # soft-CORAL ordinal loss weight
    w_consistency: float = 0.15     # dual-head consistency weight
    huber_delta: float = 1.0        # on normalised EF
    lds_kernel_sigma: float = 2.0   # label distribution smoothing (imbalanced regression)
    lds_ks: int = 5
    # v2 auxiliary loss weights (ignored when model_version == "uefnet_v1")
    w_class: float = 0.5            # softmax class-head cross-entropy (soft targets)
    w_nll: float = 0.5             # heteroscedastic gaussian NLL on EF
    w_rank: float = 0.2            # smooth pairwise EF ordering loss
    w_head_consistency: float = 0.1  # class-head vs ordinal-head agreement
    class_target_sigma: float = 4.0
    # Logit adjustment (Menon et al. 2021): shift the softmax class-head logits by
    # tau*log(class_prior) during training so rare/middle classes get a larger
    # margin.  0.0 disables (default); 0.5-1.5 typical.  Applied to the class head
    # only; at inference plain argmax is used (the shift is baked into training).
    logit_adjustment_tau: float = 0.0
    lds_weight_power: float = 0.5   # temper LDS weights (v2 keeps effective LR stable)
    effective_num_beta: float = 0.9999
    rank_min_gap: float = 3.0
    rank_temperature: float = 4.0

    # ------------------------------------------------ extra co-training data
    # Additional TRAIN manifests merged into the TRAIN split only (e.g. CAMUS,
    # rich in low/mid-EF cases).  VAL/TEST are never touched, so evaluation stays
    # on the untouched EchoNet split.  Empty tuple => behaviour unchanged.
    extra_manifests: tuple = ()

    # ----------------------------------------------------- imbalance / DRW
    use_balanced_sampler: bool = True
    # Post-hoc calibration robustness for the tiny extreme-tail (Severe) class:
    #  * calibrate the frozen decision rule on ALL of VAL (~3x more Severe cases)
    #    instead of the 30% calibration sub-split, so the Severe/Moderate boundary
    #    generalizes to TEST instead of overfitting a ~30-case subset;
    #  * widen the Severe threshold search (radius 8) so the optimizer can raise
    #    the EF=30 boundary enough to rebalance Severe<->Moderate recall;
    #  * allow a variance-expansion EF calibration that decompresses the tail
    #    (counteracts regression-to-the-mean that pushes true-Severe into Moderate).
    calibrate_on_full_val: bool = True
    threshold_search_radius: tuple = (8.0, 8.0, 6.0)
    # Deferred Re-Weighting (Cao et al. 2019): train on the NATURAL distribution
    # first so the backbone learns good features, then switch BOTH the sampler
    # and the loss re-weighting to class-balanced.  drw_epoch=15/45 is the recipe
    # that gave the best MAE (4.29) at equal min-recall in the v1 experiments;
    # balancing from epoch 0 instead hurt MAE (5.44) for no recall gain.
    drw_epoch: int = 15            # epoch to switch to class-balanced sampler + loss weights
    calibration_fraction: float = 0.30  # v2: fraction of VAL reserved for post-hoc calibration

    # ------------------------------------------------------------ optimise
    epochs: int = 45
    batch_size: int = 8
    grad_accum: int = 4             # effective batch 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 2.0
    amp: bool = True                # mixed precision (essential on 8GB)

    # ----------------------------------------------------- weight averaging
    use_ema: bool = True            # evaluate/checkpoint EMA weights (generalisation)
    ema_decay: float = 0.999        # ~ averages last 1/(1-decay) optimiser steps

    # -------------------------------------------------------------- eval
    # These defaults match the values used for every reported result, so the
    # documented command reproduces the published numbers without extra flags.
    n_tta_clips: int = 10           # multi-clip test-time augmentation
    tta_forward_batch: int = 8      # cap flattened views per forward pass (OOM-safe)
    eval_use_keyframes: bool = False  # False avoids privileged tracing annotations
    early_stop_patience: int = 18   # on val min-per-class recall

    # ------------------------------------------------------------ runtime
    # Windows spawn workers each re-import torch/torchvision (~1-2GB RSS), so keep
    # this modest; training is GPU-bound and 4 CPU workers already saturate it.
    num_workers: int = min(4, max(1, (os.cpu_count() or 4) - 2))
    device: str = "cuda"

    # Exact statistics used by this run.  Populated when training starts and
    # embedded in config.json, so evaluation never silently adopts statistics
    # from a later preprocessing run.
    norm_stats: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.PREP_DIR = self.ROOT / "preprocessing"
        self.MANIFEST = self.PREP_DIR / "artifacts" / "manifest.csv"
        self.NORM_JSON = self.PREP_DIR / "artifacts" / "norm_stats.json"
        self.CACHE_DIR = self.PREP_DIR / "cache" / "videos"

    # ---------------------------------------------------------- derived out
    @property
    def OUT_DIR(self) -> Path:      return self.TRAIN_DIR / "outputs" / self.run_name
    @property
    def CKPT_BEST(self) -> Path:    return self.OUT_DIR / "best.pt"
    @property
    def CKPT_LAST(self) -> Path:    return self.OUT_DIR / "last.pt"
    @property
    def LOG_CSV(self) -> Path:      return self.OUT_DIR / "train_log.csv"
    @property
    def THRESH_JSON(self) -> Path:  return self.OUT_DIR / "thresholds.json"
    @property
    def REPORT_JSON(self) -> Path:  return self.OUT_DIR / "test_report.json"
    @property
    def FROZEN_NORM_JSON(self) -> Path: return self.OUT_DIR / "norm_stats.json"

    @property
    def n_classes(self) -> int:     return len(self.CLASS_NAMES)

    def ensure_dirs(self):
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

    def report_path(self, split: str) -> Path:
        """Return a split-specific report path without changing legacy TEST naming."""
        split = str(split).strip().upper()
        if split not in {"TRAIN", "VAL", "TEST"}:
            raise ValueError(f"unsupported report split: {split!r}")
        return self.OUT_DIR / f"{split.lower()}_report.json"

    def validate(self, *, for_training: bool = True) -> None:
        """Fail early on invalid CLI/config combinations.

        These checks intentionally live beside the configuration so entry
        points, tests, and future scripts all enforce the same invariants.
        """
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(self.run_name)) \
                or self.run_name in {".", ".."}:
            raise ValueError("run_name must contain only letters, numbers, '.', '_' or '-' "
                             "and may not be a path")
        positive_ints = {
            "clip_len": self.clip_len,
            "sampling_period": self.sampling_period,
            "frame_size": self.frame_size,
            "in_channels": self.in_channels,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "n_tta_clips": self.n_tta_clips,
            "tta_forward_batch": self.tta_forward_batch,
        }
        if for_training:
            positive_ints.update(epochs=self.epochs,
                                 early_stop_patience=self.early_stop_patience)
        bad = [k for k, v in positive_ints.items()
               if isinstance(v, bool) or not isinstance(v, int) or v <= 0]
        if bad:
            raise ValueError("expected positive integer(s): " + ", ".join(bad))
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) \
                or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if self.motion_mode not in {"none", "tempdiff", "flow"}:
            raise ValueError("motion_mode must be one of: none, tempdiff, flow")
        if self.in_channels != (1 if self.motion_mode == "none" else 2):
            raise ValueError("in_channels must be 1 for motion_mode='none' and 2 otherwise")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if len(self.EF_THRESHOLDS) != len(self.CLASS_NAMES) - 1:
            raise ValueError("EF_THRESHOLDS must have exactly n_classes-1 entries")
        thresholds = tuple(float(v) for v in self.EF_THRESHOLDS)
        if (not all(math.isfinite(v) for v in thresholds)
                or any(b <= a for a, b in zip(thresholds, thresholds[1:]))):
            raise ValueError("EF_THRESHOLDS must be finite and strictly increasing")
        for name in ("lr_backbone", "lr_head", "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"{name} must be finite and {'non-negative' if name == 'weight_decay' else 'positive'}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.aug_pad < 0 or self.aug_intensity_jitter < 0:
            raise ValueError("augmentation magnitudes must be non-negative")
        if not 0.0 <= float(self.cycle_aware_probability) <= 1.0:
            raise ValueError("cycle_aware_probability must be in [0, 1]")

    def freeze_norm_stats(self, *, overwrite: bool = False) -> Path:
        """Copy preprocessing statistics into the run directory and use them.

        A run-local copy makes a trained checkpoint self-consistent even if
        preprocessing artifacts are regenerated later.
        """
        self.ensure_dirs()
        dest = self.FROZEN_NORM_JSON
        source = self.PREP_DIR / "artifacts" / "norm_stats.json"
        if dest.exists() and not overwrite:
            chosen = dest
            with open(chosen, "r", encoding="utf-8") as f:
                stats = json.load(f)
        elif self.norm_stats and not overwrite:
            # Resume from an embedded snapshot even if its companion file was
            # moved or omitted.
            chosen = "embedded saved config"
            stats = dict(self.norm_stats)
        else:
            chosen = source
            if not source.exists():
                raise FileNotFoundError(f"normalization statistics not found: {source}")
            with open(source, "r", encoding="utf-8") as f:
                stats = json.load(f)
        self._validate_norm_stats(stats, chosen)
        if chosen != dest:
            tmp = dest.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, dest)
        self.NORM_JSON = dest
        self.norm_stats = dict(stats)
        return dest

    def restore_for_evaluation(self, snapshot: Mapping[str, Any]) -> None:
        """Restore model/input semantics from a saved run configuration.

        Machine-specific paths and the requested run identity are deliberately
        not restored.  Unknown fields are ignored for forward compatibility.
        """
        excluded = {
            "run_name", "TRAIN_DIR", "ROOT", "PREP_DIR", "MANIFEST",
            "NORM_JSON", "CACHE_DIR", "num_workers", "device", "pretrained",
        }
        fields = self.__dataclass_fields__
        # Newer model implementations may add a version switch.  Historical
        # snapshots predate that field and must instantiate the original net.
        if "model_version" in fields and "model_version" not in snapshot:
            setattr(self, "model_version", "uefnet_v1")
        for key, value in snapshot.items():
            if key in excluded or key not in fields or not fields[key].init:
                continue
            current = getattr(self, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(self, key, value)
        # Evaluation always loads learned weights and must never download an
        # unrelated pretrained checkpoint.
        self.pretrained = False
        if self.norm_stats:
            self._validate_norm_stats(self.norm_stats, "saved config")

    @staticmethod
    def _validate_norm_stats(stats: Mapping[str, Any], source: Any) -> None:
        required = ("pixel_mean", "pixel_std", "ef_mean", "ef_std")
        missing = [k for k in required if k not in stats]
        if missing:
            raise ValueError(f"normalization statistics in {source} are missing: {missing}")
        values = {k: float(stats[k]) for k in required}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"normalization statistics in {source} must be finite")
        if values["pixel_std"] <= 0 or values["ef_std"] <= 0:
            raise ValueError(f"normalization std values in {source} must be positive")

    def ef_to_class(self, ef: float) -> int:
        for i, t in enumerate(self.EF_THRESHOLDS):
            if ef < t:
                return i
        return len(self.EF_THRESHOLDS)


CFG = TrainConfig()
