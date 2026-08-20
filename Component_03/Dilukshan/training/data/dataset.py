"""
EchoNet clip dataset.

Reads the preprocessing manifest + decoded uint8 cache, samples a cardiac-cycle
-aware clip, builds the (grayscale + motion) multichannel tensor, and applies
train-time augmentation.  The clip-sampling and motion code is LOADED DIRECTLY
from preprocessing/utils/sampling.py so there is zero train/preprocess skew.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import inspect
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CFG
from losses.lds import compute_lds_weights


# ---- reuse the EXACT preprocessing sampling/motion code (no duplication) ----
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prep = _load_module("prep_sampling", CFG.PREP_DIR / "utils" / "sampling.py")
sample_indices = _prep.sample_indices
build_multichannel = _prep.build_multichannel
_SAMPLER_SUPPORTS_VIEWS = {"view_index", "n_views"}.issubset(
    inspect.signature(sample_indices).parameters)


def load_norm_stats(cfg=CFG):
    """Load the exact run statistics, preferring an embedded snapshot."""
    s = dict(getattr(cfg, "norm_stats", {}) or {})
    if not s:
        path = Path(cfg.NORM_JSON)
        if not path.exists():
            raise FileNotFoundError(
                f"normalization statistics not found: {path}. Run preprocessing first.")
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
    required = ("pixel_mean", "pixel_std", "ef_mean", "ef_std")
    missing = [k for k in required if k not in s]
    if missing:
        raise ValueError(f"normalization statistics are missing fields: {missing}")
    out = tuple(float(s[k]) for k in required)
    if not np.all(np.isfinite(out)) or out[1] <= 0 or out[3] <= 0:
        raise ValueError("normalization statistics must be finite with positive std values")
    return out


def load_cached_video(path: Path) -> np.ndarray:
    """Load and validate an uncompressed .npy or compressed .npz cache."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cached video not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "video" in archive.files:
                video = archive["video"]
            elif len(archive.files) == 1:  # tolerate older single-array caches
                video = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"compressed cache {path} has no 'video' array; keys={archive.files}")
            video = np.asarray(video)
    elif path.suffix.lower() == ".npy":
        video = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"unsupported cache extension {path.suffix!r}: {path}")
    if video.ndim != 3 or video.shape[0] < 1:
        raise ValueError(f"cached video must have shape (T,H,W) with T>0, got {video.shape}: {path}")
    return video


def _legacy_multiview_indices(n_frames: int, clip_len: int, period: int,
                              ed_frame: int, es_frame: int,
                              view_index: int, n_views: int) -> np.ndarray:
    """Deterministic diverse eval sampling for older preprocessing modules."""
    span = clip_len * period
    if n_frames <= span:
        # Edge-safe monotonic sampling (no artificial end-to-start jump).
        return np.rint(np.linspace(0, n_frames - 1, clip_len)).astype(np.int64)
    lo, hi = 0, n_frames - span
    start_lo, start_hi = lo, hi
    if ed_frame >= 0 and es_frame >= 0:
        a, b = sorted((int(ed_frame), int(es_frame)))
        cycle_lo, cycle_hi = max(lo, b - span + 1), min(hi, a)
        if cycle_lo <= cycle_hi:
            start_lo, start_hi = cycle_lo, cycle_hi
        else:
            start_lo = start_hi = int(np.clip((a + b - span) // 2, lo, hi))
    if n_views <= 1:
        start = (start_lo + start_hi) // 2
    else:
        starts = np.rint(np.linspace(start_lo, start_hi, n_views)).astype(np.int64)
        start = int(starts[min(max(0, view_index), n_views - 1)])
    return start + np.arange(clip_len, dtype=np.int64) * period


def _augment(clip: np.ndarray, cfg, rng) -> np.ndarray:
    """Spatial pad+random-crop and mild intensity jitter on a (T,H,W) uint8 clip."""
    T, H, W = clip.shape
    p = cfg.aug_pad
    if p > 0:
        clip = np.pad(clip, ((0, 0), (p, p), (p, p)), mode="reflect")
        top = int(rng.integers(0, 2 * p + 1))
        left = int(rng.integers(0, 2 * p + 1))
        clip = clip[:, top:top + H, left:left + W]
    j = cfg.aug_intensity_jitter
    if j > 0:
        scale = 1.0 + rng.uniform(-j, j)
        shift = rng.uniform(-j, j) * 255.0
        clip = np.clip(clip.astype(np.float32) * scale + shift, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(clip)


def _read_split_manifest(path, split: str) -> "pd.DataFrame":
    """Read one manifest, normalise keys, keep only cached rows for `split`."""
    man = pd.read_csv(path)
    required_columns = {"FileName", "Split", "EF", "ef_class"}
    missing_columns = sorted(required_columns.difference(man.columns))
    if missing_columns:
        raise ValueError(f"manifest {path} is missing required columns: {missing_columns}")
    man["FileName"] = man["FileName"].astype(str)
    man["Split"] = man["Split"].astype(str).str.upper()
    if "cached_ok" in man.columns:
        cached = man["cached_ok"]
        if cached.dtype == bool:
            mask = cached.fillna(False)
        else:
            mask = cached.fillna("").astype(str).str.strip().str.lower().isin(
                {"1", "true", "t", "yes", "y"})
        man = man[mask]
    return man[man["Split"] == str(split).upper()].reset_index(drop=True)


class EchoClipDataset(Dataset):
    def __init__(self, split: str, cfg=CFG, train: bool = None,
                 n_views: int = 1, augment: bool = None, sample_random: bool = None):
        self.cfg = cfg
        self.split = split.upper()
        self.train = (self.split == "TRAIN") if train is None else train
        if isinstance(n_views, bool) or int(n_views) <= 0:
            raise ValueError("n_views must be a positive integer")
        self.n_views = int(n_views)
        self.augment = self.train if augment is None else augment
        self.sample_random = ((self.train and bool(cfg.aug_time_jitter))
                              if sample_random is None else bool(sample_random))

        man = _read_split_manifest(cfg.MANIFEST, self.split)
        if len(man) == 0:
            raise RuntimeError(f"no cached rows for split={self.split}. Run preprocessing/stage4.")

        # Opt-in extra TRAIN data (e.g. CAMUS co-training).  Merged into the
        # TRAIN split ONLY; VAL/TEST remain exactly the EchoNet split so the
        # evaluation set is never contaminated.  Default extra_manifests == ()
        # keeps this path byte-identical to an EchoNet-only run.
        extra_manifests = tuple(getattr(cfg, "extra_manifests", ()) or ())
        self.n_extra = 0
        if self.split == "TRAIN" and extra_manifests:
            frames = [man]
            for rel in extra_manifests:
                p = Path(rel)
                if not p.is_absolute():
                    p = Path(cfg.PREP_DIR) / p
                if not p.exists():
                    raise FileNotFoundError(f"extra manifest not found: {p}")
                extra = _read_split_manifest(p, "TRAIN")
                if len(extra) == 0:
                    raise RuntimeError(f"extra manifest has no cached TRAIN rows: {p}")
                self.n_extra += len(extra)
                frames.append(extra)
            man = pd.concat(frames, ignore_index=True, sort=False)
            print(f"[dataset] TRAIN merged {len(frames)-1} extra manifest(s): "
                  f"{len(man)-self.n_extra} EchoNet + {self.n_extra} extra = {len(man)} clips")

        # validate EF / ef_class over the FULL (possibly merged) split
        if not np.isfinite(pd.to_numeric(man["EF"], errors="coerce")).all():
            raise ValueError(f"manifest contains invalid EF values in split={self.split}")
        classes = pd.to_numeric(man["ef_class"], errors="coerce").to_numpy()
        if (not np.isfinite(classes).all() or (classes < 0).any()
                or (classes >= cfg.n_classes).any()
                or not np.equal(classes, np.floor(classes)).all()):
            raise ValueError(f"manifest contains invalid ef_class values in split={self.split}")
        self.df = man.reset_index(drop=True)
        man = self.df

        self.pmean, self.pstd, self.ef_mean, self.ef_std = load_norm_stats(cfg)

        # per-sample fields
        self.files = man["FileName"].tolist()
        self.ef = man["EF"].values.astype(np.float32)
        self.ef_class = man["ef_class"].values.astype(np.int64)
        self.ed = man.get("ed_frame", pd.Series([-1] * len(man))).values.astype(np.int64)
        self.es = man.get("es_frame", pd.Series([-1] * len(man))).values.astype(np.int64)
        self.sample_weight = man.get("sample_weight", pd.Series([1.0] * len(man))).values.astype(np.float32)
        self.nframes = man.get("n_frames_cached", pd.Series([0] * len(man))).values.astype(np.int64)

        # When extra manifests are merged, the per-source precomputed sampling
        # weights are on different scales, so recompute one consistent inverse-
        # frequency (class-balanced) weight over the whole merged TRAIN set.
        if self.train and getattr(self, "n_extra", 0) > 0:
            counts = np.bincount(self.ef_class, minlength=cfg.n_classes).astype(np.float64)
            inv = 1.0 / np.clip(counts, 1.0, None)
            w = inv[self.ef_class]
            self.sample_weight = (w / w.mean()).astype(np.float32)

        # LDS weights (train only; else 1.0)
        if self.train:
            self.lds_w = compute_lds_weights(self.ef, ks=cfg.lds_ks, sigma=cfg.lds_kernel_sigma)
        else:
            self.lds_w = np.ones(len(man), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, i) -> Path:
        base = Path(self.cfg.CACHE_DIR) / self.files[i]
        for suffix in (".npy", ".npz"):
            candidate = base.with_suffix(suffix)
            if candidate.exists():
                return candidate

        # Fall back to the manifest.  New manifests store paths relative to
        # PREP_DIR for portability; historical absolute paths remain valid.
        raw = self.df.iloc[i].get("cache_path", "")
        cp = "" if pd.isna(raw) else str(raw).strip()
        if cp:
            stored = Path(cp)
            candidate = stored if stored.is_absolute() else Path(self.cfg.PREP_DIR) / stored
            if candidate.exists():
                return candidate
            # Relocated legacy manifests may retain an obsolete absolute root.
            for suffix in (stored.suffix, ".npy", ".npz"):
                if suffix:
                    relocated = base.with_suffix(suffix)
                    if relocated.exists():
                        return relocated
            raise FileNotFoundError(
                f"cache_path for {self.files[i]!r} does not exist: {candidate}")
        raise FileNotFoundError(
            f"no .npy/.npz cache found for {self.files[i]!r} under {self.cfg.CACHE_DIR}")

    def _one_view(self, i, vid, nf, rng, view_index: int = 0) -> np.ndarray:
        cfg = self.cfg
        # ED/ES frames come from ground-truth contour annotations and are not
        # available for a new clinical video.  They are valid supervision for
        # train-time sampling, but evaluation is label-free by default.
        use_keyframes = bool(getattr(cfg, "eval_use_keyframes", False))
        if self.train:
            # Mix annotation-guided clips with label-free full-video clips.  A
            # model that only ever sees traced transitions acquires a train/
            # deployment sampling mismatch because tracings are unavailable at
            # inference time.
            probability = float(getattr(cfg, "cycle_aware_probability", 1.0))
            use_keyframes = bool(rng.random() < probability)
        ed_frame = int(self.ed[i]) if use_keyframes else None
        es_frame = int(self.es[i]) if use_keyframes else None
        kwargs = dict(ed_frame=ed_frame, es_frame=es_frame,
                      train=self.sample_random, rng=rng)
        if _SAMPLER_SUPPORTS_VIEWS:
            kwargs.update(view_index=view_index, n_views=self.n_views)
            idxs = sample_indices(nf, cfg.clip_len, cfg.sampling_period, **kwargs)
        elif not self.sample_random and self.n_views > 1:
            idxs = _legacy_multiview_indices(
                nf, cfg.clip_len, cfg.sampling_period,
                (-1 if ed_frame is None else ed_frame), (-1 if es_frame is None else es_frame),
                view_index, self.n_views)
        else:
            idxs = sample_indices(nf, cfg.clip_len, cfg.sampling_period, **kwargs)
        idxs = np.clip(idxs, 0, nf - 1)
        clip = np.asarray(vid[idxs], dtype=np.uint8)          # (clip_len,112,112)
        if self.augment:
            clip = _augment(clip, cfg, rng)
        result = build_multichannel(clip, self.pmean, self.pstd, cfg.motion_mode)
        if result.shape[0] != int(cfg.in_channels):
            raise ValueError(
                f"motion_mode={cfg.motion_mode!r} produced {result.shape[0]} channel(s), "
                f"but config.in_channels={cfg.in_channels}")
        return result                                              # (C,T,H,W)

    def __getitem__(self, i):
        cfg = self.cfg
        vid = load_cached_video(self._cache_path(i))          # (T,H,W) uint8
        nf = int(vid.shape[0])

        views = []
        for v in range(self.n_views):
            if self.train:
                # NumPy's process-local RNG is deterministically seeded by the
                # DataLoader worker (and captured for num_workers=0 resumes).
                # Drawing an explicit seed avoids default_rng() OS entropy.
                # dtype=int64 is required: np.random.randint defaults to C long,
                # which is 32-bit on Windows, so a uint32-max bound would overflow.
                view_seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.int64))
            else:
                view_seed = int(cfg.seed) + int(i) * 1_000_003 + v
            rng = np.random.default_rng(view_seed)
            views.append(self._one_view(i, vid, nf, rng, view_index=v))

        if self.n_views == 1:
            x = torch.from_numpy(np.ascontiguousarray(views[0])).float()   # (C,T,H,W)
        else:
            x = torch.from_numpy(np.ascontiguousarray(np.stack(views, 0))).float()  # (V,C,T,H,W)

        ef = float(self.ef[i])
        return {
            "video": x,
            "ef": torch.tensor(ef, dtype=torch.float32),
            "ef_z": torch.tensor((ef - self.ef_mean) / (self.ef_std + 1e-6), dtype=torch.float32),
            "ef_class": torch.tensor(int(self.ef_class[i]), dtype=torch.long),
            "lds_w": torch.tensor(float(self.lds_w[i]), dtype=torch.float32),
            "sample_w": torch.tensor(float(self.sample_weight[i]), dtype=torch.float32),
        }
