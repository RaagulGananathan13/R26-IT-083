"""
CAMUS -> EchoNet-compatible converter.
=====================================

Turns the official CAMUS dataset (NIfTI format, 500 patients, 2CH + 4CH apical
sequences) into the SAME on-disk format the training pipeline already consumes
for EchoNet-Dynamic:

  * per-clip uint8 (T, 112, 112) arrays in  preprocessing/cache/camus_videos/
  * a manifest  preprocessing/artifacts/camus_manifest.csv  with EXACTLY the
    columns EchoClipDataset reads (FileName, Split, EF, ef_class, ed_frame, ...)

WHY: CAMUS mean EF ~44 (vs EchoNet ~56) and ~50% of it is EF<45, so it is rich
in the Severe/Moderate/Mild cases that bound our per-class recall.  We use it as
TRAIN-only co-training data; evaluation stays on the untouched EchoNet test set.

The CAMUS "half_sequence" spans ED (first frame) -> ES (last frame), i.e. the
systolic contraction that carries the EF signal.

Run (from the preprocessing/ folder):

    python build_camus.py                       # both views, all 500 patients
    python build_camus.py --views 4CH           # 4CH only (matches EchoNet A4C)
    python build_camus.py --exclude-poor        # drop ImageQuality==Poor
    python build_camus.py --limit 5             # quick smoke test
    python build_camus.py --zip PATH            # explicit CAMUS zip location

CITATION (mandatory for any use of CAMUS):
  S. Leclerc et al., "Deep Learning for Segmentation using an Open Large-Scale
  Dataset in 2D Echocardiography", IEEE TMI 38(9):2198-2210, 2019.
"""
from __future__ import annotations
import argparse
import csv
import gc
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ZIP = HERE.parent / "Dataset" / "archive" / "download"
CACHE_DIR = HERE / "cache" / "camus_videos"
MANIFEST_OUT = HERE / "artifacts" / "camus_manifest.csv"

# MUST match preprocessing/training class boundaries (Severe/Moderate/Mild/Normal).
EF_THRESHOLDS = (30.0, 40.0, 55.0)
CLASS_NAMES = ("Severe(<30)", "Moderate(30-40)", "Mild(40-55)", "Normal(>=55)")
FRAME_SIZE = 112

# The manifest schema EchoClipDataset consumes, in order.  Extra provenance
# columns (source/view/image_quality) are appended and simply ignored by it.
COLUMNS = [
    "FileName", "Split", "usable", "EF", "ESV", "EDV", "ef_class", "class_name",
    "FPS", "NumberOfFrames", "ed_frame", "es_frame", "geometry_version",
    "class_weight", "sample_weight", "ef_density_weight",
    "cache_path", "n_frames_cached", "cached_ok",
    "source", "view", "image_quality",
]


def ef_to_class(ef: float) -> int:
    for i, t in enumerate(EF_THRESHOLDS):
        if ef < t:
            return i
    return len(EF_THRESHOLDS)


def parse_cfg(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize a single (H,W) float frame to FRAME_SIZE x FRAME_SIZE uint8."""
    import cv2
    f = np.clip(frame, 0.0, 255.0)
    # INTER_AREA is the correct choice for the large downscale (~550 -> 112).
    r = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return np.rint(r).astype(np.uint8)


def _echonet_pixel_target():
    """EchoNet global raw-pixel (0-255) mean/std, so CAMUS can be matched to it."""
    import json
    ns_path = HERE / "artifacts" / "norm_stats.json"
    if not ns_path.exists():
        return None
    with open(ns_path, "r", encoding="utf-8") as f:
        ns = json.load(f)
    return float(ns["pixel_mean"]) * 255.0, float(ns["pixel_std"]) * 255.0


def _camus_global_stats(paths):
    """Streaming global mean/std over all converted CAMUS clips."""
    s = s2 = n = 0.0
    for p in paths:
        a = np.load(p).astype(np.float64)
        s += a.sum(); s2 += float((a * a).sum()); n += a.size
    mean = s / n
    return mean, max((s2 / n - mean * mean) ** 0.5, 1e-6)


def harmonize_to_echonet(force: bool = False) -> bool:
    """Affine-match CAMUS global intensity to EchoNet (cross-scanner harmonization).

    CAMUS was acquired on different hardware (GE Vivid E95) than EchoNet, so its
    raw pixels are brighter/higher-contrast.  Because the balanced sampler
    over-samples the minority-heavy CAMUS cases, an un-normalized brightness
    offset is a shortcut the network could exploit for the minority classes -
    one that would NOT transfer to the EchoNet test set.  We remap every CAMUS
    pixel  p -> (p - cam_mean)/cam_std * echo_std + echo_mean  so both datasets
    share one global intensity distribution.  Idempotent: if CAMUS already
    matches EchoNet within tolerance the pass is skipped.
    """
    target = _echonet_pixel_target()
    if target is None:
        print("[camus][harmonize] no EchoNet norm_stats.json found; skipping")
        return False
    echo_mean, echo_std = target
    paths = sorted(p for p in CACHE_DIR.glob("*.npy"))
    if not paths:
        return False
    # A marker makes this strictly idempotent: clipping at the black background
    # means the remapped stats never EXACTLY equal the target, so a stats-based
    # skip check would re-apply (and corrupt) the clips on every re-run.
    marker = CACHE_DIR / "_harmonized.json"
    if marker.exists() and not force:
        print(f"[camus][harmonize] already applied (marker present); skipping. "
              f"Use --harmonize-only to force.")
        return False
    cam_mean, cam_std = _camus_global_stats(paths)
    scale = echo_std / cam_std
    print(f"[camus][harmonize] remapping {len(paths)} clips: CAMUS(mean={cam_mean:.1f},std={cam_std:.1f})"
          f" -> EchoNet(mean={echo_mean:.1f},std={echo_std:.1f})  scale={scale:.3f}")
    for p in paths:
        a = np.load(p).astype(np.float32)
        r = (a - cam_mean) * scale + echo_mean
        np.save(p, np.clip(np.rint(r), 0, 255).astype(np.uint8))
    new_mean, new_std = _camus_global_stats(paths)
    import json
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"source_mean": cam_mean, "source_std": cam_std,
                   "target_mean": echo_mean, "target_std": echo_std,
                   "result_mean": new_mean, "result_std": new_std}, f, indent=2)
    print(f"[camus][harmonize] done. CAMUS now mean={new_mean:.1f} std={new_std:.1f} "
          f"(marker -> {marker.name})")
    return True


def load_sequence(nii_bytes: bytes) -> np.ndarray:
    """Decode a CAMUS *_half_sequence.nii.gz into a (T,112,112) uint8 clip."""
    import nibabel as nib
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as fh:
        fh.write(nii_bytes)
        tmp = fh.name
    try:
        img = nib.load(tmp)
        arr = np.asarray(img.dataobj)          # (H, W, T) float32, range ~[0,255]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D (H,W,T) sequence, got shape {arr.shape}")
    arr = np.moveaxis(arr, 2, 0)               # -> (T, H, W)
    clip = np.stack([_resize_frame(arr[t]) for t in range(arr.shape[0])], axis=0)
    del arr
    return clip                                 # (T, 112, 112) uint8


def iter_patients(zf: zipfile.ZipFile, views):
    """Yield (patient_id, view, cfg_dict, sequence_zip_name) for available data."""
    cfg_names = sorted(n for n in zf.namelist() if n.endswith(".cfg"))
    for cfg_name in cfg_names:
        # database_nifti/patientXXXX/Info_4CH.cfg
        parts = cfg_name.split("/")
        pid = parts[-2]
        view = parts[-1].replace("Info_", "").replace(".cfg", "")  # 2CH / 4CH
        if view not in views:
            continue
        base = "/".join(parts[:-1])
        seq_name = f"{base}/{pid}_{view}_half_sequence.nii.gz"
        if seq_name not in zf.namelist():
            continue
        cfg = parse_cfg(zf.read(cfg_name).decode())
        yield pid, view, cfg, seq_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(DEFAULT_ZIP),
                    help="path to the CAMUS zip (default: Dataset/archive/download)")
    ap.add_argument("--views", nargs="+", default=["4CH", "2CH"],
                    choices=["4CH", "2CH"], help="which apical views to convert")
    ap.add_argument("--exclude-poor", action="store_true",
                    help="drop ImageQuality==Poor cases (default: keep all)")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N clips (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-convert even if the .npy exists")
    ap.add_argument("--no-harmonize", action="store_true",
                    help="skip matching CAMUS intensity to EchoNet (not recommended)")
    ap.add_argument("--harmonize-only", action="store_true",
                    help="only run intensity harmonization on already-converted clips")
    a = ap.parse_args()

    if a.harmonize_only:
        harmonize_to_echonet(force=True)
        return

    zip_path = Path(a.zip)
    if not zip_path.exists():
        sys.exit(f"[camus] zip not found: {zip_path}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    kept = skipped_poor = skipped_bad = 0
    with zipfile.ZipFile(zip_path) as zf:
        for pid, view, cfg, seq_name in iter_patients(zf, set(a.views)):
            if a.limit and kept >= a.limit:
                break
            quality = cfg.get("ImageQuality", "Unknown")
            if a.exclude_poor and quality.lower() == "poor":
                skipped_poor += 1
                continue
            try:
                ef = float(cfg["EF"])
                fps = float(cfg.get("FrameRate", 0) or 0)
            except (KeyError, ValueError):
                skipped_bad += 1
                continue
            if not (0.0 < ef <= 100.0):
                skipped_bad += 1
                continue

            file_id = f"camus_{pid}_{view}"
            npy_path = CACHE_DIR / f"{file_id}.npy"
            if npy_path.exists() and not a.force:
                clip_len = int(np.load(npy_path, mmap_mode="r").shape[0])
            else:
                try:
                    clip = load_sequence(zf.read(seq_name))
                except Exception as e:                       # keep going on any bad case
                    print(f"[camus][WARN] {file_id}: {e}")
                    skipped_bad += 1
                    continue
                if clip.shape[0] < 2:
                    skipped_bad += 1
                    continue
                np.save(npy_path, clip)
                clip_len = int(clip.shape[0])
                del clip
                gc.collect()

            cls = ef_to_class(ef)
            rows.append({
                "FileName": file_id, "Split": "TRAIN", "usable": True,
                "EF": ef, "ESV": "", "EDV": "", "ef_class": cls,
                "class_name": CLASS_NAMES[cls], "FPS": fps,
                "NumberOfFrames": clip_len, "ed_frame": 0, "es_frame": clip_len - 1,
                "geometry_version": f"camus_nifti_{view}_halfseq_v1",
                "class_weight": 1.0, "sample_weight": 1.0, "ef_density_weight": 1.0,
                "cache_path": str(npy_path), "n_frames_cached": clip_len,
                "cached_ok": True, "source": "camus", "view": view,
                "image_quality": quality,
            })
            kept += 1
            if kept % 100 == 0:
                print(f"[camus] converted {kept} clips ...")

    if not rows:
        sys.exit("[camus] no clips converted — check the zip contents/paths")

    # Cross-scanner intensity harmonization to EchoNet (unless disabled).
    if not a.no_harmonize:
        harmonize_to_echonet(force=a.force)

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # summary
    import collections
    dist = collections.Counter(r["ef_class"] for r in rows)
    print("\n================ CAMUS CONVERSION DONE ================")
    print(f"  clips written      : {kept}  (poor-skipped {skipped_poor}, bad {skipped_bad})")
    print(f"  views              : {', '.join(a.views)}")
    print(f"  cache dir          : {CACHE_DIR}")
    print(f"  manifest           : {MANIFEST_OUT}")
    print("  class distribution :")
    for c in range(len(CLASS_NAMES)):
        print(f"    {CLASS_NAMES[c]:16} {dist[c]:4d}  ({100*dist[c]/kept:.1f}%)")
    print("=======================================================")
    print("Next: train with  --extra-manifest artifacts/camus_manifest.csv")


if __name__ == "__main__":
    main()
