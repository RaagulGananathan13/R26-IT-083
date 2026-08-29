"""
STAGE 5 :: Exhaustive verification & visual QA
================================================
This is the preprocessing release gate.  It verifies every selected usable
manifest row (the complete dataset unless ``--limit`` is explicitly used),
checks train-only normalization drift on a deterministic pixel sample, and
tests two separate temporal protocols:

* TRAIN: ED/ES-aware sampling contains both keyframes whenever their separation
  fits the configured fixed-period span.  Longer transitions are reported as
  uncontainable rather than counted as successful.
* VAL/TEST: label-free deterministic multi-view sampling receives no ED/ES
  labels and covers the recording uniformly with diverse views where possible.

It also writes a sampled clip, motion preview, and genuine ED/ES tracing-overlay
image for each represented severity class.  Any failed image write is fatal.

Outputs
-------
artifacts/verification_report.json
artifacts/viz/*.png

Run:  python stage5_verify.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pandas as pd

from config import CFG
from utils.geometry import GEOMETRY_VERSION, tracing_polygon
from utils.io_utils import existing_clip_path, load_clip
from utils.sampling import build_multichannel, sample_indices
from utils.viz import draw_contour, montage, save_image


def _load_norm() -> tuple[float, float]:
    if CFG.NORM_JSON.exists():
        with open(CFG.NORM_JSON, encoding="utf-8") as f:
            stats = json.load(f)
        return (float(stats.get("pixel_mean", 0.129)),
                float(stats.get("pixel_std", 0.191)))
    return 0.129, 0.191


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return (series.fillna(str(default)).astype(str).str.strip().str.lower()
            .isin(truthy))


def _safe_slug(value: object) -> str:
    """Cross-platform filename component (no Windows-reserved punctuation)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "unnamed"


def _atomic_json_dump(payload: dict, path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _bad(file_name: str, reason: str, detail: object = "") -> dict:
    item = {"FileName": str(file_name), "reason": str(reason)}
    if detail != "":
        item["detail"] = str(detail)
    return item


def _validate_caches(manifest: pd.DataFrame, limit: int):
    if "usable" in manifest.columns:
        scoped = manifest[_as_bool(manifest["usable"], default=True)].copy()
    else:
        scoped = manifest.copy()
    if limit:
        scoped = scoped.head(limit).copy()

    bad = []
    valid_rows = []
    lengths = []
    if scoped["FileName"].duplicated().any():
        for name in scoped.loc[scoped["FileName"].duplicated(False), "FileName"].unique():
            bad.append(_bad(name, "duplicate_manifest_key"))

    cached_ok = (_as_bool(scoped["cached_ok"], default=False)
                 if "cached_ok" in scoped.columns
                 else pd.Series(True, index=scoped.index))

    for (idx, row), marked_ok in zip(scoped.iterrows(), cached_ok):
        name = str(row["FileName"])
        raw_path = str(row.get("cache_path", "")).strip()
        if not marked_ok:
            bad.append(_bad(name, "cached_ok_false"))
            continue
        if raw_path in {"", "nan", "NaN", "None"}:
            bad.append(_bad(name, "missing_cache_path"))
            continue
        try:
            path = existing_clip_path(raw_path, base_dir=CFG.PREP_DIR)
            if not path.is_file():
                raise FileNotFoundError(path)
            video = load_clip(path, mmap=True)
            if video.ndim != 3:
                raise ValueError(f"shape={video.shape}, expected (T,H,W)")
            if video.shape[0] <= 0 or video.shape[1:] != (CFG.FRAME_SIZE, CFG.FRAME_SIZE):
                raise ValueError(
                    f"shape={video.shape}, expected (T,{CFG.FRAME_SIZE},{CFG.FRAME_SIZE})")
            if video.dtype != np.uint8:
                raise ValueError(f"dtype={video.dtype}, expected uint8")
            recorded = row.get("n_frames_cached", np.nan)
            if pd.notna(recorded) and int(recorded) > 0 and int(recorded) != video.shape[0]:
                raise ValueError(
                    f"n_frames_cached={int(recorded)} but file has {video.shape[0]}")
            item = row.to_dict()
            item["cache_resolved"] = str(path)
            item["n_frames_verified"] = int(video.shape[0])
            valid_rows.append(item)
            lengths.append(int(video.shape[0]))
        except Exception as exc:
            bad.append(_bad(name, "cache_validation_failed", exc))

    return scoped, pd.DataFrame(valid_rows), bad, lengths


def _recompute_pixel_stats(valid: pd.DataFrame, sample_size: int, frames_per_video: int):
    if len(valid) == 0 or "Split" not in valid.columns:
        return 0.0, 0.0, 0, 0
    train = valid[valid["Split"].astype(str).str.upper() == "TRAIN"]
    if len(train) == 0:
        return 0.0, 0.0, 0, 0
    sample = train.sample(min(sample_size, len(train)), random_state=CFG.SEED)
    total = total_sq = 0.0
    count = 0
    for row in sample.itertuples(index=False):
        video = load_clip(row.cache_resolved, mmap=True)
        picks = np.unique(np.rint(np.linspace(
            0, video.shape[0] - 1, min(frames_per_video, video.shape[0])))
            .astype(np.int64))
        pixels = np.asarray(video[picks], dtype=np.float64) / 255.0
        total += float(pixels.sum())
        total_sq += float((pixels * pixels).sum())
        count += int(pixels.size)
    mean = total / max(count, 1)
    std = float(np.sqrt(max(total_sq / max(count, 1) - mean * mean, 0.0)))
    return mean, std, len(sample), count


def _keyframe_geometry_status() -> tuple[bool, list[str]]:
    if not CFG.KEYFRAMES_CSV.exists():
        return False, ["missing_keyframes_csv"]
    keyframes = pd.read_csv(CFG.KEYFRAMES_CSV)
    if "geometry_version" not in keyframes.columns:
        return False, ["unversioned_stale_geometry"]
    versions = sorted(keyframes["geometry_version"].dropna().astype(str).unique().tolist())
    return versions == [GEOMETRY_VERSION], versions


def _manifest_geometry_status(manifest: pd.DataFrame) -> tuple[bool, list[str]]:
    """Ensure stage 1 propagated the corrected keyframes into the manifest."""
    if "geometry_version" not in manifest.columns:
        return False, ["missing_manifest_geometry_version"]
    traced = manifest
    if "ed_frame" in manifest.columns and "es_frame" in manifest.columns:
        traced = manifest[(pd.to_numeric(manifest["ed_frame"], errors="coerce") >= 0) &
                          (pd.to_numeric(manifest["es_frame"], errors="coerce") >= 0)]
    versions = sorted(traced["geometry_version"].dropna().astype(str).unique().tolist())
    return versions == [GEOMETRY_VERSION], versions


def _verify_train_cycle_sampler(
    valid: pd.DataFrame,
    clip_len: int,
    period: int,
    views: int,
) -> dict:
    coverage = (clip_len - 1) * period
    if len(valid) == 0 or "Split" not in valid.columns:
        return dict(
            n_train_with_keyframes=0, n_containable=0, n_uncontainable=0,
            n_invalid_keyframes=0, containment_checks=0, containment_hits=0,
            containment_rate=0.0, index_failures=0,
            uncontainable_examples=[], invalid_keyframe_examples=[])
    train = valid[valid["Split"].astype(str).str.upper() == "TRAIN"]
    if "ed_frame" not in train.columns or "es_frame" not in train.columns:
        return dict(
            n_train_with_keyframes=0, n_containable=0, n_uncontainable=0,
            n_invalid_keyframes=0, containment_checks=0, containment_hits=0,
            containment_rate=0.0, index_failures=0,
            uncontainable_examples=[], invalid_keyframe_examples=[])

    with_keyframes = train[(pd.to_numeric(train["ed_frame"], errors="coerce") >= 0) &
                           (pd.to_numeric(train["es_frame"], errors="coerce") >= 0)]
    containable = uncontainable = invalid = 0
    checks = hits = index_failures = 0
    long_examples = []
    invalid_examples = []
    rng = np.random.default_rng(CFG.SEED)

    for row in with_keyframes.itertuples(index=False):
        name = str(row.FileName)
        nf = int(row.n_frames_verified)
        ed, es = int(row.ed_frame), int(row.es_frame)
        if not (0 <= ed < nf and 0 <= es < nf):
            invalid += 1
            if len(invalid_examples) < 25:
                invalid_examples.append(
                    {"FileName": name, "ed_frame": ed, "es_frame": es, "n_frames": nf})
            continue
        a, b = sorted((ed, es))
        if b - a > coverage:
            uncontainable += 1
            if len(long_examples) < 25:
                long_examples.append({
                    "FileName": name, "transition_frames": b - a,
                    "configured_coverage": coverage})
            # Still enforce range/order/stride safety for the best-effort path.
            idx = sample_indices(
                nf, clip_len, period, ed, es, train=True, rng=rng)
            if (len(idx) != clip_len or idx.min() < 0 or idx.max() >= nf or
                    np.any(np.diff(idx) < 0)):
                index_failures += 1
            continue

        containable += 1
        # Check random training placement plus deterministic endpoints of the
        # same constrained interval.  Every one must range-contain ED and ES.
        candidates = [sample_indices(
            nf, clip_len, period, ed, es, train=True, rng=rng)]
        candidates.extend(sample_indices(
            nf, clip_len, period, ed, es, train=False,
            view_index=view, n_views=views) for view in range(views))
        for idx in candidates:
            checks += 1
            safe = (len(idx) == clip_len and idx.min() >= 0 and idx.max() < nf and
                    np.all(np.diff(idx) >= 0))
            if not safe:
                index_failures += 1
                continue
            if int(idx[0]) <= a and int(idx[-1]) >= b:
                hits += 1

    return dict(
        n_train_with_keyframes=int(len(with_keyframes)),
        n_containable=containable,
        n_uncontainable=uncontainable,
        n_invalid_keyframes=invalid,
        containment_checks=checks,
        containment_hits=hits,
        containment_rate=round(hits / max(checks, 1), 6),
        index_failures=index_failures,
        uncontainable_examples=long_examples,
        invalid_keyframe_examples=invalid_examples,
    )


def _verify_label_free_eval_sampler(
    valid: pd.DataFrame,
    clip_len: int,
    period: int,
    views: int,
) -> dict:
    if len(valid) == 0 or "Split" not in valid.columns:
        return dict(
            n_eval_videos=0, n_index_failures=0, uniform_coverage_hits=0,
            uniform_coverage_rate=0.0, n_diversity_possible=0, n_diverse=0,
            diversity_rate=0.0, failure_examples=[])
    evaluation = valid[valid["Split"].astype(str).str.upper().isin(["VAL", "TEST"])]
    failures = coverage_hits = diverse = diversity_possible = 0
    examples = []

    for row in evaluation.itertuples(index=False):
        name = str(row.FileName)
        nf = int(row.n_frames_verified)
        # Deliberately do not pass ED/ES: final model evaluation must not use
        # ground-truth tracing annotations to select its input.
        sequences = [sample_indices(
            nf, clip_len, period, ed_frame=None, es_frame=None, train=False,
            view_index=view, n_views=views) for view in range(views)]
        safe = all(
            len(idx) == clip_len and idx.min() >= 0 and idx.max() < nf and
            np.all(np.diff(idx) >= 0) for idx in sequences)
        if not safe:
            failures += 1
            if len(examples) < 25:
                examples.append({"FileName": name, "reason": "unsafe_indices"})
            continue

        # Across deterministic views the first and last available frame must
        # be covered.  Short recordings are monotonic linspace samples.
        if min(int(idx[0]) for idx in sequences) == 0 and \
                max(int(idx[-1]) for idx in sequences) == nf - 1:
            coverage_hits += 1
        else:
            if len(examples) < 25:
                examples.append({"FileName": name, "reason": "incomplete_view_coverage"})

        max_start = max(0, nf - 1 - (clip_len - 1) * period)
        if views > 1 and max_start > 0:
            diversity_possible += 1
            unique = {tuple(idx.tolist()) for idx in sequences}
            if len(unique) > 1:
                diverse += 1
            elif len(examples) < 25:
                examples.append({"FileName": name, "reason": "duplicate_eval_views"})

    total = int(len(evaluation))
    return dict(
        n_eval_videos=total,
        n_index_failures=failures,
        uniform_coverage_hits=coverage_hits,
        uniform_coverage_rate=round(coverage_hits / max(total, 1), 6),
        n_diversity_possible=diversity_possible,
        n_diverse=diverse,
        diversity_rate=round(diverse / max(diversity_possible, 1), 6),
        failure_examples=examples,
    )


def _annotate(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_class_qa(
    valid: pd.DataFrame,
    pmean: float,
    pstd: float,
    clip_len: int,
    period: int,
) -> tuple[list[dict], list[dict], list[int]]:
    if not CFG.TRACINGS_CSV.exists():
        return [], [_bad("ALL", "missing_volume_tracings")], []
    if len(valid) == 0 or "ef_class" not in valid.columns:
        return [], [_bad("ALL", "no_valid_rows_for_visual_qa")], []
    traces = pd.read_csv(CFG.TRACINGS_CSV)
    traces["FileName"] = (traces["FileName"].astype(str)
                           .str.replace(".avi", "", regex=False))
    traces["Frame"] = pd.to_numeric(traces["Frame"], errors="coerce").astype("Int64")

    represented = sorted(pd.to_numeric(
        valid.get("ef_class", pd.Series(dtype=float)), errors="coerce")
        .dropna().astype(int).unique().tolist())
    made = []
    errors = []

    for class_id in represented:
        candidates = valid[pd.to_numeric(valid["ef_class"], errors="coerce") == class_id]
        if "ed_frame" not in candidates.columns or "es_frame" not in candidates.columns:
            errors.append(_bad(f"class_{class_id}", "missing_keyframe_columns"))
            continue
        candidates = candidates[(pd.to_numeric(candidates.get("ed_frame"), errors="coerce") >= 0) &
                                (pd.to_numeric(candidates.get("es_frame"), errors="coerce") >= 0)]
        candidates = candidates.sample(frac=1.0, random_state=CFG.SEED + class_id)
        class_errors = []
        for _, row in candidates.iterrows():
            try:
                name = str(row["FileName"])
                video = np.asarray(load_clip(row["cache_resolved"], mmap=False))
                nf = int(video.shape[0])
                ed, es = int(row["ed_frame"]), int(row["es_frame"])
                if not (0 <= ed < nf and 0 <= es < nf):
                    raise ValueError(f"keyframes ({ed},{es}) outside n_frames={nf}")

                idx = sample_indices(
                    nf, clip_len, period, ed, es, train=False)
                clip = video[idx]
                preview_idx = np.unique(np.rint(np.linspace(
                    0, len(clip) - 1, min(16, len(clip)))).astype(np.int64))
                preview = clip[preview_idx]

                class_name = row.get("class_name", f"class_{class_id}")
                base = f"class{class_id}_{_safe_slug(class_name)}"
                clip_path = CFG.VIZ_DIR / f"{base}_clip.png"
                motion_path = CFG.VIZ_DIR / f"{base}_motion.png"
                contour_path = CFG.VIZ_DIR / f"{base}_ed_es_contours.png"
                save_image(montage(preview, cols=8), clip_path)

                channels = build_multichannel(
                    preview, pmean, pstd, motion_mode="tempdiff")
                motion = channels[1]
                motion = ((motion - motion.min()) /
                          (np.ptp(motion) + 1e-6) * 255).astype(np.uint8)
                save_image(montage(motion, cols=8), motion_path)

                overlays = []
                for frame, label, color in (
                    (ed, "ED", (0, 255, 0)), (es, "ES", (0, 165, 255))):
                    frame_trace = traces[(traces["FileName"] == name) &
                                         (traces["Frame"] == frame)]
                    if len(frame_trace) < 3:
                        raise ValueError(f"no usable tracing for {label} frame {frame}")
                    px, py = tracing_polygon(
                        frame_trace["X1"].to_numpy(), frame_trace["Y1"].to_numpy(),
                        frame_trace["X2"].to_numpy(), frame_trace["Y2"].to_numpy())
                    overlay = draw_contour(video[frame], px, py, color=color)
                    overlays.append(_annotate(overlay, f"{label} frame {frame}"))
                separator = np.full((overlays[0].shape[0], 6, 3), 255, dtype=np.uint8)
                save_image(np.concatenate([overlays[0], separator, overlays[1]], axis=1),
                           contour_path)

                made.append({
                    "ef_class": class_id,
                    "FileName": name,
                    "clip": str(clip_path),
                    "motion": str(motion_path),
                    "contours": str(contour_path),
                })
                break
            except Exception as exc:
                class_errors.append(str(exc))
        else:
            errors.append(_bad(
                f"class_{class_id}", "visual_qa_failed",
                class_errors[:3] if class_errors else "no traced candidate"))

    return made, errors, represented


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="verify only first N usable rows (explicit smoke-test scope)")
    parser.add_argument("--clip-len", type=int, default=CFG.CLIP_LEN)
    parser.add_argument("--period", type=int, default=CFG.SAMPLING_PERIOD)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--stats-sample", type=int, default=512)
    parser.add_argument("--stats-frames", type=int, default=8)
    parser.add_argument("--mean-drift-tol", type=float, default=0.02)
    parser.add_argument("--std-drift-tol", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("limit", "clip_len", "period", "views", "stats_sample", "stats_frames"):
        value = getattr(args, name)
        if value < 0 or (name != "limit" and value == 0):
            parser.error(f"--{name.replace('_', '-')} must be {'>= 0' if name == 'limit' else '> 0'}")

    CFG.ensure_dirs()
    started = time.time()
    if not CFG.MANIFEST.exists():
        raise SystemExit("[stage5] manifest.csv missing - run stages 1 and 4 first.")
    manifest = pd.read_csv(CFG.MANIFEST)
    manifest["FileName"] = manifest["FileName"].astype(str)
    manifest["Split"] = manifest["Split"].astype(str).str.upper()

    scoped, valid, cache_bad, lengths = _validate_caches(manifest, args.limit)
    if len(scoped) == 0:
        raise SystemExit("[stage5] no usable manifest rows selected.")
    pmean, pstd = _load_norm()
    recomputed_mean, recomputed_std, stats_videos, stats_pixels = \
        _recompute_pixel_stats(valid, args.stats_sample, args.stats_frames)
    drift_mean = abs(recomputed_mean - pmean)
    drift_std = abs(recomputed_std - pstd)

    keyframe_geometry_ok, geometry_versions = _keyframe_geometry_status()
    manifest_geometry_ok, manifest_geometry_versions = _manifest_geometry_status(manifest)
    geometry_ok = keyframe_geometry_ok and manifest_geometry_ok
    train_cycle = _verify_train_cycle_sampler(
        valid, args.clip_len, args.period, args.views)
    label_free_eval = _verify_label_free_eval_sampler(
        valid, args.clip_len, args.period, args.views)
    qa_made, qa_errors, represented_classes = _write_class_qa(
        valid, pmean, pstd, args.clip_len, args.period)

    reason_counts = Counter(item["reason"] for item in cache_bad)
    integrity_ok = len(cache_bad) == 0 and len(valid) == len(scoped)
    # A limited smoke test does not estimate the full train distribution and
    # may contain no VAL/TEST row.  Those omissions are warnings, not false
    # claims of full verification.
    norm_ok = (drift_mean <= args.mean_drift_tol and drift_std <= args.std_drift_tol)
    norm_gate_ok = norm_ok or bool(args.limit)
    cycle_ok = (train_cycle["n_invalid_keyframes"] == 0 and
                train_cycle["index_failures"] == 0 and
                (train_cycle["n_containable"] == 0 or
                 train_cycle["containment_rate"] == 1.0))
    eval_present = label_free_eval["n_eval_videos"] > 0
    eval_ok = (label_free_eval["n_index_failures"] == 0 and
               (not eval_present or label_free_eval["uniform_coverage_rate"] == 1.0) and
               (label_free_eval["n_diversity_possible"] == 0 or
                label_free_eval["diversity_rate"] == 1.0) and
               (eval_present or bool(args.limit)))
    qa_ok = len(qa_errors) == 0 and len(qa_made) == len(represented_classes)

    hard_pass = all((integrity_ok, norm_gate_ok, geometry_ok, cycle_ok, eval_ok, qa_ok))
    warnings = []
    if args.limit:
        warnings.append(
            f"limited smoke-test scope: {len(scoped)} rows; not a full-dataset release gate")
    if not norm_ok:
        warnings.append("sampled normalization drift exceeds tolerance")
    if train_cycle["n_uncontainable"]:
        warnings.append(
            f"{train_cycle['n_uncontainable']} train ED/ES transitions exceed the "
            f"fixed {args.clip_len}x{args.period} coverage and use best-effort subwindows")
    if not eval_present:
        warnings.append("selected scope contains no VAL/TEST row; label-free eval sampler untested")

    overall = "FAIL"
    if hard_pass:
        overall = "PASS_WITH_WARNINGS" if warnings else "PASS"

    report = dict(
        overall=overall,
        scope="limited_smoke_test" if args.limit else "full_dataset",
        n_manifest_usable_selected=int(len(scoped)),
        n_integrity_checked=int(len(scoped)),
        n_valid_caches=int(len(valid)),
        n_bad_caches=int(len(cache_bad)),
        cache_failure_counts=dict(reason_counts),
        bad_cache_examples=cache_bad[:50],
        clip_len_min=int(min(lengths)) if lengths else 0,
        clip_len_median=int(np.median(lengths)) if lengths else 0,
        clip_len_max=int(max(lengths)) if lengths else 0,
        norm_pixel_mean=round(pmean, 6),
        norm_pixel_std=round(pstd, 6),
        recomputed_train_pixel_mean=round(recomputed_mean, 6),
        recomputed_train_pixel_std=round(recomputed_std, 6),
        norm_stats_videos=int(stats_videos),
        norm_stats_pixels=int(stats_pixels),
        drift_mean=round(drift_mean, 6),
        drift_std=round(drift_std, 6),
        mean_drift_tolerance=args.mean_drift_tol,
        std_drift_tolerance=args.std_drift_tol,
        geometry_version_expected=GEOMETRY_VERSION,
        keyframe_geometry_versions_found=geometry_versions,
        manifest_geometry_versions_found=manifest_geometry_versions,
        keyframe_geometry_version_ok=keyframe_geometry_ok,
        manifest_geometry_version_ok=manifest_geometry_ok,
        geometry_version_ok=geometry_ok,
        sampler_clip_len=args.clip_len,
        sampler_period=args.period,
        sampler_native_frame_coverage=(args.clip_len - 1) * args.period,
        sampler_eval_views=args.views,
        train_cycle_sampler=train_cycle,
        label_free_eval_sampler=label_free_eval,
        visual_qa_examples=qa_made,
        visual_qa_errors=qa_errors,
        represented_classes=represented_classes,
        viz_dir=str(CFG.VIZ_DIR),
        warnings=warnings,
        elapsed_sec=round(time.time() - started, 1),
    )
    _atomic_json_dump(report, CFG.VERIFY_JSON)

    print("\n================ STAGE 5 VERIFY ================")
    print(f"  scope                    : {report['scope']}")
    print(f"  caches valid / expected  : {len(valid)} / {len(scoped)}")
    print(f"  cache failures           : {len(cache_bad)}")
    print(f"  keyframe geometry        : {geometry_versions} (ok={keyframe_geometry_ok})")
    print(f"  manifest geometry        : {manifest_geometry_versions} "
          f"(ok={manifest_geometry_ok})")
    print(f"  sampler config           : {args.clip_len}x{args.period} "
          f"(coverage={(args.clip_len - 1) * args.period})")
    print(f"  train containment rate   : {train_cycle['containment_rate']}")
    print(f"  uncontainable transitions: {train_cycle['n_uncontainable']}")
    print(f"  label-free eval coverage : {label_free_eval['uniform_coverage_rate']}")
    print(f"  label-free TTA diversity : {label_free_eval['diversity_rate']}")
    print(f"  visual QA classes        : {len(qa_made)} / {len(represented_classes)}")
    print(f"  norm drift mean / std    : {drift_mean:.6f} / {drift_std:.6f}")
    print(f"\n  OVERALL: {overall}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print(f"  report -> {CFG.VERIFY_JSON}")
    print(f"  viz    -> {CFG.VIZ_DIR}")
    print("================================================")

    if not hard_pass:
        raise SystemExit("[stage5] verification release gate FAILED; inspect report.")


if __name__ == "__main__":
    main()
