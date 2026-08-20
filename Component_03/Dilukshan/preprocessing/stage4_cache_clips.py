"""
STAGE 4 :: Decode, standardize & cache all videos
=================================================
For every usable video this stage:
  1. fully decodes it to grayscale,
  2. resizes any resolution-outlier frames to FRAME_SIZE x FRAME_SIZE,
  3. optionally applies speckle denoising (CFG.DENOISE / --denoise),
  4. writes a (T,H,W) uint8 array to cache/videos/<FileName>.npy   (mmap-able),
  5. records the TRUE decoded frame count,
  6. accumulates EXACT train-split pixel statistics and refines norm_stats.json.

Caching decoded frames removes AVI decoding from the training loop entirely,
which is the single biggest throughput win on an 8-core / RTX-4060 box.

Disk budget: ~112x112 uint8 x mean 176 frames x 10,030 videos ~= 22 GB
uncompressed (fastest).  Use --compress for ~2-3x smaller .npz (slower load),
or --max-frames N to cap very long videos.

Outputs
-------
cache/videos/*.npy                 decoded clips
artifacts/manifest.csv (updated)   cache_path + n_frames_cached columns
artifacts/norm_stats.json (refined exact pixel stats, if not --resume)

Run (full)   : python stage4_cache_clips.py
Run (test)   : python stage4_cache_clips.py --limit 50
Run (resume) : python stage4_cache_clips.py --resume
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import CFG
from utils.io_utils import decode_video, save_clip, load_clip, resolve_cache_path
from utils.denoise import denoise_video


def _process_one(args):
    (fname, split, video_dir, cache_dir, size, denoise, median_k, nlm_h,
     compress, max_frames, resume) = args
    ext = ".npz" if compress else ".npy"
    out = os.path.join(cache_dir, fname + ext)

    if resume and os.path.exists(out):
        try:
            # Validate shape/dtype and record the true length for both .npy and
            # compressed .npz caches.  Never trust a merely existing file.
            cached = load_clip(out, mmap=True)
            if (cached.ndim != 3 or cached.shape[0] <= 0 or
                    cached.shape[1:] != (size, size) or cached.dtype != np.uint8):
                raise ValueError(
                    f"invalid cached array shape={cached.shape}, dtype={cached.dtype}")
            n = int(cached.shape[0])
            return dict(FileName=fname, cache_path=out, n_frames_cached=n,
                        ok=True, skipped=True, s=0.0, ss=0.0, cnt=0)
        except Exception:
            pass  # fall through and re-cache

    try:
        path = os.path.join(video_dir, fname + ".avi")
        vid, meta = decode_video(path, size=size, grayscale=True,
                                 max_frames=max_frames)
        if denoise != "none":
            vid = denoise_video(vid, denoise, median_k, nlm_h)
        save_clip(vid, os.path.join(cache_dir, fname), compress=compress)

        # exact train stats accumulation (normalised [0,1])
        s = ss = 0.0; cnt = 0
        if split == "TRAIN":
            g = vid.astype(np.float64) / 255.0
            s = float(g.sum()); ss = float((g * g).sum()); cnt = int(g.size)
        return dict(FileName=fname, cache_path=out,
                    n_frames_cached=int(vid.shape[0]), ok=True, skipped=False,
                    s=s, ss=ss, cnt=cnt)
    except Exception as e:
        return dict(FileName=fname, cache_path="", n_frames_cached=0,
                    ok=False, skipped=False, error=str(e), s=0.0, ss=0.0, cnt=0)


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    """Parse CSV booleans without treating the string ``'False'`` as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return series.fillna(str(default)).astype(str).str.strip().str.lower().isin(truthy)


def _portable_cache_path(path: str) -> str:
    """Store paths relative to preprocessing root when possible."""
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(CFG.PREP_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _merge_cache_results(man: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Overlay successful results without erasing unprocessed manifest rows."""
    out = man.copy()
    if "cache_path" not in out.columns:
        out["cache_path"] = ""
    if "n_frames_cached" not in out.columns:
        out["n_frames_cached"] = 0
    if "cached_ok" not in out.columns:
        out["cached_ok"] = False

    # FileName is the stable key.  A failed recache leaves previous valid
    # metadata untouched, while the non-zero process exit still surfaces the
    # failure to the orchestrator.
    for row in res.loc[res["ok"]].itertuples(index=False):
        mask = out["FileName"] == str(row.FileName)
        out.loc[mask, "cache_path"] = _portable_cache_path(str(row.cache_path))
        out.loc[mask, "n_frames_cached"] = int(row.n_frames_cached)
        out.loc[mask, "cached_ok"] = True
    return out


def _incomplete_manifest_rows(man: pd.DataFrame) -> pd.DataFrame:
    """Find usable rows without complete, existing cache metadata."""
    usable = (_as_bool(man["usable"], default=True)
              if "usable" in man.columns else pd.Series(True, index=man.index))
    ok = (_as_bool(man["cached_ok"], default=False)
          if "cached_ok" in man.columns else pd.Series(False, index=man.index))
    paths = man.get("cache_path", pd.Series("", index=man.index)).fillna("").astype(str)

    exists = []
    for value in paths:
        try:
            p = resolve_cache_path(value, base_dir=CFG.PREP_DIR)
            exists.append(p.is_file())
        except (TypeError, ValueError, OSError):
            exists.append(False)
    complete = ok & paths.str.strip().ne("") & pd.Series(exists, index=man.index)
    return man.loc[usable & ~complete]


def _write_manifest_atomic(man: pd.DataFrame) -> None:
    tmp = CFG.MANIFEST.with_suffix(".csv.tmp")
    man.to_csv(tmp, index=False)
    os.replace(tmp, CFG.MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise", default=CFG.DENOISE,
                    choices=["none", "median", "nlm"])
    ap.add_argument("--compress", action="store_true", default=CFG.COMPRESS_CACHE)
    ap.add_argument("--max-frames", type=int, default=CFG.STORE_MAX_FRAMES)
    ap.add_argument("--workers", type=int, default=CFG.NUM_WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N usable videos (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip already-cached files (disables exact stat refine)")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit must be >= 0")
    if args.workers <= 0:
        ap.error("--workers must be positive")
    if args.max_frames < 0:
        ap.error("--max-frames must be >= 0")

    CFG.ensure_dirs()
    t0 = time.time()

    if not CFG.MANIFEST.exists():
        sys.exit("[stage4] manifest.csv missing - run stage1_labels.py first.")
    man = pd.read_csv(CFG.MANIFEST)
    man["FileName"] = man["FileName"].astype(str)
    if "usable" in man.columns:
        work = man[_as_bool(man["usable"], default=True)].copy()
    else:
        work = man.copy()
    if args.limit:
        work = work.head(args.limit)
    if len(work) == 0:
        sys.exit("[stage4] no usable videos selected for caching.")

    print(f"[stage4] caching {len(work)} videos | workers={args.workers} | "
          f"denoise={args.denoise} | compress={args.compress} | "
          f"max_frames={args.max_frames or 'ALL'} | resume={args.resume}")

    jobs = [
        (r.FileName, r.Split, str(CFG.VIDEO_DIR), str(CFG.CACHE_DIR),
         CFG.FRAME_SIZE, args.denoise, CFG.DENOISE_MEDIAN_K, CFG.NLM_H,
         args.compress, args.max_frames, args.resume)
        for r in work.itertuples(index=False)
    ]

    results = []
    S = SS = 0.0; N = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="cache"):
            r = f.result()
            results.append(r)
            S += r["s"]; SS += r["ss"]; N += r["cnt"]

    res = pd.DataFrame(results)
    n_ok = int(res["ok"].sum())
    n_skip = int(res.get("skipped", pd.Series(dtype=bool)).sum())
    n_fail = int((~res["ok"]).sum())

    # ---- update manifest ---------------------------------------------------
    # Overlay only attempted successes.  This is essential for --limit smoke
    # tests and interrupted/resumed runs: unprocessed cache metadata survives.
    man = _merge_cache_results(man, res)
    _write_manifest_atomic(man)

    # ---- refine exact pixel stats -----------------------------------------
    refined = False
    full_success = not args.limit and n_fail == 0 and len(work) == len(
        man[_as_bool(man["usable"], default=True)]
        if "usable" in man.columns else man)
    if not args.resume and full_success and N > 0:
        pix_mean = S / N
        pix_std = float(np.sqrt(max(SS / N - pix_mean ** 2, 0.0)))
        stats = {}
        if CFG.NORM_JSON.exists():
            with open(CFG.NORM_JSON, encoding="utf-8") as f:
                stats = json.load(f)
        stats.update(pixel_mean=round(pix_mean, 6), pixel_std=round(pix_std, 6),
                     source="exact_cached_train_pixels", n_train_pixels=int(N),
                     cache_denoise=args.denoise,
                     cache_max_frames=int(args.max_frames),
                     cache_format="npz" if args.compress else "npy")
        tmp = CFG.NORM_JSON.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, CFG.NORM_JSON)
        refined = True

    # A full run is only complete when every usable row has positive cache
    # metadata and an existing file.  Limited smoke tests intentionally check
    # only their selected subset and therefore skip this global assertion.
    incomplete = _incomplete_manifest_rows(man) if not args.limit else man.iloc[0:0]

    print("\n================ STAGE 4 CACHE ================")
    print(f"  cached ok : {n_ok}   skipped: {n_skip}   failed: {n_fail}")
    if n_fail:
        fails = res.loc[~res["ok"]]
        ex_fail = fails["FileName"].head(10).tolist()
        print(f"  failed examples: {ex_fail}")
        if "error" in fails.columns:
            for msg in fails["error"].dropna().head(3):
                print(f"    error: {msg}")
    if refined:
        print(f"  refined exact pixel stats -> {CFG.NORM_JSON}")
    elif args.limit:
        print("  norm stats unchanged (limited smoke test is not full-dataset statistics)")
    elif args.resume:
        print("  norm stats unchanged (--resume does not re-accumulate skipped pixels)")
    elif n_fail:
        print("  norm stats unchanged (cache run was incomplete)")
    tot_frames = int(res["n_frames_cached"].sum())
    print(f"  total frames cached : {tot_frames:,}")
    print(f"  cache dir : {CFG.CACHE_DIR}")
    print(f"  manifest  : {CFG.MANIFEST}")
    print(f"  elapsed   : {time.time()-t0:.1f}s")
    print("===============================================")

    if n_fail:
        raise SystemExit(f"[stage4] FAILED: {n_fail} selected videos could not be cached.")
    if len(incomplete):
        examples = incomplete["FileName"].astype(str).head(10).tolist()
        raise SystemExit(
            f"[stage4] FAILED: {len(incomplete)} usable manifest rows are not fully "
            f"cached; examples={examples}")


if __name__ == "__main__":
    main()
