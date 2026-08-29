"""
STAGE 3 :: Normalization statistics
===================================
Computes the constants every model needs, from the TRAIN split only (no leakage):

  * EF target stats     : mean / std of EF  -> for standardised regression head
  * pixel intensity     : grayscale mean / std in [0,1]  -> input normalization

Intensity stats are estimated from a uniform temporal sample of frames per
train video (fast, ~16 frames each).  Stage-4 optionally refines these to the
*exact* full-frame statistics while it is decoding everything anyway; if you
skip stage-4 refinement, this estimate is already within ~1e-3.

Outputs
-------
artifacts/norm_stats.json
    {ef_mean, ef_std, pixel_mean, pixel_std, source, n_videos_sampled}

Run:  python stage3_norm_stats.py
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import CFG

_N_SAMPLE = 16   # frames sampled per video for the intensity estimate


def _sample_stats(args):
    """Return (sum, sumsq, count) of normalised intensities for one video."""
    fname, video_dir, size = args
    path = os.path.join(video_dir, fname + ".avi")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0.0, 0.0, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    picks = np.linspace(0, max(total - 1, 0), num=min(_N_SAMPLE, max(total, 1)),
                        dtype=int) if total > 0 else [0]
    s = ss = 0.0
    n = 0
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if g.shape[0] != size or g.shape[1] != size:
            g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float64) / 255.0
        s += g.sum(); ss += (g * g).sum(); n += g.size
    cap.release()
    return (s, ss, n)


def main():
    CFG.ensure_dirs()
    t0 = time.time()
    fl = pd.read_csv(CFG.FILELIST_CSV)
    fl["FileName"] = fl["FileName"].astype(str)
    fl["Split"] = fl["Split"].str.upper()
    train = fl[fl["Split"] == "TRAIN"].dropna(subset=["EF"])

    ef = train["EF"].values.astype(np.float64)
    ef_mean, ef_std = float(ef.mean()), float(ef.std())

    print(f"[stage3] estimating pixel stats from {len(train)} train videos "
          f"({_N_SAMPLE} frames each) ...")
    args = [(n, str(CFG.VIDEO_DIR), CFG.FRAME_SIZE) for n in train["FileName"]]
    S = SS = 0.0
    N = 0
    with ProcessPoolExecutor(max_workers=CFG.NUM_WORKERS) as ex:
        futs = [ex.submit(_sample_stats, a) for a in args]
        for f in tqdm(as_completed(futs), total=len(futs), desc="pixels"):
            s, ss, n = f.result()
            S += s; SS += ss; N += n

    pix_mean = S / max(N, 1)
    pix_var = max(SS / max(N, 1) - pix_mean ** 2, 0.0)
    pix_std = float(np.sqrt(pix_var))

    stats = dict(
        ef_mean=round(ef_mean, 6), ef_std=round(ef_std, 6),
        pixel_mean=round(pix_mean, 6), pixel_std=round(pix_std, 6),
        source="estimate_sampled_frames", n_frames_per_video=_N_SAMPLE,
        n_videos_sampled=len(train),
    )
    with open(CFG.NORM_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n================ STAGE 3 NORM STATS ================")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")
    print(f"  norm_stats -> {CFG.NORM_JSON}")
    print(f"  elapsed    : {time.time()-t0:.1f}s")
    print("  (EchoNet reference pixel mean/std ~ 0.129 / 0.191 - sanity check)")
    print("====================================================")


if __name__ == "__main__":
    main()
