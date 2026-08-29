"""
STAGE 2 :: End-Diastole / End-Systole key-frame extraction
==========================================================
Parses VolumeTracings.csv, reconstructs the LV contour for each traced frame,
computes its shoelace area, and labels the larger-area frame ED and the
smaller-area frame ES.  These indices drive the cardiac-cycle-aware sampler
and the verification overlays.

Outputs
-------
artifacts/keyframes.csv : FileName, ed_frame, es_frame, ed_area, es_area,
                          area_ratio (ES/ED, a rough EF sanity proxy)

Run:  python stage2_keyframes.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG
from utils.geometry import tracing_area, GEOMETRY_VERSION


def main():
    CFG.ensure_dirs()
    t0 = time.time()
    print(f"[stage2] reading {CFG.TRACINGS_CSV}")
    tr = pd.read_csv(CFG.TRACINGS_CSV)
    # normalise FileName (tracings use .avi extension)
    tr["FileName"] = tr["FileName"].astype(str).str.replace(".avi", "", regex=False)

    rows = []
    for fname, g in tqdm(tr.groupby("FileName"), desc="tracings"):
        areas = {}
        for frame_no, fg in g.groupby("Frame"):
            a = tracing_area(fg["X1"].values, fg["Y1"].values,
                             fg["X2"].values, fg["Y2"].values)
            areas[int(frame_no)] = a
        if not areas:
            continue
        # ED = max area (fullest chamber), ES = min area (most contracted)
        ed = max(areas, key=areas.get)
        es = min(areas, key=areas.get)
        ed_a, es_a = areas[ed], areas[es]
        rows.append(dict(
            FileName=fname, ed_frame=ed, es_frame=es,
            ed_area=round(ed_a, 3), es_area=round(es_a, 3),
            area_ratio=round(es_a / ed_a, 4) if ed_a > 0 else np.nan,
            n_traced=len(areas),
            geometry_version=GEOMETRY_VERSION,
        ))

    kf = pd.DataFrame(rows)
    kf.to_csv(CFG.KEYFRAMES_CSV, index=False)

    print("\n================ STAGE 2 KEYFRAMES ================")
    print(f"  videos with tracings : {len(kf)}")
    print(f"  mean ED area         : {kf['ed_area'].mean():.1f}")
    print(f"  mean ES area         : {kf['es_area'].mean():.1f}")
    print(f"  mean ES/ED area ratio: {kf['area_ratio'].mean():.3f}  "
          f"(lower ~ lower EF; sanity proxy)")
    print(f"  geometry version     : {GEOMETRY_VERSION}")
    print(f"  keyframes -> {CFG.KEYFRAMES_CSV}")
    print(f"  elapsed  : {time.time()-t0:.1f}s")
    print("==================================================")


if __name__ == "__main__":
    main()
