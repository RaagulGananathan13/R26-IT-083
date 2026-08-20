"""
STAGE 0 :: Dataset audit & integrity verification
=================================================
Cross-checks FileList.csv against the actual .avi files:
  * every labelled file exists and opens,
  * records true container resolution / fps / frame-count,
  * flags resolution outliers (anything not FRAME_SIZE x FRAME_SIZE),
  * flags CSV<->container metadata disagreements,
  * flags missing videos and unlabelled orphan videos.

Outputs
-------
artifacts/video_index.csv   one row per labelled file with probe results
artifacts/audit_report.json machine-readable summary of all anomalies

Run:  python stage0_audit.py            (from the preprocessing/ folder)
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import CFG
from utils.io_utils import probe_video


def _probe_one(args):
    fname, video_dir = args
    path = os.path.join(video_dir, fname if fname.endswith(".avi") else fname + ".avi")
    exists = os.path.exists(path)
    rec = dict(FileName=fname, path=path, exists=exists)
    if not exists:
        rec.update(ok=False, error="missing_file")
        return rec
    info = probe_video(path)
    rec.update(info)
    return rec


def main():
    CFG.ensure_dirs()
    t0 = time.time()
    print(f"[stage0] reading {CFG.FILELIST_CSV}")
    fl = pd.read_csv(CFG.FILELIST_CSV)
    # FileName in FileList has no extension; Videos have .avi
    names = fl["FileName"].astype(str).tolist()

    # discover orphan videos (present on disk but absent from FileList)
    disk = {p for p in os.listdir(CFG.VIDEO_DIR) if p.lower().endswith(".avi")}
    labelled = {n if n.endswith(".avi") else n + ".avi" for n in names}
    orphans = sorted(disk - labelled)

    print(f"[stage0] probing {len(names)} videos with {CFG.NUM_WORKERS} workers ...")
    args = [(n, str(CFG.VIDEO_DIR)) for n in names]
    records = []
    with ProcessPoolExecutor(max_workers=CFG.NUM_WORKERS) as ex:
        futs = [ex.submit(_probe_one, a) for a in args]
        for f in tqdm(as_completed(futs), total=len(futs), desc="probe"):
            records.append(f.result())

    idx = pd.DataFrame(records)
    # merge CSV metadata for cross-checks
    meta = fl[["FileName", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames"]].copy()
    idx = idx.merge(meta, on="FileName", how="left")

    # anomaly flags
    idx["res_outlier"] = (idx["height"] != CFG.FRAME_SIZE) | (idx["width"] != CFG.FRAME_SIZE)
    idx["res_outlier"] = idx["res_outlier"].fillna(True)
    idx["meta_mismatch"] = (
        (idx["height"] != idx["FrameHeight"]) | (idx["width"] != idx["FrameWidth"])
    ).fillna(True)

    idx.to_csv(CFG.VIDEO_INDEX, index=False)

    n_missing = int((~idx["exists"]).sum())
    n_unreadable = int((idx["exists"] & ~idx["ok"].astype(bool)).sum())
    n_res_out = int(idx["res_outlier"].sum())
    n_meta_mis = int(idx["meta_mismatch"].sum())

    report = dict(
        n_labelled=len(names),
        n_on_disk=len(disk),
        n_missing_files=n_missing,
        n_unreadable=n_unreadable,
        n_resolution_outliers=n_res_out,
        n_metadata_mismatch=n_meta_mis,
        n_orphans=len(orphans),
        orphan_examples=orphans[:20],
        resolution_outlier_examples=idx.loc[idx["res_outlier"], "FileName"].head(20).tolist(),
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(CFG.AUDIT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("\n================ STAGE 0 AUDIT ================")
    for k, v in report.items():
        if "examples" in k:
            continue
        print(f"  {k:28s}: {v}")
    print(f"  video_index -> {CFG.VIDEO_INDEX}")
    print(f"  report      -> {CFG.AUDIT_JSON}")
    print("===============================================")
    if n_missing or n_unreadable:
        print(f"[stage0][WARN] {n_missing} missing / {n_unreadable} unreadable "
              f"videos will be dropped downstream.")


if __name__ == "__main__":
    main()
