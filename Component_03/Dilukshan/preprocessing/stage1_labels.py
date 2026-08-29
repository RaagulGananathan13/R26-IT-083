"""
STAGE 1 :: Label engineering & imbalance analysis
=================================================
Builds the master manifest by fusing:
  * FileList.csv        (EF / ESV / EDV / Split / fps / frames)
  * video_index.csv     (stage-0 integrity + true resolution)   [optional]
  * keyframes.csv       (stage-2 ED/ES indices)                 [optional]

and derives every target/weight the training stage needs:

  ef_class          4-class severity label (from CFG.EF_THRESHOLDS)
  class_name        human-readable class
  ef                regression target (raw EF)
  class_weight      Cui-2019 effective-number class re-weighting (train freq)
  sample_weight     per-row weight for a WeightedRandomSampler (balances the
                    4 classes so minority classes are seen 75%+ of the time)
  ef_density_weight balanced-REGRESSION weight = inverse EF-histogram density,
                    so rare (low-EF) cases dominate the MAE gradient less
                    unequally  ->  lowers worst-case / minority-class MAE.

Only the TRAIN split is used to fit weights; VAL/TEST inherit the mapping but
their weights are set to 1.0 (never resampled).

Outputs
-------
artifacts/manifest.csv       master table (cache_path filled by stage-4)
prints the full 4x{train,val,test} class contingency table.

Run:  python stage1_labels.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import CFG


def effective_number_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    """Class-balanced weights (Cui et al., CVPR-2019).  Normalised to mean 1."""
    counts = np.asarray(counts, dtype=np.float64)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    return w / w.mean()


def main():
    CFG.ensure_dirs()
    t0 = time.time()
    # Label/keyframe regeneration must not discard an expensive completed
    # cache.  Carry cache metadata forward by FileName when a manifest exists.
    previous_cache = None
    if CFG.MANIFEST.exists():
        previous = pd.read_csv(CFG.MANIFEST)
        previous["FileName"] = previous["FileName"].astype(str)
        cache_cols = [c for c in
                      ["FileName", "cache_path", "n_frames_cached", "cached_ok"]
                      if c in previous.columns]
        if "cache_path" in cache_cols:
            previous_cache = previous[cache_cols].drop_duplicates(
                subset=["FileName"], keep="last")

    fl = pd.read_csv(CFG.FILELIST_CSV)
    fl["FileName"] = fl["FileName"].astype(str)
    fl = fl.dropna(subset=["EF"]).copy()

    # ---- 4-class label -----------------------------------------------------
    fl["ef_class"] = fl["EF"].apply(CFG.ef_to_class).astype(int)
    fl["class_name"] = fl["ef_class"].map(dict(enumerate(CFG.CLASS_NAMES)))
    fl["Split"] = fl["Split"].str.upper()

    # ---- merge stage-0 integrity ------------------------------------------
    if CFG.VIDEO_INDEX.exists():
        vi = pd.read_csv(CFG.VIDEO_INDEX)[
            ["FileName", "exists", "ok", "n_frames_prop", "height", "width",
             "res_outlier"]]
        fl = fl.merge(vi, on="FileName", how="left")
        fl["usable"] = fl["ok"].fillna(False).astype(bool) & fl["exists"].fillna(False).astype(bool)
    else:
        print("[stage1][WARN] video_index.csv not found - run stage0 first for "
              "integrity checks. Assuming all files usable.")
        fl["usable"] = True

    # ---- merge stage-2 keyframes ------------------------------------------
    if CFG.KEYFRAMES_CSV.exists():
        kf_all = pd.read_csv(CFG.KEYFRAMES_CSV)
        kf_cols = ["FileName", "ed_frame", "es_frame", "ed_area", "es_area",
                   "area_ratio"]
        if "geometry_version" in kf_all.columns:
            kf_cols.append("geometry_version")
        kf = kf_all[kf_cols]
        fl = fl.merge(kf, on="FileName", how="left")
    else:
        print("[stage1][WARN] keyframes.csv not found - run stage2 for "
              "cycle-aware sampling. Filling ED/ES with -1.")
        for c in ["ed_frame", "es_frame"]:
            fl[c] = -1
    for c in ["ed_frame", "es_frame"]:
        fl[c] = fl[c].fillna(-1).astype(int)
    if "geometry_version" not in fl.columns:
        fl["geometry_version"] = "unknown"
    fl["geometry_version"] = fl["geometry_version"].fillna("untraced")

    # ---- class weights (train frequencies only) ---------------------------
    train = fl[(fl["Split"] == "TRAIN") & fl["usable"]]
    counts = np.array([(train["ef_class"] == c).sum() for c in range(CFG.N_CLASSES)])
    cls_w = effective_number_weights(counts, CFG.EFFECTIVE_NUM_BETA)
    cls_w_map = {c: cls_w[c] for c in range(CFG.N_CLASSES)}
    fl["class_weight"] = fl["ef_class"].map(cls_w_map)

    # ---- per-sample sampler weight ----------------------------------------
    # Inverse train frequency so each class is drawn equally often.
    inv_freq = {c: (len(train) / max(counts[c], 1)) for c in range(CFG.N_CLASSES)}
    inv_series = fl["ef_class"].map(inv_freq)
    fl["sample_weight"] = np.where(fl["Split"] == "TRAIN", inv_series, 1.0)
    # normalise train weights to mean 1 for numerical comfort
    m = fl.loc[fl["Split"] == "TRAIN", "sample_weight"].mean()
    fl.loc[fl["Split"] == "TRAIN", "sample_weight"] /= m

    # ---- balanced-regression density weight -------------------------------
    ef_train = train["EF"].values
    hist, edges = np.histogram(ef_train, bins=CFG.EF_DENSITY_BINS,
                               range=(0, 100), density=False)
    hist = hist.astype(np.float64) + 1.0            # Laplace smoothing
    dens = hist / hist.sum()
    bin_idx = np.clip(np.digitize(fl["EF"].values, edges) - 1, 0, CFG.EF_DENSITY_BINS - 1)
    w = 1.0 / dens[bin_idx]
    w = w / w[fl["Split"].values == "TRAIN"].mean()  # normalise on train
    fl["ef_density_weight"] = np.where(fl["Split"] == "TRAIN", w, 1.0)
    # cap extreme weights to keep gradients stable
    cap = 8.0
    fl["ef_density_weight"] = fl["ef_density_weight"].clip(upper=cap)

    keep = ["FileName", "Split", "usable", "EF", "ESV", "EDV",
            "ef_class", "class_name", "FPS", "NumberOfFrames",
            "ed_frame", "es_frame", "geometry_version", "class_weight", "sample_weight",
            "ef_density_weight"]
    keep = [c for c in keep if c in fl.columns]
    manifest = fl[keep].copy()
    if previous_cache is not None:
        manifest = manifest.merge(previous_cache, on="FileName", how="left")
        manifest["cache_path"] = manifest["cache_path"].fillna("")
        if "n_frames_cached" in manifest.columns:
            manifest["n_frames_cached"] = manifest["n_frames_cached"].fillna(0)
        if "cached_ok" in manifest.columns:
            manifest["cached_ok"] = manifest["cached_ok"].fillna(False)
    else:
        manifest["cache_path"] = ""       # filled by stage-4

    tmp = CFG.MANIFEST.with_suffix(".csv.tmp")
    manifest.to_csv(tmp, index=False)
    os.replace(tmp, CFG.MANIFEST)

    # ---- report ------------------------------------------------------------
    print("\n================ STAGE 1 LABELS ================")
    print(f"  usable / total : {int(fl['usable'].sum())} / {len(fl)}")
    ct = pd.crosstab(fl["class_name"], fl["Split"])
    order = list(CFG.CLASS_NAMES)
    ct = ct.reindex(order)
    print("\n  Class x Split contingency:")
    print(ct.to_string())
    print("\n  Train class counts :", counts.tolist())
    print("  Effective-num class weights :",
          {CFG.CLASS_NAMES[c]: round(cls_w[c], 3) for c in range(CFG.N_CLASSES)})
    print("  Sampler inverse-freq weights:",
          {CFG.CLASS_NAMES[c]: round(inv_freq[c] / m, 3) for c in range(CFG.N_CLASSES)})
    print(f"\n  manifest -> {CFG.MANIFEST}")
    print(f"  elapsed  : {time.time()-t0:.1f}s")
    print("================================================")


if __name__ == "__main__":
    main()
