"""
================================================================================
 EchoNet-Dynamic  ::  Preprocessing Configuration
================================================================================
Single source of truth for every path and hyper-parameter used by the
preprocessing pipeline.  Import `CFG` everywhere; never hard-code a path.

Task recap
----------
Dataset : EchoNet-Dynamic (10,030 apical-4-chamber echocardiogram videos).
Target  : Left-ventricular Ejection Fraction (EF).
Heads   : (a) regression  -> minimise MAE on EF
          (b) 4-class classification of EF severity -> 75%+ accuracy / class.

Clinical 4-class scheme (ASE / AHA severity grading of LV systolic function):
    class 0  Severe    reduction   EF <  30
    class 1  Moderate  reduction   30 <= EF < 40
    class 2  Mild      reduction   40 <= EF < 55
    class 3  Normal    (preserved) EF >= 55

Author : Research pipeline authored for Component_03 (PP2 deliverable).
================================================================================
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    # ----------------------------------------------------------------- paths
    PREP_DIR: Path = Path(__file__).resolve().parent          # .../Dilukshan/preprocessing
    ROOT: Path = Path(__file__).resolve().parent.parent       # .../Dilukshan

    # ------------------------------------------------------------ label maps
    # Upper-exclusive thresholds. len == n_classes-1.
    EF_THRESHOLDS: tuple = (30.0, 40.0, 55.0)
    CLASS_NAMES: tuple = (
        "Severe(<30)",
        "Moderate(30-40)",
        "Mild(40-55)",
        "Normal(>=55)",
    )

    # -------------------------------------------------------- video geometry
    FRAME_SIZE: int = 112          # EchoNet native square resolution
    TO_GRAYSCALE: bool = True      # echo is single-channel

    # ------------------------------------------------------- clip sampling
    # These are consumed at TRAINING time by the cardiac-cycle-aware sampler,
    # but are defined here so preprocessing/verification stay consistent.
    # Keep these aligned with training/config.py.  The inclusive temporal
    # coverage is (CLIP_LEN - 1) * SAMPLING_PERIOD = 62 native-frame gaps.
    CLIP_LEN: int = 32             # frames per training clip
    SAMPLING_PERIOD: int = 2       # temporal stride between sampled frames
    MOTION_MODE: str = "tempdiff"  # "none" | "tempdiff" | "flow"

    # ---------------------------------------------------------- denoising
    # Ultrasound speckle handling applied when caching.  Kept OFF by default:
    # decoded frames are stored losslessly so any denoise can be re-derived
    # cheaply at train time; toggle to bake it in.
    DENOISE: str = "none"          # "none" | "median" | "nlm"
    DENOISE_MEDIAN_K: int = 3
    NLM_H: float = 7.0

    # ------------------------------------------------------------- caching
    STORE_MAX_FRAMES: int = 0      # 0 => keep full video; else cap length
    COMPRESS_CACHE: bool = False   # False => .npy (mmap-able, fastest I/O)

    # --------------------------------------------------- imbalance handling
    EFFECTIVE_NUM_BETA: float = 0.9999   # Cui et al. 2019 class re-weighting
    EF_DENSITY_BINS: int = 20            # for balanced-regression sample weights

    # ------------------------------------------------------------- runtime
    NUM_WORKERS: int = max(1, (os.cpu_count() or 4) - 2)
    SEED: int = 1337

    # ---------------------------------------------------------- derived out
    @property
    def DATASET(self) -> Path:      return self.ROOT / "Dataset"
    @property
    def VIDEO_DIR(self) -> Path:    return self.DATASET / "Videos"
    @property
    def FILELIST_CSV(self) -> Path: return self.DATASET / "FileList.csv"
    @property
    def TRACINGS_CSV(self) -> Path: return self.DATASET / "VolumeTracings.csv"

    @property
    def ARTIFACTS(self) -> Path:    return self.PREP_DIR / "artifacts"
    @property
    def CACHE_DIR(self) -> Path:    return self.PREP_DIR / "cache" / "videos"
    @property
    def VIZ_DIR(self) -> Path:      return self.ARTIFACTS / "viz"
    @property
    def LOG_DIR(self) -> Path:      return self.PREP_DIR / "logs"

    # artifact files
    @property
    def AUDIT_JSON(self) -> Path:   return self.ARTIFACTS / "audit_report.json"
    @property
    def VIDEO_INDEX(self) -> Path:  return self.ARTIFACTS / "video_index.csv"
    @property
    def KEYFRAMES_CSV(self) -> Path:return self.ARTIFACTS / "keyframes.csv"
    @property
    def NORM_JSON(self) -> Path:    return self.ARTIFACTS / "norm_stats.json"
    @property
    def MANIFEST(self) -> Path:     return self.ARTIFACTS / "manifest.csv"
    @property
    def VERIFY_JSON(self) -> Path:  return self.ARTIFACTS / "verification_report.json"

    @property
    def N_CLASSES(self) -> int:     return len(self.CLASS_NAMES)

    def ensure_dirs(self) -> None:
        for d in (self.ARTIFACTS, self.CACHE_DIR, self.VIZ_DIR, self.LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)

    def ef_to_class(self, ef: float) -> int:
        """Map a continuous EF value to its 4-class severity label."""
        for i, t in enumerate(self.EF_THRESHOLDS):
            if ef < t:
                return i
        return len(self.EF_THRESHOLDS)


CFG = Config()
