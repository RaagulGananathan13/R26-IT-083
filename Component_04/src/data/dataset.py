"""
Component 04 — split-aware dataset assembly.

Anything that must be *learned* from data (the TF-IDF vocabulary, the SVD
basis) is fitted on the training fold only and then applied to val/test.
Fitting a vectoriser on the pooled corpus is a quiet transductive leak; it is
avoided here by construction.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, LABEL_MAP, MODEL_DIR, load_json
import text_features as TF
from utils import kv, section

FOLDS = ("train", "val", "test")


@dataclass
class Bundle:
    """One horizon, one cohort setting, split three ways."""
    X: Dict[str, pd.DataFrame]
    y: Dict[str, np.ndarray]          # 4-class label 0..3
    meta: Dict[str, pd.DataFrame]
    features: List[str]
    modality: Dict[str, str]
    horizon: int
    embedder: TF.TextEmbedder | None

    @property
    def n_features(self) -> int:
        return len(self.features)

    def binary(self, fold: str) -> np.ndarray:
        return (self.y[fold] > 0).astype(int)

    def acs_only(self, fold: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Rows with a true ACS label, relabelled 0=UA 1=NSTEMI 2=STEMI."""
        m = self.y[fold] > 0
        return self.X[fold][m], (self.y[fold][m] - 1), self.meta[fold][m]

    def groups(self, fold: str) -> np.ndarray:
        return self.meta[fold]["subject_id"].to_numpy()


def load_bundle(horizon: int | None = None, cohort_only: bool = True,
                use_text_svd: bool = True, verbose: bool = True) -> Bundle:
    horizon = CFG.primary_horizon if horizon is None else horizon
    fpath = os.path.join(DATA_DIR, f"features_H{horizon}.parquet")
    ipath = os.path.join(DATA_DIR, f"features_H{horizon}_info.json")
    spath = os.path.join(DATA_DIR, "split_assignment.parquet")
    for p in (fpath, ipath, spath):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} — run preprocess.py then split.py first")

    info = load_json(ipath)
    df = pd.read_parquet(fpath)
    split = pd.read_parquet(spath)
    df = df.merge(split[["stay_id", "fold"]], on="stay_id", how="inner")

    if cohort_only:
        df = df[df.in_cohort == 1].reset_index(drop=True)

    base_features: List[str] = list(info["feature_names"])
    modality: Dict[str, str] = dict(info["modality_map"])

    meta_cols = ["subject_id", "hadm_id", "stay_id", "acs_label", "intime",
                 "ed_los_h", "in_cohort", "chiefcomplaint_raw",
                 "chiefcomplaint_norm", "chiefcomplaint_model", "fold"]

    X: Dict[str, pd.DataFrame] = {}
    y: Dict[str, np.ndarray] = {}
    meta: Dict[str, pd.DataFrame] = {}
    for f in FOLDS:
        d = df[df.fold == f].reset_index(drop=True)
        X[f] = d[base_features].copy()
        y[f] = d["acs_label"].to_numpy(dtype=int)
        meta[f] = d[[c for c in meta_cols if c in d.columns]].copy()

    # ---- text embedding: fit on TRAIN ONLY -------------------------------
    embedder = None
    if use_text_svd:
        embedder = TF.TextEmbedder(
            n_components=int(CFG.get("text.svd_components", 24)),
            word_max=int(CFG.get("text.tfidf_word_max_features", 6000)),
            char_max=int(CFG.get("text.tfidf_char_max_features", 6000)),
            min_df=int(CFG.get("text.min_df", 3)),
            seed=CFG.seed,
        )
        embedder.fit(meta["train"]["chiefcomplaint_model"])
        for f in FOLDS:
            Z = embedder.transform(meta[f]["chiefcomplaint_model"])
            Z.index = X[f].index
            X[f] = pd.concat([X[f], Z], axis=1)
        for c in X["train"].columns:
            if c.startswith("cc_svd_"):
                modality[c] = "text"
        joblib.dump(embedder, os.path.join(MODEL_DIR, f"text_embedder_H{horizon}.joblib"))

    features = list(X["train"].columns)
    for f in FOLDS:
        X[f] = X[f][features].astype(np.float32)

    if verbose:
        section(f"Dataset  H={horizon}h  cohort_only={cohort_only}")
        kv("features", f"{len(features)} "
                       f"({len(base_features)} engineered + "
                       f"{len(features)-len(base_features)} text-SVD)")
        if embedder is not None:
            kv("SVD explained variance", f"{embedder.explained_variance*100:.1f}%")
        print(f"\n  {'fold':<8}{'n':>9}{'patients':>10}" +
              "".join(f"{LABEL_MAP[k]:>9}" for k in sorted(LABEL_MAP)))
        print("  " + "-" * 63)
        for f in FOLDS:
            print(f"  {f:<8}{len(y[f]):>9,}{meta[f].subject_id.nunique():>10,}" +
                  "".join(f"{int((y[f]==k).sum()):>9,}" for k in sorted(LABEL_MAP)))
        # guard: the split must remain patient-disjoint after every filter
        s = {f: set(meta[f].subject_id) for f in FOLDS}
        assert not (s["train"] & s["test"]) and not (s["train"] & s["val"]) \
            and not (s["val"] & s["test"]), "patient leakage after filtering"
        kv("\n  patient-disjoint check", "PASS")

    return Bundle(X=X, y=y, meta=meta, features=features, modality=modality,
                  horizon=horizon, embedder=embedder)


def modality_groups(bundle: Bundle) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in bundle.features:
        out.setdefault(bundle.modality.get(c, "other"), []).append(c)
    return out
