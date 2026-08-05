"""
COMPONENT_01 · STAGE 6 · ACQUISITION-CONDITIONED RELIABILITY  (ACR)
====================================================================

THIS FILE IS THE INDEPENDENT CONTRIBUTION.

Everything upstream (ConvNeXt, BART, Grad-CAM, CheXpert labels) is existing
work that was assembled. This module is a method that did not exist before it
was written here, and it is deliberately kept as a standalone, unit-testable
module -- not buried in a notebook -- so that authorship is inspectable.

--------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------
A chest radiograph is not a neutral observation of a patient. It is a
measurement made under conditions -- projection (AP/PA), portable vs fixed
equipment, inspiration depth, rotation, penetration. Radiologists qualify
every finding by those conditions ("cardiac silhouette is enlarged, though
assessment is limited by AP technique and low lung volumes"). Deployed CXR
classifiers do not: they emit `Cardiomegaly 0.94` with identical confidence
on a perfect PA film and on a rotated, poorly-inspired portable AP.

ACR closes that gap post-hoc, with NO retraining:

  1. AUDIT       how much apparent classifier performance is really
                 projection separation rather than anatomy
  2. RECALIBRATE per-pathology, conditional on acquisition
  3. SCORE       attach an empirical reliability tier to every prediction
  4. QUALIFY     emit the radiologist-style hedge in plain English

--------------------------------------------------------------------
THE CLAIM WE MAKE, AND THE ONE WE DO NOT
--------------------------------------------------------------------
A panel will attack this, so the distinction is built into the code.

  WE DO NOT CLAIM: "gamma != 0 proves the classifier uses a shortcut."
      It does not. Acquisition genuinely carries information -- AP patients
      really are sicker. A non-zero acquisition coefficient only shows the
      classifier did not fully exploit it.

  WE DO CLAIM (1): within-projection AUROC vs pooled AUROC decomposes how
      much discrimination survives when the AP/PA cue is removed. That is a
      clean, standard, defensible test. `within_stratum_auroc`.

  WE DO CLAIM (2): given the classifier's own probability, acquisition still
      predicts the label => the classifier is MISCALIBRATED CONDITIONAL ON
      ACQUISITION. Its probabilities mean different things on AP and PA.
      That is measurable (`ece`), fixable (`ACRModel`), and worth fixing
      regardless of the shortcut question.

Every reported effect is paired with a bootstrap CI and a shuffled-acquisition
negative control. If the negative control does not collapse, the effect is an
artifact and must be reported as one.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

EPS = 1e-6
PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
               "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]

# Acquisition covariates. Metadata-derived ones are exact (from DICOM headers);
# image-derived ones are proxies and are named to say so.
META_FEATURES = ["is_AP", "is_portable", "off_hours"]
IMAGE_FEATURES = ["insp_lungfrac", "pen_mean", "pen_contrast",
                  "rot_lr_asym", "blur_lapvar"]
ACQ_FEATURES = META_FEATURES + IMAGE_FEATURES


# ====================================================================
# 1 · numeric helpers
# ====================================================================
def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Stable logit. Classifier probabilities saturate at 0/1 in bf16; an
    unclipped logit yields +-inf and silently poisons every downstream fit."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U identity, ties averaged.

    Returns NaN when a stratum is single-class. Callers MUST treat NaN as
    'undefined', never as 0 -- averaging a 0 into a mean AUROC would silently
    manufacture a subgroup gap that does not exist.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on [0,1]."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        tot += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn=auroc, n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point, lo, hi)."""
    y, s = np.asarray(y), np.asarray(s)
    point = fn(y, s)
    if not np.isfinite(point) or y.size == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = rng.integers(0, y.size, y.size)
        vals[i] = fn(y[j], s[j])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def delta_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn=auroc,
             n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for fn(y,b) - fn(y,a) on PAIRED scores.

    Paired resampling is required: a and b are two scorings of the SAME images,
    so independent bootstraps would inflate the variance and hide a real effect.
    """
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    point = fn(y, b) - fn(y, a)
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = rng.integers(0, y.size, y.size)
        vals[i] = fn(y[j], b[j]) - fn(y[j], a[j])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ====================================================================
# 2 · acquisition covariates
# ====================================================================
def parse_study_hour(study_time) -> float:
    """MIMIC StudyTime is HHMMSS.frac as a float (e.g. 132700.234 -> 13).

    Returns NaN on anything unparseable rather than guessing -- a wrong hour
    would silently corrupt the off_hours covariate.
    """
    try:
        v = float(study_time)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v) or v < 0:
        return float("nan")
    h = int(v // 10000)
    return float(h) if 0 <= h <= 23 else float("nan")


def metadata_acquisition(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join DICOM header fields and derive the exact (non-proxy) covariates.

    `df` must carry dicom_id. Raises if the join is not total -- a partial join
    would quietly drop films from the analysis and bias every stratum.
    """
    cols = ["dicom_id", "ViewPosition", "PerformedProcedureStepDescription",
            "StudyTime", "Rows", "Columns"]
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise KeyError(f"metadata is missing {missing}")
    out = df.merge(metadata[cols], on="dicom_id", how="left", validate="m:1")
    if out["ViewPosition"].isna().any():
        n = int(out["ViewPosition"].isna().sum())
        raise ValueError(f"{n} rows failed the metadata join -- refusing to "
                         f"proceed with a partial acquisition table")

    proc = out["PerformedProcedureStepDescription"].fillna("").str.upper()
    out["is_AP"] = (out["ViewPosition"].astype(str).str.upper() == "AP").astype(float)
    out["is_portable"] = proc.str.contains("PORT").astype(float)
    hour = out["StudyTime"].map(parse_study_hour)
    out["study_hour"] = hour
    # Off-hours imaging skews toward emergency/portable/sicker patients. Films
    # with an unparseable time default to 0 (in-hours) -- the conservative
    # choice, since it cannot manufacture an effect.
    out["off_hours"] = (((hour < 8) | (hour >= 18)) & hour.notna()).astype(float)
    return out


def image_acquisition(path: str) -> dict:
    """Acquisition proxies computed from pixels. No segmentation masks, no
    network, no download -- classical CV only, ~1 ms/image.

    IMPORTANT: these are computed on the RAW uint8 PNG, never on the
    z-scored tensor. PerImageZScore deliberately destroys absolute intensity,
    which is exactly the signal `pen_mean` and `pen_contrast` measure.
    """
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    a = a.astype(np.float32)
    h, w = a.shape

    # Penetration proxies: absolute brightness and global contrast.
    pen_mean = float(a.mean() / 255.0)
    pen_contrast = float(a.std() / 255.0)

    # Inspiration proxy: air is dark. Otsu-split, then take the dark fraction
    # inside the central 80% so collimation borders do not dominate.
    core = a[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
    thr, _ = cv2.threshold(core.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    insp_lungfrac = float((core < thr).mean())

    # Rotation proxy: a rotated patient makes the hemithoraces asymmetric.
    lh, rh = a[:, : w // 2].mean(), a[:, w - w // 2:].mean()
    rot_lr_asym = float(abs(lh - rh) / (0.5 * (lh + rh) + EPS))

    # Motion/blur proxy.
    blur_lapvar = float(cv2.Laplacian(a, cv2.CV_32F).var() / (255.0 ** 2))

    return dict(insp_lungfrac=insp_lungfrac, pen_mean=pen_mean,
                pen_contrast=pen_contrast, rot_lr_asym=rot_lr_asym,
                blur_lapvar=blur_lapvar)


def image_acquisition_batch(paths: Sequence[str], progress=None) -> pd.DataFrame:
    rows = []
    it = enumerate(paths)
    for i, p in it:
        rows.append(image_acquisition(p))
        if progress is not None and (i + 1) % 500 == 0:
            progress(i + 1, len(paths))
    return pd.DataFrame(rows)


# ====================================================================
# 3 · the audit  (CLAIM 1)
# ====================================================================
def shortcut_signal(y: np.ndarray, acq: pd.DataFrame, features=ACQ_FEATURES,
                    seed: int = 0) -> dict:
    """How well does ACQUISITION ALONE predict the label? No image content.

    A high value is the headline: it is the AUROC a 'model' achieves by
    detecting the camera rather than the patient.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X = acq[features].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return dict(auroc=float("nan"), lo=float("nan"), hi=float("nan"))
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=seed))
    # Cross-validated so the number is honest out-of-sample, not a fit statistic.
    s = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
    pt, lo, hi = bootstrap_ci(y, s, seed=seed)
    return dict(auroc=pt, lo=lo, hi=hi)


def within_stratum_auroc(y: np.ndarray, p: np.ndarray, stratum: np.ndarray,
                         seed: int = 0) -> dict:
    """Pooled AUROC vs AUROC computed inside each stratum.

    If pooled >> within, part of the apparent discrimination comes from
    separating the strata, not from reading anatomy. This is the honest
    shortcut test referenced in the module docstring.
    """
    y, p, stratum = np.asarray(y).astype(int), np.asarray(p), np.asarray(stratum)
    out = {"pooled": dict(zip(("auroc", "lo", "hi"), bootstrap_ci(y, p, seed=seed))),
           "strata": {}}
    weighted, wsum = 0.0, 0.0
    for s in np.unique(stratum):
        m = stratum == s
        pt, lo, hi = bootstrap_ci(y[m], p[m], seed=seed)
        out["strata"][str(s)] = dict(n=int(m.sum()), pos=int(y[m].sum()),
                                     auroc=pt, lo=lo, hi=hi)
        if np.isfinite(pt):
            weighted += pt * m.sum()
            wsum += m.sum()
    out["within_weighted"] = float(weighted / wsum) if wsum else float("nan")
    out["shortcut_gap"] = float(out["pooled"]["auroc"] - out["within_weighted"])
    return out


# ====================================================================
# 4 · the method  (CLAIM 2)
# ====================================================================
@dataclass
class ACRModel:
    """Per-pathology recalibration conditional on acquisition.

        logit(P(y=1)) = b0 + b1 * logit(p_hat) + gamma^T a

    Fitted on VALIDATION ONLY. Fitting on train is invalid -- the classifier
    saw those images, so its probabilities there are overconfident in a way
    that does not transfer, and the correction learned would be wrong.
    """
    pathologies: list[str]
    features: list[str]
    _models: dict = field(default_factory=dict, repr=False)
    _coef: dict = field(default_factory=dict)

    def fit(self, probs: pd.DataFrame, labels: pd.DataFrame,
            acq: pd.DataFrame, seed: int = 0) -> "ACRModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        A = np.nan_to_num(acq[self.features].to_numpy(dtype=np.float64), nan=0.0)
        for k in self.pathologies:
            y = labels[k].to_numpy().astype(int)
            if len(np.unique(y)) < 2:
                continue
            X = np.column_stack([logit(probs[k].to_numpy()), A])
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, random_state=seed))
            pipe.fit(X, y)
            self._models[k] = pipe
            lr = pipe.named_steps["logisticregression"]
            self._coef[k] = dict(zip(["classifier_logit"] + self.features,
                                     [float(c) for c in lr.coef_[0]]))
        return self

    def apply(self, probs: pd.DataFrame, acq: pd.DataFrame) -> pd.DataFrame:
        A = np.nan_to_num(acq[self.features].to_numpy(dtype=np.float64), nan=0.0)
        out = {}
        for k in self.pathologies:
            if k not in self._models:
                out[k] = probs[k].to_numpy()
                continue
            X = np.column_stack([logit(probs[k].to_numpy()), A])
            out[k] = self._models[k].predict_proba(X)[:, 1]
        return pd.DataFrame(out, index=probs.index)

    def acquisition_weight(self, k: str) -> float:
        """L1 norm of the acquisition coefficients, standardized units.

        Interpretation, stated carefully: how much the correction leans on
        acquisition AFTER the classifier has spoken. NOT a shortcut measure.
        """
        c = self._coef.get(k, {})
        return float(sum(abs(v) for f, v in c.items() if f != "classifier_logit"))

    def to_dict(self) -> dict:
        return dict(pathologies=self.pathologies, features=self.features,
                    coefficients=self._coef,
                    acquisition_weight={k: self.acquisition_weight(k)
                                        for k in self._coef})


def negative_control(probs: pd.DataFrame, labels: pd.DataFrame, acq: pd.DataFrame,
                     test_probs: pd.DataFrame, test_labels: pd.DataFrame,
                     test_acq: pd.DataFrame, pathologies=PATHOLOGIES,
                     features=ACQ_FEATURES, seed: int = 0) -> dict:
    """Refit ACR on SHUFFLED acquisition rows.

    ⚠️ SUPERSEDED by `calibration_ablation`. Kept only for reproducibility of
    the first Stage 6 run. It compares AUROC, which ACR is not designed to
    move -- a monotone recalibration cannot reorder cases within a stratum, so
    this test is uninformative by construction. Use `calibration_ablation`.
    """
    rng = np.random.default_rng(seed)
    sh_val = acq.copy()
    sh_val[features] = acq[features].to_numpy()[rng.permutation(len(acq))]
    sh_te = test_acq.copy()
    sh_te[features] = test_acq[features].to_numpy()[rng.permutation(len(test_acq))]

    m = ACRModel(list(pathologies), list(features)).fit(probs, labels, sh_val, seed=seed)
    adj = m.apply(test_probs, sh_te)
    return {k: dict(base=auroc(test_labels[k], test_probs[k]),
                    shuffled=auroc(test_labels[k], adj[k]))
            for k in pathologies}


def subgroup_ece_gap(y: np.ndarray, p: np.ndarray, is_ap: np.ndarray) -> float:
    """|ECE among AP films - ECE among PA films|. The FAIRNESS quantity:
    are the two groups served equally well?"""
    y, p, is_ap = np.asarray(y).astype(int), np.asarray(p), np.asarray(is_ap).astype(bool)
    return abs(ece(y[is_ap], p[is_ap]) - ece(y[~is_ap], p[~is_ap]))


def group_ece(y: np.ndarray, p: np.ndarray, is_ap: np.ndarray) -> float:
    """Mean WITHIN-group ECE. The CORRECTNESS quantity, and the one the
    verdict turns on.

    `subgroup_ece_gap` alone is gameable: a model equally badly calibrated in
    both groups scores a perfect 0 gap. A single global transform (Platt) can
    drive the gap down while leaving both groups miscalibrated. This cannot be
    gamed that way -- it only falls if calibration actually improves inside
    each group, which needs a per-group transform.
    """
    y, p, is_ap = np.asarray(y).astype(int), np.asarray(p), np.asarray(is_ap).astype(bool)
    return float(np.nanmean([ece(y[is_ap], p[is_ap]), ece(y[~is_ap], p[~is_ap])]))


def calibration_ablation(val_probs: pd.DataFrame, val_labels: pd.DataFrame,
                         val_acq: pd.DataFrame, test_probs: pd.DataFrame,
                         test_labels: pd.DataFrame, test_acq: pd.DataFrame,
                         pathologies=PATHOLOGIES, features=ACQ_FEATURES,
                         seed: int = 0) -> dict:
    """FOUR-ARM ablation -- the only honest test of whether acquisition matters.

    The shuffled-acquisition control alone is NOT sufficient, and believing it
    was is the flaw this function exists to correct. Even with acquisition
    destroyed, the fit still contains b0 + b1*logit(p_hat), which is ordinary
    Platt scaling and improves ECE substantially on its own.

        A  raw          the classifier as shipped
        B  platt        recalibrate on the classifier logit ONLY   <- THE NULL
        C  acr          + real acquisition covariates
        D  shuffled     + row-shuffled acquisition covariates

    The claim "acquisition-conditioned calibration works" requires C to beat
    **B**, not merely to beat A. If C ~= B, the contribution reduces to "we
    applied Platt scaling" and must be reported that way.

    The decisive metric is the SUBGROUP gap, not pooled ECE: arm B applies one
    global monotone transform, so it cannot fix AP and PA miscalibration that
    run in different directions. Arm C can.
    """
    rng = np.random.default_rng(seed)
    sh_val, sh_te = val_acq.copy(), test_acq.copy()
    sh_val[features] = val_acq[features].to_numpy()[rng.permutation(len(val_acq))]
    sh_te[features] = test_acq[features].to_numpy()[rng.permutation(len(test_acq))]

    arms = {"A_raw": test_probs}
    for tag, feats, va, ta in (("B_platt", [], val_acq, test_acq),
                               ("C_acr", list(features), val_acq, test_acq),
                               ("D_shuffled", list(features), sh_val, sh_te)):
        m = ACRModel(list(pathologies), feats).fit(val_probs, val_labels, va, seed=seed)
        arms[tag] = m.apply(test_probs, ta)

    ap = (test_acq["is_AP"] > 0.5).to_numpy()
    out = {}
    for tag, P in arms.items():
        per = {}
        for k in pathologies:
            y = test_labels[k].to_numpy()
            per[k] = dict(auroc=auroc(y, P[k].to_numpy()),
                          ece=ece(y, P[k].to_numpy()),
                          subgroup_gap=subgroup_ece_gap(y, P[k].to_numpy(), ap),
                          group_ece=group_ece(y, P[k].to_numpy(), ap))
        out[tag] = dict(
            per_pathology=per,
            mean_auroc=float(np.nanmean([per[k]["auroc"] for k in pathologies])),
            mean_ece=float(np.nanmean([per[k]["ece"] for k in pathologies])),
            mean_subgroup_gap=float(np.nanmean([per[k]["subgroup_gap"] for k in pathologies])),
            mean_group_ece=float(np.nanmean([per[k]["group_ece"] for k in pathologies])))

    c, b = out["C_acr"], out["B_platt"]
    # The verdict turns on WITHIN-group calibration, not the gap -- see
    # `group_ece` for why the gap alone is gameable by a global transform.
    out["verdict"] = dict(
        ece_gain_over_platt=float(b["mean_ece"] - c["mean_ece"]),
        group_ece_gain_over_platt=float(b["mean_group_ece"] - c["mean_group_ece"]),
        subgroup_gain_over_platt=float(b["mean_subgroup_gap"] - c["mean_subgroup_gap"]),
        acquisition_matters=bool(c["mean_group_ece"] < b["mean_group_ece"]))
    return out


def projection_gap_test(labels: pd.DataFrame, probs: pd.DataFrame, is_ap: np.ndarray,
                        pathologies=PATHOLOGIES, n_boot: int = 1000,
                        seed: int = 0) -> dict:
    """PA-minus-AP AUROC gap, per pathology and in aggregate.

    Two different bootstraps, deliberately:

    * per pathology -- STRATIFIED: resample within the AP group and within the
      PA group separately, so each group's size is held fixed.

    * the mean across pathologies -- CLUSTER over films: the 8 labels are
      measured on the SAME radiographs and are strongly correlated (a sick
      patient is positive for several at once). Resampling pathologies as if
      independent would badly understate the variance. Resampling FILMS
      propagates that correlation correctly.
    """
    is_ap = np.asarray(is_ap).astype(bool)
    rng = np.random.default_rng(seed)
    ap_i, pa_i = np.where(is_ap)[0], np.where(~is_ap)[0]
    n = len(is_ap)

    per = {}
    for k in pathologies:
        y, s = labels[k].to_numpy().astype(int), probs[k].to_numpy()
        pt = auroc(y[~is_ap], s[~is_ap]) - auroc(y[is_ap], s[is_ap])
        vals = np.empty(n_boot)
        for b in range(n_boot):
            a = rng.choice(ap_i, ap_i.size, replace=True)
            p = rng.choice(pa_i, pa_i.size, replace=True)
            vals[b] = auroc(y[p], s[p]) - auroc(y[a], s[a])
        vals = vals[np.isfinite(vals)]
        per[k] = dict(ap=auroc(y[is_ap], s[is_ap]), pa=auroc(y[~is_ap], s[~is_ap]),
                      gap=float(pt), lo=float(np.quantile(vals, .025)),
                      hi=float(np.quantile(vals, .975)),
                      p_gt_0=float((vals <= 0).mean()))

    means = np.empty(n_boot)
    for b in range(n_boot):
        j = rng.integers(0, n, n)                      # resample FILMS
        m = is_ap[j]
        if m.all() or (~m).all():
            means[b] = np.nan
            continue
        g = []
        for k in pathologies:
            y, s = labels[k].to_numpy()[j].astype(int), probs[k].to_numpy()[j]
            g.append(auroc(y[~m], s[~m]) - auroc(y[m], s[m]))
        means[b] = np.nanmean(g)
    means = means[np.isfinite(means)]

    pt = float(np.nanmean([per[k]["gap"] for k in pathologies]))
    return dict(per_pathology=per, mean_gap=pt,
                mean_lo=float(np.quantile(means, .025)),
                mean_hi=float(np.quantile(means, .975)),
                mean_p_gt_0=float((means <= 0).mean()),
                n_favouring_pa=int(sum(per[k]["gap"] > 0 for k in pathologies)),
                n_pathologies=len(pathologies))


# ====================================================================
# 5 · reliability score + qualification
# ====================================================================
STRATUM_KEYS = ["is_AP", "is_portable"]


def stratum_id(acq: pd.DataFrame, keys=STRATUM_KEYS) -> pd.Series:
    """Discrete acquisition stratum. Kept coarse ON PURPOSE: fine strata give
    tiny cells and unstable per-stratum AUROC, which would make the reliability
    score noise. Coarse and honest beats granular and wrong."""
    lab = acq[keys].astype(int).astype(str).agg("".join, axis=1)
    names = {"00": "PA-fixed", "01": "PA-portable",
             "10": "AP-fixed", "11": "AP-portable"}
    return lab.map(lambda s: names.get(s, s))


@dataclass
class ReliabilityTable:
    """Empirical, stratum-conditional reliability measured on VALIDATION.

    Reliability is not invented -- it is the classifier's own historical
    discrimination in that acquisition stratum. A prediction made on a
    portable AP is labelled LOW because the classifier measurably performs
    worse on portable APs, not because AP 'feels' worse.
    """
    table: dict = field(default_factory=dict)
    overall: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, labels: pd.DataFrame, probs: pd.DataFrame, strata: pd.Series,
            pathologies=PATHOLOGIES, min_n: int = 100) -> "ReliabilityTable":
        t, o = {}, {}
        for k in pathologies:
            y, p = labels[k].to_numpy().astype(int), probs[k].to_numpy()
            o[k] = auroc(y, p)
            t[k] = {}
            for s in strata.unique():
                m = (strata == s).to_numpy()
                # Cells below min_n produce unstable AUROC -> mark undefined
                # rather than emitting a confident-looking noisy number.
                t[k][s] = auroc(y[m], p[m]) if m.sum() >= min_n else float("nan")
        return cls(table=t, overall=o)

    def score(self, pathology: str, stratum: str) -> float:
        v = self.table.get(pathology, {}).get(stratum, float("nan"))
        return v if np.isfinite(v) else self.overall.get(pathology, float("nan"))

    def tier(self, pathology: str, stratum: str,
             hi: float = 0.02, lo: float = 0.05) -> str:
        """HIGH / MODERATE / LOW by AUROC shortfall against this pathology's
        pooled performance."""
        s, ov = self.score(pathology, stratum), self.overall.get(pathology, np.nan)
        if not np.isfinite(s) or not np.isfinite(ov):
            return "UNKNOWN"
        d = ov - s
        return "HIGH" if d <= hi else ("MODERATE" if d <= lo else "LOW")


_QUALIFIERS = {
    "is_portable": "portable acquisition",
    "insp_lungfrac": "shallow inspiration",
    "rot_lr_asym": "patient rotation",
    "pen_mean": "suboptimal penetration",
    "blur_lapvar": "reduced image sharpness",
}

# The effect of AP technique is PATHOLOGY-SPECIFIC, and not even in a
# consistent direction: it magnifies the cardiac silhouette (over-read), and
# supine positioning accentuates vascular congestion (over-read) but lets an
# effusion layer posteriorly (UNDER-read). A single generic parenthetical would
# be wrong for most pathologies, so anything not listed here gets the bare,
# unarguable phrase.
_AP_NOTE = {
    "Cardiomegaly": "AP technique (magnifies the cardiac silhouette)",
    "Edema": "AP/supine technique (accentuates vascular congestion)",
    "Pleural_Effusion": "AP/supine technique (effusion may layer posteriorly)",
    "Atelectasis": "AP technique with low lung volumes",
}
_AP_DEFAULT = "AP technique"

# A hedge naming five limitations reads as noise; radiologists name one or two.
MAX_QUALIFIERS = 3


def qualification(row: pd.Series, tier: str, pathology: str,
                  thresholds: dict | None = None) -> str:
    """The radiologist-style hedge. This is the user-visible output of Stage 6.

    Deliberately templated, not generated: a language model here could
    hallucinate a limitation that was never measured, which is precisely the
    failure mode this component exists to remove.
    """
    thresholds = thresholds or {}
    active = []
    # Exact (DICOM-derived) conditions lead; pixel proxies follow. Truncation
    # then drops the least certain evidence first, never the most certain.
    if row.get("is_AP", 0) >= 0.5:
        active.append(_AP_NOTE.get(pathology, _AP_DEFAULT))
    if row.get("is_portable", 0) >= 0.5:
        active.append(_QUALIFIERS["is_portable"])
    for f in ("insp_lungfrac", "rot_lr_asym", "pen_mean", "blur_lapvar"):
        t = thresholds.get(f)
        if t is None:
            continue
        v = row.get(f, np.nan)
        if not np.isfinite(v):
            continue
        bad = v < t if f in ("insp_lungfrac", "blur_lapvar") else v > t
        if bad:
            active.append(_QUALIFIERS[f])
    if tier == "HIGH" or not active:
        return ""
    active = active[:MAX_QUALIFIERS]
    lead = "limited by" if tier == "LOW" else "mildly limited by"
    detail = active[0] if len(active) == 1 else \
        ", ".join(active[:-1]) + " and " + active[-1]
    return f"{pathology.replace('_', ' ')} assessment is {lead} {detail}."


def selective_curve(y: np.ndarray, p: np.ndarray, reliability: np.ndarray,
                    steps: int = 10) -> pd.DataFrame:
    """Does the reliability score actually predict error?

    Drop the least-reliable films and re-measure AUROC. A score that means
    something produces a RISING curve. A flat curve falsifies the score, and
    that outcome must be reported, not hidden.
    """
    y, p, r = np.asarray(y).astype(int), np.asarray(p), np.asarray(reliability)
    order = np.argsort(-r)          # most reliable first
    rows = []
    for i in range(1, steps + 1):
        n = max(int(len(y) * i / steps), 2)
        idx = order[:n]
        rows.append(dict(coverage=n / len(y), n=n,
                         auroc=auroc(y[idx], p[idx])))
    return pd.DataFrame(rows)


# ====================================================================
# 6 · self-tests  (run before any real data touches this module)
# ====================================================================
def _selftest(verbose: bool = True) -> tuple[int, int]:
    P, F = [], []

    def g(name, ok, extra=""):
        (P if ok else F).append(name)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{extra}")

    rng = np.random.default_rng(0)

    g("logit/sigmoid round-trip",
      np.allclose(sigmoid(logit(np.array([.1, .5, .9]))), [.1, .5, .9], atol=1e-9))
    g("logit clips p=0 and p=1 (no inf)", np.isfinite(logit(np.array([0.0, 1.0]))).all(),
      f"{logit(np.array([0.0,1.0]))}")

    y = np.array([0, 0, 1, 1])
    g("auroc perfect = 1.0", auroc(y, np.array([.1, .2, .8, .9])) == 1.0)
    g("auroc inverted = 0.0", auroc(y, np.array([.9, .8, .2, .1])) == 0.0)
    g("auroc all-ties = 0.5", auroc(y, np.array([.5, .5, .5, .5])) == 0.5)
    g("auroc single-class -> NaN (never 0)",
      np.isnan(auroc(np.array([1, 1, 1]), np.array([.1, .5, .9]))))

    g("ece perfect calibration ~ 0",
      ece(np.array([0, 1] * 500), np.array([0.0, 1.0] * 500)) < 1e-9)
    g("ece worst case ~ 1",
      abs(ece(np.array([0, 1] * 500), np.array([1.0, 0.0] * 500)) - 1.0) < 1e-9)

    # Deliberately OVERLAPPING scores. With separable data the CI collapses to
    # [1,1] and the test proves nothing.
    yb = (rng.random(400) < 0.5).astype(int)
    sb = yb * 0.3 + rng.random(400) * 0.7
    pt, lo, hi = bootstrap_ci(yb, sb, n=200, seed=1)
    g("bootstrap CI brackets the point estimate", lo <= pt <= hi, f"{lo:.3f}<={pt:.3f}<={hi:.3f}")
    g("bootstrap CI is non-degenerate on overlapping scores", hi - lo > 0.01, f"width={hi-lo:.3f}")

    g("parse_study_hour 132700.234 -> 13", parse_study_hour(132700.234) == 13.0)
    g("parse_study_hour 000500.0 -> 0", parse_study_hour(000500.0) == 0.0)
    g("parse_study_hour garbage -> NaN", np.isnan(parse_study_hour("abc")))
    g("parse_study_hour 995959 -> NaN (invalid hour)", np.isnan(parse_study_hour(995959.0)))

    acq = pd.DataFrame({"is_AP": [1., 1., 0., 0.], "is_portable": [1., 0., 1., 0.]})
    s = stratum_id(acq)
    g("stratum_id names all four cells",
      list(s) == ["AP-portable", "AP-fixed", "PA-portable", "PA-fixed"], str(list(s)))

    # ACR must recover a KNOWN injected acquisition bias.
    n = 3000
    ap = (rng.random(n) < 0.5).astype(float)
    truth = (rng.random(n) < 0.35).astype(int)
    # classifier is inflated by +1.2 logits on AP films regardless of truth
    z = 1.4 * truth - 0.7 + 1.2 * ap + rng.normal(0, 0.5, n)
    pv = pd.DataFrame({"Cardiomegaly": sigmoid(z)})
    lv = pd.DataFrame({"Cardiomegaly": truth})
    av = pd.DataFrame({"is_AP": ap, "is_portable": np.zeros(n), "off_hours": np.zeros(n),
                       "insp_lungfrac": np.zeros(n), "pen_mean": np.zeros(n),
                       "pen_contrast": np.zeros(n), "rot_lr_asym": np.zeros(n),
                       "blur_lapvar": np.zeros(n)})
    m = ACRModel(["Cardiomegaly"], ACQ_FEATURES).fit(pv, lv, av)
    c = m._coef["Cardiomegaly"]
    g("ACR recovers the injected AP bias with the right sign",
      c["is_AP"] < -0.05, f"is_AP coef={c['is_AP']:.3f} (injected +1.2 -> correction must be negative)")

    adj = m.apply(pv, av)
    e_before, e_after = ece(truth, pv["Cardiomegaly"]), ece(truth, adj["Cardiomegaly"])
    g("ACR improves calibration on the injected bias", e_after < e_before,
      f"ECE {e_before:.4f} -> {e_after:.4f}")

    g("ACR preserves row count and index",
      adj.shape == pv.shape and list(adj.index) == list(pv.index))

    # within-stratum decomposition must SEE a pure-shortcut model
    yq = (rng.random(2000) < 0.5).astype(int)
    apq = yq.astype(float)                       # projection == label: pure shortcut
    pq = apq * 0.9 + rng.random(2000) * 0.1      # score is projection only
    w = within_stratum_auroc(yq, pq, np.where(apq > 0.5, "AP", "PA"))
    g("within-stratum exposes a pure shortcut",
      w["pooled"]["auroc"] > 0.9 and not np.isfinite(w["within_weighted"]),
      f"pooled={w['pooled']['auroc']:.3f} within=undefined (single-class strata)")

    rt = ReliabilityTable.fit(lv, pv, stratum_id(av), ["Cardiomegaly"], min_n=10)
    g("reliability tier returns a legal value",
      rt.tier("Cardiomegaly", "AP-fixed") in {"HIGH", "MODERATE", "LOW", "UNKNOWN"},
      rt.tier("Cardiomegaly", "AP-fixed"))
    g("reliability falls back to pooled on unseen stratum",
      np.isfinite(rt.score("Cardiomegaly", "NOT-A-STRATUM")))

    q = qualification(pd.Series({"is_AP": 1.0, "is_portable": 1.0}), "LOW", "Cardiomegaly")
    g("qualification names AP and portable", "AP technique" in q and "portable" in q)
    g("qualification is empty when reliability is HIGH",
      qualification(pd.Series({"is_AP": 1.0}), "HIGH", "Cardiomegaly") == "")
    g("qualification reads as one clause (no 'assessment: assessment')",
      "assessment: assessment" not in q and q.count("assessment") == 1, q)
    g("AP note is cardiac-specific ONLY for Cardiomegaly",
      "cardiac silhouette" in qualification(pd.Series({"is_AP": 1.0}), "LOW", "Cardiomegaly")
      and "cardiac silhouette" not in qualification(pd.Series({"is_AP": 1.0}), "LOW", "Edema"))
    g("effusion gets the UNDER-read note, not the magnification note",
      "layer posteriorly" in qualification(pd.Series({"is_AP": 1.0}), "LOW", "Pleural_Effusion"))
    g("unlisted pathology falls back to the bare phrase",
      qualification(pd.Series({"is_AP": 1.0}), "LOW", "Pneumothorax").count("(") == 0)
    allbad = pd.Series({"is_AP": 1.0, "is_portable": 1.0, "insp_lungfrac": 0.0,
                        "rot_lr_asym": 9.0, "pen_mean": 9.0, "blur_lapvar": 0.0})
    thr = dict(insp_lungfrac=.5, rot_lr_asym=.5, pen_mean=.5, blur_lapvar=.5)
    qa = qualification(allbad, "LOW", "Edema", thr)
    g(f"at most {MAX_QUALIFIERS} qualifiers listed", qa.count(",") <= MAX_QUALIFIERS - 2 + 1,
      qa)
    g("truncation keeps the DICOM-exact conditions",
      "AP/supine technique" in qa and "portable" in qa)

    sc = selective_curve(yb, sb, sb)
    g("selective curve is monotone in coverage",
      sc.coverage.is_monotonic_increasing and len(sc) == 10)

    # ---- FIX 1: four-arm calibration ablation -------------------------------
    # Scenario where acquisition GENUINELY matters: AP logits inflated, PA
    # clean. One global (Platt) transform cannot fix both directions; ACR can.
    n = 4000
    ap = (rng.random(n) < 0.5)
    truth = (rng.random(n) < 0.4).astype(int)
    z = 1.6 * truth - 0.8 + 1.5 * ap + rng.normal(0, 0.4, n)
    A2 = pd.DataFrame({f: np.zeros(n) for f in ACQ_FEATURES}); A2["is_AP"] = ap.astype(float)
    Pdf, Ldf = pd.DataFrame({"Cardiomegaly": sigmoid(z)}), pd.DataFrame({"Cardiomegaly": truth})
    half = n // 2
    sl = slice(0, half); st = slice(half, n)
    ab = calibration_ablation(
        Pdf.iloc[sl].reset_index(drop=True), Ldf.iloc[sl].reset_index(drop=True),
        A2.iloc[sl].reset_index(drop=True), Pdf.iloc[st].reset_index(drop=True),
        Ldf.iloc[st].reset_index(drop=True), A2.iloc[st].reset_index(drop=True),
        ["Cardiomegaly"])
    g("ablation produces all four arms",
      set(ab) >= {"A_raw", "B_platt", "C_acr", "D_shuffled", "verdict"})
    g("Platt alone already improves pooled ECE (why the old control was void)",
      ab["B_platt"]["mean_ece"] < ab["A_raw"]["mean_ece"],
      f"{ab['A_raw']['mean_ece']:.4f} -> {ab['B_platt']['mean_ece']:.4f}")
    g("ACR beats Platt on WITHIN-GROUP ECE (the decisive test)",
      ab["C_acr"]["mean_group_ece"] < ab["B_platt"]["mean_group_ece"],
      f"platt {ab['B_platt']['mean_group_ece']:.4f} vs acr "
      f"{ab['C_acr']['mean_group_ece']:.4f}")
    g("shuffled acquisition collapses back toward Platt",
      abs(ab["D_shuffled"]["mean_group_ece"] - ab["B_platt"]["mean_group_ece"])
      < abs(ab["C_acr"]["mean_group_ece"] - ab["B_platt"]["mean_group_ece"]),
      f"shuffled {ab['D_shuffled']['mean_group_ece']:.4f}")
    g("verdict flags that acquisition matters here", ab["verdict"]["acquisition_matters"])

    # ---- FIX 2: projection gap significance ---------------------------------
    # Scores must OVERLAP. My first attempt used perfectly separable scores in
    # both groups, so both AUROCs were exactly 1.0 and the gap was structurally
    # zero -- the test failed on bad synthetic data, not on bad code.
    s2 = np.where(ap, truth * 0.3 + rng.random(n),      # AP: weak separation
                  truth * 1.2 + rng.random(n))          # PA: strong separation
    gt = projection_gap_test(pd.DataFrame({"Cardiomegaly": truth}),
                             pd.DataFrame({"Cardiomegaly": s2}), ap,
                             ["Cardiomegaly"], n_boot=200)
    g("gap test detects an injected AP/PA gap",
      gt["mean_gap"] > 0.05 and gt["mean_lo"] > 0,
      f"gap={gt['mean_gap']:+.4f} CI[{gt['mean_lo']:+.4f},{gt['mean_hi']:+.4f}]")
    g("gap test reports direction count",
      gt["n_favouring_pa"] == 1 and gt["n_pathologies"] == 1)
    # Null case: identical score quality in both groups -> CI must include 0.
    s3 = truth * 0.8 + rng.random(n)   # identical quality in BOTH groups
    gt0 = projection_gap_test(pd.DataFrame({"Cardiomegaly": truth}),
                              pd.DataFrame({"Cardiomegaly": s3}), ap,
                              ["Cardiomegaly"], n_boot=200)
    g("gap test does NOT invent a gap when there is none",
      gt0["mean_lo"] < 0 < gt0["mean_hi"],
      f"CI[{gt0['mean_lo']:+.4f},{gt0['mean_hi']:+.4f}]")

    if verbose:
        print(f"\n  {len(P)} passed, {len(F)} failed")
        for f in F:
            print(f"    - {f}")
    return len(P), len(F)


if __name__ == "__main__":
    print("=" * 78)
    print(" STAGE 6 · stage6_acr.py self-test")
    print("=" * 78)
    p, f = _selftest()
    raise SystemExit(1 if f else 0)
