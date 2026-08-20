"""
Cardiac-cycle-aware clip sampling + motion-channel construction.
================================================================

NOVELTY (methodological contribution for PP2)
---------------------------------------------
1. Cycle-aware sampling.  EF is defined by the ED->ES volume change, so a clip
   that misses the systolic contraction is uninformative.  Instead of blind
   random windows, we constrain each clip to contain the complete ED<->ES
   transition whenever it fits the configured temporal span.  A transition
   longer than that span is sampled in contiguous, stride-preserving sections
   and is explicitly reported by stage-5 verification; it is never presented
   as a successful full-transition sample.  Videos without tracings fall back
   to uniform sampling.

2. Motion channel.  Wall motion is the physical substrate of EF.  We augment
   the grayscale clip with a temporal-gradient channel (frame differencing) or
   Farneback optical-flow magnitude, giving the network an explicit motion cue
   without any extra labels.

Both routines are pure functions so they are reused identically by the training
Dataset and by stage-5 verification -> no train/preprocess skew.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2


# --------------------------------------------------------------------------- #
#  Temporal sampling                                                          #
# --------------------------------------------------------------------------- #
def sample_indices(
    n_frames: int,
    clip_len: int,
    period: int = 1,
    ed_frame: Optional[int] = None,
    es_frame: Optional[int] = None,
    train: bool = True,
    rng: Optional[np.random.Generator] = None,
    view_index: Optional[int] = None,
    n_views: int = 1,
) -> np.ndarray:
    """
    Choose `clip_len` frame indices with stride `period`.

    If valid ED/ES key frames are provided and their separation fits the
    configured span, the window start is constrained so the returned index
    range contains both frames.  If the transition is longer than the span,
    a best-effort window is placed within the transition without changing the
    requested stride.

    Training starts are sampled randomly.  Evaluation is deterministic:
    ``view_index`` in ``[0, n_views)`` places multiple views evenly across the
    valid start interval (or returns its centre when no view index is given).

    A short video that cannot support ``clip_len`` samples at ``period`` is
    sampled monotonically from first to last frame, with repeated boundary-
    adjacent indices where necessary.  This avoids modulo wraparound, which
    would introduce an artificial last-frame -> first-frame motion jump.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if n_views <= 0:
        raise ValueError(f"n_views must be positive, got {n_views}")
    if view_index is not None and not 0 <= int(view_index) < n_views:
        raise ValueError(
            f"view_index must be in [0, {n_views}), got {view_index}")

    rng = rng if rng is not None else np.random.default_rng()

    # The inclusive index range is [start, start + coverage].  For example,
    # 32 samples at period 2 cover indices 0..62, not 0..64.
    coverage = (clip_len - 1) * period

    if n_frames <= coverage:
        # The requested cadence cannot fit.  Preserve temporal order and cover
        # the available recording once rather than wrapping back to frame 0.
        return np.rint(np.linspace(0, n_frames - 1, clip_len)).astype(np.int64)

    min_start, max_start = 0, n_frames - 1 - coverage

    def _valid_keyframe(value: Optional[int]) -> bool:
        if value is None:
            return False
        try:
            return bool(np.isfinite(value) and 0 <= int(value) < n_frames)
        except (TypeError, ValueError, OverflowError):
            return False

    has_keyframes = _valid_keyframe(ed_frame) and _valid_keyframe(es_frame)
    if has_keyframes:
        a, b = sorted((int(ed_frame), int(es_frame)))
        if b - a <= coverage:
            # Exact range containment: start <= a and start+coverage >= b.
            start_lo = max(min_start, b - coverage)
            start_hi = min(max_start, a)
        else:
            # No fixed-period window can contain both frames.  Move a
            # stride-preserving subwindow through the transition: the first
            # deterministic view starts at/near ED and the last ends at/near
            # ES, while a single view is centred between those positions.
            start_lo = int(np.clip(a, min_start, max_start))
            start_hi = int(np.clip(b - coverage, min_start, max_start))
            if start_lo > start_hi:
                start_lo, start_hi = start_hi, start_lo
    else:
        start_lo, start_hi = min_start, max_start

    if train:
        start = int(rng.integers(start_lo, start_hi + 1))
    elif view_index is None or n_views == 1:
        start = (start_lo + start_hi) // 2
    else:
        # Integer rounding can duplicate views when fewer valid starts than
        # requested views exist; that is unavoidable and remains deterministic.
        fraction = int(view_index) / (n_views - 1)
        start = int(round(start_lo + fraction * (start_hi - start_lo)))

    return (start + np.arange(clip_len, dtype=np.int64) * period).astype(np.int64)


# --------------------------------------------------------------------------- #
#  Motion channel                                                             #
# --------------------------------------------------------------------------- #
def temporal_difference(clip: np.ndarray) -> np.ndarray:
    """
    Signed temporal gradient of a (T,H,W) float clip, same length as input
    (first frame difference is replicated).  Output roughly in [-1, 1] scale
    if the input is in [0,1].
    """
    clip = clip.astype(np.float32)
    if clip.ndim != 3 or clip.shape[0] == 0:
        raise ValueError(f"clip must have shape (T,H,W) with T > 0, got {clip.shape}")
    diff = np.empty_like(clip)
    if clip.shape[0] == 1:
        diff[0] = 0.0
        return diff
    diff[1:] = clip[1:] - clip[:-1]
    diff[0] = diff[1]
    return diff


def optical_flow_magnitude(clip_uint8: np.ndarray) -> np.ndarray:
    """
    Farneback optical-flow magnitude per frame for a (T,H,W) uint8 clip.
    Heavier than temporal differencing; use when MOTION_MODE == 'flow'.
    """
    if clip_uint8.ndim != 3 or clip_uint8.shape[0] == 0:
        raise ValueError(
            f"clip_uint8 must have shape (T,H,W) with T > 0, got {clip_uint8.shape}")
    T = clip_uint8.shape[0]
    out = np.zeros(clip_uint8.shape, dtype=np.float32)
    if T == 1:
        return out
    prev = clip_uint8[0]
    for t in range(1, T):
        cur = clip_uint8[t]
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out[t] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        prev = cur
    out[0] = out[1]
    return out


def build_multichannel(
    clip_uint8: np.ndarray,
    norm_mean: float,
    norm_std: float,
    motion_mode: str = "tempdiff",
) -> np.ndarray:
    """
    Convert a (T,H,W) uint8 clip into a normalised (C,T,H,W) float32 tensor.
    Channel 0 = standardized grayscale.  Channel 1 (optional) = motion.
    """
    g = clip_uint8.astype(np.float32) / 255.0
    g_norm = (g - norm_mean) / (norm_std + 1e-6)

    if motion_mode == "none":
        return g_norm[None]                        # (1,T,H,W)

    if motion_mode == "tempdiff":
        m = temporal_difference(g)                 # already in [0,1] scale
    elif motion_mode == "flow":
        m = optical_flow_magnitude(clip_uint8)
        m = m / (m.std() + 1e-6)
    else:
        raise ValueError(f"unknown motion_mode: {motion_mode}")

    return np.stack([g_norm, m.astype(np.float32)], axis=0)  # (2,T,H,W)
