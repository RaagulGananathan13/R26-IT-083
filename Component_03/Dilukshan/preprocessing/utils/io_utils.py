"""
Robust video I/O for EchoNet AVIs.

Everything funnels through OpenCV (PyAV not required).  All decoders return
grayscale uint8 arrays shaped (T, H, W) so downstream stages never worry about
colour channels or backend quirks.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


def probe_video(path: Path) -> dict:
    """
    Cheap metadata probe using container properties (no full decode).
    Returns keys: ok, n_frames_prop, width, height, fps, first_frame_ok.
    """
    info = dict(ok=False, n_frames_prop=0, width=0, height=0,
                fps=0.0, first_frame_ok=False, error="")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "cannot_open"
            return info
        info["n_frames_prop"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        ok, frame = cap.read()
        info["first_frame_ok"] = bool(ok and frame is not None)
        info["ok"] = info["first_frame_ok"]
        if not info["ok"]:
            info["error"] = "cannot_read_first_frame"
    finally:
        cap.release()
    return info


def decode_video(
    path: Path,
    size: Optional[int] = None,
    grayscale: bool = True,
    max_frames: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Fully decode a video to a (T, H, W) uint8 array.

    Parameters
    ----------
    size       : if given, every frame is resized to (size, size).
    grayscale  : convert BGR -> single-channel luma.
    max_frames : 0 keeps all frames; otherwise stop after this many.

    Returns
    -------
    frames : np.ndarray uint8, shape (T, H, W) if grayscale else (T, H, W, 3).
    meta   : dict with true decoded counts and resize flag.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    resized = False
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if size is not None and (frame.shape[0] != size or frame.shape[1] != size):
                interp = cv2.INTER_AREA if frame.shape[0] > size else cv2.INTER_CUBIC
                frame = cv2.resize(frame, (size, size), interpolation=interp)
                resized = True
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if len(frames) == 0:
        raise IOError(f"decoded 0 frames: {path}")

    arr = np.stack(frames, axis=0).astype(np.uint8)
    meta = dict(n_frames=arr.shape[0], height=arr.shape[1],
                width=arr.shape[2], resized=resized)
    return arr, meta


def save_clip(arr: np.ndarray, path, compress: bool = False) -> None:
    """Atomically persist a clip as mmap-able ``.npy`` or compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(".npz" if compress else ".npy")
    # A failed/interrupted writer must not leave a file that --resume later
    # mistakes for a valid cache.  os.replace is atomic on the same volume.
    tmp = target.with_name(
        f".{target.stem}.{os.getpid()}.tmp{target.suffix}")
    try:
        if compress:
            np.savez_compressed(tmp, video=arr)
        else:
            np.save(tmp, arr)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def resolve_cache_path(path: Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve an absolute or portable manifest cache path."""
    raw = str(path).strip()
    if raw in {"", "nan", "NaN", "None"}:
        raise ValueError("cache path is empty")
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def existing_clip_path(path: Path, base_dir: Optional[Path] = None) -> Path:
    """Return the existing ``.npy``/``.npz`` cache path, if either exists."""
    resolved = resolve_cache_path(path, base_dir=base_dir)
    if resolved.suffix.lower() in {".npy", ".npz"}:
        candidates = [resolved]
    else:
        candidates = [resolved.with_suffix(".npy"), resolved.with_suffix(".npz")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Preserve the requested path in the eventual FileNotFoundError.
    return candidates[0]


def cached_clip_shape(path: Path, base_dir: Optional[Path] = None) -> tuple:
    """Read and validate cached clip metadata, including compressed caches."""
    resolved = existing_clip_path(path, base_dir=base_dir)
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as z:
            if "video" not in z.files:
                raise ValueError(f"compressed cache has no 'video' array: {resolved}")
            return tuple(z["video"].shape)
    return tuple(np.load(resolved, mmap_mode="r", allow_pickle=False).shape)


def load_clip(
    path: Path,
    mmap: bool = True,
    base_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load an absolute or portable ``.npy``/``.npz`` cached clip."""
    resolved = existing_clip_path(path, base_dir=base_dir)
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as z:
            if "video" not in z.files:
                raise ValueError(f"compressed cache has no 'video' array: {resolved}")
            return z["video"]
    return np.load(
        resolved, mmap_mode="r" if mmap else None, allow_pickle=False)
