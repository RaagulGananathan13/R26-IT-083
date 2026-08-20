"""
Lightweight visualization helpers for verification montages.
Uses only OpenCV so there is no matplotlib dependency in the hot path.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def montage(frames: np.ndarray, cols: int = 8, pad: int = 2,
            upscale: int = 2) -> np.ndarray:
    """Tile a (N,H,W) uint8 stack into a single grid image."""
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"frames must have non-empty shape (N,H,W), got {frames.shape}")
    if cols <= 0:
        raise ValueError(f"cols must be positive, got {cols}")
    n, h, w = frames.shape
    rows = int(np.ceil(n / cols))
    canvas = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad),
                      dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = frames[i]
    if upscale > 1:
        canvas = cv2.resize(canvas, None, fx=upscale, fy=upscale,
                            interpolation=cv2.INTER_NEAREST)
    return canvas


def draw_contour(frame_gray: np.ndarray, px, py,
                 color=(0, 255, 0)) -> np.ndarray:
    """Overlay an LV contour polygon on a grayscale frame (returns BGR)."""
    if frame_gray.ndim != 2:
        raise ValueError(f"frame_gray must have shape (H,W), got {frame_gray.shape}")
    bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    pts = np.stack([np.asarray(px), np.asarray(py)], axis=1).astype(np.int32)
    if len(pts) < 3:
        raise ValueError("at least three contour points are required")
    cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=1,
                  lineType=cv2.LINE_AA)
    return bgr


def save_image(img: np.ndarray, path: Path) -> None:
    """Write an image and fail loudly if OpenCV cannot encode or persist it."""
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("cannot save an empty image")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok or not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"failed to write image: {path}")
