"""
Geometry helpers for the LV tracings in VolumeTracings.csv.

Each traced frame is stored as a set of line segments (X1,Y1)-(X2,Y2):
the first segment is the LV long axis, the remaining ~20 are short-axis
chords (Simpson's method-of-disks).  Chaining the P1 endpoints down one wall
and the P2 endpoints back up the other wall yields the closed LV contour,
whose shoelace area is a faithful proxy for chamber volume.  The larger-area
traced frame is End-Diastole (ED); the smaller is End-Systole (ES).
"""
from __future__ import annotations
import numpy as np


# Increment whenever keyframe-defining contour geometry changes.  Stage 2
# records this in keyframes.csv so verification can reject stale artifacts.
GEOMETRY_VERSION = "echonet_dynamic_no_long_axis_v1"


def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """Absolute polygon area via the shoelace formula."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def tracing_polygon(x1, y1, x2, y2):
    """
    Build the closed LV contour from the four coordinate columns of one
    traced frame.  The first coordinate pair is the LV long axis and is not a
    boundary chord; EchoNet-Dynamic's official mask construction therefore
    uses ``x1[1:]`` and reversed ``x2[1:]`` (and likewise for y).  Returns
    ``(px, py)`` ordered around the polygon.
    """
    x1 = np.asarray(x1, dtype=np.float64); y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64); y2 = np.asarray(y2, dtype=np.float64)
    lengths = {len(x1), len(y1), len(x2), len(y2)}
    if len(lengths) != 1:
        raise ValueError("tracing coordinate columns must have equal lengths")
    if len(x1) < 3:
        raise ValueError(
            "a tracing requires one long-axis pair and at least two boundary chords")
    if not all(np.isfinite(v).all() for v in (x1, y1, x2, y2)):
        raise ValueError("tracing coordinates must be finite")
    px = np.concatenate([x1[1:], x2[1:][::-1]])
    py = np.concatenate([y1[1:], y2[1:][::-1]])
    return px, py


def tracing_area(x1, y1, x2, y2) -> float:
    """Enclosed LV area for one traced frame (relative volume proxy)."""
    px, py = tracing_polygon(x1, y1, x2, y2)
    return shoelace_area(px, py)
