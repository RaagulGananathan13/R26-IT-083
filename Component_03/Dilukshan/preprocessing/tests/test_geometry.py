from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PREP = Path(__file__).resolve().parents[1]
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))

from utils.geometry import tracing_area, tracing_polygon  # noqa: E402


class TracingGeometryTests(unittest.TestCase):
    def test_official_long_axis_row_is_excluded(self):
        # The extreme first pair represents the long axis and would dominate
        # the polygon if a regression accidentally included it.
        x1 = np.array([1000.0, 0.0, 0.0])
        y1 = np.array([1000.0, 0.0, 1.0])
        x2 = np.array([-1000.0, 1.0, 1.0])
        y2 = np.array([-1000.0, 0.0, 1.0])
        px, py = tracing_polygon(x1, y1, x2, y2)
        np.testing.assert_array_equal(px, [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(py, [0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(tracing_area(x1, y1, x2, y2), 1.0)

    def test_malformed_coordinates_fail_closed(self):
        with self.assertRaises(ValueError):
            tracing_polygon([0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2])
        with self.assertRaises(ValueError):
            tracing_polygon([0, 1], [0, 1], [0, 1], [0, 1])


if __name__ == "__main__":
    unittest.main()
