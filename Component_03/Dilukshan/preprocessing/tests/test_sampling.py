from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PREP = Path(__file__).resolve().parents[1]
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))

from utils.sampling import sample_indices, temporal_difference  # noqa: E402


class SampleIndicesTests(unittest.TestCase):
    def test_feasible_cycle_is_contained_at_exact_stride(self):
        for seed in range(20):
            idx = sample_indices(
                n_frames=200, clip_len=32, period=2,
                ed_frame=50, es_frame=100, train=True,
                rng=np.random.default_rng(seed))
            self.assertEqual(len(idx), 32)
            np.testing.assert_array_equal(np.diff(idx), np.full(31, 2))
            self.assertLessEqual(int(idx[0]), 50)
            self.assertGreaterEqual(int(idx[-1]), 100)

    def test_label_free_eval_views_are_deterministic_and_cover_video(self):
        views = [sample_indices(
            200, 32, 2, train=False, view_index=v, n_views=5)
            for v in range(5)]
        self.assertEqual(int(views[0][0]), 0)
        self.assertEqual(int(views[-1][-1]), 199)
        self.assertGreater(len({tuple(v) for v in views}), 1)
        again = sample_indices(200, 32, 2, train=False, view_index=2, n_views=5)
        np.testing.assert_array_equal(views[2], again)

    def test_short_video_is_monotonic_without_modulo_wrap(self):
        idx = sample_indices(17, 32, 2, train=False)
        self.assertEqual(int(idx[0]), 0)
        self.assertEqual(int(idx[-1]), 16)
        self.assertTrue(np.all(np.diff(idx) >= 0))
        self.assertTrue(np.all((idx >= 0) & (idx < 17)))

    def test_long_transition_is_not_falsely_claimed_as_contained(self):
        first = sample_indices(
            200, 32, 2, 20, 120, train=False, view_index=0, n_views=5)
        last = sample_indices(
            200, 32, 2, 20, 120, train=False, view_index=4, n_views=5)
        np.testing.assert_array_equal(np.diff(first), np.full(31, 2))
        self.assertEqual(int(first[0]), 20)
        self.assertLess(int(first[-1]), 120)
        self.assertGreater(int(last[0]), 20)
        self.assertEqual(int(last[-1]), 120)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            sample_indices(0, 32, 2)
        with self.assertRaises(ValueError):
            sample_indices(100, 32, 0)
        with self.assertRaises(ValueError):
            sample_indices(100, 32, 2, train=False, view_index=5, n_views=5)

    def test_singleton_temporal_difference_is_zero(self):
        diff = temporal_difference(np.ones((1, 3, 3), dtype=np.float32))
        np.testing.assert_array_equal(diff, np.zeros_like(diff))


if __name__ == "__main__":
    unittest.main()
