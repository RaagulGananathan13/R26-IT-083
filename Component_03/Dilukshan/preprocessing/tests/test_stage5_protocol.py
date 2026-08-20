from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

PREP = Path(__file__).resolve().parents[1]
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))

from stage5_verify import (  # noqa: E402
    _safe_slug, _verify_label_free_eval_sampler, _verify_train_cycle_sampler)


class Stage5ProtocolTests(unittest.TestCase):
    def test_train_sampler_separates_containable_and_long_transitions(self):
        rows = pd.DataFrame([
            dict(FileName="fit", Split="TRAIN", n_frames_verified=150,
                 ed_frame=20, es_frame=60),
            dict(FileName="long", Split="TRAIN", n_frames_verified=150,
                 ed_frame=10, es_frame=100),
        ])
        result = _verify_train_cycle_sampler(rows, clip_len=32, period=2, views=5)
        self.assertEqual(result["n_containable"], 1)
        self.assertEqual(result["n_uncontainable"], 1)
        self.assertEqual(result["containment_rate"], 1.0)
        self.assertEqual(result["index_failures"], 0)

    def test_eval_sampler_is_uniform_and_does_not_need_keyframes(self):
        rows = pd.DataFrame([
            dict(FileName="val", Split="VAL", n_frames_verified=200,
                 ed_frame=9999, es_frame=9999),
            dict(FileName="test", Split="TEST", n_frames_verified=17,
                 ed_frame=-1, es_frame=-1),
        ])
        result = _verify_label_free_eval_sampler(
            rows, clip_len=32, period=2, views=5)
        self.assertEqual(result["n_eval_videos"], 2)
        self.assertEqual(result["n_index_failures"], 0)
        self.assertEqual(result["uniform_coverage_rate"], 1.0)
        self.assertEqual(result["diversity_rate"], 1.0)

    def test_visual_filename_slug_is_windows_safe(self):
        slug = _safe_slug("Severe(<30): A/B?*")
        self.assertEqual(slug, "Severe_30_A_B")
        self.assertFalse(any(char in slug for char in '<>:"/\\|?*'))


if __name__ == "__main__":
    unittest.main()
