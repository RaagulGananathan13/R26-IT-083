from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

PREP = Path(__file__).resolve().parents[1]
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))

from config import CFG  # noqa: E402
from stage4_cache_clips import (  # noqa: E402
    _merge_cache_results, _portable_cache_path, _process_one)
from utils.io_utils import load_clip, save_clip  # noqa: E402


class CacheRobustnessTests(unittest.TestCase):
    def test_limited_result_overlay_preserves_other_rows(self):
        manifest = pd.DataFrame([
            dict(FileName="A", cache_path="old_a.npy", n_frames_cached=4,
                 cached_ok=True),
            dict(FileName="B", cache_path="old_b.npy", n_frames_cached=7,
                 cached_ok=True),
        ])
        result = pd.DataFrame([
            dict(FileName="A", cache_path=str(CFG.CACHE_DIR / "A.npy"),
                 n_frames_cached=9, ok=True),
        ])
        updated = _merge_cache_results(manifest, result)
        row_b = updated[updated["FileName"] == "B"].iloc[0]
        self.assertEqual(row_b["cache_path"], "old_b.npy")
        self.assertEqual(int(row_b["n_frames_cached"]), 7)
        self.assertTrue(bool(row_b["cached_ok"]))
        self.assertEqual(
            updated.loc[updated["FileName"] == "A", "cache_path"].iloc[0],
            "cache/videos/A.npy")

    def test_failed_result_does_not_erase_previous_valid_metadata(self):
        manifest = pd.DataFrame([dict(
            FileName="A", cache_path="old.npy", n_frames_cached=4, cached_ok=True)])
        result = pd.DataFrame([dict(
            FileName="A", cache_path="", n_frames_cached=0, ok=False)])
        updated = _merge_cache_results(manifest, result)
        self.assertEqual(updated.iloc[0]["cache_path"], "old.npy")
        self.assertEqual(int(updated.iloc[0]["n_frames_cached"]), 4)

    def test_compressed_resume_records_true_frame_count(self):
        with tempfile.TemporaryDirectory(dir=PREP / "tests") as temp:
            cache_dir = Path(temp) / "cache"
            cache_dir.mkdir()
            clip = np.zeros((3, 112, 112), dtype=np.uint8)
            save_clip(clip, cache_dir / "A", compress=True)
            result = _process_one((
                "A", "TRAIN", str(Path(temp) / "missing_videos"), str(cache_dir),
                112, "none", 3, 7.0, True, 0, True))
            self.assertTrue(result["ok"])
            self.assertTrue(result["skipped"])
            self.assertEqual(result["n_frames_cached"], 3)
            loaded = load_clip(cache_dir / "A.npz")
            np.testing.assert_array_equal(loaded, clip)

    def test_portable_path_is_relative_to_preprocessing_root(self):
        path = CFG.PREP_DIR / "cache" / "videos" / "sample.npy"
        self.assertEqual(_portable_cache_path(str(path)),
                         "cache/videos/sample.npy")


if __name__ == "__main__":
    unittest.main()
