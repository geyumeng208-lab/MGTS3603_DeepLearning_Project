from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import parse_optional_float, parse_optional_int, split_indices_by_user


class DataPipelineTest(unittest.TestCase):
    def test_user_split_has_no_overlap(self) -> None:
        samples = [
            {"user_id": 1},
            {"user_id": 1},
            {"user_id": 2},
            {"user_id": 2},
            {"user_id": 3},
            {"user_id": 4},
        ]
        train_indices, valid_indices = split_indices_by_user(samples, valid_ratio=0.5, seed=2026)
        train_users = {samples[idx]["user_id"] for idx in train_indices}
        valid_users = {samples[idx]["user_id"] for idx in valid_indices}
        self.assertTrue(train_users)
        self.assertTrue(valid_users)
        self.assertTrue(train_users.isdisjoint(valid_users))

    def test_optional_feature_parsers_are_robust(self) -> None:
        self.assertEqual(parse_optional_int("3.0"), 3)
        self.assertEqual(parse_optional_int(""), 0)
        self.assertEqual(parse_optional_int("missing"), 0)
        self.assertAlmostEqual(parse_optional_float("12.5"), 12.5)
        self.assertEqual(parse_optional_float(""), 0.0)
        self.assertEqual(parse_optional_float("missing"), 0.0)


if __name__ == "__main__":
    unittest.main()
