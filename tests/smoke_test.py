from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FieldDims
from src.models import build_model
from src.utils import Config


class ModelSmokeTest(unittest.TestCase):
    def test_all_models_forward(self) -> None:
        field_dims = FieldDims(num_users=16, num_items=64, num_categories=8)
        batch = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([10, 20]),
            "cate_id": torch.tensor([3, 4]),
            "label": torch.tensor([1.0, 0.0]),
            "hist_item_ids": torch.tensor([[10, 11, 12, 0, 0], [20, 21, 22, 23, 0]]),
            "hist_cate_ids": torch.tensor([[3, 3, 2, 0, 0], [4, 5, 4, 6, 0]]),
            "hist_mask": torch.tensor([[True, True, True, False, False], [True, True, True, True, False]]),
            "hist_time_gaps": torch.tensor([[3600.0, 1200.0, 60.0, 0.0, 0.0], [7200.0, 3600.0, 600.0, 30.0, 0.0]]),
            "hist_time_deltas": torch.tensor([[0.0, 2400.0, 1140.0, 0.0, 0.0], [0.0, 3600.0, 3000.0, 570.0, 0.0]]),
            "hist_btags": torch.tensor([[1, 2, 3, 0, 0], [1, 1, 2, 4, 0]]),
            "user_static_ids": torch.tensor([[2, 3, 1, 3, 0, 2], [1, 4, 2, 2, 1, 3]]),
            "item_static_values": torch.tensor([[99.0, 12.0, 120.0, 50.0], [19.0, 8.0, 80.0, 30.0]]),
        }

        for name in [
            "base",
            "sim",
            "eta",
            "twin",
            "hyformer",
            "hyformer_opt",
            "hyformer_time",
            "hyformer_event",
            "hyformer_multigrain",
            "hyformer_session",
            "hyformer_static",
            "hyformer_hier",
            "hyformer_dynamic",
            "hyformer_offline_long",
        ]:
            with self.subTest(model=name):
                cfg = Config(
                    model=name,
                    embedding_dim=8,
                    lstm_hidden_dim=8,
                    hidden_dims=[16],
                    top_k=3,
                    hash_bits=16,
                    compressed_dim=4,
                    twin_heads=2,
                    twin_cross_features=6,
                    hyformer_heads=2,
                    hyformer_layers=1,
                    hyformer_ff_dim=16,
                    hyformer_non_seq_tokens=2,
                    hyformer_query_tokens=1,
                    hyformer_short_seq_len=2,
                    max_seq_len=5,
                )
                model = build_model(cfg, field_dims)
                logits = model(batch)
                self.assertEqual(tuple(logits.shape), (2,))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
