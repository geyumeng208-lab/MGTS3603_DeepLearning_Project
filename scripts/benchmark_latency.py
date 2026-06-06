from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FieldDims
from src.models import build_model
from src.utils import Config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model forward latency on synthetic batches.")
    parser.add_argument("--models", nargs="+", default=["twin", "hyformer_static", "hyformer_hier"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_batch(batch_size: int, seq_len: int, field_dims: FieldDims, device: torch.device) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(2026 + batch_size + seq_len)
    hist_len = rng.integers(max(2, seq_len // 2), seq_len + 1, size=batch_size)
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    for idx, length in enumerate(hist_len):
        mask[idx, -length:] = True

    time_gaps = np.zeros((batch_size, seq_len), dtype=np.float32)
    time_deltas = np.zeros((batch_size, seq_len), dtype=np.float32)
    for idx in range(batch_size):
        deltas = rng.integers(0, 3600, size=seq_len).astype(np.float32)
        gaps = np.cumsum(deltas[::-1])[::-1]
        time_deltas[idx] = deltas
        time_gaps[idx] = gaps

    batch = {
        "user_id": torch.tensor(rng.integers(1, field_dims.num_users, size=batch_size), dtype=torch.long),
        "item_id": torch.tensor(rng.integers(1, field_dims.num_items, size=batch_size), dtype=torch.long),
        "cate_id": torch.tensor(rng.integers(1, field_dims.num_categories, size=batch_size), dtype=torch.long),
        "label": torch.zeros(batch_size, dtype=torch.float32),
        "hist_item_ids": torch.tensor(rng.integers(1, field_dims.num_items, size=(batch_size, seq_len)), dtype=torch.long),
        "hist_cate_ids": torch.tensor(rng.integers(1, field_dims.num_categories, size=(batch_size, seq_len)), dtype=torch.long),
        "hist_mask": torch.tensor(mask, dtype=torch.bool),
        "hist_time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
        "hist_time_deltas": torch.tensor(time_deltas, dtype=torch.float32),
        "hist_btags": torch.tensor(rng.integers(1, 5, size=(batch_size, seq_len)), dtype=torch.long),
        "user_static_ids": torch.tensor(rng.integers(0, 8, size=(batch_size, 6)), dtype=torch.long),
        "item_static_values": torch.tensor(rng.random((batch_size, 4)) * 100.0, dtype=torch.float32),
    }
    return {key: value.to(device) for key, value in batch.items()}


def benchmark_one(
    model_name: str,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float | str | int]:
    cfg = Config(
        model=model_name,
        max_seq_len=seq_len,
        top_k=min(50, seq_len),
        batch_size=batch_size,
        device=str(device),
        embedding_dim=32,
        compressed_dim=16,
        hidden_dims=[128, 64],
        hyformer_layers=2,
        hyformer_heads=4,
        hyformer_ff_dim=128,
        hyformer_non_seq_tokens=3,
        hyformer_query_tokens=1,
        hyformer_short_seq_len=8,
        recent_seq_len=min(100, seq_len),
        long_num_chunks=8,
    )
    field_dims = FieldDims(num_users=1_100_000, num_items=500_000, num_categories=20_000)
    model = build_model(cfg, field_dims).to(device)
    model.eval()
    batch = make_batch(batch_size, seq_len, field_dims, device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "model": model_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "per_sample_ms": float(np.mean(timings) / batch_size),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print("| model | batch_size | seq_len | mean_ms | p50_ms | p95_ms | per_sample_ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name in args.models:
        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lens:
                row = benchmark_one(model_name, batch_size, seq_len, args.warmup, args.iters, device)
                print(
                    f"| {row['model']} | {row['batch_size']} | {row['seq_len']} | "
                    f"{row['mean_ms']:.2f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
                    f"{row['per_sample_ms']:.2f} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
