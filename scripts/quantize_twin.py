"""
Phase 3, Step 14: INT8 Quantization (PTQ) for TWIN model.

Uses torchao.quantization.quantize_ with Int8WeightOnlyConfig
to quantize all Linear layers to INT8 weights (no engine dependency).

Usage:
    python scripts/quantize_twin.py --model twin --data_path data/purchase_sequence_100k_static_long500.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import build_dataloaders
from src.metrics import auc_score, gauc_score
from src.models import build_model
from src.utils import Config, set_seed
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INT8 quantization benchmark for TWIN")
    parser.add_argument("--model", type=str, default="twin")
    parser.add_argument("--data_path", type=str, default="data/purchase_sequence_100k_static_long500.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_users, all_labels, all_preds = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch)
        preds = torch.sigmoid(logits)
        all_users.append(batch["user_id"].detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
    users = np.concatenate(all_users)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {"auc": auc_score(labels, preds), "gauc": gauc_score(users, labels, preds)}


def measure_latency(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device, num_runs: int = 50) -> float:
    batch = {k: v.to(device) for k, v in batch.items()}
    model.eval()
    for _ in range(10):
        _ = model(batch)
    torch.manual_seed(42)
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model(batch)
    end = time.perf_counter()
    return (end - start) / num_runs * 1000


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = Config(
        model=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    set_seed(cfg.seed)
    train_loader, valid_loader, field_dims = build_dataloaders(cfg)

    ckpt_path = Path(f"checkpoints/{args.model}_fp32.pt")

    # === 1. FP32 Baseline ===
    print("=" * 60)
    print("1. FP32 BASELINE")
    print("=" * 60)
    model_fp32 = build_model(cfg, field_dims).to(device)
    if ckpt_path.exists():
        model_fp32.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found — training a quick model instead...")
        trainer = Trainer(model_fp32, cfg)
        trainer.fit(train_loader, valid_loader)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_fp32.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    fp32_metrics = evaluate(model_fp32, valid_loader, device)
    sample_batch = next(iter(valid_loader))
    fp32_latency = measure_latency(model_fp32, sample_batch, device)
    fp32_size = model_size_mb(model_fp32)
    print(f"FP32  | AUC={fp32_metrics['auc']:.4f} GAUC={fp32_metrics['gauc']:.4f} "
          f"Latency={fp32_latency:.2f}ms Size={fp32_size:.2f}MB")

    # === 2. INT8 Weight-Only Quantization (torchao) ===
    print()
    print("=" * 60)
    print("2. INT8 WEIGHT-ONLY QUANTIZATION (torchao)")
    print("=" * 60)
    model_int8 = build_model(cfg, field_dims).to("cpu")
    model_int8.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    quantize_(model_int8, Int8WeightOnlyConfig())
    model_int8 = model_int8.to(device)

    int8_metrics = evaluate(model_int8, valid_loader, device)
    int8_latency = measure_latency(model_int8, sample_batch, device)
    int8_size = model_size_mb(model_int8)
    print(f"INT8  | AUC={int8_metrics['auc']:.4f} GAUC={int8_metrics['gauc']:.4f} "
          f"Latency={int8_latency:.2f}ms Size={int8_size:.2f}MB")

    # === Summary ===
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<16} {'AUC':<8} {'GAUC':<8} {'Latency(ms)':<12} {'Size(MB)':<10}")
    print("-" * 60)
    print(f"{'FP32':<16} {fp32_metrics['auc']:.4f}   {fp32_metrics['gauc']:.4f}   {fp32_latency:<10.2f} {fp32_size:<10.2f}")
    print(f"{'INT8':<16} {int8_metrics['auc']:.4f}   {int8_metrics['gauc']:.4f}   {int8_latency:<10.2f} {int8_size:<10.2f}")


if __name__ == "__main__":
    main()
