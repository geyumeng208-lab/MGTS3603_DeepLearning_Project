"""
Phase 4, Step 18: Two-stage ranking pipeline (TWIN → HyFormer).

Simulates a real-world recommendation system architecture:
  Stage 1 (coarse) : TWIN scores all candidates quickly.
  Stage 2 (fine)   : HyFormer re-ranks the top candidates.

In our training data, each sample has one target item, so we
simulate the pipeline by:
  1. Running TWIN on all samples (fast pass).
  2. Selecting top-K samples by TWIN confidence.
  3. Running HyFormer only on those top-K samples (fine pass).
  4. Measuring latency, throughput, and accuracy at each stage.

Usage:
    python scripts/two_stage_benchmark.py --data_path data/purchase_sequence_100k_static_long500.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage ranking: TWIN (coarse) → HyFormer (fine)")
    parser.add_argument("--data_path", type=str, default="data/purchase_sequence_100k_static_long500.csv")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k_ratio", type=float, default=0.5,
                        help="Fraction of samples to pass to HyFormer for fine scoring")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Standard single-model evaluation."""
    model.eval()
    all_users, all_labels, all_preds = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch)
        preds = torch.sigmoid(logits)
        all_users.append(batch["user_id"].detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
    users = np.concatenate(all_users)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {"auc": auc_score(labels, preds), "gauc": gauc_score(users, labels, preds)}


def collect_all_preds(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect all predictions, labels, and user IDs."""
    model.eval()
    all_users, all_labels, all_preds = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_users.append(batch["user_id"].cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_preds.append(preds)
    return np.concatenate(all_users), np.concatenate(all_labels), np.concatenate(all_preds)


def measure_latency(model: nn.Module, loader: DataLoader, device: torch.device, num_batches: int = 20) -> tuple[float, float]:
    """Measure average per-batch latency (total and P99)."""
    model.eval()
    latencies = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        for _ in range(3):  # warmup
            _ = model(batch)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    avg = float(np.mean(latencies))
    p99 = float(np.percentile(latencies, 99))
    return avg, p99


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # === Load config & data ===
    cfg = Config(
        model="twin",
        data_path=args.data_path,
        epochs=1,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    set_seed(cfg.seed)
    _, valid_loader, field_dims = build_dataloaders(cfg)

    # === Build models ===
    print("=" * 65)
    print("  TWO-STAGE RANKING BENCHMARK: TWIN (coarse) → HyFormer (fine)")
    print("=" * 65)

    # Coarse: TWIN
    print("\n[Stage 1] Loading TWIN (coarse model)...")
    cfg_twin = Config(**{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})
    cfg_twin.model = "twin"
    model_twin = build_model(cfg_twin, field_dims).to(device)
    ckpt_twin = Path("checkpoints/twin_fp32.pt")
    if ckpt_twin.exists():
        model_twin.load_state_dict(torch.load(ckpt_twin, map_location=device))
    else:
        print("  No TWIN checkpoint found. Training a quick TWIN...")
        from src.trainer import Trainer
        Trainer(model_twin, cfg_twin).fit(*build_dataloaders(cfg)[:2])

    # Fine: HyFormer-Hierarchical
    print("[Stage 2] Loading HyFormer-Hierarchical (fine model)...")
    cfg_hyformer = Config(**{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})
    cfg_hyformer.model = "hyformer_hierarchical"
    model_hyformer = build_model(cfg_hyformer, field_dims).to(device)
    ckpt_hyformer = Path("checkpoints/hyformer_hierarchical.pt")
    if ckpt_hyformer.exists():
        model_hyformer.load_state_dict(torch.load(ckpt_hyformer, map_location=device))
        print(f"  Loaded checkpoint: {ckpt_hyformer}")
    else:
        print("  No checkpoint found, training a HyFormer model (this will take a while)...")
        from src.trainer import Trainer
        hyformer_cfg = Config(**{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})
        hyformer_cfg.model = "hyformer_hierarchical"
        hyformer_cfg.epochs = 3
        train_l, valid_l, _ = build_dataloaders(hyformer_cfg)
        Trainer(model_hyformer, hyformer_cfg).fit(train_l, valid_l)
        torch.save(model_hyformer.state_dict(), ckpt_hyformer)
        print(f"  Saved checkpoint: {ckpt_hyformer}")

    # === 1. Single-model baselines ===
    print("\n" + "─" * 65)
    print("  BASELINES: Single-model evaluation")
    print("─" * 65)

    print("\n--- TWIN (coarse, stage-1) ---")
    twin_latency_avg, twin_latency_p99 = measure_latency(model_twin, valid_loader, device)
    twin_metrics = evaluate(model_twin, valid_loader, device)
    print(f"  AUC={twin_metrics['auc']:.4f} GAUC={twin_metrics['gauc']:.4f}")
    print(f"  Latency: avg={twin_latency_avg:.2f}ms  P99={twin_latency_p99:.2f}ms")

    print("\n--- HyFormer-Hierarchical (fine, stage-2) ---")
    hyformer_latency_avg, hyformer_latency_p99 = measure_latency(model_hyformer, valid_loader, device)
    hyformer_metrics = evaluate(model_hyformer, valid_loader, device)
    print(f"  AUC={hyformer_metrics['auc']:.4f} GAUC={hyformer_metrics['gauc']:.4f}")
    print(f"  Latency: avg={hyformer_latency_avg:.2f}ms  P99={hyformer_latency_p99:.2f}ms")

    # === 2. Two-stage pipeline ===
    print("\n" + "─" * 65)
    print(f"  TWO-STAGE PIPELINE (top_k_ratio={args.top_k_ratio})")
    print("─" * 65)

    # Collect all TWIN predictions first
    all_users, all_labels, all_twin_preds = collect_all_preds(model_twin, valid_loader, device)

    # Select top-k_ratio samples by TWIN confidence (absolute prediction)
    # Higher confidence → more ambiguous → more benefit from fine scoring
    n_total = len(all_twin_preds)
    n_fine = int(n_total * args.top_k_ratio)
    # Pick samples where TWIN is most uncertain (closest to 0.5)
    uncertainty = np.abs(all_twin_preds - 0.5)
    fine_indices = np.argsort(uncertainty)[:n_fine]  # most uncertain → need HyFormer
    coarse_indices = np.setdiff1d(np.arange(n_total), fine_indices)

    # Build final predictions: coarse (TWIN) for most, fine (HyFormer) for uncertain ones
    all_final_preds = all_twin_preds.copy()

    # Run HyFormer only on the uncertain subset
    print(f"  TWIN handles {len(coarse_indices)} samples ({(1-args.top_k_ratio)*100:.0f}%)")
    print(f"  HyFormer refines {len(fine_indices)} samples ({args.top_k_ratio*100:.0f}%)")

    # Simulate by running inference on the full set and then blending
    _, _, all_hyformer_preds = collect_all_preds(model_hyformer, valid_loader, device)
    all_final_preds[fine_indices] = all_hyformer_preds[fine_indices]

    # Compute metrics
    pipeline_auc = float(auc_score(all_labels, all_final_preds))
    pipeline_gauc = float(gauc_score(all_users, all_labels, all_final_preds))

    # Estimate pipeline latency:
    # TWIN runs on ALL samples, HyFormer runs on top_k_ratio of samples
    n_batches = len(valid_loader)
    total_batches = n_batches + int(n_batches * args.top_k_ratio)
    pipeline_avg = (twin_latency_avg + hyformer_latency_avg * args.top_k_ratio) if args.top_k_ratio > 0 else twin_latency_avg
    pipeline_p99 = (twin_latency_p99 + hyformer_latency_p99 * args.top_k_ratio) if args.top_k_ratio > 0 else twin_latency_p99

    print(f"  Pipeline | AUC={pipeline_auc:.4f} GAUC={pipeline_gauc:.4f}")
    print(f"  Latency: avg={pipeline_avg:.2f}ms  P99={pipeline_p99:.2f}ms")

    # === 3. Summary ===
    print("\n" + "=" * 65)
    print("  SUMMARY: Latency vs. Accuracy Trade-off")
    print("=" * 65)
    print(f"{'Method':<30} {'AUC':<8} {'GAUC':<8} {'Avg(ms)':<10} {'P99(ms)':<10}")
    print("-" * 65)
    print(f"{'TWIN (stage-1 coarse)':<30} {twin_metrics['auc']:.4f}   {twin_metrics['gauc']:.4f}   {twin_latency_avg:<10.2f} {twin_latency_p99:<10.2f}")
    print(f"{'HyFormer (stage-2 fine)':<30} {hyformer_metrics['auc']:.4f}   {hyformer_metrics['gauc']:.4f}   {hyformer_latency_avg:<10.2f} {hyformer_latency_p99:<10.2f}")
    print(f"{'TWIN + HyFormer (pipeline)':<30} {pipeline_auc:.4f}   {pipeline_gauc:.4f}   {pipeline_avg:<10.2f} {pipeline_p99:<10.2f}")

    # Speedup vs accuracy gain
    speedup_vs_hyformer = hyformer_latency_avg / max(pipeline_avg, 0.01)
    auc_gain_vs_twin = (pipeline_auc - twin_metrics['auc']) * 100
    print(f"\n  Pipeline vs TWIN-only:       +{auc_gain_vs_twin:.2f}% AUC (trade: +{pipeline_avg - twin_latency_avg:.1f}ms)")
    print(f"  Pipeline vs HyFormer-only:   {speedup_vs_hyformer:.1f}x faster (trade: {(hyformer_metrics['auc'] - pipeline_auc)*100:.2f}% AUC loss)")


if __name__ == "__main__":
    main()
