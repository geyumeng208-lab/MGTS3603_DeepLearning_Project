"""
Phase 3, Step 15: Embedding Pruning (L1-norm based) for TWIN model.

Approach:
1. Load a fully trained TWIN model (embedding_dim=32).
2. Compute L1 norm of each embedding dimension across all Embedding layers.
3. Sparsify the bottom `prune_ratio` dimensions (zero them out).
4. Fine-tune the sparsified model on the original task.
5. Evaluate AUC/GAUC vs. compression ratio.

Usage:
    # Prune 25% of embedding dims, then fine-tune
    python scripts/prune_embedding.py --model twin --prune_ratio 0.25 --data_path data/purchase_sequence_100k_static_long500.csv

    # Prune 50%, no fine-tune
    python scripts/prune_embedding.py --model twin --prune_ratio 0.5 --finetune_epochs 0 --data_path ...
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
from src.models.base import CTRBaseModel
from src.utils import Config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L1-norm embedding pruning for TWIN")
    parser.add_argument("--model", type=str, default="twin")
    parser.add_argument("--data_path", type=str, default="data/purchase_sequence_100k_static_long500.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prune_ratio", type=float, default=0.25,
                        help="Fraction of embedding dimensions to zero out")
    parser.add_argument("--finetune_epochs", type=int, default=2,
                        help="Number of fine-tuning epochs after pruning")
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


def compute_embedding_l1_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    """Compute L1 norm of each embedding dimension.

    Returns a dict mapping layer name → L1 norm vector (shape: [embedding_dim]).
    """
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 1:
            # L1 norm across the vocabulary dimension: [vocab, dim] → [dim]
            l1 = module.weight.abs().mean(dim=0)  # shape: [embedding_dim]
            norms[name] = l1
    return norms


def sparsify_embeddings(model: nn.Module, prune_ratio: float) -> tuple[nn.Module, int]:
    """Zero out the bottom `prune_ratio` dimensions across all embedding layers.

    Returns the sparsified model and the number of dimensions kept.
    """
    norms = compute_embedding_l1_norms(model)

    # Concatenate all L1 norms to get a global ranking
    all_norms = torch.cat([n for n in norms.values()])  # shape: [total_dimensions]
    total_dims = all_norms.numel()
    n_prune = int(total_dims * prune_ratio)

    if n_prune == 0:
        print("  prune_ratio=0, no pruning")
        return model, total_dims

    # Find the global threshold at the `n_prune`-th smallest value
    sorted_norms, _ = all_norms.sort()
    threshold = sorted_norms[n_prune - 1] if n_prune > 0 else -1.0

    # Apply mask: for each embedding layer, zero out dims ≤ threshold
    n_kept = total_dims
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 1:
            mask = module.weight.abs().mean(dim=0) > threshold  # [dim]
            module.weight.data[:, ~mask] = 0.0
            n_kept -= (~mask).sum().item()

    print(f"  Pruned {total_dims - n_kept}/{total_dims} embedding dims ({prune_ratio*100:.0f}%)")
    return model, n_kept


def count_active_embedding_dims(model: nn.Module) -> int:
    """Count how many embedding dims have non-zero weights."""
    total = 0
    active = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 1:
            total += module.embedding_dim
            nonzero = (module.weight.abs().sum(dim=0) > 0).sum().item()
            active += nonzero
    return active, total


def compute_model_size_mb(model: nn.Module) -> float:
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


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

    # === 1. Load trained model ===
    print("=" * 60)
    print(f"1. LOAD TRAINED MODEL (embedding_dim={cfg.embedding_dim})")
    print("=" * 60)
    model = build_model(cfg, field_dims).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    pre_metrics = evaluate(model, valid_loader, device)
    pre_size = compute_model_size_mb(model)
    active, total = count_active_embedding_dims(model)
    print(f"  Pre-prune  | AUC={pre_metrics['auc']:.4f} GAUC={pre_metrics['gauc']:.4f} "
          f"Size={pre_size:.2f}MB ActiveEmbedDims={active}/{total}")

    # === 2. Prune ===
    print()
    print("=" * 60)
    print(f"2. PRUNE EMBEDDING DIMS (ratio={args.prune_ratio})")
    print("=" * 60)
    model, kept_dims = sparsify_embeddings(model, args.prune_ratio)
    prune_metrics = evaluate(model, valid_loader, device)
    prune_size = compute_model_size_mb(model)
    active, total = count_active_embedding_dims(model)
    print(f"  Post-prune | AUC={prune_metrics['auc']:.4f} GAUC={prune_metrics['gauc']:.4f} "
          f"Size={prune_size:.2f}MB ActiveEmbedDims={active}/{total}")

    # === 3. Fine-tune ===
    if args.finetune_epochs > 0:
        print()
        print("=" * 60)
        print(f"3. FINE-TUNE ({args.finetune_epochs} epochs)")
        print("=" * 60)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 0.1, weight_decay=cfg.weight_decay)
        for epoch in range(1, args.finetune_epochs + 1):
            model.train()
            total_loss = 0.0
            total_count = 0
            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                logits = model(batch)
                loss = criterion(logits, batch["label"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Don't update pruned dims — freeze zeroed-out columns
                for module in model.modules():
                    if isinstance(module, nn.Embedding) and module.num_embeddings > 1:
                        pruned_dims = module.weight.abs().sum(dim=0) == 0  # [dim]
                        module.weight.grad[:, pruned_dims] = 0.0
                optimizer.step()
                batch_size = batch["label"].size(0)
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size
            metrics = evaluate(model, valid_loader, device)
            print(f"  finetune epoch={epoch:02d} loss={total_loss/max(total_count,1):.4f} "
                  f"auc={metrics['auc']:.4f} gauc={metrics['gauc']:.4f}")

    # === 4. Final evaluation ===
    final_metrics = evaluate(model, valid_loader, device)
    final_size = compute_model_size_mb(model)
    active, total = count_active_embedding_dims(model)

    print()
    print("=" * 60)
    print("SUMMARY: EMBEDDING PRUNING")
    print("=" * 60)
    print(f"{'Stage':<16} {'AUC':<8} {'GAUC':<8} {'Size(MB)':<10} {'ActiveDims':<12}")
    print("-" * 60)
    print(f"{'Pre-prune':<16} {pre_metrics['auc']:.4f}   {pre_metrics['gauc']:.4f}   {pre_size:<10.2f} {total:<12}")
    print(f"{'Post-prune':<16} {prune_metrics['auc']:.4f}   {prune_metrics['gauc']:.4f}   {prune_size:<10.2f} {active:<12}")
    print(f"{'After finetune':<16} {final_metrics['auc']:.4f}   {final_metrics['gauc']:.4f}   {final_size:<10.2f}")


if __name__ == "__main__":
    main()
