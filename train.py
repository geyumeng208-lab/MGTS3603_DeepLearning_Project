from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from src.data import build_dataloaders
from src.models import build_model
from src.trainer import Trainer
from src.utils import Config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train long user behavior sequence CTR models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "base",
            "lstm",
            "lstm_attn",
            "lstm_attention",
            "transformer_baseline",
            "full_transformer_baseline",
            "sim",
            "eta",
            "twin",
            "twin_lite",
            "twin_old",
            "hyformer",
            "hybrid_transformer",
            "hyformer_opt",
            "hyformer_optimized",
            "hyformer_time",
            "hyformer_temporal",
            "hyformer_event",
            "hyformer_btag",
            "hyformer_multigrain",
            "hyformer_multi",
            "hyformer_session",
            "hyformer_sessional",
            "hyformer_static",
            "hyformer_profile",
            "hyformer_hier",
            "hyformer_hierarchical",
            "hyformer_dynamic",
            "hyformer_dyn",
            "hyformer_topk",
            "hyformer_filter",
            "hyformer_offline_long",
            "hyformer_cached_long",
        ],
        default=None,
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--synthetic_samples", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--session_gap_minutes", type=float, default=None)
    parser.add_argument("--adaptive_session_gap", action="store_true")
    parser.add_argument("--multitask_loss_weight", type=float, default=None)
    parser.add_argument("--auto_pos_weight", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--lstm_hidden_dim", type=int, default=None)
    parser.add_argument("--hyformer_layers", type=int, default=None)
    parser.add_argument("--hyformer_heads", type=int, default=None)
    parser.add_argument("--hyformer_ff_dim", type=int, default=None)
    parser.add_argument("--twin_heads", type=int, default=None)
    parser.add_argument("--recent_seq_len", type=int, default=None)
    parser.add_argument("--long_num_chunks", type=int, default=None)
    parser.add_argument("--dynamic_recent_len", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--hyformer_encoder_type",
        type=str,
        choices=["longer", "full_transformer", "swiglu"],
        default=None,
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    with Path(args.config).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for key in [
        "model",
        "data_path",
        "epochs",
        "batch_size",
        "max_seq_len",
        "top_k",
        "synthetic_samples",
        "pos_weight",
        "session_gap_minutes",
        "adaptive_session_gap",
        "multitask_loss_weight",
        "auto_pos_weight",
        "embedding_dim",
        "lstm_hidden_dim",
        "hyformer_layers",
        "hyformer_heads",
        "hyformer_ff_dim",
        "twin_heads",
        "recent_seq_len",
        "long_num_chunks",
        "dynamic_recent_len",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "hyformer_encoder_type",
        "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
    known_fields = {field.name for field in fields(Config)}
    unknown_fields = sorted(set(raw) - known_fields)
    if unknown_fields:
        raise ValueError(f"配置文件包含未知字段: {unknown_fields}")
    return Config(**raw)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg.seed)

    train_loader, valid_loader, field_dims = build_dataloaders(cfg)
    model = build_model(cfg, field_dims)
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, valid_loader)


if __name__ == "__main__":
    main()
