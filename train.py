from __future__ import annotations

import argparse
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
        "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
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
