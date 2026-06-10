from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Config:
    seed: int = 2026
    model: str = "twin"
    data_path: Optional[str] = None
    synthetic_samples: int = 3000
    valid_ratio: float = 0.2

    num_users: int = 2000
    num_items: int = 5000
    num_categories: int = 80
    max_seq_len: int = 1000
    min_seq_len: int = 20
    top_k: int = 50

    embedding_dim: int = 32
    lstm_hidden_dim: int = 32
    lstm_layers: int = 1
    hidden_dims: tuple[int, ...] | list[int] = (128, 64)
    dropout: float = 0.15
    hash_bits: int = 64
    compressed_dim: int = 16
    twin_heads: int = 4
    twin_cross_features: int = 6
    hyformer_layers: int = 2
    hyformer_heads: int = 4
    hyformer_ff_dim: int = 128
    hyformer_kernel_size: int = 3
    hyformer_non_seq_tokens: int = 3
    hyformer_query_tokens: int = 1
    hyformer_encoder_type: str = "longer"
    hyformer_short_seq_len: int = 8
    time_num_bins: int = 12
    time_decay_hours: float = 24.0
    btag_num_types: int = 5
    session_gap_minutes: float = 30.0
    adaptive_session_gap: bool = False
    static_feature_vocab_size: int = 64
    recent_seq_len: int = 100
    long_num_chunks: int = 8
    dynamic_low_activity_len: int = 50
    dynamic_recent_len: int = 100
    multitask_loss_weight: float = 0.0
    auto_pos_weight: bool = False

    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    pos_weight: float = 0.0
    num_workers: int = 0
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
