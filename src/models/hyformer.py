from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel
from src.utils import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYFORMER_ROOT = PROJECT_ROOT / "external" / "Hyformer_Pytorch"
if str(HYFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(HYFORMER_ROOT))

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool  # noqa: E402


class MLPQueryGenerator(nn.Module):
    """Query-token generator from WestbrookLong/Hyformer_Pytorch's TAAC wrapper."""

    def __init__(self, global_info_dim: int, num_query_tokens: int, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Linear(global_info_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_query_tokens * d_model),
        )

    def forward(self, global_info: torch.Tensor) -> torch.Tensor:
        return self.ffn(global_info).view(global_info.size(0), self.num_query_tokens, self.d_model)


class HyFormerModel(CTRBaseModel):
    """HyFormer CTR model adapted from WestbrookLong/Hyformer_Pytorch.

    The external implementation models heterogeneous inputs with:
    sequence representation encoders, sequence-specific query tokens,
    non-sequence tokens, and QueryBoostMixer. Here we map the e-commerce
    purchase-prediction sample into one behavior sequence plus non-sequence
    user/target context tokens.
    """

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.d_model = cfg.embedding_dim
        self.num_sequences = 1
        self.num_non_seq_tokens = cfg.hyformer_non_seq_tokens
        self.num_query_tokens = cfg.hyformer_query_tokens

        non_seq_dim = cfg.embedding_dim * 3
        global_info_dim = non_seq_dim + self.num_sequences * cfg.embedding_dim
        self.non_seq_tokenizer = nn.Linear(non_seq_dim, self.num_non_seq_tokens * cfg.embedding_dim)
        self.query_generators = nn.ModuleList(
            [
                MLPQueryGenerator(
                    global_info_dim=global_info_dim,
                    num_query_tokens=self.num_query_tokens,
                    d_model=cfg.embedding_dim,
                    hidden_dim=cfg.hyformer_ff_dim,
                )
                for _ in range(self.num_sequences)
            ]
        )
        self.backbone = HyFormerBackbone(
            num_layers=cfg.hyformer_layers,
            num_sequences=self.num_sequences,
            num_queries_per_sequence=self.num_query_tokens,
            num_non_seq_tokens=self.num_non_seq_tokens,
            d_model=cfg.embedding_dim,
            num_heads=cfg.hyformer_heads,
            ffn_hidden=cfg.hyformer_ff_dim,
            encoder_type=cfg.hyformer_encoder_type,
            short_seq_len=cfg.hyformer_short_seq_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embedding_dim),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        mask = ensure_non_empty_mask(batch["hist_mask"])
        history = history * mask.unsqueeze(-1).float()

        user = self.user_emb(batch["user_id"])
        history_mean = masked_mean(history, mask)
        non_seq_x = torch.cat([user, target, history_mean], dim=-1)
        non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(
            non_seq_x.size(0), self.num_non_seq_tokens, self.d_model
        )

        sequence_tokens = [history]
        sequence_masks = [mask]
        pooled_sequences = [masked_mean_pool(history, mask)]
        global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
        query_tokens = [generator(global_info) for generator in self.query_generators]

        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        pooled = boosted_tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)
