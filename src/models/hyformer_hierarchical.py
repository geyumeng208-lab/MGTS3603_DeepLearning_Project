from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.base import MLP
from src.models.hyformer import MLPQueryGenerator
from src.models.hyformer_static import StaticFeatureHyFormerModel
from src.utils import Config

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool


class HierarchicalHyFormerModel(StaticFeatureHyFormerModel):
    """Long-sequence HyFormer with recent fine-grained events and compressed history."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.num_sequences = 3
        self.recent_seq_len = cfg.recent_seq_len
        self.long_num_chunks = cfg.long_num_chunks

        non_seq_dim = cfg.embedding_dim * 8 + 6
        global_info_dim = non_seq_dim + self.num_sequences * cfg.embedding_dim
        self.non_seq_tokenizer = nn.Sequential(
            nn.Linear(non_seq_dim, cfg.hyformer_ff_dim),
            nn.SiLU(),
            nn.Linear(cfg.hyformer_ff_dim, self.num_non_seq_tokens * cfg.embedding_dim),
        )
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
        self.head = MLP(cfg.embedding_dim * 7 + 6, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = ensure_non_empty_mask(batch["hist_mask"])
        current_mask, _ = self.split_session(mask, batch["hist_time_deltas"])
        recent_mask, long_mask = self.split_recent_long(mask)

        user = self.user_emb(batch["user_id"])
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))
        user_static, item_static = self.build_static_features(batch)

        time_gaps = batch["hist_time_gaps"].clamp_min(0.0)
        time_deltas = batch["hist_time_deltas"].clamp_min(0.0)
        temporal = self.temporal_encoding(time_gaps, time_deltas, mask)
        event = self.btag_emb(batch["hist_btags"].clamp_min(0))

        hist_item = self.item_emb(batch["hist_item_ids"])
        hist_cate = self.cate_emb(batch["hist_cate_ids"])
        hist_pair = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))
        event_tokens = (hist_pair + event + temporal) * mask.unsqueeze(-1).float()

        current_tokens = event_tokens * current_mask.unsqueeze(-1).float()
        recent_tokens = event_tokens * recent_mask.unsqueeze(-1).float()
        long_tokens = event_tokens * long_mask.unsqueeze(-1).float()
        long_chunks, long_chunk_mask = chunk_pool(long_tokens, long_mask, self.long_num_chunks)

        sequence_tokens = [current_tokens, recent_tokens, long_chunks]
        sequence_masks = [
            ensure_non_empty_mask(current_mask),
            ensure_non_empty_mask(recent_mask),
            ensure_non_empty_mask(long_chunk_mask),
        ]
        pooled_sequences = [
            masked_mean_pool(tokens, seq_mask) for tokens, seq_mask in zip(sequence_tokens, sequence_masks)
        ]

        all_mean = self.masked_avg(event_tokens, mask)
        current_mean = self.masked_avg(current_tokens, current_mask)
        recent_mean = self.masked_avg(recent_tokens, recent_mask)
        time_summary = self.time_summary(time_gaps, time_deltas, mask)
        session_summary = self.session_summary(mask, current_mask, long_mask)
        context_stats = torch.cat([time_summary, session_summary], dim=-1)

        non_seq_x = torch.cat(
            [
                user,
                target_item,
                target_cate,
                all_mean,
                current_mean,
                recent_mean,
                user_static,
                item_static,
                context_stats,
            ],
            dim=-1,
        )
        non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(
            non_seq_x.size(0), self.num_non_seq_tokens, self.d_model
        )
        global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
        query_tokens = [generator(global_info) for generator in self.query_generators]

        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        boosted = boosted_tokens.mean(dim=1)
        features = torch.cat(
            [user, target, all_mean, current_mean, boosted, user_static, item_static, context_stats],
            dim=-1,
        )
        return self.head(features)

    def split_recent_long(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = mask.size(1)
        positions = torch.arange(seq_len, device=mask.device).unsqueeze(0)
        valid_counts = mask.long().sum(dim=1, keepdim=True)
        recent_start = (valid_counts - self.recent_seq_len).clamp_min(0)
        recent_mask = mask & (positions >= recent_start)
        long_mask = mask & ~recent_mask
        return recent_mask, long_mask


def chunk_pool(tokens: torch.Tensor, mask: torch.Tensor, num_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, dim = tokens.shape
    num_chunks = max(1, min(num_chunks, seq_len))
    pooled = torch.zeros(batch_size, num_chunks, dim, device=tokens.device, dtype=tokens.dtype)
    pooled_mask = torch.zeros(batch_size, num_chunks, device=tokens.device, dtype=torch.bool)
    for idx in range(num_chunks):
        start = int(idx * seq_len / num_chunks)
        end = max(start + 1, int((idx + 1) * seq_len / num_chunks))
        chunk_mask = mask[:, start:end]
        chunk_tokens = tokens[:, start:end, :]
        pooled[:, idx, :] = masked_mean_or_zero(chunk_tokens, chunk_mask)
        pooled_mask[:, idx] = chunk_mask.any(dim=1)
    return pooled, pooled_mask


def masked_mean_or_zero(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom
