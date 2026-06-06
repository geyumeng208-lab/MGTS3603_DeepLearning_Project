from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel, MLP
from src.models.hyformer import MLPQueryGenerator
from src.models.hyformer_time import time_to_bucket
from src.utils import Config

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool


class SessionAwareHyFormerModel(CTRBaseModel):
    """HyFormer with current-session and long-term behavior sequences."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.d_model = cfg.embedding_dim
        self.num_sequences = 2
        self.num_non_seq_tokens = cfg.hyformer_non_seq_tokens
        self.num_query_tokens = cfg.hyformer_query_tokens
        self.decay_hours = cfg.time_decay_hours
        self.session_gap_seconds = cfg.session_gap_minutes * 60.0

        self.btag_emb = nn.Embedding(cfg.btag_num_types, cfg.embedding_dim, padding_idx=0)
        self.time_gap_emb = nn.Embedding(cfg.time_num_bins, cfg.embedding_dim)
        self.time_delta_emb = nn.Embedding(cfg.time_num_bins, cfg.embedding_dim)
        self.time_gate = nn.Sequential(nn.Linear(2, cfg.embedding_dim), nn.Sigmoid())

        non_seq_dim = cfg.embedding_dim * 6 + 6
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
        self.head = MLP(cfg.embedding_dim * 5 + 6, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = ensure_non_empty_mask(batch["hist_mask"])
        current_mask, long_mask = self.split_session(mask, batch["hist_time_deltas"])

        user = self.user_emb(batch["user_id"])
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))

        time_gaps = batch["hist_time_gaps"].clamp_min(0.0)
        time_deltas = batch["hist_time_deltas"].clamp_min(0.0)
        temporal = self.temporal_encoding(time_gaps, time_deltas, mask)
        event = self.btag_emb(batch["hist_btags"].clamp_min(0))

        hist_item = self.item_emb(batch["hist_item_ids"])
        hist_cate = self.cate_emb(batch["hist_cate_ids"])
        hist_pair = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))
        event_tokens = (hist_pair + event + temporal) * mask.unsqueeze(-1).float()

        current_tokens = event_tokens * current_mask.unsqueeze(-1).float()
        long_tokens = event_tokens * long_mask.unsqueeze(-1).float()
        sequence_tokens = [current_tokens, long_tokens]
        sequence_masks = [ensure_non_empty_mask(current_mask), ensure_non_empty_mask(long_mask)]
        pooled_sequences = [masked_mean_pool(tokens, seq_mask) for tokens, seq_mask in zip(sequence_tokens, sequence_masks)]

        all_mean = masked_mean(event_tokens, mask)
        current_mean = masked_mean(current_tokens, current_mask)
        long_mean = masked_mean(long_tokens, long_mask)
        time_summary = self.time_summary(time_gaps, time_deltas, mask)
        session_summary = self.session_summary(mask, current_mask, long_mask)
        context_stats = torch.cat([time_summary, session_summary], dim=-1)

        non_seq_x = torch.cat([user, target_item, target_cate, all_mean, current_mean, long_mean, context_stats], dim=-1)
        non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(non_seq_x.size(0), self.num_non_seq_tokens, self.d_model)
        global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
        query_tokens = [generator(global_info) for generator in self.query_generators]

        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        boosted = boosted_tokens.mean(dim=1)
        features = torch.cat([user, target, all_mean, current_mean, boosted, context_stats], dim=-1)
        return self.head(features)

    def split_session(self, mask: torch.Tensor, time_deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        breaks = (time_deltas > self.session_gap_seconds) & mask
        has_future_break = torch.flip(torch.cumsum(torch.flip(breaks.long(), dims=[1]), dim=1), dims=[1]) > 0
        current_mask = mask & ~has_future_break
        # Keep at least the most recent valid event in current session.
        empty_current = ~current_mask.any(dim=1)
        if empty_current.any():
            last_idx = mask.long().argmax(dim=1)
            for row in torch.where(empty_current)[0]:
                valid_positions = torch.where(mask[row])[0]
                if valid_positions.numel() > 0:
                    last_idx[row] = valid_positions[-1]
                    current_mask[row, last_idx[row]] = True
        long_mask = mask & ~current_mask
        return current_mask, long_mask

    def temporal_encoding(self, time_gaps: torch.Tensor, time_deltas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gap_bins = time_to_bucket(time_gaps, self.cfg.time_num_bins)
        delta_bins = time_to_bucket(time_deltas, self.cfg.time_num_bins)
        gap_hours = time_gaps / 3600.0
        delta_hours = time_deltas / 3600.0
        decay = torch.exp(-gap_hours / max(self.decay_hours, 1e-6)).unsqueeze(-1)
        gate_input = torch.stack([torch.log1p(gap_hours) / 24.0, torch.log1p(delta_hours) / 24.0], dim=-1)
        return (self.time_gap_emb(gap_bins) + self.time_delta_emb(delta_bins)) * self.time_gate(gate_input) * decay * mask.unsqueeze(-1).float()

    def time_summary(self, time_gaps: torch.Tensor, time_deltas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        gap_hours = time_gaps / 3600.0
        delta_hours = time_deltas / 3600.0
        recency_weight = torch.exp(-gap_hours / max(self.decay_hours, 1e-6)) * mask_f
        return torch.stack(
            [
                (torch.log1p(gap_hours) * mask_f).sum(dim=1) / denom,
                (torch.log1p(delta_hours) * mask_f).sum(dim=1) / denom,
                recency_weight.sum(dim=1) / denom,
                (time_gaps.eq(0).float() * mask_f).sum(dim=1) / denom,
            ],
            dim=-1,
        )

    def session_summary(self, mask: torch.Tensor, current_mask: torch.Tensor, long_mask: torch.Tensor) -> torch.Tensor:
        total = mask.float().sum(dim=1).clamp_min(1.0)
        current_len = current_mask.float().sum(dim=1)
        long_len = long_mask.float().sum(dim=1)
        return torch.stack([current_len / total, long_len / total], dim=-1)
