from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.base import MLP
from src.models.hyformer_session import SessionAwareHyFormerModel
from src.utils import Config


class StaticFeatureHyFormerModel(SessionAwareHyFormerModel):
    """HyFormer-Session with user profile and product-side static features."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.static_user_emb = nn.Embedding(cfg.static_feature_vocab_size, cfg.embedding_dim, padding_idx=0)
        self.item_static_proj = nn.Sequential(
            nn.Linear(4, cfg.embedding_dim),
            nn.SiLU(),
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )

        non_seq_dim = cfg.embedding_dim * 8 + 6
        global_info_dim = non_seq_dim + self.num_sequences * cfg.embedding_dim
        self.non_seq_tokenizer = nn.Sequential(
            nn.Linear(non_seq_dim, cfg.hyformer_ff_dim),
            nn.SiLU(),
            nn.Linear(cfg.hyformer_ff_dim, self.num_non_seq_tokens * cfg.embedding_dim),
        )
        self.query_generators = nn.ModuleList(
            [
                type(self.query_generators[0])(
                    global_info_dim=global_info_dim,
                    num_query_tokens=self.num_query_tokens,
                    d_model=cfg.embedding_dim,
                    hidden_dim=cfg.hyformer_ff_dim,
                )
                for _ in range(self.num_sequences)
            ]
        )
        self.head = MLP(cfg.embedding_dim * 7 + 6, cfg.hidden_dims, cfg.dropout)
        self.btag_head = nn.Linear(cfg.embedding_dim * 7 + 6, cfg.btag_num_types)

    def build_static_features(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids = batch["user_static_ids"].clamp(0, self.cfg.static_feature_vocab_size - 1)
        user_static = self.static_user_emb(user_ids).mean(dim=1)
        item_values = torch.log1p(batch["item_static_values"].clamp_min(0.0))
        item_static = self.item_static_proj(item_values)
        return user_static, item_static

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Reuse the session split and temporal/event token construction from the parent,
        # but inject static tokens into non-sequence context and the output head.
        mask = self.ensure_mask(batch["hist_mask"])
        if self.adaptive_session_gap:
            gap = batch.get("session_gap_threshold")
            if gap is not None and (gap > 0).any():
                gap = gap.unsqueeze(-1)
            else:
                gap = self.session_gap_seconds
        else:
            gap = self.session_gap_seconds
        current_mask, long_mask = self.split_session(mask, batch["hist_time_deltas"], gap)

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
        long_tokens = event_tokens * long_mask.unsqueeze(-1).float()
        sequence_tokens = [current_tokens, long_tokens]
        sequence_masks = [self.ensure_mask(current_mask), self.ensure_mask(long_mask)]
        pooled_sequences = [
            self.masked_pool(tokens, seq_mask) for tokens, seq_mask in zip(sequence_tokens, sequence_masks)
        ]

        all_mean = self.masked_avg(event_tokens, mask)
        current_mean = self.masked_avg(current_tokens, current_mask)
        long_mean = self.masked_avg(long_tokens, long_mask)
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
                long_mean,
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
        logits = self.head(features)
        if self.training and self.multitask_loss_weight > 0:
            btag_logits = self.btag_head(features)
            return {"logits": logits, "btag_logits": btag_logits, "btag_labels": batch.get("btag", torch.zeros_like(batch["label"]).long())}
        return logits

    @staticmethod
    def ensure_mask(mask: torch.Tensor) -> torch.Tensor:
        from main_pytorch import ensure_non_empty_mask

        return ensure_non_empty_mask(mask)

    @staticmethod
    def masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from main_pytorch import masked_mean_pool

        return masked_mean_pool(tokens, mask)

    @staticmethod
    def masked_avg(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from src.models.attention import masked_mean

        return masked_mean(tokens, mask)
