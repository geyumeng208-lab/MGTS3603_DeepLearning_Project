from __future__ import annotations

import torch

from src.data import FieldDims
from src.models.hyformer_hierarchical import HierarchicalHyFormerModel
from src.utils import Config

from main_pytorch import ensure_non_empty_mask, masked_mean_pool


class OfflineLongTermHyFormerModel(HierarchicalHyFormerModel):
    """HyFormer with a cached long-term interest token.

    The online path consumes current-session tokens, recent-history tokens, and
    a precomputed long-term embedding. If the batch does not provide
    ``long_term_embedding``, the model falls back to mean-pooling older history,
    which keeps training and smoke tests self-contained.
    """

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.long_term_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(cfg.embedding_dim),
            torch.nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        full_mask = ensure_non_empty_mask(batch["hist_mask"])
        full_current_mask, _ = self.split_session(full_mask, batch["hist_time_deltas"])
        full_recent_mask, long_mask = self.split_recent_long(full_mask)
        cached_long_term = batch.get("long_term_embedding")

        if cached_long_term is not None:
            seq_slice = slice(-self.recent_seq_len, None)
            mask = ensure_non_empty_mask(full_mask[:, seq_slice])
            current_mask = full_current_mask[:, seq_slice]
            recent_mask = full_recent_mask[:, seq_slice]
            hist_item_ids = batch["hist_item_ids"][:, seq_slice]
            hist_cate_ids = batch["hist_cate_ids"][:, seq_slice]
            hist_btags = batch["hist_btags"][:, seq_slice]
            time_gaps = batch["hist_time_gaps"][:, seq_slice].clamp_min(0.0)
            time_deltas = batch["hist_time_deltas"][:, seq_slice].clamp_min(0.0)
        else:
            mask = full_mask
            current_mask = full_current_mask
            recent_mask = full_recent_mask
            hist_item_ids = batch["hist_item_ids"]
            hist_cate_ids = batch["hist_cate_ids"]
            hist_btags = batch["hist_btags"]
            time_gaps = batch["hist_time_gaps"].clamp_min(0.0)
            time_deltas = batch["hist_time_deltas"].clamp_min(0.0)

        user = self.user_emb(batch["user_id"])
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))
        user_static, item_static = self.build_static_features(batch)

        temporal = self.temporal_encoding(time_gaps, time_deltas, mask)
        event = self.btag_emb(hist_btags.clamp_min(0))

        hist_item = self.item_emb(hist_item_ids)
        hist_cate = self.cate_emb(hist_cate_ids)
        hist_pair = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))
        event_tokens = (hist_pair + event + temporal) * mask.unsqueeze(-1).float()

        current_tokens = event_tokens * current_mask.unsqueeze(-1).float()
        recent_tokens = event_tokens * recent_mask.unsqueeze(-1).float()
        current_tokens, current_seq_mask = crop_recent(current_tokens, current_mask, self.recent_seq_len)
        recent_tokens, recent_seq_mask = crop_recent(recent_tokens, recent_mask, self.recent_seq_len)

        long_term = self.get_long_term_embedding(batch, event_tokens, long_mask)
        long_term = self.long_term_proj(long_term)
        long_tokens = long_term.unsqueeze(1)
        long_token_mask = torch.ones(long_tokens.size(0), 1, device=long_tokens.device, dtype=torch.bool)

        sequence_tokens = [current_tokens, recent_tokens, long_tokens]
        sequence_masks = [
            ensure_non_empty_mask(current_seq_mask),
            ensure_non_empty_mask(recent_seq_mask),
            long_token_mask,
        ]
        pooled_sequences = [
            masked_mean_pool(tokens, seq_mask) for tokens, seq_mask in zip(sequence_tokens, sequence_masks)
        ]

        current_mean = self.masked_avg(current_tokens, current_seq_mask)
        recent_mean = self.masked_avg(recent_tokens, recent_seq_mask)
        all_mean = (recent_mean + long_term) * 0.5
        time_summary = self.time_summary(time_gaps, time_deltas, mask)
        session_summary = self.session_summary(full_mask, full_current_mask, long_mask)
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

    def get_long_term_embedding(
        self,
        batch: dict[str, torch.Tensor],
        event_tokens: torch.Tensor,
        long_mask: torch.Tensor,
    ) -> torch.Tensor:
        cached = batch.get("long_term_embedding")
        if cached is not None:
            return cached.to(device=event_tokens.device, dtype=event_tokens.dtype)
        return self.masked_avg(event_tokens * long_mask.unsqueeze(-1).float(), long_mask)


def crop_recent(tokens: torch.Tensor, mask: torch.Tensor, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.size(1) <= max_len:
        return tokens, mask
    return tokens[:, -max_len:, :], mask[:, -max_len:]
