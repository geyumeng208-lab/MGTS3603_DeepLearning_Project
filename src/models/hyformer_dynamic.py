from __future__ import annotations

import torch

from src.data import FieldDims
from src.models.hyformer_hierarchical import HierarchicalHyFormerModel, chunk_pool
from src.utils import Config

from main_pytorch import ensure_non_empty_mask, masked_mean_pool


class DynamicHyFormerModel(HierarchicalHyFormerModel):
    """Hierarchical HyFormer experiment with dynamic history and lightweight GSU.

    Unlike the plain hierarchical model, this version crops current/recent
    sequences and filters older behaviors by target brand/category before
    chunk pooling. The dense PyTorch implementation is intended for ablation,
    not as a production latency optimization.
    """

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.low_activity_len = cfg.dynamic_low_activity_len
        self.dynamic_recent_len = cfg.dynamic_recent_len

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = ensure_non_empty_mask(batch["hist_mask"])
        current_mask, _ = self.split_session(mask, batch["hist_time_deltas"])
        recent_mask, long_mask = self.split_recent_long(mask)
        long_mask = self.filter_long_history(batch, long_mask)

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

        current_tokens, current_seq_mask = crop_recent(current_tokens, current_mask, self.dynamic_recent_len)
        recent_tokens, recent_seq_mask = crop_recent(recent_tokens, recent_mask, self.dynamic_recent_len)
        long_chunks, long_chunk_mask = chunk_pool(long_tokens, long_mask, self.long_num_chunks)

        sequence_tokens = [current_tokens, recent_tokens, long_chunks]
        sequence_masks = [
            ensure_non_empty_mask(current_seq_mask),
            ensure_non_empty_mask(recent_seq_mask),
            ensure_non_empty_mask(long_chunk_mask),
        ]
        pooled_sequences = [
            masked_mean_pool(tokens, seq_mask) for tokens, seq_mask in zip(sequence_tokens, sequence_masks)
        ]

        all_mean = self.masked_avg(event_tokens, mask)
        current_mean = self.masked_avg(event_tokens * current_mask.unsqueeze(-1).float(), current_mask)
        recent_mean = self.masked_avg(event_tokens * recent_mask.unsqueeze(-1).float(), recent_mask)
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
        recent_len = torch.where(
            valid_counts <= self.low_activity_len,
            valid_counts,
            torch.full_like(valid_counts, self.dynamic_recent_len),
        )
        recent_start = (valid_counts - recent_len).clamp_min(0)
        recent_mask = mask & (positions >= recent_start)
        long_mask = mask & ~recent_mask & (valid_counts > self.dynamic_recent_len)
        return recent_mask, long_mask

    def filter_long_history(self, batch: dict[str, torch.Tensor], long_mask: torch.Tensor) -> torch.Tensor:
        item_match = batch["hist_item_ids"].eq(batch["item_id"].unsqueeze(1))
        cate_match = batch["hist_cate_ids"].eq(batch["cate_id"].unsqueeze(1))
        filtered = long_mask & (item_match | cate_match)
        empty_filtered = ~filtered.any(dim=1)
        has_long = long_mask.any(dim=1)
        fallback_rows = empty_filtered & has_long
        if fallback_rows.any():
            filtered[fallback_rows] = long_mask[fallback_rows]
        return filtered


def crop_recent(tokens: torch.Tensor, mask: torch.Tensor, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.size(1) <= max_len:
        return tokens, mask
    return tokens[:, -max_len:, :], mask[:, -max_len:]
