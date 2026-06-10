from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel, MLP
from src.models.sim import gather_by_index
from src.utils import Config


class TWINModelCF4(CTRBaseModel):
    """TWIN with twin_cross_features=4."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.num_heads = cfg.twin_heads
        self.head_dim = cfg.compressed_dim
        if cfg.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by twin_heads")

        inherent_dim = cfg.embedding_dim * 2
        self.query_proj = nn.Linear(inherent_dim, self.num_heads * self.head_dim)
        self.key_proj = nn.Linear(inherent_dim, self.num_heads * self.head_dim)
        self.value_proj = nn.Linear(inherent_dim, self.num_heads * cfg.embedding_dim)

        # Fixed cross_features=4: item_match, cate_match, both_match, item_sim
        self.cross_bias = nn.Linear(4, self.num_heads, bias=False)
        self.head_weights = nn.Parameter(torch.zeros(self.num_heads))
        self.out_proj = nn.Linear(self.num_heads * cfg.embedding_dim, cfg.embedding_dim)
        self.mlp = MLP(cfg.embedding_dim * 5, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        hist_item = self.item_emb(batch["hist_item_ids"])
        hist_cate = self.cate_emb(batch["hist_cate_ids"])

        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))
        history = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))
        selected, selected_mask, selected_scores = self.cp_gsu(
            batch=batch,
            target_item=target_item,
            target_cate=target_cate,
            hist_item=hist_item,
            hist_cate=hist_cate,
        )
        interest = self.esu(selected, selected_mask, selected_scores)

        user = self.user_emb(batch["user_id"])
        history_mean = masked_mean(history, batch["hist_mask"])
        match_features = self.aggregate_match_features(batch)
        features = torch.cat([user, target, interest, history_mean, match_features], dim=-1)
        return self.mlp(features)

    def cp_gsu(
        self,
        batch: dict[str, torch.Tensor],
        target_item: torch.Tensor,
        target_cate: torch.Tensor,
        hist_item: torch.Tensor,
        hist_cate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_inherent = torch.cat([target_item, target_cate], dim=-1)
        history_inherent = torch.cat([hist_item, hist_cate], dim=-1)
        batch_size, seq_len, _ = history_inherent.shape

        query = self.query_proj(target_inherent).view(batch_size, self.num_heads, self.head_dim)
        keys = self.key_proj(history_inherent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(history_inherent).view(
            batch_size, seq_len, self.num_heads * self.cfg.embedding_dim
        )

        attn_scores = torch.einsum("bhd,bshd->bsh", query, keys) / math.sqrt(self.head_dim)
        cross = self.build_cross_features(batch, target_item, target_cate, hist_item, hist_cate)
        per_head_scores = attn_scores + self.cross_bias(cross)
        head_weights = torch.softmax(self.head_weights, dim=0)
        scores = (per_head_scores * head_weights.view(1, 1, -1)).sum(dim=-1)
        scores = scores.masked_fill(~batch["hist_mask"], torch.finfo(scores.dtype).min)

        top_k = min(self.cfg.top_k, seq_len)
        selected_scores, indices = torch.topk(scores, k=top_k, dim=1)
        selected_values, selected_mask = gather_by_index(values, batch["hist_mask"], indices)
        return selected_values, selected_mask, selected_scores

    def esu(
        self,
        selected_values: torch.Tensor,
        selected_mask: torch.Tensor,
        selected_scores: torch.Tensor,
    ) -> torch.Tensor:
        scores = selected_scores.masked_fill(~selected_mask, torch.finfo(selected_scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(selected_mask, weights, torch.zeros_like(weights))
        interest = torch.bmm(weights.unsqueeze(1), selected_values).squeeze(1)
        return self.out_proj(interest)

    def build_cross_features(
        self,
        batch: dict[str, torch.Tensor],
        target_item: torch.Tensor,
        target_cate: torch.Tensor,
        hist_item: torch.Tensor,
        hist_cate: torch.Tensor,
    ) -> torch.Tensor:
        item_match = batch["hist_item_ids"].eq(batch["item_id"].unsqueeze(1)).float()
        cate_match = batch["hist_cate_ids"].eq(batch["cate_id"].unsqueeze(1)).float()
        both_match = item_match * cate_match
        item_sim = F.cosine_similarity(hist_item, target_item.unsqueeze(1), dim=-1)
        return torch.stack([item_match, cate_match, both_match, item_sim], dim=-1)

    def aggregate_match_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["hist_mask"].float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        item_match = batch["hist_item_ids"].eq(batch["item_id"].unsqueeze(1)).float()
        cate_match = batch["hist_cate_ids"].eq(batch["cate_id"].unsqueeze(1)).float()
        both_match = item_match * cate_match
        stats = torch.stack(
            [
                (item_match * mask).sum(dim=1) / denom.squeeze(1),
                (cate_match * mask).sum(dim=1) / denom.squeeze(1),
                (both_match * mask).sum(dim=1) / denom.squeeze(1),
                mask.sum(dim=1) / max(float(batch["hist_mask"].size(1)), 1.0),
            ],
            dim=-1,
        )
        if stats.size(1) < self.cfg.embedding_dim:
            stats = F.pad(stats, (0, self.cfg.embedding_dim - stats.size(1)))
        return stats[:, : self.cfg.embedding_dim]
