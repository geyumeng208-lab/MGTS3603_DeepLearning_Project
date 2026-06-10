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


class TWINModelGateFusion(CTRBaseModel):
    """TWIN with learnable GSU-ESU gate fusion (Step 3).

    In vanilla TWIN, GSU (top-K hard filter) and ESU (softmax weighted sum)
    are hard-wired: GSU selects candidates, ESU aggregates them with no
    learned gate.  This version inserts a **per-sample gate** between the
    hard-filtered values and the attention pool, letting the model learn
    when to rely on the hard-topK subset and when to fall back to a full
    history mean.

    Key changes:
    - GSU now returns both hard-selected values and a full-history mean.
    - A sigmoid-gated scalar ``alpha`` blends them element-wise.
    - ``alpha`` is derived from the same user+target features that the
      prediction head sees, making it content-aware.

    Params added: gate_proj (embedding_dim*4 → 1), negligible overhead.
    """

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

        self.cross_bias = nn.Linear(cfg.twin_cross_features, self.num_heads, bias=False)
        self.head_weights = nn.Parameter(torch.zeros(self.num_heads))
        self.out_proj = nn.Linear(self.num_heads * cfg.embedding_dim, cfg.embedding_dim)
        self.mlp = MLP(cfg.embedding_dim * 5, cfg.hidden_dims, cfg.dropout)

        # --- Step 3: Gate for GSU-ESU fusion ---
        # Gate: user(32) + target(32) + hist_mean_proj(32) → scalar alpha.
        gate_input_dim = cfg.embedding_dim * 3  # 32 + 32 + 32 = 96
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, cfg.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.embedding_dim, 1),
        )
        # Project history dim (embedding_dim, after item_proj) to 128-d (num_heads * embedding_dim).
        self.hist_mean_to_selected = nn.Linear(cfg.embedding_dim, self.num_heads * cfg.embedding_dim)
        # --- End Step 3 ---

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        hist_item = self.item_emb(batch["hist_item_ids"])
        hist_cate = self.cate_emb(batch["hist_cate_ids"])

        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))
        history = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))

        # GSU returns hard-selected + full-history mean + scores.
        selected, selected_mask, selected_scores, hist_mean = self.cp_gsu(
            batch=batch,
            target_item=target_item,
            target_cate=target_cate,
            hist_item=hist_item,
            hist_cate=hist_cate,
            history=history,
        )

        # --- Step 3: Gate fusion between GSU (hard-selected) and ESU (full mean) ---
        alpha = self.gate_from_target(batch, target_item, target_cate, hist_mean)
        # alpha shape: [B] (scalar per sample, broadcasts to all dims)
        # selected shape: [B, top_k, 128] → mean over top_k
        selected_mean = (selected * selected_mask.unsqueeze(-1)).sum(dim=1) / selected_mask.sum(dim=1, keepdim=True).clamp_min(1)
        # Project 64-d hist_mean to 128-d to match selected_mean
        projected_hist_mean = self.hist_mean_to_selected(hist_mean)
        # alpha: [B] vs selected_mean/projected_hist_mean: [B, 128]
        interest = alpha.unsqueeze(1) * selected_mean + (1 - alpha).unsqueeze(1) * projected_hist_mean
        interest = self.out_proj(interest)
        # --- End Step 3 ---

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
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # --- Step 3: Compute full-history mean for gate fallback ---
        hist_mean = masked_mean(history, batch["hist_mask"])
        # --- End Step 3 ---

        return selected_values, selected_mask, selected_scores, hist_mean

    def gate_from_target(
        self,
        batch: dict[str, torch.Tensor],
        target_item: torch.Tensor,
        target_cate: torch.Tensor,
        hist_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample gate alpha ∈ [0, 1].

        hist_mean comes from masked_mean(history, mask) where history is
        [B, S, 2*D] (item+cate concat), so its last dim is 2*D.
        We project it back to D before concatenating with user and target.
        """
        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))
        user = self.user_emb(batch["user_id"])

        # Project hist_mean from 2*embedding_dim back to embedding_dim.
        if hist_mean.size(-1) != self.cfg.embedding_dim:
            if not hasattr(self, "hist_mean_proj"):
                self.hist_mean_proj = nn.Linear(hist_mean.size(-1), self.cfg.embedding_dim)
            hist_mean = self.hist_mean_proj(hist_mean)

        gate_input = torch.cat([user, target, hist_mean], dim=-1)
        alpha = torch.sigmoid(self.gate_proj(gate_input).squeeze(-1))
        return alpha  # [B]

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
        cate_sim = F.cosine_similarity(hist_cate, target_cate.unsqueeze(1), dim=-1)
        seq_len = batch["hist_mask"].size(1)
        recency = torch.linspace(0.0, 1.0, steps=seq_len, device=hist_item.device).unsqueeze(0)
        recency = recency.expand_as(item_match)
        return torch.stack([item_match, cate_match, both_match, item_sim, cate_sim, recency], dim=-1)

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
        # Pad to embedding_dim so the downstream MLP shape remains stable.
        if stats.size(1) < self.cfg.embedding_dim:
            stats = F.pad(stats, (0, self.cfg.embedding_dim - stats.size(1)))
        return stats[:, : self.cfg.embedding_dim]
