from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import TargetAttention
from src.models.base import CTRBaseModel, MLP
from src.models.sim import gather_by_index
from src.utils import Config


class ETAModel(CTRBaseModel):
    """ETA: SimHash retrieval using Hamming distance, then target attention."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.hash_proj = nn.Linear(cfg.embedding_dim, cfg.hash_bits, bias=False)
        self.attention = TargetAttention(cfg.embedding_dim, cfg.embedding_dim * 2)
        self.mlp = MLP(cfg.embedding_dim * 4, cfg.hidden_dims, cfg.dropout)
        nn.init.normal_(self.hash_proj.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        selected_history, selected_mask = self.gsu(target, history, batch["hist_mask"])
        interest, _ = self.attention(target, selected_history, selected_mask)
        return self.mlp(self.common_features(batch, target, interest))

    def gsu(
        self, target: torch.Tensor, history: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_hash = self.hash_proj(target).gt(0)
        history_hash = self.hash_proj(history).gt(0)
        distance = torch.logical_xor(history_hash, target_hash.unsqueeze(1)).float().sum(dim=-1)
        scores = -distance.masked_fill(~mask, float(self.cfg.hash_bits + 1))
        top_k = min(self.cfg.top_k, history.size(1))
        _, indices = torch.topk(scores, k=top_k, dim=1)
        return gather_by_index(history, mask, indices)

