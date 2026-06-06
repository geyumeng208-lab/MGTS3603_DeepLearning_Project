from __future__ import annotations

import torch

from src.data import FieldDims
from src.models.attention import TargetAttention
from src.models.base import CTRBaseModel, MLP
from src.utils import Config


class SIMModel(CTRBaseModel):
    """SIM: category-based GSU retrieval followed by target attention ESU."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.attention = TargetAttention(cfg.embedding_dim, cfg.embedding_dim * 2)
        self.mlp = MLP(cfg.embedding_dim * 4, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        selected_history, selected_mask = self.gsu(batch, history)
        interest, _ = self.attention(target, selected_history, selected_mask)
        return self.mlp(self.common_features(batch, target, interest))

    def gsu(
        self, batch: dict[str, torch.Tensor], history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        same_cate = batch["hist_cate_ids"].eq(batch["cate_id"].unsqueeze(1))
        valid = batch["hist_mask"] & same_cate
        scores = torch.arange(history.size(1), device=history.device).float().unsqueeze(0)
        scores = scores.masked_fill(~valid, -1.0)
        top_k = min(self.cfg.top_k, history.size(1))
        _, indices = torch.topk(scores, k=top_k, dim=1)
        return gather_by_index(history, valid, indices)


def gather_by_index(
    history: torch.Tensor, valid: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = history.size(-1)
    gathered = torch.gather(history, 1, indices.unsqueeze(-1).expand(-1, -1, dim))
    gathered_mask = torch.gather(valid, 1, indices)
    return gathered, gathered_mask

