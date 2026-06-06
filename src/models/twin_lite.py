from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.data import FieldDims
from src.models.base import CTRBaseModel, MLP
from src.models.sim import gather_by_index
from src.utils import Config


class TWINModel(CTRBaseModel):
    """TWIN: unified similarity for GSU retrieval and ESU attention."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.compressed_dim)
        self.key_proj = nn.Linear(cfg.embedding_dim, cfg.compressed_dim)
        self.value_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        self.temperature = nn.Parameter(torch.tensor(cfg.compressed_dim**0.5))
        self.mlp = MLP(cfg.embedding_dim * 4, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        selected_history, selected_mask, selected_scores = self.gsu_and_scores(
            target, history, batch["hist_mask"]
        )
        interest = self.esu(selected_history, selected_mask, selected_scores)
        return self.mlp(self.common_features(batch, target, interest))

    def gsu_and_scores(
        self, target: torch.Tensor, history: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = F.normalize(self.query_proj(target), dim=-1)
        keys = F.normalize(self.key_proj(history), dim=-1)
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / self.temperature.clamp_min(1e-3)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        top_k = min(self.cfg.top_k, history.size(1))
        selected_scores, indices = torch.topk(scores, k=top_k, dim=1)
        selected_history, selected_mask = gather_by_index(history, mask, indices)
        return selected_history, selected_mask, selected_scores

    def esu(
        self, selected_history: torch.Tensor, selected_mask: torch.Tensor, selected_scores: torch.Tensor
    ) -> torch.Tensor:
        scores = selected_scores.masked_fill(~selected_mask, torch.finfo(selected_scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(selected_mask, weights, torch.zeros_like(weights))
        values = self.value_proj(selected_history)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)

