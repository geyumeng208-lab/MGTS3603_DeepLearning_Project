from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.utils import Config


class CTRBaseModel(nn.Module):
    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.user_emb = nn.Embedding(field_dims.num_users, cfg.embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(field_dims.num_items, cfg.embedding_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(field_dims.num_categories, cfg.embedding_dim, padding_idx=0)
        self.item_proj = nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim)
        self.btag_num_types = cfg.btag_num_types

    def embed_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item = self.item_emb(batch["item_id"])
        cate = self.cate_emb(batch["cate_id"])
        return self.item_proj(torch.cat([item, cate], dim=-1))

    def embed_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item = self.item_emb(batch["hist_item_ids"])
        cate = self.cate_emb(batch["hist_cate_ids"])
        return self.item_proj(torch.cat([item, cate], dim=-1))

    def common_features(
        self, batch: dict[str, torch.Tensor], target: torch.Tensor, interest: torch.Tensor
    ) -> torch.Tensor:
        user = self.user_emb(batch["user_id"])
        history = self.embed_history(batch)
        history_mean = masked_mean(history, batch["hist_mask"])
        return torch.cat([user, target, interest, history_mean], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

