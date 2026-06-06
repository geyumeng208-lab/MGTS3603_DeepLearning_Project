from __future__ import annotations

import torch
from torch import nn


class TargetAttention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, target: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_expanded = target.unsqueeze(1).expand_as(sequence)
        features = torch.cat(
            [
                sequence,
                target_expanded,
                sequence * target_expanded,
                sequence - target_expanded,
            ],
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        interest = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return interest, weights


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (sequence * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom

