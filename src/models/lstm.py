from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel, MLP
from src.utils import Config


class LSTMBaseModel(CTRBaseModel):
    """LSTM baseline: encode the full behavior sequence with recurrent modeling."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        input_dim = cfg.embedding_dim * 3 + cfg.lstm_hidden_dim
        self.mlp = MLP(input_dim, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        sequence_state = self.encode_sequence(history, batch["hist_mask"])
        user = self.user_emb(batch["user_id"])
        history_mean = masked_mean(history, batch["hist_mask"])
        features = torch.cat([user, target, sequence_state, history_mean], dim=-1)
        return self.mlp(features)

    def encode_sequence(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]
