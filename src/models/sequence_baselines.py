from __future__ import annotations

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel, MLP
from src.utils import Config


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.attn(sequence))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


class LSTMAttentionModel(CTRBaseModel):
    """LSTM baseline with masked self-attention pooling.

    Adapted from the teammate DIGINETICA implementation and integrated into
    the unified CTR input format, metrics, and user/session-level split.
    """

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.attn_pool = MaskedAttentionPooling(cfg.lstm_hidden_dim)
        input_dim = cfg.embedding_dim * 3 + cfg.lstm_hidden_dim
        self.mlp = MLP(input_dim, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        mask = batch["hist_mask"]
        lengths = mask.sum(dim=1).clamp_min(1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=history.size(1),
        )
        interest = self.attn_pool(lstm_out, mask)
        user = self.user_emb(batch["user_id"])
        history_mean = masked_mean(history, mask)
        features = torch.cat([user, target, interest, history_mean], dim=-1)
        return self.mlp(features)


class TransformerBaselineModel(CTRBaseModel):
    """Full self-attention Transformer encoder baseline."""

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.hyformer_heads,
            dim_feedforward=cfg.hyformer_ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.hyformer_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.mlp = MLP(cfg.embedding_dim * 4, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.embed_target(batch)
        history = self.embed_history(batch)
        mask = batch["hist_mask"]
        history = history + self.pos_emb[:, : history.size(1), :]
        history = self.dropout(history)
        encoded = self.encoder(history, src_key_padding_mask=~mask)
        interest = masked_mean(encoded, mask)
        user = self.user_emb(batch["user_id"])
        history_mean = masked_mean(self.embed_history(batch), mask)
        features = torch.cat([user, target, interest, history_mean], dim=-1)
        return self.mlp(features)
