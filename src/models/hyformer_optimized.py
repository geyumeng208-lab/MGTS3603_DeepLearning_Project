from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from src.data import FieldDims
from src.models.attention import masked_mean
from src.models.base import CTRBaseModel, MLP
from src.models.hyformer import MLPQueryGenerator
from src.utils import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYFORMER_ROOT = PROJECT_ROOT / "external" / "Hyformer_Pytorch"
if str(HYFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(HYFORMER_ROOT))

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool  # noqa: E402


class OptimizedHyFormerModel(CTRBaseModel):
    """Task-adapted HyFormer for e-commerce purchase prediction.

    Compared with the fixed repository-adapted HyFormer, this version keeps the
    same HyFormerBackbone but maps the current dataset more carefully:
    brand history and category history are treated as two heterogeneous
    sequences, while user/target/history summary features are tokenized as
    non-sequence context.
    """

    def __init__(self, cfg: Config, field_dims: FieldDims):
        super().__init__(cfg, field_dims)
        self.d_model = cfg.embedding_dim
        self.num_sequences = 2
        self.num_non_seq_tokens = cfg.hyformer_non_seq_tokens
        self.num_query_tokens = cfg.hyformer_query_tokens

        non_seq_dim = cfg.embedding_dim * 5
        global_info_dim = non_seq_dim + self.num_sequences * cfg.embedding_dim
        self.non_seq_tokenizer = nn.Sequential(
            nn.Linear(non_seq_dim, cfg.hyformer_ff_dim),
            nn.SiLU(),
            nn.Linear(cfg.hyformer_ff_dim, self.num_non_seq_tokens * cfg.embedding_dim),
        )
        self.query_generators = nn.ModuleList(
            [
                MLPQueryGenerator(
                    global_info_dim=global_info_dim,
                    num_query_tokens=self.num_query_tokens,
                    d_model=cfg.embedding_dim,
                    hidden_dim=cfg.hyformer_ff_dim,
                )
                for _ in range(self.num_sequences)
            ]
        )
        self.backbone = HyFormerBackbone(
            num_layers=cfg.hyformer_layers,
            num_sequences=self.num_sequences,
            num_queries_per_sequence=self.num_query_tokens,
            num_non_seq_tokens=self.num_non_seq_tokens,
            d_model=cfg.embedding_dim,
            num_heads=cfg.hyformer_heads,
            ffn_hidden=cfg.hyformer_ff_dim,
            encoder_type=cfg.hyformer_encoder_type,
            short_seq_len=cfg.hyformer_short_seq_len,
        )
        self.head = MLP(cfg.embedding_dim * 4, cfg.hidden_dims, cfg.dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = ensure_non_empty_mask(batch["hist_mask"])
        user = self.user_emb(batch["user_id"])
        target_item = self.item_emb(batch["item_id"])
        target_cate = self.cate_emb(batch["cate_id"])
        target = self.item_proj(torch.cat([target_item, target_cate], dim=-1))

        hist_item = self.item_emb(batch["hist_item_ids"]) * mask.unsqueeze(-1).float()
        hist_cate = self.cate_emb(batch["hist_cate_ids"]) * mask.unsqueeze(-1).float()
        hist_pair = self.item_proj(torch.cat([hist_item, hist_cate], dim=-1))
        hist_mean = masked_mean(hist_pair, mask)
        item_mean = masked_mean(hist_item, mask)
        cate_mean = masked_mean(hist_cate, mask)

        sequence_tokens = [hist_item, hist_cate]
        sequence_masks = [mask, mask]
        pooled_sequences = [masked_mean_pool(seq, mask) for seq in sequence_tokens]

        non_seq_x = torch.cat([user, target_item, target_cate, item_mean, cate_mean], dim=-1)
        non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(
            non_seq_x.size(0), self.num_non_seq_tokens, self.d_model
        )
        global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
        query_tokens = [generator(global_info) for generator in self.query_generators]

        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        boosted = boosted_tokens.mean(dim=1)
        features = torch.cat([user, target, hist_mean, boosted], dim=-1)
        return self.head(features)
