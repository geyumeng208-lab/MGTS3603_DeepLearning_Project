from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import auc_score, gauc_score
from src.utils import Config, resolve_device


class Trainer:
    def __init__(self, model: nn.Module, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        if cfg.pos_weight and cfg.pos_weight > 0:
            pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        print(f"device={self.device} model={self.cfg.model}")
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            metrics = self.evaluate(valid_loader)
            print(
                f"epoch={epoch:02d} "
                f"loss={train_loss:.4f} "
                f"val_auc={metrics['auc']:.4f} "
                f"val_gauc={metrics['gauc']:.4f}"
            )

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = self.move_batch(batch)
            logits = self.model(batch)
            loss = self.criterion(logits, batch["label"])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = batch["label"].size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_users: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        for batch in loader:
            batch = self.move_batch(batch)
            logits = self.model(batch)
            preds = torch.sigmoid(logits)
            all_users.append(batch["user_id"].detach().cpu().numpy())
            all_labels.append(batch["label"].detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

        users = np.concatenate(all_users)
        labels = np.concatenate(all_labels)
        preds = np.concatenate(all_preds)
        return {"auc": auc_score(labels, preds), "gauc": gauc_score(users, labels, preds)}

    def move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
