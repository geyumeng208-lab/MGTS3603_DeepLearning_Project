"""
Phase 2: Knowledge Distillation — Teacher (HyFormer-Hierarchical) → Student (TWIN).

Usage:
    python train_distill.py --model twin --teacher hyformer_hierarchical
    python train_distill.py --model twin --teacher hyformer_hierarchical --alpha 0.5 --T 2.0

The script:
    1. Loads Teacher model (HyFormer-Hierarchical or HyFormer-Static), freezes it.
    2. Loads Student model (TWIN).
    3. Trains Student with: Loss = CE(student, labels) + alpha * KL(teacher_soft, student_soft)
    4. Evaluates Student on validation set.

Teacher model must match the model type specified by --teacher.
Student model is specified by --model (e.g., twin, twin_gate_fusion, twin_nonlinear_sim).
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import build_dataloaders
from src.metrics import auc_score, gauc_score
from src.models import build_model
from src.utils import Config, set_seed, resolve_device
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: Train TWIN (student) with HyFormer (teacher) soft labels."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="twin",
                        help="Student model name")
    parser.add_argument("--teacher", type=str, default="hyformer_hierarchical",
                        help="Teacher model name (frozen)")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss. alpha=0 → CE only.")
    parser.add_argument("--T", type=float, default=2.0,
                        help="Temperature for soft-label distillation.")
    parser.add_argument("--device", type=str, default=None)
    # Add all common args from train.py
    for key in ["synthetic_samples", "top_k", "pos_weight", "session_gap_minutes",
                "embedding_dim", "lstm_hidden_dim", "hyformer_layers", "hyformer_heads",
                "hyformer_ff_dim", "twin_heads", "recent_seq_len", "long_num_chunks",
                "dynamic_recent_len", "learning_rate", "weight_decay", "num_workers",
                "hyformer_encoder_type"]:
        parser.add_argument(f"--{key}", type=None if key in ("learning_rate", "weight_decay", "pos_weight",
                            "session_gap_minutes", "embedding_dim", "lstm_hidden_dim", "lstm_layers",
                            "hyformer_ff_dim") else int if key not in ("learning_rate", "weight_decay",
                            "pos_weight", "session_gap_minutes") else float, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    import yaml
    from dataclasses import fields

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    for key in ["model", "data_path", "epochs", "batch_size", "max_seq_len", "top_k",
                "synthetic_samples", "pos_weight", "session_gap_minutes", "embedding_dim",
                "lstm_hidden_dim", "hyformer_layers", "hyformer_heads", "hyformer_ff_dim",
                "twin_heads", "recent_seq_len", "long_num_chunks", "dynamic_recent_len",
                "learning_rate", "weight_decay", "num_workers", "hyformer_encoder_type", "device"]:
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
    known_fields = {field.name for field in fields(Config)}
    unknown_fields = sorted(set(raw) - known_fields)
    if unknown_fields:
        raise ValueError(f"Unknown config fields: {unknown_fields}")
    return Config(**raw)


class DistillTrainer:
    """Teacher-Student knowledge distillation trainer."""

    def __init__(self, student_model: nn.Module, teacher_model: nn.Module | None, cfg: Config, alpha: float, T: float,
                 student_name: str = "", teacher_name: str = ""):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.student = student_model.to(self.device)
        self.teacher = None
        if teacher_model is not None:
            self.teacher = teacher_model.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.T = T
        self.student_model_name = student_name or cfg.model
        self.teacher_model_name = teacher_name or "None"
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        teacher_name = self.teacher_model_name or "None"
        print(f"device={self.device} model=distill student={self.student_model_name} teacher={teacher_name} alpha={self.alpha} T={self.T}")
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
        self.student.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                teacher_logits = self.teacher(batch) if self.teacher is not None else None
            student_logits = self.student(batch)

            ce_loss = self.criterion(student_logits, batch["label"])
            if self.alpha > 0 and teacher_logits is not None:
                # Soft-label KL divergence
                t_prob = torch.sigmoid(teacher_logits / self.T)
                s_prob = torch.sigmoid(student_logits / self.T)
                kd_loss = -(t_prob * torch.log(s_prob + 1e-9) + (1 - t_prob) * torch.log(1 - s_prob + 1e-9))
                kd_loss = (kd_loss * self.T * self.T).mean()
                loss = ce_loss + self.alpha * kd_loss
            else:
                loss = ce_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            bs = batch["label"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.student.eval()
        all_users, all_labels, all_preds = [], [], []
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            logits = self.student(batch)
            preds = torch.sigmoid(logits)
            all_users.append(batch["user_id"].detach().cpu().numpy())
            all_labels.append(batch["label"].detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
        users = __import__("numpy").concatenate(all_users)
        labels = __import__("numpy").concatenate(all_labels)
        preds = __import__("numpy").concatenate(all_preds)
        return {"auc": auc_score(labels, preds), "gauc": gauc_score(users, labels, preds)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg.seed)

    train_loader, valid_loader, field_dims = build_dataloaders(cfg)

    # Load teacher model.
    teacher_model = build_model(cfg, field_dims)
    teacher_model.eval()

    # Load student model (different from teacher).
    student_cfg = Config(**dict(cfg.__dict__))
    student_cfg.model = args.model
    student_model = build_model(student_cfg, field_dims)

    trainer = DistillTrainer(student_model, teacher_model, cfg, alpha=args.alpha, T=args.T,
                             student_name=args.model, teacher_name=args.teacher)
    trainer.fit(train_loader, valid_loader)


if __name__ == "__main__":
    main()
