from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils import Config


@dataclass(frozen=True)
class FieldDims:
    num_users: int
    num_items: int
    num_categories: int


class TaobaoAdDataset(Dataset):
    def __init__(self, samples: list[dict], max_seq_len: int):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.samples[index]
        hist_items, hist_cates, mask = pad_history(
            row["hist_item_ids"], row["hist_cate_ids"], self.max_seq_len
        )
        hist_time_gaps = pad_numeric_history(row.get("hist_time_gaps", []), self.max_seq_len)
        hist_time_deltas = pad_numeric_history(row.get("hist_time_deltas", []), self.max_seq_len)
        hist_btags = pad_id_history(row.get("hist_btags", []), self.max_seq_len)
        return {
            "user_id": torch.tensor(row["user_id"], dtype=torch.long),
            "item_id": torch.tensor(row["item_id"], dtype=torch.long),
            "cate_id": torch.tensor(row["cate_id"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "hist_item_ids": torch.tensor(hist_items, dtype=torch.long),
            "hist_cate_ids": torch.tensor(hist_cates, dtype=torch.long),
            "hist_mask": torch.tensor(mask, dtype=torch.bool),
            "hist_time_gaps": torch.tensor(hist_time_gaps, dtype=torch.float32),
            "hist_time_deltas": torch.tensor(hist_time_deltas, dtype=torch.float32),
            "hist_btags": torch.tensor(hist_btags, dtype=torch.long),
            "user_static_ids": torch.tensor(row.get("user_static_ids", [0] * 6), dtype=torch.long),
            "item_static_values": torch.tensor(row.get("item_static_values", [0.0] * 4), dtype=torch.float32),
        }


def pad_history(
    item_ids: Iterable[int], cate_ids: Iterable[int], max_seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = np.asarray(list(item_ids)[-max_seq_len:], dtype=np.int64)
    cates = np.asarray(list(cate_ids)[-max_seq_len:], dtype=np.int64)
    length = min(len(items), len(cates), max_seq_len)

    padded_items = np.zeros(max_seq_len, dtype=np.int64)
    padded_cates = np.zeros(max_seq_len, dtype=np.int64)
    mask = np.zeros(max_seq_len, dtype=bool)
    if length > 0:
        padded_items[-length:] = items[-length:]
        padded_cates[-length:] = cates[-length:]
        mask[-length:] = True
    return padded_items, padded_cates, mask


def pad_numeric_history(values: Iterable[float], max_seq_len: int) -> np.ndarray:
    array = np.asarray(list(values)[-max_seq_len:], dtype=np.float32)
    length = min(len(array), max_seq_len)
    padded = np.zeros(max_seq_len, dtype=np.float32)
    if length > 0:
        padded[-length:] = array[-length:]
    return padded


def pad_id_history(values: Iterable[int], max_seq_len: int) -> np.ndarray:
    array = np.asarray(list(values)[-max_seq_len:], dtype=np.int64)
    length = min(len(array), max_seq_len)
    padded = np.zeros(max_seq_len, dtype=np.int64)
    if length > 0:
        padded[-length:] = array[-length:]
    return padded


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, FieldDims]:
    if cfg.data_path:
        samples, field_dims = load_csv(Path(cfg.data_path), cfg)
    else:
        samples, field_dims = generate_synthetic_samples(cfg)

    dataset = TaobaoAdDataset(samples, cfg.max_seq_len)
    valid_size = max(1, int(len(dataset) * cfg.valid_ratio))
    train_size = len(dataset) - valid_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, field_dims


def load_csv(path: Path, cfg: Config) -> tuple[list[dict], FieldDims]:
    samples: list[dict] = []
    max_user = max_item = max_cate = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user_id", "ad_id", "cate_id", "label", "hist_ad_ids", "hist_cate_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        for row in reader:
            user_id = int(row["user_id"])
            item_id = int(row["ad_id"])
            cate_id = int(row["cate_id"])
            hist_items = parse_sequence(row["hist_ad_ids"])
            hist_cates = parse_sequence(row["hist_cate_ids"])
            hist_time_gaps = parse_float_sequence(row.get("hist_time_gaps", ""))
            hist_time_deltas = parse_float_sequence(row.get("hist_time_deltas", ""))
            hist_btags = parse_sequence(row.get("hist_btags", ""))
            user_static_ids = [
                int(row.get("user_gender", 0) or 0),
                int(row.get("user_age", 0) or 0),
                int(row.get("user_pvalue", 0) or 0),
                int(row.get("user_shopping", 0) or 0),
                int(row.get("user_occupation", 0) or 0),
                int(row.get("user_new_level", 0) or 0),
            ]
            item_static_values = [
                float(row.get("brand_price_mean", 0.0) or 0.0),
                float(row.get("brand_ad_count", 0.0) or 0.0),
                float(row.get("cate_price_mean", 0.0) or 0.0),
                float(row.get("cate_ad_count", 0.0) or 0.0),
            ]
            max_user = max(max_user, user_id)
            max_item = max(max_item, item_id, *(hist_items or [0]))
            max_cate = max(max_cate, cate_id, *(hist_cates or [0]))
            samples.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "cate_id": cate_id,
                    "label": float(row["label"]),
                    "hist_item_ids": hist_items,
                    "hist_cate_ids": hist_cates,
                    "hist_time_gaps": hist_time_gaps,
                    "hist_time_deltas": hist_time_deltas,
                    "hist_btags": hist_btags,
                    "user_static_ids": user_static_ids,
                    "item_static_values": item_static_values,
                }
            )

    if not samples:
        raise ValueError(f"{path} 中没有可训练样本")

    field_dims = FieldDims(
        num_users=max(max_user + 1, cfg.num_users + 1),
        num_items=max(max_item + 1, cfg.num_items + 1),
        num_categories=max(max_cate + 1, cfg.num_categories + 1),
    )
    return samples, field_dims


def parse_sequence(value: str) -> list[int]:
    if not value:
        return []
    return [int(token) for token in value.replace(",", " ").split() if token]


def parse_float_sequence(value: str) -> list[float]:
    if not value:
        return []
    return [float(token) for token in value.replace(",", " ").split() if token]


def generate_synthetic_samples(cfg: Config) -> tuple[list[dict], FieldDims]:
    rng = np.random.default_rng(cfg.seed)
    user_interest = rng.integers(1, cfg.num_categories + 1, size=cfg.num_users + 1)
    samples: list[dict] = []

    for _ in range(cfg.synthetic_samples):
        user_id = int(rng.integers(1, cfg.num_users + 1))
        main_cate = int(user_interest[user_id])
        length = int(rng.integers(cfg.min_seq_len, cfg.max_seq_len + 1))

        hist_cates = rng.integers(1, cfg.num_categories + 1, size=length)
        interest_positions = rng.random(length) < 0.55
        hist_cates[interest_positions] = main_cate
        hist_items = ((hist_cates - 1) * 64 + rng.integers(1, 64, size=length))
        hist_items = np.clip(hist_items, 1, cfg.num_items)

        positive = rng.random() < 0.5
        if positive:
            cate_id = main_cate if rng.random() < 0.8 else int(rng.choice(hist_cates))
        else:
            cate_id = int(rng.integers(1, cfg.num_categories + 1))
            if cate_id == main_cate:
                cate_id = (cate_id % cfg.num_categories) + 1

        item_id = int(np.clip((cate_id - 1) * 64 + rng.integers(1, 64), 1, cfg.num_items))
        recent_match = np.mean(hist_cates[-min(80, length) :] == cate_id)
        logit = -1.2 + 4.0 * recent_match + (1.0 if cate_id == main_cate else -0.4)
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = float(rng.random() < prob)

        samples.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "cate_id": cate_id,
                "label": label,
                "hist_item_ids": hist_items.astype(int).tolist(),
                "hist_cate_ids": hist_cates.astype(int).tolist(),
            }
        )

    field_dims = FieldDims(cfg.num_users + 1, cfg.num_items + 1, cfg.num_categories + 1)
    return samples, field_dims
