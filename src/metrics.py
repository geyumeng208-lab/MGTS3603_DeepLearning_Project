from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, preds))


def gauc_score(user_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> float:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for user_id, label, pred in zip(user_ids, labels, preds):
        grouped[int(user_id)].append((float(label), float(pred)))

    weighted_auc = 0.0
    total_weight = 0
    for values in grouped.values():
        group_labels = np.asarray([v[0] for v in values])
        group_preds = np.asarray([v[1] for v in values])
        if len(np.unique(group_labels)) < 2:
            continue
        weight = len(group_labels)
        weighted_auc += auc_score(group_labels, group_preds) * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return float(weighted_auc / total_weight)

