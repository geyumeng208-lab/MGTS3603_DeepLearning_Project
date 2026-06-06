from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build purchase-prediction sequences from DIGINETICA.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/dataset-train-diginetica"))
    parser.add_argument("--output", type=Path, default=Path("data/diginetica_sequence_100k.csv"))
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_history", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=2)
    parser.add_argument("--neg_per_pos", type=int, default=2)
    return parser.parse_args()


def load_categories(path: Path) -> dict[int, int]:
    categories: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            item = safe_int(row.get("itemId", ""))
            cate = safe_int(row.get("categoryId", ""))
            if item > 0 and cate > 0 and item not in categories:
                categories[item] = cate
    return categories


def load_prices(path: Path) -> dict[int, float]:
    prices: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            item = safe_int(row.get("itemId", ""))
            price = safe_float(row.get("pricelog2", ""))
            if item > 0:
                prices[item] = price
    return prices


def load_purchases(path: Path) -> dict[int, set[int]]:
    purchases: dict[int, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            session = safe_int(row.get("sessionId", ""))
            item = safe_int(row.get("itemId", ""))
            if session > 0 and item > 0:
                purchases[session].add(item)
    return purchases


def load_views(path: Path) -> dict[int, list[tuple[int, int]]]:
    views: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            session = safe_int(row.get("sessionId", ""))
            item = safe_int(row.get("itemId", ""))
            timeframe = safe_int(row.get("timeframe", ""))
            if session > 0 and item > 0:
                views[session].append((timeframe, item))
    for session in list(views):
        views[session].sort(key=lambda x: x[0])
    return views


def write_samples(
    output: Path,
    views: dict[int, list[tuple[int, int]]],
    purchases: dict[int, set[int]],
    categories: dict[int, int],
    prices: dict[int, float],
    max_samples: int,
    max_history: int,
    min_history: int,
    neg_per_pos: int,
) -> tuple[int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = positives = negatives = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "user_id",
            "ad_id",
            "cate_id",
            "label",
            "hist_ad_ids",
            "hist_cate_ids",
            "hist_btags",
            "hist_time_gaps",
            "hist_time_deltas",
            "user_gender",
            "user_age",
            "user_pvalue",
            "user_shopping",
            "user_occupation",
            "user_new_level",
            "brand_price_mean",
            "brand_ad_count",
            "cate_price_mean",
            "cate_ad_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for session, events in views.items():
            purchased = purchases.get(session, set())
            if len(events) <= min_history:
                continue
            session_negatives = 0
            for idx in range(min_history, len(events)):
                timestamp, item = events[idx]
                label = 1 if item in purchased else 0
                if label == 0:
                    if session_negatives >= max(1, len(purchased) * neg_per_pos):
                        continue
                    session_negatives += 1

                history = events[max(0, idx - max_history) : idx]
                hist_times = [t for t, _ in history]
                hist_items = [it for _, it in history]
                hist_cates = [categories.get(it, 0) for it in hist_items]
                gaps = [max(timestamp - t, 0) for t in hist_times]
                deltas = [0 if j == 0 else max(hist_times[j] - hist_times[j - 1], 0) for j in range(len(hist_times))]
                cate = categories.get(item, 0)

                writer.writerow(
                    {
                        "user_id": session,
                        "ad_id": item,
                        "cate_id": cate,
                        "label": label,
                        "hist_ad_ids": " ".join(str(x) for x in hist_items),
                        "hist_cate_ids": " ".join(str(x) for x in hist_cates),
                        "hist_btags": " ".join("1" for _ in hist_items),
                        "hist_time_gaps": " ".join(str(x) for x in gaps),
                        "hist_time_deltas": " ".join(str(x) for x in deltas),
                        "user_gender": 0,
                        "user_age": 0,
                        "user_pvalue": 0,
                        "user_shopping": 0,
                        "user_occupation": 0,
                        "user_new_level": 0,
                        "brand_price_mean": prices.get(item, 0.0),
                        "brand_ad_count": 1.0,
                        "cate_price_mean": 0.0,
                        "cate_ad_count": 1.0 if cate > 0 else 0.0,
                    }
                )
                written += 1
                positives += label
                negatives += 1 - label
                if written >= max_samples:
                    return written, positives, negatives
    return written, positives, negatives


def safe_int(value: str) -> int:
    value = (value or "").strip()
    if not value or value.upper() == "NA":
        return 0
    return int(float(value))


def safe_float(value: str) -> float:
    value = (value or "").strip()
    if not value or value.upper() == "NA":
        return 0.0
    return float(value)


def main() -> None:
    args = parse_args()
    categories = load_categories(args.input_dir / "product-categories.csv")
    prices = load_prices(args.input_dir / "products.csv")
    purchases = load_purchases(args.input_dir / "train-purchases.csv")
    views = load_views(args.input_dir / "train-item-views.csv")
    written, positives, negatives = write_samples(
        output=args.output,
        views=views,
        purchases=purchases,
        categories=categories,
        prices=prices,
        max_samples=args.max_samples,
        max_history=args.max_history,
        min_history=args.min_history,
        neg_per_pos=args.neg_per_pos,
    )
    print(
        f"wrote={written:,} positives={positives:,} negatives={negatives:,} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
