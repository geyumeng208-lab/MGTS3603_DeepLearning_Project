from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


BTAG_WEIGHT = {
    "pv": 1,
    "fav": 2,
    "cart": 3,
    "buy": 4,
}
BTAG_ID = {
    "pv": 1,
    "fav": 2,
    "cart": 3,
    "buy": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build purchase-prediction sequences from behavior_log.csv.")
    parser.add_argument("--behavior_log", type=Path, default=Path("src/sampled_10pct/behavior_log.csv"))
    parser.add_argument("--user_profile", type=Path, default=Path("src/sampled_10pct/user_profile.csv"))
    parser.add_argument("--ad_feature", type=Path, default=Path("src/sampled_10pct/ad_feature.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/purchase_sequence_100k.csv"))
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_history", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=5)
    parser.add_argument("--neg_sample_rate", type=float, default=0.02)
    parser.add_argument("--neg_strategy", type=str, choices=["uniform", "hard"], default="uniform")
    parser.add_argument("--pv_neg_rate", type=float, default=0.005)
    parser.add_argument("--fav_neg_rate", type=float, default=0.15)
    parser.add_argument("--cart_neg_rate", type=float, default=0.25)
    parser.add_argument("--with_static_features", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--min_timestamp", type=int, default=1400000000)
    parser.add_argument("--max_timestamp", type=int, default=1600000000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def safe_int(value: str) -> int:
    value = (value or "").strip()
    if not value or value.upper() == "NULL":
        return 0
    return int(float(value))


def load_user_profiles(path: Path) -> dict[str, dict[str, int]]:
    profiles: dict[str, dict[str, int]] = {}
    if not path.exists():
        return profiles
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles[row["userid"]] = {
                "user_gender": safe_int(row.get("final_gender_code", "")),
                "user_age": safe_int(row.get("age_level", "")),
                "user_pvalue": safe_int(row.get("pvalue_level", "")),
                "user_shopping": safe_int(row.get("shopping_level", "")),
                "user_occupation": safe_int(row.get("occupation", "")),
                "user_new_level": safe_int(row.get("new_user_class_level ", row.get("new_user_class_level", ""))),
            }
    return profiles


def load_product_stats(path: Path) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    brand_prices: dict[int, list[float]] = defaultdict(list)
    cate_prices: dict[int, list[float]] = defaultdict(list)
    brand_counts: Counter[int] = Counter()
    cate_counts: Counter[int] = Counter()
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = safe_int(row.get("brand", ""))
            cate = safe_int(row.get("cate_id", ""))
            price = safe_float(row.get("price", ""))
            if brand > 0:
                brand_counts[brand] += 1
                if price > 0:
                    brand_prices[brand].append(price)
            if cate > 0:
                cate_counts[cate] += 1
                if price > 0:
                    cate_prices[cate].append(price)
    brand_stats = {
        brand: {
            "brand_price_mean": sum(brand_prices.get(brand, [0.0])) / max(len(brand_prices.get(brand, [])), 1),
            "brand_ad_count": float(count),
        }
        for brand, count in brand_counts.items()
    }
    cate_stats = {
        cate: {
            "cate_price_mean": sum(cate_prices.get(cate, [0.0])) / max(len(cate_prices.get(cate, [])), 1),
            "cate_ad_count": float(count),
        }
        for cate, count in cate_counts.items()
    }
    return brand_stats, cate_stats


def safe_float(value: str) -> float:
    value = (value or "").strip()
    if not value or value.upper() == "NULL":
        return 0.0
    return float(value)


def should_sample_negative(btag: str, args: argparse.Namespace, rng: random.Random) -> bool:
    if btag == "buy":
        return False
    if args.neg_strategy == "uniform":
        return rng.random() < args.neg_sample_rate
    rates = {
        "pv": args.pv_neg_rate,
        "fav": args.fav_neg_rate,
        "cart": args.cart_neg_rate,
    }
    return rng.random() < rates.get(btag, 0.0)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    user_profiles = load_user_profiles(args.user_profile) if args.with_static_features else {}
    brand_stats, cate_stats = load_product_stats(args.ad_feature) if args.with_static_features else ({}, {})

    histories: dict[str, deque[tuple[int, int, int, int]]] = defaultdict(lambda: deque(maxlen=args.max_history))
    scanned = 0
    written = 0
    positives = 0
    negatives = 0

    with args.behavior_log.open("r", encoding="utf-8-sig", newline="") as fin, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
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
        ]
        if args.with_static_features:
            fieldnames.extend(
                [
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
            )
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            scanned += 1
            if args.max_rows > 0 and scanned > args.max_rows:
                break

            user = row["user"]
            timestamp = safe_int(row["time_stamp"])
            if timestamp < args.min_timestamp or timestamp > args.max_timestamp:
                continue
            btag = row["btag"]
            cate = safe_int(row["cate"])
            brand = max(safe_int(row["brand"]), 1)
            if cate <= 0 or btag not in BTAG_WEIGHT:
                continue

            history = histories[user]
            label = 1 if btag == "buy" else 0
            should_write = len(history) >= args.min_history and (
                label == 1 or should_sample_negative(btag, args, rng)
            )

            if should_write:
                selected = list(history)[-args.max_history :]
                timestamps = [item[0] for item in selected]
                time_gaps = [max(timestamp - hist_ts, 0) for hist_ts in timestamps]
                time_deltas = [
                    0 if idx == 0 else max(timestamps[idx] - timestamps[idx - 1], 0)
                    for idx in range(len(timestamps))
                ]
                output_row = {
                    "user_id": user,
                    "ad_id": brand,
                    "cate_id": cate,
                    "label": label,
                    "hist_ad_ids": " ".join(str(item[1]) for item in selected),
                    "hist_cate_ids": " ".join(str(item[2]) for item in selected),
                    "hist_btags": " ".join(str(item[3]) for item in selected),
                    "hist_time_gaps": " ".join(str(value) for value in time_gaps),
                    "hist_time_deltas": " ".join(str(value) for value in time_deltas),
                }
                if args.with_static_features:
                    profile = user_profiles.get(user, {})
                    bstats = brand_stats.get(brand, {})
                    cstats = cate_stats.get(cate, {})
                    output_row.update(
                        {
                            "user_gender": profile.get("user_gender", 0),
                            "user_age": profile.get("user_age", 0),
                            "user_pvalue": profile.get("user_pvalue", 0),
                            "user_shopping": profile.get("user_shopping", 0),
                            "user_occupation": profile.get("user_occupation", 0),
                            "user_new_level": profile.get("user_new_level", 0),
                            "brand_price_mean": bstats.get("brand_price_mean", 0.0),
                            "brand_ad_count": bstats.get("brand_ad_count", 0.0),
                            "cate_price_mean": cstats.get("cate_price_mean", 0.0),
                            "cate_ad_count": cstats.get("cate_ad_count", 0.0),
                        }
                    )
                writer.writerow(output_row)
                written += 1
                positives += label
                negatives += 1 - label
                if written >= args.max_samples:
                    break

            # Encode behavior strength into repeated history by keeping the same brand/category
            # more visible for high-intent actions without changing the existing model interface.
            repeat = BTAG_WEIGHT[btag]
            for _ in range(repeat):
                history.append((timestamp, brand, cate, BTAG_ID[btag]))

            if scanned % 1_000_000 == 0:
                print(
                    f"scanned={scanned:,} written={written:,} "
                    f"pos={positives:,} neg={negatives:,}",
                    flush=True,
                )

    print(
        f"done scanned={scanned:,} written={written:,} "
        f"positives={positives:,} negatives={negatives:,} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
