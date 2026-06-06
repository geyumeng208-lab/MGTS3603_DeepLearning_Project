from __future__ import annotations

import argparse
import bisect
import csv
import random
from collections import defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Taobao four-table CTR data into the project's sequence CSV format."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("src/sampled_10pct"))
    parser.add_argument("--output", type=Path, default=Path("data/taobao_sequence_sample.csv"))
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--max_history", type=int, default=200)
    parser.add_argument("--max_behavior_rows", type=int, default=0)
    parser.add_argument("--min_timestamp", type=int, default=1400000000)
    parser.add_argument("--max_timestamp", type=int, default=1600000000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_raw_rows(path: Path, max_samples: int, seed: int) -> tuple[list[dict[str, str]], set[str], set[str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if len(rows) < max_samples:
                rows.append(row)
            else:
                replace_index = rng.randint(0, total - 1)
                if replace_index < max_samples:
                    rows[replace_index] = row
            if total % 1_000_000 == 0:
                print(f"raw_sample scanned={total:,} reservoir={len(rows):,}", flush=True)
    users = {row["user"] for row in rows}
    adgroups = {row["adgroup_id"] for row in rows}
    return rows, users, adgroups


def load_ad_features(path: Path, adgroups: set[str]) -> dict[str, dict[str, str]]:
    features: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["adgroup_id"] in adgroups:
                features[row["adgroup_id"]] = row
    return features


def load_user_histories(
    path: Path,
    users: set[str],
    max_history: int,
    max_rows: int,
    min_timestamp: int,
    max_timestamp: int,
) -> dict[str, list[tuple[int, int, int]]]:
    histories: dict[str, deque[tuple[int, int, int]]] = defaultdict(lambda: deque(maxlen=max_history * 3))
    scanned = 0
    kept = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned += 1
            if max_rows > 0 and scanned > max_rows:
                break
            user = row["user"]
            if user not in users:
                continue
            timestamp = int(row["time_stamp"])
            if timestamp < min_timestamp or timestamp > max_timestamp:
                continue
            cate = safe_int(row["cate"])
            brand = safe_int(row["brand"])
            if cate <= 0:
                continue
            histories[user].append((timestamp, max(brand, 1), cate))
            kept += 1
            if scanned % 1_000_000 == 0:
                print(f"behavior_log scanned={scanned:,} kept_for_sample_users={kept:,}", flush=True)

    return {user: sorted(values, key=lambda x: x[0]) for user, values in histories.items()}


def write_sequence_csv(
    rows: list[dict[str, str]],
    ad_features: dict[str, dict[str, str]],
    histories: dict[str, list[tuple[int, int, int]]],
    output: Path,
    max_history: int,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_ad = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["user_id", "ad_id", "cate_id", "label", "hist_ad_ids", "hist_cate_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            ad_id = row["adgroup_id"]
            ad = ad_features.get(ad_id)
            if not ad:
                skipped_no_ad += 1
                continue

            exposure_ts = int(row["time_stamp"])
            user_history = histories.get(row["user"], [])
            cutoff = bisect.bisect_left([item[0] for item in user_history], exposure_ts)
            selected = user_history[max(0, cutoff - max_history) : cutoff]
            hist_items = [str(item[1]) for item in selected]
            hist_cates = [str(item[2]) for item in selected]

            writer.writerow(
                {
                    "user_id": row["user"],
                    "ad_id": max(safe_int(ad["brand"]), 1),
                    "cate_id": ad["cate_id"],
                    "label": row["clk"],
                    "hist_ad_ids": " ".join(hist_items),
                    "hist_cate_ids": " ".join(hist_cates),
                }
            )
            written += 1
    return written, skipped_no_ad


def safe_int(value: str) -> int:
    value = (value or "").strip()
    if not value or value.upper() == "NULL":
        return 0
    return int(float(value))


def main() -> None:
    args = parse_args()
    raw_rows, users, adgroups = load_raw_rows(args.input_dir / "raw_sample.csv", args.max_samples, args.seed)
    print(f"raw rows={len(raw_rows):,} users={len(users):,} adgroups={len(adgroups):,}", flush=True)

    ad_features = load_ad_features(args.input_dir / "ad_feature.csv", adgroups)
    print(f"matched ad features={len(ad_features):,}", flush=True)

    histories = load_user_histories(
        args.input_dir / "behavior_log.csv",
        users,
        args.max_history,
        args.max_behavior_rows,
        args.min_timestamp,
        args.max_timestamp,
    )
    print(f"users with histories={len(histories):,}", flush=True)

    written, skipped_no_ad = write_sequence_csv(raw_rows, ad_features, histories, args.output, args.max_history)
    print(f"wrote={written:,} skipped_no_ad={skipped_no_ad:,} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
