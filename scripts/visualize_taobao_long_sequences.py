from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize raw Taobao user behavior sequence lengths.")
    parser.add_argument("--behavior_log", type=Path, default=Path("data/sampled_10pct/behavior_log.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def load_user_lengths(path: Path) -> pd.Series:
    counts: dict[int, int] = {}
    for chunk in pd.read_csv(path, usecols=["user"], chunksize=500000):
        value_counts = chunk["user"].value_counts()
        for user, count in value_counts.items():
            user_id = int(user)
            counts[user_id] = counts.get(user_id, 0) + int(count)
    return pd.Series(counts, name="behavior_count")


def save_threshold_plot(lengths: pd.Series, output_dir: Path) -> pd.DataFrame:
    thresholds = [50, 500]
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": f">{threshold}",
                "user_percentage": float((lengths > threshold).mean() * 100.0),
                "user_count": int((lengths > threshold).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "taobao_long_sequence_thresholds.csv", index=False)

    plt.figure(figsize=(6.8, 4.2))
    bars = plt.bar(summary["threshold"], summary["user_percentage"], color=["#5b8db8", "#e58b55"])
    plt.title("Taobao Raw User Behavior Sequence Length")
    plt.xlabel("Sequence length threshold")
    plt.ylabel("Users above threshold (%)")
    plt.ylim(0, max(100.0, summary["user_percentage"].max() * 1.2))
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, summary["user_percentage"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "taobao_long_sequence_thresholds.png", dpi=180)
    plt.close()
    return summary


def save_distribution_plot(lengths: pd.Series, output_dir: Path) -> None:
    clipped = lengths.clip(upper=1000)
    plt.figure(figsize=(7.4, 4.2))
    plt.hist(clipped, bins=60, color="#5b8db8", alpha=0.78)
    plt.axvline(50, color="#d94f30", linestyle="--", linewidth=1.6, label="50")
    plt.axvline(500, color="#6b4e9b", linestyle="--", linewidth=1.6, label="500")
    plt.title("Taobao Raw User Sequence Length Distribution")
    plt.xlabel("number of behaviors per user (clipped at 1000)")
    plt.ylabel("number of users")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "taobao_user_sequence_length_distribution.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = load_user_lengths(args.behavior_log)
    summary = save_threshold_plot(lengths, args.output_dir)
    save_distribution_plot(lengths, args.output_dir)
    stats = {
        "users": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p90": float(lengths.quantile(0.9)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }
    pd.DataFrame([stats]).to_csv(args.output_dir / "taobao_user_sequence_stats.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.Series(stats).to_string())
    print(f"Saved Taobao long-sequence figures to {args.output_dir}")


if __name__ == "__main__":
    main()
