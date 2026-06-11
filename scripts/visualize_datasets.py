from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BTAG_NAMES = {
    0: "pad/unknown",
    1: "pv/view",
    2: "fav/click",
    3: "cart",
    4: "buy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset visualizations for Taobao and DIGINETICA.")
    parser.add_argument("--taobao", type=Path, default=Path("data/purchase_sequence_100k_static_long500.csv"))
    parser.add_argument("--diginetica", type=Path, default=Path("data/diginetica_sequence_100k.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--max_rows", type=int, default=100000)
    return parser.parse_args()


def sequence_length(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len(value.replace(",", " ").split())


def parse_number_sequence(value: object) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return []
    parsed: list[float] = []
    for token in value.replace(",", " ").split():
        try:
            parsed.append(float(token))
        except ValueError:
            continue
    return parsed


def load_dataset(path: Path, name: str, max_rows: int) -> pd.DataFrame:
    usecols = ["label", "hist_ad_ids", "hist_btags", "hist_time_gaps", "hist_time_deltas"]
    df = pd.read_csv(path, usecols=usecols, nrows=max_rows)
    df["dataset"] = name
    df["hist_len"] = df["hist_ad_ids"].map(sequence_length)
    return df


def summarize(df: pd.DataFrame) -> dict[str, float | str]:
    return {
        "dataset": str(df["dataset"].iloc[0]),
        "samples": int(len(df)),
        "positive_rate": float(df["label"].mean()),
        "hist_len_mean": float(df["hist_len"].mean()),
        "hist_len_median": float(df["hist_len"].median()),
        "hist_len_p90": float(df["hist_len"].quantile(0.9)),
        "hist_len_p99": float(df["hist_len"].quantile(0.99)),
    }


def save_label_distribution(combined: pd.DataFrame, output_dir: Path) -> None:
    counts = combined.groupby(["dataset", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[0.0, 1.0], fill_value=0)
    rates = counts.div(counts.sum(axis=1), axis=0)
    ax = rates.plot(kind="bar", stacked=True, color=["#7aa6c2", "#f08a5d"], figsize=(7.2, 4.2))
    ax.set_title("Label Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.legend(["negative", "positive"], loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=180)
    plt.close()


def save_length_distribution(taobao: pd.DataFrame, diginetica: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7.4, 4.2))
    bins = np.arange(0, max(taobao["hist_len"].max(), diginetica["hist_len"].max()) + 10, 10)
    plt.hist(taobao["hist_len"], bins=bins, alpha=0.62, label="Taobao", density=True, color="#5b8db8")
    plt.hist(diginetica["hist_len"], bins=bins, alpha=0.62, label="DIGINETICA", density=True, color="#e58b55")
    plt.title("Historical Sequence Length Distribution")
    plt.xlabel("history length")
    plt.ylabel("density")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.boxplot(
        [taobao["hist_len"], diginetica["hist_len"]],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d8e8f3"},
        medianprops={"color": "#d94f30"},
    )
    plt.xticks([1, 2], ["Taobao", "DIGINETICA"])
    plt.title("Historical Sequence Length Boxplot")
    plt.ylabel("history length")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_boxplot.png", dpi=180)
    plt.close()


def save_btag_distribution(combined: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for dataset, group in combined.groupby("dataset"):
        counts: dict[int, int] = {}
        for value in group["hist_btags"]:
            for btag in parse_number_sequence(value):
                key = int(btag)
                counts[key] = counts.get(key, 0) + 1
        total = max(sum(counts.values()), 1)
        for key, count in counts.items():
            rows.append({"dataset": dataset, "behavior": BTAG_NAMES.get(key, str(key)), "rate": count / total})

    if not rows:
        return
    plot_df = pd.DataFrame(rows).pivot(index="behavior", columns="dataset", values="rate").fillna(0.0)
    ax = plot_df.plot(kind="bar", figsize=(7.4, 4.2), color=["#5b8db8", "#e58b55"])
    ax.set_title("Historical Behavior Type Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "behavior_type_distribution.png", dpi=180)
    plt.close()


def sample_sequence_values(series: pd.Series, max_values: int = 200000) -> np.ndarray:
    values: list[float] = []
    for raw in series:
        values.extend(parse_number_sequence(raw))
        if len(values) >= max_values:
            break
    if not values:
        return np.asarray([], dtype=float)
    return np.asarray(values[:max_values], dtype=float)


def save_time_gap_distribution(taobao: pd.DataFrame, diginetica: pd.DataFrame, output_dir: Path) -> None:
    taobao_gaps = np.log1p(sample_sequence_values(taobao["hist_time_gaps"]))
    digi_gaps = np.log1p(sample_sequence_values(diginetica["hist_time_gaps"]))
    if taobao_gaps.size == 0 or digi_gaps.size == 0:
        return
    plt.figure(figsize=(7.4, 4.2))
    bins = np.linspace(0, max(taobao_gaps.max(), digi_gaps.max()), 50)
    plt.hist(taobao_gaps, bins=bins, alpha=0.62, density=True, label="Taobao", color="#5b8db8")
    plt.hist(digi_gaps, bins=bins, alpha=0.62, density=True, label="DIGINETICA", color="#e58b55")
    plt.title("Recency Gap Distribution")
    plt.xlabel("log(1 + time gap)")
    plt.ylabel("density")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "time_gap_distribution.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taobao = load_dataset(args.taobao, "Taobao", args.max_rows)
    diginetica = load_dataset(args.diginetica, "DIGINETICA", args.max_rows)
    combined = pd.concat([taobao, diginetica], ignore_index=True)

    summary = pd.DataFrame([summarize(taobao), summarize(diginetica)])
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)

    save_label_distribution(combined, args.output_dir)
    save_length_distribution(taobao, diginetica, args.output_dir)
    save_btag_distribution(combined, args.output_dir)
    save_time_gap_distribution(taobao, diginetica, args.output_dir)

    print(f"Saved figures and summary to {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
