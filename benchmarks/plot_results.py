from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark outputs")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to raw_results.csv or summary_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default=None,
        help="Directory for generated plots (defaults to results file directory)",
    )
    return parser.parse_args()


def _to_summary(df: pd.DataFrame) -> pd.DataFrame:
    if {"solver", "n_cities", "trial", "cost", "runtime_s"}.issubset(df.columns):
        summary = (
            df.groupby(["solver", "n_cities"], as_index=False)
            .agg(
                mean_cost=("cost", "mean"),
                mean_runtime_s=("runtime_s", "mean"),
                mean_optimality_gap_pct=("optimality_gap_pct", "mean"),
            )
            .sort_values(["solver", "n_cities"])
        )
        return summary

    required = {"solver", "n_cities", "mean_cost", "mean_runtime_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.copy()


def _plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver, group in summary.groupby("solver"):
        ax.plot(group["n_cities"], group[metric], marker="o", linewidth=2, label=solver)
    ax.set_xlabel("Number of Cities (N)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hybrid_win_rate(raw: pd.DataFrame, out_path: Path) -> None:
    pivot = raw.pivot(index=["n_cities", "trial"], columns="solver", values="cost")
    if {"hybrid_qaoa", "simulated_annealing"} - set(pivot.columns):
        return

    win_rate = (
        (pivot["hybrid_qaoa"] < pivot["simulated_annealing"])
        .groupby("n_cities")
        .mean()
        .reset_index(name="hybrid_win_rate")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(win_rate["n_cities"], 100.0 * win_rate["hybrid_win_rate"], marker="o", linewidth=2)
    ax.set_xlabel("Number of Cities (N)")
    ax.set_ylabel("Hybrid Win Rate vs SA (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    summary = _to_summary(df)

    _plot_metric(summary, "mean_cost", "Mean Tour Cost", out_dir / "solution_quality_vs_n.png")
    _plot_metric(summary, "mean_runtime_s", "Mean Runtime (s)", out_dir / "runtime_vs_n.png")

    if "mean_optimality_gap_pct" in summary.columns and np.isfinite(summary["mean_optimality_gap_pct"]).any():
        _plot_metric(
            summary[np.isfinite(summary["mean_optimality_gap_pct"])],
            "mean_optimality_gap_pct",
            "Mean Optimality Gap (%)",
            out_dir / "optimality_gap_vs_n.png",
        )

    if {"solver", "trial", "cost"}.issubset(df.columns):
        _plot_hybrid_win_rate(df, out_dir / "hybrid_win_rate_vs_n.png")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
