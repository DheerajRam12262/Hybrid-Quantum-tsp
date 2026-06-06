"""Generate benchmark figures from a raw_results.csv produced by the harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..benchmark.harness import BASELINE_SOLVER  # noqa: E402


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if {"solver", "n_cities", "trial", "cost", "runtime_s"}.issubset(df.columns):
        return (
            df.groupby(["solver", "n_cities"], as_index=False)
            .agg(
                mean_cost=("cost", "mean"),
                mean_runtime_s=("runtime_s", "mean"),
                mean_optimality_gap_pct=("optimality_gap_pct", "mean"),
            )
            .sort_values(["solver", "n_cities"])
        )
    required = {"solver", "n_cities", "mean_cost", "mean_runtime_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.copy()


def _plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver, group in summary.groupby("solver"):
        ax.plot(group["n_cities"], group[metric], marker="o", linewidth=2, label=solver)
    ax.set_xlabel("Number of cities (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_match_or_beat(raw: pd.DataFrame, out_path: Path) -> None:
    """For each heuristic, the % of instances where it matches/beats OR-Tools, per N."""
    pivot = raw.pivot_table(index=["n_cities", "trial"], columns="solver", values="cost")
    if BASELINE_SOLVER not in pivot.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in pivot.columns:
        if solver == BASELINE_SOLVER:
            continue
        beat = (pivot[solver] <= pivot[BASELINE_SOLVER] * (1 + 1e-9)).groupby("n_cities").mean()
        ax.plot(beat.index, 100.0 * beat.to_numpy(), marker="o", linewidth=2, label=solver)

    ax.set_xlabel("Number of cities (N)")
    ax.set_ylabel(f"Match-or-beat {BASELINE_SOLVER} (%)")
    ax.set_title(f"Heuristic quality vs {BASELINE_SOLVER} baseline")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_plots(results_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_path)
    summary = _summarize(df)
    written: list[Path] = []

    p = out_dir / "solution_quality_vs_n.png"
    _plot_metric(summary, "mean_cost", "Mean tour cost", p)
    written.append(p)

    p = out_dir / "runtime_vs_n.png"
    _plot_metric(summary, "mean_runtime_s", "Mean runtime (s)", p)
    written.append(p)

    if (
        "mean_optimality_gap_pct" in summary.columns
        and np.isfinite(summary["mean_optimality_gap_pct"]).any()
    ):
        p = out_dir / "optimality_gap_vs_n.png"
        _plot_metric(
            summary[np.isfinite(summary["mean_optimality_gap_pct"])],
            "mean_optimality_gap_pct",
            "Mean optimality gap vs exact (%)",
            p,
        )
        written.append(p)

    if {"solver", "trial", "cost"}.issubset(df.columns):
        p = out_dir / "match_or_beat_baseline_vs_n.png"
        _plot_match_or_beat(df, p)
        if p.exists():
            written.append(p)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark outputs")
    parser.add_argument("--results", required=True, help="Path to raw_results.csv")
    parser.add_argument("--output-dir", default=None, help="Defaults to the results file directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    written = generate_plots(results_path, out_dir)
    print(f"Saved {len(written)} plots to {out_dir}")


if __name__ == "__main__":
    main()
