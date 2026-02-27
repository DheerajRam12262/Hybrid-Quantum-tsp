from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qaoa_tsp_benchmark.classical import (  # noqa: E402
    SimulatedAnnealingConfig,
    SimulatedAnnealingSolver,
    exact_tsp_if_small,
)
from qaoa_tsp_benchmark.metrics import optimality_gap_pct, scaling_exponent  # noqa: E402
from qaoa_tsp_benchmark.quantum import HybridQAOAConfig, HybridQAOASolver  # noqa: E402
from qaoa_tsp_benchmark.utils import generate_euclidean_instance, save_instance  # noqa: E402


DEFAULT_SIZES = [5, 8, 10, 12, 15, 20, 25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA vs hybrid QAOA-style TSP benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--time-budget", type=float, default=0.25, help="Per-solver budget (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-exact-cities", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--save-instances", action="store_true")

    parser.add_argument("--sa-initial-temp", type=float, default=10_000.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.9995)

    parser.add_argument("--hybrid-reps", type=int, default=2)
    parser.add_argument("--hybrid-n-chains", type=int, default=8)
    parser.add_argument("--hybrid-initial-temp", type=float, default=8_000.0)
    parser.add_argument("--hybrid-cooling-rate", type=float, default=0.99935)

    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    sa_solver = SimulatedAnnealingSolver(
        SimulatedAnnealingConfig(
            initial_temp=args.sa_initial_temp,
            cooling_rate=args.sa_cooling_rate,
            min_temp=1e-3,
            use_two_opt=False,
        )
    )
    hybrid_solver = HybridQAOASolver(
        HybridQAOAConfig(
            reps=args.hybrid_reps,
            n_chains=args.hybrid_n_chains,
            initial_temp=args.hybrid_initial_temp,
            cooling_rate=args.hybrid_cooling_rate,
        )
    )

    rows: list[dict[str, float | int | str]] = []

    for n_cities in args.sizes:
        for trial in range(args.trials):
            instance_seed = args.seed + n_cities * 1_000 + trial
            instance = generate_euclidean_instance(n_cities=n_cities, seed=instance_seed)

            if args.save_instances:
                instance_dir = Path(args.output_dir) / "instances"
                save_instance(instance, instance_dir / f"tsp_n{n_cities}_trial{trial}.json")

            optimal_cost, _optimal_tour = exact_tsp_if_small(
                instance.distance_matrix,
                max_exact_cities=args.max_exact_cities,
            )

            sa_result = sa_solver.solve(
                instance.distance_matrix,
                time_budget_s=args.time_budget,
                seed=instance_seed + 11,
            )
            hybrid_result = hybrid_solver.solve(
                instance.distance_matrix,
                time_budget_s=args.time_budget,
                seed=instance_seed + 97,
            )

            for solver_name, result in [
                ("simulated_annealing", sa_result),
                ("hybrid_qaoa", hybrid_result),
            ]:
                rows.append(
                    {
                        "n_cities": n_cities,
                        "trial": trial,
                        "solver": solver_name,
                        "instance_seed": instance_seed,
                        "time_budget_s": args.time_budget,
                        "cost": result.cost,
                        "runtime_s": result.runtime_s,
                        "optimal_cost": optimal_cost if optimal_cost is not None else np.nan,
                        "optimality_gap_pct": (
                            optimality_gap_pct(result.cost, optimal_cost)
                            if optimal_cost is not None
                            else np.nan
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["solver", "n_cities"], as_index=False)
        .agg(
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_runtime_s=("runtime_s", "mean"),
            std_runtime_s=("runtime_s", "std"),
            mean_optimality_gap_pct=("optimality_gap_pct", "mean"),
            std_optimality_gap_pct=("optimality_gap_pct", "std"),
        )
        .sort_values(["solver", "n_cities"])
    )

    pivot = raw.pivot(index=["n_cities", "trial"], columns="solver", values="cost")
    if {"hybrid_qaoa", "simulated_annealing"}.issubset(set(pivot.columns)):
        win_rate = (
            (pivot["hybrid_qaoa"] < pivot["simulated_annealing"])
            .groupby("n_cities")
            .mean()
            .rename("hybrid_win_rate")
            .reset_index()
        )
        summary = summary.merge(win_rate, on="n_cities", how="left")
    else:
        summary["hybrid_win_rate"] = np.nan

    trends: dict[str, float | str] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    for solver_name in sorted(summary["solver"].unique()):
        solver_summary = summary[summary["solver"] == solver_name]
        k_cost = scaling_exponent(
            solver_summary["n_cities"].to_numpy(),
            solver_summary["mean_cost"].to_numpy(),
        )
        k_runtime = scaling_exponent(
            solver_summary["n_cities"].to_numpy(),
            solver_summary["mean_runtime_s"].to_numpy(),
        )
        if k_cost is not None:
            trends[f"{solver_name}_cost_scaling_exponent"] = k_cost
        if k_runtime is not None:
            trends[f"{solver_name}_runtime_scaling_exponent"] = k_runtime

    small_mask = raw["n_cities"] <= max(12, args.max_exact_cities)
    small = raw[small_mask]
    if not small.empty:
        small_pivot = small.pivot(index=["n_cities", "trial"], columns="solver", values="cost")
        if {"hybrid_qaoa", "simulated_annealing"}.issubset(set(small_pivot.columns)):
            trends["small_graph_hybrid_win_rate"] = float(
                (small_pivot["hybrid_qaoa"] < small_pivot["simulated_annealing"]).mean()
            )

    nontrivial_small = raw[(raw["n_cities"] >= 10) & (raw["n_cities"] <= args.max_exact_cities)]
    if not nontrivial_small.empty:
        ns_pivot = nontrivial_small.pivot(
            index=["n_cities", "trial"],
            columns="solver",
            values="cost",
        )
        if {"hybrid_qaoa", "simulated_annealing"}.issubset(set(ns_pivot.columns)):
            trends["nontrivial_small_graph_hybrid_win_rate"] = float(
                (ns_pivot["hybrid_qaoa"] < ns_pivot["simulated_annealing"]).mean()
            )

    return raw, summary, trends


def save_outputs(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    trends: dict[str, float | str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "raw_results.csv", index=False)
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    (output_dir / "trend_metrics.json").write_text(json.dumps(trends, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / tag

    raw, summary, trends = benchmark(args)
    save_outputs(raw, summary, trends, output_dir)

    print(f"Saved benchmark outputs to: {output_dir}")
    print("\nTop summary rows:")
    print(summary.head(12).to_string(index=False))
    print("\nTrend metrics:")
    print(json.dumps(trends, indent=2))


if __name__ == "__main__":
    main()
