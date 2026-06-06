"""Fair benchmark harness.

Every solver is run on the *same* seeded instances under the *same* wall-clock
budget. OR-Tools is the honest, strong baseline; exact ground truth (Held-Karp)
is computed for small N so we can report a true optimality gap. Heuristic
quality is reported both as a gap-to-optimum and as a match-or-beat rate against
OR-Tools -- which, honestly, the heuristics rarely achieve as N grows.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..problems.tsp import generate_euclidean_instance, save_instance
from ..solvers.exact import exact_tsp_if_small
from ..solvers.quantum_inspired import QuantumInspiredConfig, QuantumInspiredSolver
from ..solvers.simulated_annealing import SimulatedAnnealingConfig, SimulatedAnnealingSolver
from ..types import SolverResult
from .metrics import optimality_gap_pct, scaling_exponent

# Name of the honest, strong baseline that heuristics are scored against.
BASELINE_SOLVER = "ortools"

SolverFn = Callable[[np.ndarray, float, int], SolverResult]


@dataclass(slots=True)
class BenchmarkConfig:
    sizes: list[int] = field(default_factory=lambda: [5, 8, 10, 12, 15, 20, 25])
    trials: int = 5
    time_budget_s: float = 0.25
    seed: int = 42
    max_exact_cities: int = 12
    include_ortools: bool = True
    save_instances: bool = False
    output_dir: Path | None = None
    sa: SimulatedAnnealingConfig = field(default_factory=SimulatedAnnealingConfig)
    qi: QuantumInspiredConfig = field(default_factory=QuantumInspiredConfig)


@dataclass(slots=True)
class BenchmarkOutput:
    raw: pd.DataFrame
    summary: pd.DataFrame
    trends: dict[str, float | str]


def _build_solvers(config: BenchmarkConfig) -> dict[str, SolverFn]:
    sa = SimulatedAnnealingSolver(config.sa)
    qi = QuantumInspiredSolver(config.qi)

    solvers: dict[str, SolverFn] = {
        "simulated_annealing": lambda dm, t, s: sa.solve(dm, time_budget_s=t, seed=s),
        "quantum_inspired": lambda dm, t, s: qi.solve(dm, time_budget_s=t, seed=s),
    }

    if config.include_ortools:
        try:
            from ..solvers.ortools_solver import solve_with_or_tools

            solvers[BASELINE_SOLVER] = lambda dm, t, s: solve_with_or_tools(
                dm, time_budget_s=t, seed=s
            )
        except ImportError:
            pass  # OR-Tools extra not installed; benchmark runs without the baseline.

    return solvers


def run_benchmark(config: BenchmarkConfig) -> BenchmarkOutput:
    solvers = _build_solvers(config)
    rows: list[dict[str, Any]] = []

    for n_cities in config.sizes:
        for trial in range(config.trials):
            instance_seed = config.seed + n_cities * 1_000 + trial
            instance = generate_euclidean_instance(n_cities=n_cities, seed=instance_seed)

            if config.save_instances and config.output_dir is not None:
                save_instance(
                    instance,
                    config.output_dir / "instances" / f"tsp_n{n_cities}_trial{trial}.json",
                )

            optimal_cost, _ = exact_tsp_if_small(
                instance.distance_matrix, max_exact_cities=config.max_exact_cities
            )

            # Distinct, deterministic seed per solver so they don't share an RNG stream.
            for offset, (solver_name, solver_fn) in enumerate(solvers.items()):
                result = solver_fn(
                    instance.distance_matrix, config.time_budget_s, instance_seed + 11 + offset
                )
                rows.append(
                    {
                        "n_cities": n_cities,
                        "trial": trial,
                        "solver": solver_name,
                        "instance_seed": instance_seed,
                        "time_budget_s": config.time_budget_s,
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
    summary = _summarize(raw)
    trends = _trends(raw, summary, config)
    return BenchmarkOutput(raw=raw, summary=summary, trends=trends)


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
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

    # Match-or-beat rate of each heuristic against the OR-Tools baseline, per N.
    pivot = raw.pivot_table(index=["n_cities", "trial"], columns="solver", values="cost")
    if BASELINE_SOLVER in pivot.columns:
        rates = []
        for solver_name in pivot.columns:
            if solver_name == BASELINE_SOLVER:
                continue
            beat = pivot[solver_name] <= pivot[BASELINE_SOLVER] * (1 + 1e-9)
            rate = (
                beat.groupby("n_cities").mean().rename("match_or_beat_baseline_rate").reset_index()
            )
            rate["solver"] = solver_name
            rates.append(rate)
        if rates:
            summary = summary.merge(pd.concat(rates), on=["solver", "n_cities"], how="left")
    if "match_or_beat_baseline_rate" not in summary.columns:
        summary["match_or_beat_baseline_rate"] = np.nan
    return summary


def _trends(
    raw: pd.DataFrame, summary: pd.DataFrame, config: BenchmarkConfig
) -> dict[str, float | str]:
    trends: dict[str, float | str] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_solver": BASELINE_SOLVER if BASELINE_SOLVER in raw["solver"].unique() else "none",
        "time_budget_s": config.time_budget_s,
        "trials": config.trials,
    }

    for solver_name in sorted(summary["solver"].unique()):
        s = summary[summary["solver"] == solver_name]
        k_cost = scaling_exponent(s["n_cities"].to_numpy(), s["mean_cost"].to_numpy())
        k_runtime = scaling_exponent(s["n_cities"].to_numpy(), s["mean_runtime_s"].to_numpy())
        if k_cost is not None:
            trends[f"{solver_name}_cost_scaling_exponent"] = k_cost
        if k_runtime is not None:
            trends[f"{solver_name}_runtime_scaling_exponent"] = k_runtime

    # Mean optimality gap on instances with exact ground truth (the headline metric).
    exact = raw[np.isfinite(raw["optimality_gap_pct"])]
    if not exact.empty:
        for solver_name, gap in exact.groupby("solver")["optimality_gap_pct"].mean().items():
            trends[f"{solver_name}_mean_gap_pct_exact"] = float(gap)

    return trends


def save_outputs(output: BenchmarkOutput, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output.raw.to_csv(output_dir / "raw_results.csv", index=False)
    output.summary.to_csv(output_dir / "summary_results.csv", index=False)
    (output_dir / "trend_metrics.json").write_text(
        json.dumps(output.trends, indent=2), encoding="utf-8"
    )
