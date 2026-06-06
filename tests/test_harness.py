from __future__ import annotations

import numpy as np

from quantum_classical_benchmark.benchmark.harness import (
    BASELINE_SOLVER,
    BenchmarkConfig,
    run_benchmark,
)


def _tiny_config() -> BenchmarkConfig:
    return BenchmarkConfig(sizes=[5, 8], trials=2, time_budget_s=0.05, max_exact_cities=12)


def test_harness_runs_all_solvers_and_reports_columns() -> None:
    out = run_benchmark(_tiny_config())

    solvers = set(out.raw["solver"].unique())
    assert {"simulated_annealing", "quantum_inspired"} <= solvers
    # 2 sizes * 2 trials * n_solvers rows
    assert len(out.raw) == 2 * 2 * len(solvers)

    for col in [
        "mean_cost",
        "mean_runtime_s",
        "mean_optimality_gap_pct",
        "match_or_beat_baseline_rate",
    ]:
        assert col in out.summary.columns


def test_ortools_baseline_is_near_optimal_when_available() -> None:
    out = run_benchmark(_tiny_config())
    if BASELINE_SOLVER not in set(out.raw["solver"].unique()):
        return  # OR-Tools extra not installed in this environment

    ortools_gap = out.raw[out.raw["solver"] == BASELINE_SOLVER]["optimality_gap_pct"]
    # On these tiny instances OR-Tools should reach the exact optimum (gap ~ 0).
    assert np.nanmax(ortools_gap.to_numpy()) < 1.0


def test_budget_is_matched_across_solvers() -> None:
    out = run_benchmark(_tiny_config())
    # No solver may exceed the shared budget by more than scheduling slack.
    assert (out.raw["runtime_s"] <= out.raw["time_budget_s"] + 0.2).all()
