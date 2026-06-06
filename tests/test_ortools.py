from __future__ import annotations

import pytest

from quantum_classical_benchmark.problems.tsp import generate_euclidean_instance, is_valid_tour
from quantum_classical_benchmark.solvers.exact import held_karp_tsp

pytest.importorskip("ortools", reason="OR-Tools is an optional extra ([ortools]).")

from quantum_classical_benchmark.solvers.ortools_solver import solve_with_or_tools  # noqa: E402


def test_ortools_returns_valid_tour_matching_exact_optimum() -> None:
    instance = generate_euclidean_instance(n_cities=8, seed=42)
    result = solve_with_or_tools(instance.distance_matrix, time_budget_s=1.0, seed=42)

    assert is_valid_tour(result.tour, 8)
    assert result.runtime_s > 0.0

    optimal, _ = held_karp_tsp(instance.distance_matrix)
    # OR-Tools with GLS should reach the exact optimum on a tiny instance.
    assert result.cost <= optimal * 1.001
