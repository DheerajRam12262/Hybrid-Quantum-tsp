from __future__ import annotations

from qaoa_tsp_benchmark.classical.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSolver,
)
from qaoa_tsp_benchmark.quantum.hybrid_optimizer import HybridQAOAConfig, HybridQAOASolver
from qaoa_tsp_benchmark.utils import generate_euclidean_instance, is_valid_tour


def test_hybrid_returns_valid_solution() -> None:
    instance = generate_euclidean_instance(n_cities=10, seed=14)
    solver = HybridQAOASolver(HybridQAOAConfig(reps=2, n_chains=6))
    result = solver.solve(instance.distance_matrix, time_budget_s=0.06, seed=3)
    assert is_valid_tour(result.tour, 10)


def test_hybrid_not_significantly_worse_than_sa_on_fixed_seed() -> None:
    instance = generate_euclidean_instance(n_cities=10, seed=20)

    sa = SimulatedAnnealingSolver(
        SimulatedAnnealingConfig(initial_temp=9_000.0, cooling_rate=0.9995, min_temp=1e-3)
    )
    hybrid = HybridQAOASolver(HybridQAOAConfig(reps=2, n_chains=8))

    sa_result = sa.solve(instance.distance_matrix, time_budget_s=0.08, seed=77)
    hybrid_result = hybrid.solve(instance.distance_matrix, time_budget_s=0.08, seed=77)

    assert hybrid_result.cost <= sa_result.cost * 1.15
