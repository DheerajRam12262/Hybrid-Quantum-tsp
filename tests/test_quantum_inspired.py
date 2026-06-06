from __future__ import annotations

from quantum_classical_benchmark.problems.tsp import generate_euclidean_instance, is_valid_tour
from quantum_classical_benchmark.solvers.quantum_inspired import (
    QuantumInspiredConfig,
    QuantumInspiredSolver,
)
from quantum_classical_benchmark.solvers.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSolver,
)


def test_quantum_inspired_returns_valid_solution() -> None:
    instance = generate_euclidean_instance(n_cities=10, seed=14)
    solver = QuantumInspiredSolver(QuantumInspiredConfig(reps=2, n_chains=6))
    result = solver.solve(instance.distance_matrix, time_budget_s=0.06, seed=3)
    assert is_valid_tour(result.tour, 10)


def test_quantum_inspired_not_significantly_worse_than_sa_on_fixed_seed() -> None:
    instance = generate_euclidean_instance(n_cities=10, seed=20)

    sa = SimulatedAnnealingSolver(
        SimulatedAnnealingConfig(initial_temp=9_000.0, cooling_rate=0.9995, min_temp=1e-3)
    )
    qi = QuantumInspiredSolver(QuantumInspiredConfig(reps=2, n_chains=8))

    sa_result = sa.solve(instance.distance_matrix, time_budget_s=0.08, seed=77)
    qi_result = qi.solve(instance.distance_matrix, time_budget_s=0.08, seed=77)

    assert qi_result.cost <= sa_result.cost * 1.15
