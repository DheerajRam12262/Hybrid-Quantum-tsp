from __future__ import annotations

import numpy as np

from qaoa_tsp_benchmark.classical.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSolver,
)
from qaoa_tsp_benchmark.utils import generate_euclidean_instance, is_valid_tour, random_tour, tour_cost


def test_sa_returns_valid_tour_and_improves_random() -> None:
    instance = generate_euclidean_instance(n_cities=8, seed=10)
    solver = SimulatedAnnealingSolver(
        SimulatedAnnealingConfig(initial_temp=5_000.0, cooling_rate=0.9994, min_temp=1e-3)
    )
    result = solver.solve(instance.distance_matrix, time_budget_s=0.05, seed=7)

    assert is_valid_tour(result.tour, 8)

    rng = np.random.default_rng(7)
    random_baseline = tour_cost(random_tour(rng, 8), instance.distance_matrix)
    assert result.cost <= random_baseline
