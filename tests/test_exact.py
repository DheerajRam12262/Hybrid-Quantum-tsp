from __future__ import annotations

import numpy as np

from quantum_classical_benchmark.problems.tsp import generate_euclidean_instance, is_valid_tour
from quantum_classical_benchmark.solvers.exact import (
    brute_force_tsp,
    exact_tsp_if_small,
    held_karp_tsp,
)


def test_held_karp_matches_known_optimum_on_unit_square() -> None:
    # Four corners of a unit square: optimal tour is the perimeter, cost 4.0.
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))

    cost, tour = held_karp_tsp(dist)
    assert np.isclose(cost, 4.0)
    assert is_valid_tour(tour, 4)


def test_held_karp_agrees_with_brute_force() -> None:
    instance = generate_euclidean_instance(n_cities=7, seed=5)
    hk_cost, _ = held_karp_tsp(instance.distance_matrix)
    bf_cost, _ = brute_force_tsp(instance.distance_matrix)
    assert np.isclose(hk_cost, bf_cost)


def test_exact_returns_none_above_threshold() -> None:
    instance = generate_euclidean_instance(n_cities=15, seed=1)
    cost, tour = exact_tsp_if_small(instance.distance_matrix, max_exact_cities=12)
    assert cost is None and tour is None
