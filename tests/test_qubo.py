from __future__ import annotations

import numpy as np

from quantum_classical_benchmark.encoding.qubo import (
    build_tsp_qubo,
    ising_energy,
    qubo_energy,
    qubo_to_ising,
)
from quantum_classical_benchmark.problems.tsp import generate_euclidean_instance


def _route_to_bitstring(route: tuple[int, ...], n: int) -> np.ndarray:
    x = np.zeros(n * n, dtype=float)
    for pos, city in enumerate(route):
        x[city * n + pos] = 1.0
    return x


def test_qubo_dimensions_and_penalty_signal() -> None:
    instance = generate_euclidean_instance(n_cities=4, seed=123)
    qubo, offset, meta = build_tsp_qubo(instance.distance_matrix)

    assert qubo.shape == (16, 16)
    assert meta["n_variables"] == 16

    valid = _route_to_bitstring((0, 1, 2, 3), 4)
    invalid = np.zeros(16, dtype=float)
    invalid[[0, 1, 2, 3]] = 1.0  # all four cities in position 0 -> infeasible

    assert qubo_energy(valid, qubo, offset) < qubo_energy(invalid, qubo, offset)


def test_qubo_to_ising_energy_equivalence() -> None:
    """Ising energy of spins z must equal QUBO energy of x = (1 - z) / 2."""
    instance = generate_euclidean_instance(n_cities=3, seed=7)
    qubo, offset, _ = build_tsp_qubo(instance.distance_matrix)
    h, J, const = qubo_to_ising(qubo, offset)

    rng = np.random.default_rng(0)
    for _ in range(64):
        x = rng.integers(0, 2, size=qubo.shape[0])
        z = 1 - 2 * x  # x in {0,1} -> z in {+1,-1}
        assert np.isclose(ising_energy(z, h, J, const), qubo_energy(x, qubo, offset))
