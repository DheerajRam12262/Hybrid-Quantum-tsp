from __future__ import annotations

import numpy as np

from qaoa_tsp_benchmark.quantum.qubo_encoder import build_tsp_qubo, qubo_energy
from qaoa_tsp_benchmark.utils import generate_euclidean_instance


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
    invalid[0] = 1.0
    invalid[1] = 1.0
    invalid[2] = 1.0
    invalid[3] = 1.0

    e_valid = qubo_energy(valid, qubo, offset)
    e_invalid = qubo_energy(invalid, qubo, offset)

    assert e_valid < e_invalid
