from __future__ import annotations

import numpy as np
import pytest

from quantum_classical_benchmark.encoding.qubo import build_tsp_qubo, qubo_energy
from quantum_classical_benchmark.problems.tsp import generate_euclidean_instance
from quantum_classical_benchmark.solvers.qaoa import (
    build_qaoa_ansatz_from_qubo,
    cost_hamiltonian_terms,
    qiskit_ready,
)


def _pauli_diag_energy(terms: list[tuple[str, float]], x: np.ndarray) -> float:
    """Diagonal energy of a Z-only Pauli Hamiltonian on basis state x via z = 1 - 2x."""
    z = 1 - 2 * np.asarray(x, dtype=int)
    energy = 0.0
    for label, coeff in terms:
        val = coeff
        for pos, p in enumerate(label[::-1]):  # little-endian: pos 0 is right-most
            if p == "Z":
                val *= z[pos]
        energy += val
    return float(energy)


def test_cost_hamiltonian_matches_qubo_energy() -> None:
    """The QAOA cost Hamiltonian's diagonal must equal the QUBO objective."""
    instance = generate_euclidean_instance(n_cities=3, seed=11)
    qubo, offset, _ = build_tsp_qubo(instance.distance_matrix)
    terms = cost_hamiltonian_terms(qubo, offset)

    rng = np.random.default_rng(1)
    for _ in range(32):
        x = rng.integers(0, 2, size=qubo.shape[0])
        assert np.isclose(_pauli_diag_energy(terms, x), qubo_energy(x, qubo, offset))


def test_qaoa_builder_requires_qiskit() -> None:
    qubo = np.eye(4)
    if qiskit_ready():
        assert build_qaoa_ansatz_from_qubo(qubo, reps=2) is not None
    else:
        with pytest.raises(RuntimeError):
            build_qaoa_ansatz_from_qubo(qubo, reps=2)
