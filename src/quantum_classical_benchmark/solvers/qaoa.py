"""QAOA solver for TSP via its QUBO/Ising encoding (Qiskit, simulator-only).

This is the *real quantum* path. It is honest about its limits:

* It runs on a statevector **simulator**, not quantum hardware.
* The TSP QUBO uses ``n^2`` binary variables, so an ``n``-city instance needs
  ``n^2`` qubits. Statevector simulation is exponential in qubit count, so this
  is practical only for **n <= 4 cities (<= 16 qubits)**. It does not scale and
  is *not* competitive with the classical baselines -- it exists to demonstrate
  a correct end-to-end QUBO -> Ising -> QAOA -> decoded-tour pipeline.

Everything Qiskit-dependent is imported lazily so the rest of the package (and
the benchmark) runs without the optional ``[quantum]`` extra installed.
"""

from __future__ import annotations

import time

import numpy as np

from ..encoding.qubo import build_tsp_qubo, qubo_to_ising
from ..problems.tsp import is_valid_tour, normalize_tour, tour_cost
from ..types import SolverResult

# n^2 qubits; 16 qubits (4 cities) is already ~1M amplitudes. Guard against more.
DEFAULT_MAX_QUBITS = 16


def qiskit_ready() -> bool:
    """True iff Qiskit is importable (the optional ``[quantum]`` extra is installed)."""
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return False
    return True


def cost_hamiltonian_terms(qubo: np.ndarray, offset: float = 0.0) -> list[tuple[str, float]]:
    """Pauli terms of the Ising cost Hamiltonian for a QUBO (no Qiskit needed).

    Returns ``[(pauli_label, coefficient), ...]`` using Qiskit's little-endian
    convention (qubit 0 is the right-most character). Variable ``i`` maps to a
    ``Z`` at string position ``n - 1 - i``. The diagonal of this Hamiltonian on
    computational basis state ``z`` equals :func:`ising_energy`, which in turn
    equals the QUBO energy of ``x = (1 - z) / 2`` -- verified in the tests.
    """
    h, J, const = qubo_to_ising(qubo, offset)
    n = len(h)
    terms: list[tuple[str, float]] = [("I" * n, const)]

    for i in range(n):
        if abs(h[i]) < 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        terms.append(("".join(label), float(h[i])))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) < 1e-12:
                continue
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"
            terms.append(("".join(label), float(J[i, j])))

    return terms


def build_cost_hamiltonian(qubo: np.ndarray, offset: float = 0.0):
    """Build the Ising cost Hamiltonian as a Qiskit ``SparsePauliOp``."""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qiskit is not installed. Install it with 'pip install .[quantum]'."
        ) from exc
    return SparsePauliOp.from_list(cost_hamiltonian_terms(qubo, offset))


def build_qaoa_ansatz_from_qubo(qubo: np.ndarray, reps: int = 2, offset: float = 0.0):
    """Build a QAOA ansatz whose cost operator is the full TSP Ising Hamiltonian."""
    try:
        from qiskit.circuit.library import QAOAAnsatz
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qiskit is not installed. Install it with 'pip install .[quantum]'."
        ) from exc
    return QAOAAnsatz(cost_operator=build_cost_hamiltonian(qubo, offset), reps=reps)


def _decode_bitstring_to_tour(bits: np.ndarray, n: int) -> tuple[int, ...] | None:
    """Decode an ``n^2`` permutation-matrix bitstring ``x_{city, pos}`` into a tour."""
    matrix = np.asarray(bits, dtype=int).reshape(n, n)  # rows = cities, cols = positions
    if matrix.sum() != n or (matrix.sum(axis=0) != 1).any() or (matrix.sum(axis=1) != 1).any():
        return None
    tour = [int(np.argmax(matrix[:, pos])) for pos in range(n)]
    if not is_valid_tour(tour, n):
        return None
    return normalize_tour(tour)


def solve_qaoa_tsp(
    distance_matrix: np.ndarray,
    reps: int = 2,
    seed: int = 42,
    max_qubits: int = DEFAULT_MAX_QUBITS,
    optimizer_maxiter: int = 75,
    shots: int = 4096,
) -> SolverResult:
    """Run QAOA on a statevector simulator and decode the best feasible tour.

    Raises ``ValueError`` if the instance needs more than ``max_qubits`` qubits.
    Requires the ``[quantum]`` extra (Qiskit + SciPy).
    """
    try:
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qiskit is not installed. Install it with 'pip install .[quantum]'."
        ) from exc

    n = distance_matrix.shape[0]
    n_qubits = n * n
    if n_qubits > max_qubits:
        raise ValueError(
            f"{n} cities need {n_qubits} qubits > max_qubits={max_qubits}. "
            "QAOA here is simulator-only and does not scale; use a classical solver."
        )

    start = time.perf_counter()
    qubo, offset, _meta = build_tsp_qubo(distance_matrix)
    hamiltonian = build_cost_hamiltonian(qubo, offset)
    ansatz = build_qaoa_ansatz_from_qubo(qubo, reps=reps, offset=offset).decompose()

    estimator = StatevectorEstimator()
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi, size=ansatz.num_parameters)

    def expectation(params: np.ndarray) -> float:
        result = estimator.run([(ansatz, hamiltonian, [params])]).result()
        return float(result[0].data.evs[0])

    opt = minimize(expectation, x0, method="COBYLA", options={"maxiter": optimizer_maxiter})

    sampler = StatevectorSampler()
    measured = ansatz.copy()
    measured.measure_all()
    samples = sampler.run([(measured, [opt.x])], shots=shots).result()
    counts = samples[0].data.meas.get_counts()

    best_tour: tuple[int, ...] | None = None
    best_cost = float("inf")
    for bitstring in counts:
        bits = np.array([int(b) for b in bitstring[::-1]], dtype=int)  # little-endian -> var order
        tour = _decode_bitstring_to_tour(bits, n)
        if tour is None:
            continue
        cost = tour_cost(tour, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    runtime = time.perf_counter() - start
    feasible_fraction = sum(
        c
        for b, c in counts.items()
        if _decode_bitstring_to_tour(np.array([int(x) for x in b[::-1]], dtype=int), n) is not None
    ) / max(1, sum(counts.values()))

    if best_tour is None:
        raise RuntimeError(
            f"QAOA produced no feasible tour in {shots} shots (feasible fraction 0). "
            "This is expected for under-optimised QAOA on TSP; increase reps/shots."
        )

    return SolverResult(
        tour=best_tour,
        cost=best_cost,
        runtime_s=runtime,
        meta={
            "n_qubits": n_qubits,
            "reps": reps,
            "feasible_fraction": feasible_fraction,
            "optimizer_evals": int(opt.nfev),
            "backend": "statevector_simulator",
        },
    )
