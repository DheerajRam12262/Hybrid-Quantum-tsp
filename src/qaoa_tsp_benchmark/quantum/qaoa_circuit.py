from __future__ import annotations

import numpy as np


def qiskit_ready() -> bool:
    try:
        import qiskit  # noqa: F401

        return True
    except Exception:
        return False


def build_qaoa_ansatz_from_qubo(qubo: np.ndarray, reps: int = 2):
    """Build a QAOA ansatz when Qiskit is available.

    This helper keeps integration optional; benchmarks can run without Qiskit.
    """
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.quantum_info import SparsePauliOp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Qiskit is not installed. Install optional deps with `pip install .[quantum]`."
        ) from exc

    n = qubo.shape[0]
    paulis: list[tuple[str, float]] = []

    # Lightweight Ising surrogate: map diagonal QUBO terms to Z operators.
    # Full QUBO->Ising conversion can be added later for hardware experiments.
    for i in range(n):
        coeff = float(qubo[i, i])
        if abs(coeff) < 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        paulis.append(("".join(label), coeff))

    if not paulis:
        paulis = [("I" * n, 0.0)]

    hamiltonian = SparsePauliOp.from_list(paulis)
    return QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
