from __future__ import annotations

import numpy as np
import pytest

from qaoa_tsp_benchmark.quantum.qaoa_circuit import build_qaoa_ansatz_from_qubo, qiskit_ready


def test_qaoa_builder_behavior() -> None:
    qubo = np.eye(4)
    if not qiskit_ready():
        with pytest.raises(RuntimeError):
            build_qaoa_ansatz_from_qubo(qubo, reps=2)
    else:
        ansatz = build_qaoa_ansatz_from_qubo(qubo, reps=2)
        assert ansatz is not None
