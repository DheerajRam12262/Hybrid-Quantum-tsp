"""Quantum-Classical Benchmark.

A reproducible benchmark comparing classical, quantum, and quantum-inspired
optimizers on the Travelling Salesperson Problem (TSP) under a matched
wall-clock budget, with exact ground truth and honest statistics.
"""

from __future__ import annotations

from .types import SolverResult, TSPInstance

__version__ = "0.1.0"

__all__ = ["SolverResult", "TSPInstance", "__version__"]
