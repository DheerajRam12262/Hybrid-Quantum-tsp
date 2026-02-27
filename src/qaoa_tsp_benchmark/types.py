from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TSPInstance:
    """Container for a generated TSP instance."""

    n_cities: int
    seed: int
    coordinates: np.ndarray
    distance_matrix: np.ndarray


@dataclass(slots=True)
class SolverResult:
    """Standard solver response used by classical and hybrid solvers."""

    tour: tuple[int, ...]
    cost: float
    runtime_s: float
    meta: dict[str, Any] = field(default_factory=dict)
