from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VRPInstance:
    distance_matrix: np.ndarray
    demands: np.ndarray
    n_vehicles: int
    vehicle_capacity: float


def build_simple_vrp_penalty_matrix(instance: VRPInstance, penalty_capacity: float = 100.0) -> np.ndarray:
    """Return a lightweight capacity-penalty matrix for future VRP QUBO extensions."""
    n = instance.distance_matrix.shape[0]
    penalty = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        if instance.demands[i] > instance.vehicle_capacity:
            penalty[i, i] += penalty_capacity * (instance.demands[i] - instance.vehicle_capacity)
    return penalty
