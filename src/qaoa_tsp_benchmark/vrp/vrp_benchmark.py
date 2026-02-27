from __future__ import annotations

import numpy as np

from .vrp_encoder import VRPInstance


def greedy_vrp_routes(instance: VRPInstance) -> list[list[int]]:
    """Simple greedy allocator used as a placeholder baseline for VRP experiments."""
    n = instance.distance_matrix.shape[0]
    customers = list(range(1, n))
    routes: list[list[int]] = [[] for _ in range(instance.n_vehicles)]
    remaining = np.full(instance.n_vehicles, instance.vehicle_capacity, dtype=float)

    vehicle = 0
    for customer in customers:
        demand = instance.demands[customer]
        placed = False
        for _ in range(instance.n_vehicles):
            if remaining[vehicle] >= demand:
                routes[vehicle].append(customer)
                remaining[vehicle] -= demand
                placed = True
                break
            vehicle = (vehicle + 1) % instance.n_vehicles
        if not placed:
            raise ValueError("Insufficient capacity for greedy assignment")
        vehicle = (vehicle + 1) % instance.n_vehicles
    return routes
