from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


def held_karp_tsp(distance_matrix: np.ndarray, start: int = 0) -> tuple[float, tuple[int, ...]]:
    """Exact dynamic-programming TSP solver with O(n^2 2^n) complexity."""
    n = distance_matrix.shape[0]
    if n < 2:
        return 0.0, (0,)

    cities = tuple(i for i in range(n) if i != start)

    @lru_cache(maxsize=None)
    def dp(last: int, remaining: frozenset[int]) -> tuple[float, tuple[int, ...]]:
        if not remaining:
            return distance_matrix[last, start], (start,)

        best_cost = float("inf")
        best_path: tuple[int, ...] = ()
        for nxt in remaining:
            cost_tail, path_tail = dp(nxt, remaining - {nxt})
            cost = distance_matrix[last, nxt] + cost_tail
            if cost < best_cost:
                best_cost = cost
                best_path = (nxt,) + path_tail
        return best_cost, best_path

    best_total = float("inf")
    best_route: tuple[int, ...] = ()
    for first in cities:
        remaining = frozenset(c for c in cities if c != first)
        tail_cost, tail_path = dp(first, remaining)
        total = distance_matrix[start, first] + tail_cost
        if total < best_total:
            best_total = total
            best_route = (start, first) + tail_path[:-1]

    return float(best_total), best_route


def brute_force_tsp(distance_matrix: np.ndarray, start: int = 0) -> tuple[float, tuple[int, ...]]:
    """Permutation brute-force solver; useful only for very small n."""
    n = distance_matrix.shape[0]
    nodes = [i for i in range(n) if i != start]
    best_cost = float("inf")
    best_route: tuple[int, ...] = ()

    for perm in itertools.permutations(nodes):
        route = (start,) + perm
        nxt = route[1:] + (start,)
        cost = float(np.sum(distance_matrix[np.array(route), np.array(nxt)]))
        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_cost, best_route


def exact_tsp_if_small(
    distance_matrix: np.ndarray,
    max_exact_cities: int = 12,
) -> tuple[float | None, tuple[int, ...] | None]:
    """Return exact optimum for small instances, else None."""
    n = distance_matrix.shape[0]
    if n > max_exact_cities:
        return None, None
    return held_karp_tsp(distance_matrix)
