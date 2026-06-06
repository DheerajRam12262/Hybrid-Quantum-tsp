from __future__ import annotations

import time

import numpy as np

from ..problems.tsp import normalize_tour, tour_cost
from ..types import SolverResult

# Distances are floats; OR-Tools routing needs integer arc costs, so we scale
# before rounding. 1e6 keeps sub-unit distance differences meaningful.
_DISTANCE_SCALE = 1_000_000


class ORToolsUnavailableError(RuntimeError):
    """Raised when the optional OR-Tools dependency is not installed."""


def solve_with_or_tools(
    distance_matrix: np.ndarray,
    time_budget_s: float = 1.0,
    seed: int | None = None,
) -> SolverResult:
    """Honest classical baseline: Google OR-Tools routing with Guided Local Search.

    OR-Tools is the *strong* baseline for this benchmark. It is given the same
    wall-clock budget as every other solver; Guided Local Search keeps improving
    until the budget expires (it may converge and stop earlier on tiny instances,
    which is reported honestly as a short runtime).
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ORToolsUnavailableError(
            "OR-Tools is not installed. Install it with 'pip install .[ortools]'."
        ) from exc

    n = distance_matrix.shape[0]
    start = time.perf_counter()

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    scaled = np.rint(distance_matrix * _DISTANCE_SCALE).astype(np.int64)

    def distance_callback(from_index: int, to_index: int) -> int:
        return int(scaled[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # Match the shared wall-clock budget (millisecond resolution).
    params.time_limit.FromMilliseconds(max(1, int(time_budget_s * 1000)))
    if seed is not None:
        params.log_search = False

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools failed to produce a solution")

    index = routing.Start(0)
    route: list[int] = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    runtime = time.perf_counter() - start
    normalized = normalize_tour(route)
    return SolverResult(
        tour=normalized,
        cost=tour_cost(normalized, distance_matrix),
        runtime_s=runtime,
        meta={"metaheuristic": "guided_local_search"},
    )
