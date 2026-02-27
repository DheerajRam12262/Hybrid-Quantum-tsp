from __future__ import annotations

import numpy as np

from ..types import SolverResult
from ..utils import normalize_tour, tour_cost


class ORToolsUnavailableError(RuntimeError):
    pass


def solve_with_or_tools(distance_matrix: np.ndarray) -> SolverResult:
    """Optional OR-Tools baseline for users who install the dependency."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ORToolsUnavailableError(
            "OR-Tools is not installed. Install project extras with 'pip install .[quantum]'."
        ) from exc

    n = distance_matrix.shape[0]
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    scaled = np.rint(distance_matrix * 1000).astype(int)

    def distance_callback(from_index: int, to_index: int) -> int:
        return int(
            scaled[
                manager.IndexToNode(from_index),
                manager.IndexToNode(to_index),
            ]
        )

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools failed to produce a solution")

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    normalized = normalize_tour(route)
    return SolverResult(tour=normalized, cost=tour_cost(normalized, distance_matrix), runtime_s=0.0)
