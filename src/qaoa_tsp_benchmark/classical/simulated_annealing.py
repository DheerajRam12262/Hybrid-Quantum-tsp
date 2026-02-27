from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

from ..types import SolverResult
from ..utils import is_valid_tour, load_instance, normalize_tour, random_tour, tour_cost


@dataclass(slots=True)
class SimulatedAnnealingConfig:
    initial_temp: float = 10_000.0
    cooling_rate: float = 0.9995
    min_temp: float = 1e-3
    use_two_opt: bool = False


class SimulatedAnnealingSolver:
    """Simple SA baseline used for fixed-budget comparisons."""

    def __init__(self, config: SimulatedAnnealingConfig | None = None) -> None:
        self.config = config or SimulatedAnnealingConfig()

    def solve(
        self,
        distance_matrix: np.ndarray,
        time_budget_s: float,
        seed: int,
        initial_tour: np.ndarray | None = None,
    ) -> SolverResult:
        n = distance_matrix.shape[0]
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()
        deadline = start_time + time_budget_s

        current = initial_tour.copy() if initial_tour is not None else random_tour(rng, n)
        current_cost = tour_cost(current, distance_matrix)
        best = current.copy()
        best_cost = current_cost

        temp = self.config.initial_temp
        accepted = 0
        iterations = 0

        while time.perf_counter() < deadline:
            proposal = self._propose_neighbor(current, rng)
            proposal_cost = tour_cost(proposal, distance_matrix)
            delta = proposal_cost - current_cost

            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current = proposal
                current_cost = proposal_cost
                accepted += 1
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

            temp = max(self.config.min_temp, temp * self.config.cooling_rate)
            iterations += 1

        runtime = time.perf_counter() - start_time
        normalized = normalize_tour(best)
        assert is_valid_tour(normalized, n)
        return SolverResult(
            tour=normalized,
            cost=best_cost,
            runtime_s=runtime,
            meta={
                "iterations": iterations,
                "accepted": accepted,
                "acceptance_rate": accepted / iterations if iterations else 0.0,
                "final_temp": temp,
            },
        )

    def _propose_neighbor(
        self,
        tour: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidate = tour.copy()
        n = len(candidate)

        if self.config.use_two_opt and rng.random() < 0.4 and n > 4:
            i, j = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
            candidate[i : j + 1] = candidate[i : j + 1][::-1]
            return candidate

        i, j = rng.choice(np.arange(1, n), size=2, replace=False)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated annealing baseline for TSP")
    parser.add_argument("--input", required=True, help="Input TSP instance JSON")
    parser.add_argument("--time-budget", type=float, default=1.0, help="Time budget in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--initial-temp", type=float, default=10_000.0)
    parser.add_argument("--cooling-rate", type=float, default=0.9995)
    parser.add_argument("--min-temp", type=float, default=1e-3)
    parser.add_argument("--use-two-opt", action="store_true", help="Enable occasional 2-opt moves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = load_instance(args.input)
    solver = SimulatedAnnealingSolver(
        SimulatedAnnealingConfig(
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            min_temp=args.min_temp,
            use_two_opt=args.use_two_opt,
        )
    )
    result = solver.solve(instance.distance_matrix, args.time_budget, seed=args.seed)
    print(f"Best tour: {result.tour}")
    print(f"Cost: {result.cost:.4f}")
    print(f"Runtime: {result.runtime_s:.4f}s")
    print(f"Meta: {result.meta}")


if __name__ == "__main__":
    main()
