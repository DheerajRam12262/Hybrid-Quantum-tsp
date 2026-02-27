from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import SolverResult
from ..utils import is_valid_tour, load_instance, normalize_tour, random_tour, tour_cost


@dataclass(slots=True)
class HybridQAOAConfig:
    reps: int = 2
    n_chains: int = 8
    initial_temp: float = 8_000.0
    cooling_rate: float = 0.99935
    sync_interval: int = 48
    local_refine_every: int = 64
    local_refine_steps: int = 200


class HybridQAOASolver:
    """QAOA-inspired hybrid optimizer under a fixed wall-clock budget.

    The implementation mixes multiple stochastic chains (exploration) with
    periodic deterministic 2-opt refinement (exploitation), inspired by
    quantum-classical alternating optimization loops.
    """

    def __init__(self, config: HybridQAOAConfig | None = None) -> None:
        self.config = config or HybridQAOAConfig()

    def solve(self, distance_matrix: np.ndarray, time_budget_s: float, seed: int) -> SolverResult:
        n = distance_matrix.shape[0]
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        deadline = start + time_budget_s

        chains = [random_tour(rng, n) for _ in range(self.config.n_chains)]
        costs = [tour_cost(route, distance_matrix) for route in chains]
        temps = np.geomspace(
            self.config.initial_temp * 1.4,
            max(10.0, self.config.initial_temp * 0.06),
            self.config.n_chains,
        )

        best_idx = int(np.argmin(costs))
        best_route = chains[best_idx].copy()
        best_cost = float(costs[best_idx])

        iterations = 0
        sync_events = 0
        improve_events = 0

        while time.perf_counter() < deadline:
            beta = 0.25 + 0.7 * (0.5 + 0.5 * np.sin(iterations / max(8.0, 10.0 * self.config.reps)))

            for chain_idx in range(self.config.n_chains):
                proposal = self._mixed_neighbor(chains[chain_idx], rng, beta)
                proposal_cost = tour_cost(proposal, distance_matrix)
                delta = proposal_cost - costs[chain_idx]

                if delta <= 0.0 or rng.random() < np.exp(-delta / max(temps[chain_idx], 1e-8)):
                    chains[chain_idx] = proposal
                    costs[chain_idx] = proposal_cost
                    if proposal_cost < best_cost:
                        best_route = proposal.copy()
                        best_cost = float(proposal_cost)
                        improve_events += 1

                temps[chain_idx] = max(1e-3, temps[chain_idx] * self.config.cooling_rate)

            if iterations % self.config.sync_interval == 0:
                self._sync_chains(chains, costs, best_route, distance_matrix, rng)
                sync_events += 1

            if iterations % self.config.local_refine_every == 0:
                refined, refined_cost = self._two_opt_refine(
                    best_route,
                    best_cost,
                    distance_matrix,
                    max_steps=self.config.local_refine_steps,
                )
                if refined_cost < best_cost:
                    best_route = refined
                    best_cost = refined_cost
                    improve_events += 1

            iterations += 1

        runtime = time.perf_counter() - start
        normalized = normalize_tour(best_route)
        assert is_valid_tour(normalized, n)

        return SolverResult(
            tour=normalized,
            cost=best_cost,
            runtime_s=runtime,
            meta={
                "iterations": iterations,
                "sync_events": sync_events,
                "improve_events": improve_events,
                "reps": self.config.reps,
                "n_chains": self.config.n_chains,
            },
        )

    @staticmethod
    def _mixed_neighbor(tour: np.ndarray, rng: np.random.Generator, beta: float) -> np.ndarray:
        candidate = tour.copy()
        n = len(candidate)

        # beta acts like a mixer: lower beta favors global swaps; higher beta favors 2-opt.
        if n > 4 and rng.random() < beta:
            i, j = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
            candidate[i : j + 1] = candidate[i : j + 1][::-1]
            return candidate

        i, j = rng.choice(np.arange(1, n), size=2, replace=False)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        return candidate

    @staticmethod
    def _sync_chains(
        chains: list[np.ndarray],
        costs: list[float],
        best_route: np.ndarray,
        distance_matrix: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        order = np.argsort(costs)
        half = max(1, len(chains) // 2)
        for idx in order[-half:]:
            candidate = best_route.copy()
            for _ in range(2):
                i, j = rng.choice(np.arange(1, len(candidate)), size=2, replace=False)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            chains[int(idx)] = candidate
            costs[int(idx)] = tour_cost(candidate, distance_matrix)

    @staticmethod
    def _two_opt_refine(
        route: np.ndarray,
        current_cost: float,
        distance_matrix: np.ndarray,
        max_steps: int,
    ) -> tuple[np.ndarray, float]:
        best = route.copy()
        best_cost = current_cost
        n = len(route)
        steps = 0

        improved = True
        while improved and steps < max_steps:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    candidate = best.copy()
                    candidate[i : j + 1] = candidate[i : j + 1][::-1]
                    cost = tour_cost(candidate, distance_matrix)
                    steps += 1
                    if cost + 1e-9 < best_cost:
                        best = candidate
                        best_cost = cost
                        improved = True
                        break
                if improved or steps >= max_steps:
                    break

        return best, best_cost


def _load_distance_from_qubo(path: str | Path) -> np.ndarray:
    data = np.load(path)
    if "distance_matrix" not in data:
        raise ValueError("QUBO file does not contain distance_matrix; cannot decode routes")
    return np.asarray(data["distance_matrix"], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QAOA-inspired hybrid TSP optimizer")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="TSP instance JSON")
    src.add_argument("--qubo", type=str, help="QUBO .npz output (must include distance matrix)")

    parser.add_argument("--time-budget", type=float, default=1.0, help="Time budget in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--initial-temp", type=float, default=8_000.0)
    parser.add_argument("--cooling-rate", type=float, default=0.99935)
    parser.add_argument("--sync-interval", type=int, default=48)
    parser.add_argument("--local-refine-every", type=int, default=64)
    parser.add_argument("--local-refine-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        distance_matrix = load_instance(args.input).distance_matrix
    else:
        distance_matrix = _load_distance_from_qubo(args.qubo)

    config = HybridQAOAConfig(
        reps=args.reps,
        n_chains=args.n_chains,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        sync_interval=args.sync_interval,
        local_refine_every=args.local_refine_every,
        local_refine_steps=args.local_refine_steps,
    )

    solver = HybridQAOASolver(config)
    result = solver.solve(distance_matrix, time_budget_s=args.time_budget, seed=args.seed)

    print(f"Best tour: {result.tour}")
    print(f"Cost: {result.cost:.4f}")
    print(f"Runtime: {result.runtime_s:.4f}s")
    print(f"Meta: {result.meta}")


if __name__ == "__main__":
    main()
