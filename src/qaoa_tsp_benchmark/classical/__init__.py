"""Classical solvers and exact baselines."""

from __future__ import annotations

from typing import Any

__all__ = ["exact_tsp_if_small", "SimulatedAnnealingConfig", "SimulatedAnnealingSolver"]


def __getattr__(name: str) -> Any:
    if name == "exact_tsp_if_small":
        from .brute_force import exact_tsp_if_small

        return exact_tsp_if_small
    if name in {"SimulatedAnnealingConfig", "SimulatedAnnealingSolver"}:
        from .simulated_annealing import SimulatedAnnealingConfig, SimulatedAnnealingSolver

        return {
            "SimulatedAnnealingConfig": SimulatedAnnealingConfig,
            "SimulatedAnnealingSolver": SimulatedAnnealingSolver,
        }[name]
    raise AttributeError(name)
