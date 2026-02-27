"""Quantum and hybrid optimization modules."""

from __future__ import annotations

from typing import Any

__all__ = ["HybridQAOAConfig", "HybridQAOASolver", "build_tsp_qubo"]


def __getattr__(name: str) -> Any:
    if name in {"HybridQAOAConfig", "HybridQAOASolver"}:
        from .hybrid_optimizer import HybridQAOAConfig, HybridQAOASolver

        return {"HybridQAOAConfig": HybridQAOAConfig, "HybridQAOASolver": HybridQAOASolver}[name]
    if name == "build_tsp_qubo":
        from .qubo_encoder import build_tsp_qubo

        return build_tsp_qubo
    raise AttributeError(name)
