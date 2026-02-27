from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..utils import load_instance


def _index(city: int, position: int, n_cities: int) -> int:
    return city * n_cities + position


def _add_qubo_term(Q: np.ndarray, i: int, j: int, coeff: float) -> None:
    """Store a QUBO term in a symmetric matrix used with x^T Q x."""
    if i == j:
        Q[i, i] += coeff
        return
    half = coeff * 0.5
    Q[i, j] += half
    Q[j, i] += half


def build_tsp_qubo(
    distance_matrix: np.ndarray,
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Encode TSP into a dense QUBO matrix."""
    n = distance_matrix.shape[0]
    if penalty_a is None:
        penalty_a = float(np.max(distance_matrix) * 10.0)

    n_vars = n * n
    Q = np.zeros((n_vars, n_vars), dtype=float)
    offset = 0.0

    # Each city appears exactly once: A * sum_i (1 - sum_p x_{i,p})^2
    for city in range(n):
        offset += penalty_a
        for pos in range(n):
            idx = _index(city, pos, n)
            _add_qubo_term(Q, idx, idx, -penalty_a)
        for pos_a in range(n):
            for pos_b in range(pos_a + 1, n):
                idx_a = _index(city, pos_a, n)
                idx_b = _index(city, pos_b, n)
                _add_qubo_term(Q, idx_a, idx_b, 2.0 * penalty_a)

    # Each position has exactly one city: A * sum_p (1 - sum_i x_{i,p})^2
    for pos in range(n):
        offset += penalty_a
        for city in range(n):
            idx = _index(city, pos, n)
            _add_qubo_term(Q, idx, idx, -penalty_a)
        for city_a in range(n):
            for city_b in range(city_a + 1, n):
                idx_a = _index(city_a, pos, n)
                idx_b = _index(city_b, pos, n)
                _add_qubo_term(Q, idx_a, idx_b, 2.0 * penalty_a)

    # Tour length term: B * sum_{i,j,p} W_ij x_{i,p} x_{j,p+1}
    for pos in range(n):
        nxt = (pos + 1) % n
        for city_i in range(n):
            idx_i = _index(city_i, pos, n)
            for city_j in range(n):
                if city_i == city_j:
                    continue
                idx_j = _index(city_j, nxt, n)
                _add_qubo_term(Q, idx_i, idx_j, penalty_b * float(distance_matrix[city_i, city_j]))

    meta = {
        "n_cities": float(n),
        "n_variables": float(n_vars),
        "penalty_a": float(penalty_a),
        "penalty_b": float(penalty_b),
    }
    return Q, offset, meta


def qubo_energy(bitstring: np.ndarray, qubo: np.ndarray, offset: float = 0.0) -> float:
    x = np.asarray(bitstring, dtype=float)
    return float(x @ qubo @ x + offset)


def save_qubo(
    output_path: str | Path,
    qubo: np.ndarray,
    offset: float,
    distance_matrix: np.ndarray,
    meta: dict[str, float],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qubo=qubo,
        offset=np.array([offset], dtype=float),
        distance_matrix=distance_matrix,
        **{k: np.array([v], dtype=float) for k, v in meta.items()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a TSP instance as QUBO")
    parser.add_argument("--input", required=True, help="Input TSP instance JSON")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--penalty-a", type=float, default=None, help="Constraint penalty")
    parser.add_argument("--penalty-b", type=float, default=1.0, help="Distance weight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = load_instance(args.input)
    qubo, offset, meta = build_tsp_qubo(
        instance.distance_matrix,
        penalty_a=args.penalty_a,
        penalty_b=args.penalty_b,
    )
    save_qubo(args.output, qubo, offset, instance.distance_matrix, meta)
    print(f"Saved QUBO ({qubo.shape[0]} vars) to {args.output}")
    print(f"penalty_a={meta['penalty_a']:.3f}, penalty_b={meta['penalty_b']:.3f}, offset={offset:.3f}")


if __name__ == "__main__":
    main()
