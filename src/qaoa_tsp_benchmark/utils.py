from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import TSPInstance


def generate_euclidean_instance(n_cities: int, seed: int, scale: float = 100.0) -> TSPInstance:
    """Generate reproducible random coordinates and pairwise Euclidean distances."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, scale, size=(n_cities, 2))
    distances = _pairwise_distances(coords)
    return TSPInstance(
        n_cities=n_cities,
        seed=seed,
        coordinates=coords,
        distance_matrix=distances,
    )


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2)).astype(float)
    np.fill_diagonal(distances, 0.0)
    return distances


def random_tour(rng: np.random.Generator, n_cities: int) -> np.ndarray:
    """Generate a tour with city 0 fixed as the first position."""
    tail = np.arange(1, n_cities)
    rng.shuffle(tail)
    return np.concatenate(([0], tail))


def normalize_tour(tour: Iterable[int]) -> tuple[int, ...]:
    """Rotate/reverse a tour so equivalent cycles map to a canonical representation."""
    route = list(tour)
    if route[0] != 0:
        idx0 = route.index(0)
        route = route[idx0:] + route[:idx0]

    reversed_route = [route[0]] + list(reversed(route[1:]))
    return tuple(min(route, reversed_route))


def tour_cost(tour: Iterable[int], distance_matrix: np.ndarray) -> float:
    route = np.asarray(list(tour), dtype=int)
    nxt = np.roll(route, -1)
    return float(np.sum(distance_matrix[route, nxt]))


def is_valid_tour(tour: Iterable[int], n_cities: int) -> bool:
    route = list(tour)
    return len(route) == n_cities and set(route) == set(range(n_cities))


def save_instance(instance: TSPInstance, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cities": instance.n_cities,
        "seed": instance.seed,
        "coordinates": instance.coordinates.tolist(),
        "distance_matrix": instance.distance_matrix.tolist(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_instance(path: str | Path) -> TSPInstance:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    coords = np.asarray(payload["coordinates"], dtype=float)
    distances = np.asarray(payload["distance_matrix"], dtype=float)
    return TSPInstance(
        n_cities=int(payload["n_cities"]),
        seed=int(payload["seed"]),
        coordinates=coords,
        distance_matrix=distances,
    )
