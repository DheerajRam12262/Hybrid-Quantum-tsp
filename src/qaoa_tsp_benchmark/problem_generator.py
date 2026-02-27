from __future__ import annotations

import argparse

from .utils import generate_euclidean_instance, save_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a random Euclidean TSP instance")
    parser.add_argument("--cities", type=int, required=True, help="Number of cities")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument("--scale", type=float, default=100.0, help="Coordinate range")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = generate_euclidean_instance(args.cities, args.seed, scale=args.scale)
    save_instance(instance, args.output)
    print(f"Saved TSP instance with {args.cities} cities to {args.output}")


if __name__ == "__main__":
    main()
