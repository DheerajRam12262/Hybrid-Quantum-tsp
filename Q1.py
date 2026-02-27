from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick demo runner for SA vs hybrid benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 8, 10, 12])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="quick_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    cmd = [
        "python",
        "benchmarks/run_benchmark.py",
        "--sizes",
        *[str(s) for s in args.sizes],
        "--trials",
        str(args.trials),
        "--time-budget",
        str(args.time_budget),
        "--tag",
        args.tag,
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)
    print(f"Quick demo finished. Results: {repo_root / 'benchmarks' / 'results' / args.tag}")


if __name__ == "__main__":
    main()
