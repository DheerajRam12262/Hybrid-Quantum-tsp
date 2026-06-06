"""CLI entry point for the fair TSP benchmark.

Thin wrapper around :mod:`quantum_classical_benchmark.benchmark.harness`. Run
with no arguments for the built-in defaults, point ``--config`` at a YAML file
(see ``benchmarks/config.yaml``), or override individual values on the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_classical_benchmark.benchmark.harness import (  # noqa: E402
    BenchmarkConfig,
    run_benchmark,
    save_outputs,
)
from quantum_classical_benchmark.solvers.quantum_inspired import QuantumInspiredConfig  # noqa: E402
from quantum_classical_benchmark.solvers.simulated_annealing import (  # noqa: E402
    SimulatedAnnealingConfig,
)

BUILTIN_DEFAULTS: dict[str, Any] = {
    "sizes": [5, 8, 10, 12, 15, 20, 25],
    "trials": 5,
    "time_budget": 0.25,
    "seed": 42,
    "max_exact_cities": 12,
    "no_ortools": False,
    "sa_initial_temp": 10_000.0,
    "sa_cooling_rate": 0.9995,
    "qi_reps": 2,
    "qi_n_chains": 8,
    "qi_initial_temp": 8_000.0,
    "qi_cooling_rate": 0.99935,
}


def _load_config_file(path: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    bench = data.get("benchmark", {})
    solvers = data.get("solvers", {})
    sa = solvers.get("simulated_annealing", {})
    qi = solvers.get("quantum_inspired", {})
    flat: dict[str, Any] = {
        "sizes": bench.get("sizes"),
        "trials": bench.get("trials"),
        "time_budget": bench.get("time_budget_s"),
        "seed": bench.get("seed"),
        "max_exact_cities": bench.get("max_exact_cities"),
        "no_ortools": (not bench["include_ortools"]) if "include_ortools" in bench else None,
        "sa_initial_temp": sa.get("initial_temp"),
        "sa_cooling_rate": sa.get("cooling_rate"),
        "qi_reps": qi.get("reps"),
        "qi_n_chains": qi.get("n_chains"),
        "qi_initial_temp": qi.get("initial_temp"),
        "qi_cooling_rate": qi.get("cooling_rate"),
    }
    return {k: v for k, v in flat.items() if v is not None}


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()

    defaults = dict(BUILTIN_DEFAULTS)
    if known.config:
        defaults.update(_load_config_file(known.config))

    parser = argparse.ArgumentParser(
        parents=[pre],
        description="Run the fair TSP benchmark (classical vs quantum-inspired vs OR-Tools)",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=defaults["sizes"])
    parser.add_argument("--trials", type=int, default=defaults["trials"])
    parser.add_argument("--time-budget", type=float, default=defaults["time_budget"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max-exact-cities", type=int, default=defaults["max_exact_cities"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--save-instances", action="store_true")
    parser.add_argument(
        "--no-ortools",
        action="store_true",
        default=defaults["no_ortools"],
        help="Disable the OR-Tools baseline",
    )
    parser.add_argument("--sa-initial-temp", type=float, default=defaults["sa_initial_temp"])
    parser.add_argument("--sa-cooling-rate", type=float, default=defaults["sa_cooling_rate"])
    parser.add_argument("--qi-reps", type=int, default=defaults["qi_reps"])
    parser.add_argument("--qi-n-chains", type=int, default=defaults["qi_n_chains"])
    parser.add_argument("--qi-initial-temp", type=float, default=defaults["qi_initial_temp"])
    parser.add_argument("--qi-cooling-rate", type=float, default=defaults["qi_cooling_rate"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / tag

    config = BenchmarkConfig(
        sizes=args.sizes,
        trials=args.trials,
        time_budget_s=args.time_budget,
        seed=args.seed,
        max_exact_cities=args.max_exact_cities,
        include_ortools=not args.no_ortools,
        save_instances=args.save_instances,
        output_dir=output_dir,
        sa=SimulatedAnnealingConfig(
            initial_temp=args.sa_initial_temp, cooling_rate=args.sa_cooling_rate
        ),
        qi=QuantumInspiredConfig(
            reps=args.qi_reps,
            n_chains=args.qi_n_chains,
            initial_temp=args.qi_initial_temp,
            cooling_rate=args.qi_cooling_rate,
        ),
    )

    output = run_benchmark(config)
    save_outputs(output, output_dir)

    print(f"Saved benchmark outputs to: {output_dir}")
    print("\nSummary:")
    print(output.summary.to_string(index=False))
    print("\nTrend metrics:")
    print(json.dumps(output.trends, indent=2))


if __name__ == "__main__":
    main()
