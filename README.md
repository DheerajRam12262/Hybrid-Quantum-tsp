# QAOA TSP Benchmark

Hybrid-vs-classical benchmark for TSP under a **matched per-solver wall-clock budget**.

Project moto:

> Hybrid beats a simple classical baseline under the same time budget on small graphs, and you show the trend.

## What This Repository Demonstrates

- Reproducible Euclidean TSP generation with fixed seeds.
- QUBO encoding of TSP (`x_{i,p}` formulation).
- Classical baseline: simple Simulated Annealing (SA).
- Hybrid solver: QAOA-inspired multi-chain stochastic search + local refinement.
- Fair benchmark loop where SA and hybrid receive exactly the same time budget.
- Trend outputs: raw CSVs, aggregate CSVs, scaling exponents, win-rate and plots.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── src/qaoa_tsp_benchmark/
│   ├── problem_generator.py
│   ├── metrics.py
│   ├── utils.py
│   ├── classical/
│   │   ├── simulated_annealing.py
│   │   ├── brute_force.py
│   │   └── or_tools_solver.py
│   ├── quantum/
│   │   ├── qubo_encoder.py
│   │   ├── qaoa_circuit.py
│   │   ├── hybrid_optimizer.py
│   │   └── dwave_solver.py
│   └── vrp/
│       ├── vrp_encoder.py
│       └── vrp_benchmark.py
├── benchmarks/
│   ├── config.yaml
│   ├── run_benchmark.py
│   ├── plot_results.py
│   └── results/
└── tests/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Optional stacks:

```bash
pip install -e .[quantum]
pip install -e .[annealer]
pip install -e .[classical_plus]
```

## End-to-End Workflow

### 1) Generate Problem Instance

```bash
python -m qaoa_tsp_benchmark.problem_generator \
  --cities 15 --seed 42 --output data/tsp_15.json
```

### 2) Encode as QUBO

```bash
python -m qaoa_tsp_benchmark.quantum.qubo_encoder \
  --input data/tsp_15.json --output data/qubo_15.npz
```

### 3) Run SA Baseline

```bash
python -m qaoa_tsp_benchmark.classical.simulated_annealing \
  --input data/tsp_15.json --time-budget 1.0 --seed 42
```

### 4) Run Hybrid Optimizer

```bash
python -m qaoa_tsp_benchmark.quantum.hybrid_optimizer \
  --input data/tsp_15.json --time-budget 1.0 --seed 42 --reps 2
```

### 5) Run Benchmark Trend Study

```bash
python benchmarks/run_benchmark.py \
  --sizes 5 8 10 12 15 20 25 --trials 5 --time-budget 0.25
```

### 6) Plot Results

```bash
python benchmarks/plot_results.py \
  --results benchmarks/results/<run_tag>/raw_results.csv
```

Generated artifacts:

- `raw_results.csv`
- `summary_results.csv`
- `trend_metrics.json`
- `solution_quality_vs_n.png`
- `runtime_vs_n.png`
- `optimality_gap_vs_n.png` (when exact ground truth is available)
- `hybrid_win_rate_vs_n.png`

## Benchmark Fairness Policy

- SA and hybrid each get the same `--time-budget` per instance.
- Same instance seed is used for both solvers.
- For `N <= max_exact_cities`, exact optimum is computed for objective gap reporting.
- All runs are repeated for multiple trials and summarized statistically.

## Notes on Quantum Integration

- Default local workflow is dependency-light and reproducible.
- `qaoa_circuit.py` exposes optional Qiskit integration hooks.
- D-Wave integration is intentionally stubbed unless credentials and SDK are configured.

## Running Tests

```bash
pytest
```

## GitHub Hosting Checklist

```bash
git init
git add .
git commit -m "Initial benchmark scaffold: SA vs hybrid QAOA trend pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/qaoa-tsp-benchmark.git
git push -u origin main
```

