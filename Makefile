# Quantum-Classical Benchmark — common tasks.
# Activate your venv first, or override: `make PYTHON=.venv/bin/python repro`.
PYTHON ?= python
CONFIG := benchmarks/config.yaml
RESULTS := results/benchmark

.PHONY: help install data bench plots repro test lint format type check clean

help:
	@echo "install  Install the package with dev + ortools extras"
	@echo "data     Generate a small seeded set of TSP instances into data/"
	@echo "bench    Run the benchmark (timestamped run under results/)"
	@echo "plots    Render figures from the canonical run ($(RESULTS))"
	@echo "repro    Regenerate the canonical results + figures from scratch"
	@echo "test     Run the test suite"
	@echo "lint     Ruff lint check"
	@echo "format   Apply black + ruff --fix"
	@echo "type     Mypy type check"
	@echo "check    lint + type + test (what CI runs)"
	@echo "clean    Remove caches and ad-hoc benchmark runs"

install:
	$(PYTHON) -m pip install -e ".[dev,ortools]"

data:
	@mkdir -p data
	$(PYTHON) -m quantum_classical_benchmark.problems.tsp_cli --cities 5  --seed 42 --output data/tsp_05.json
	$(PYTHON) -m quantum_classical_benchmark.problems.tsp_cli --cities 12 --seed 42 --output data/tsp_12.json
	$(PYTHON) -m quantum_classical_benchmark.problems.tsp_cli --cities 20 --seed 42 --output data/tsp_20.json

bench:
	$(PYTHON) benchmarks/run_benchmark.py --config $(CONFIG)

# Reproduce every committed figure and table from scratch.
repro:
	$(PYTHON) benchmarks/run_benchmark.py --config $(CONFIG) --output-dir results --tag benchmark
	$(PYTHON) -m quantum_classical_benchmark.plotting.plots --results $(RESULTS)/raw_results.csv

plots:
	$(PYTHON) -m quantum_classical_benchmark.plotting.plots --results $(RESULTS)/raw_results.csv

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests benchmarks

format:
	black src tests benchmarks
	ruff check --fix src tests benchmarks

type:
	mypy

check: lint type test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf results/run_*
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
