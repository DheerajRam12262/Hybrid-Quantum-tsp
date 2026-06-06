# Quantum vs Classical Optimization for TSP — A Fair Benchmark

**Author:** Dheeraj Lagudu · **Domain:** Combinatorial Optimization · Quantum
Computing · Operations Research

> This report supersedes an earlier draft that claimed a hybrid-quantum *advantage*
> over classical solvers. The measurements do not support that claim, so the report
> was rewritten to state what the experiments actually show. Scientific honesty —
> including reporting where quantum-inspired methods lose — is the point of the
> project.

## 1. Abstract

We benchmark classical, quantum, and quantum-inspired optimizers on the Travelling
Salesperson Problem (TSP), encoded as a QUBO, under a **matched wall-clock budget**
with **exact ground truth**. The central question is whether a quantum-inspired
metaheuristic, or QAOA, can rival a strong classical baseline (Google OR-Tools) on
classical hardware. **It cannot.** OR-Tools reaches the exact optimum on every
verifiable instance; the metaheuristics match it only for very small N and then
diverge, with the heavier quantum-inspired method degrading fastest under the time
budget. The contribution is a rigorous, reproducible *methodology* for such
comparisons — one that would register a genuine quantum win if one existed.

## 2. Problem & encoding

TSP asks for the minimum-length Hamiltonian cycle over N cities. We use Euclidean
instances with seeded random coordinates. For the quantum/QUBO formulation we use
the permutation-matrix encoding with binary variables `x_{i,p}` (city *i* at
position *p*), giving N² variables and the objective in
[DECISIONS.md §3](DECISIONS.md). Constraint penalties force feasibility; the
distance term encodes tour length. The QUBO is converted to an Ising Hamiltonian
(`x_i = (1−z_i)/2`) for QAOA, with the full linear (`Z`) and quadratic (`ZZ`) terms.

## 3. Solvers

| Solver | Type | Role |
|---|---|---|
| **Held-Karp** | Exact (O(N²·2ᴺ)) | Ground truth for N ≤ 12 |
| **OR-Tools** (Guided Local Search) | Classical | **Honest strong baseline** |
| **Simulated Annealing** | Classical heuristic | Simple metaheuristic baseline |
| **Quantum-Inspired** | Classical (parallel tempering) | Quantum-*structured*, no qubits |
| **QAOA** | Quantum (statevector simulator) | Correctness demo, N ≤ 4 only |
| D-Wave | Quantum annealer | Stubbed; needs Ocean credentials |

## 4. Methodology

- **Matched budget:** identical per-solver wall-clock time per instance.
- **Same seeded instances** across all solvers; every RNG seeded.
- **Exact optimality gap** for N ≤ 12 via Held-Karp.
- **Multiple trials** (5) per size; mean ± std reported.
- **Scaling sweep** N = 5…25; cost/runtime scaling exponents fitted in log-space.
- **Quality vs the baseline** measured as a match-or-beat rate against OR-Tools.

## 5. Results

Canonical run: N = 5…25, 5 trials, 0.25 s budget. Figures in
[results/benchmark/](../results/benchmark/).

**Mean tour cost** and **gap vs exact** (N ≤ 12):

| N | OR-Tools | SA | Quantum-Inspired | SA gap | QI gap |
|--:|--:|--:|--:|--:|--:|
| 5  | 192.5 | 192.5 | 192.5 | 0.00% | 0.00% |
| 8  | 267.6 | 267.6 | 267.6 | 0.00% | 0.00% |
| 10 | 307.4 | 308.0 | 307.4 | 0.21% | 0.00% |
| 12 | 321.4 | 328.8 | 334.1 | 2.26% | 4.03% |
| 15 | 329.3 | 347.9 | 353.3 | — | — |
| 20 | 391.7 | 436.0 | 447.3 | — | — |
| 25 | 425.8 | 485.9 | 516.6 | — | — |

- **OR-Tools mean gap vs exact ≈ 0%** at every size with ground truth.
- **Crossover:** all methods tie up to N ≈ 10 (small enough that even SA finds the
  optimum within budget). From N = 12 the heuristics separate from OR-Tools.
- **At N = 25**, SA is +14% over OR-Tools and quantum-inspired is +21%.
- **Cost-scaling exponent** (cost ∝ Nᵏ): OR-Tools 0.47 < SA 0.56 < QI 0.59.

## 6. Analysis — why the quantum-inspired method loses

The quantum-inspired solver maintains 8 replicas and periodically runs an O(N²)
2-opt refinement. Under a *fixed wall-clock budget*, that richer per-iteration work
buys fewer total moves than plain SA's single-swap loop, so at large N it explores
less and ends up **worse than even simple SA** (517 vs 486 at N = 25). This is a
genuine, useful result about algorithmic overhead under time budgets — and a clean
illustration of why matched-budget benchmarking matters. OR-Tools' specialised
routing search and guided local search simply dominate both.

## 7. Conclusion

On classical hardware, neither the quantum-inspired metaheuristic nor simulator
QAOA beats a strong classical baseline on TSP, and they scale worse. This is the
expected, honest outcome; the project's worth is the **fair, reproducible
methodology** (matched budget, exact ground truth, seeded trials, scaling study)
rather than a manufactured advantage.

## 8. Limitations & future work

- Wall-clock results are hardware-dependent; trends reproduce, absolute numbers
  vary (see [DECISIONS.md §7](DECISIONS.md)).
- QAOA is simulator-only and capped at N ≤ 4 by the N²-qubit encoding.
- **Future:** capacitated VRP under the same harness; locality-reduced QUBO
  encodings; a `p`-sweep QAOA study on tiny instances; an LKH baseline for an even
  stronger classical reference.
