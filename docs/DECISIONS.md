# Design decisions & trade-offs

This document records *why* the benchmark is built the way it is — the questions a
reviewer would ask. It is the companion to [REPORT.md](REPORT.md) (results) and the
[README](../README.md) (overview).

## 1. OR-Tools is the baseline (not a weak SA)

A benchmark that beats a deliberately hobbled baseline proves nothing. Google
OR-Tools with **Guided Local Search** is a genuinely strong, industry-standard TSP
solver. Making it the baseline means any heuristic has to earn its keep against
something real. It does not: OR-Tools reaches the exact optimum on every instance
we can verify, and the metaheuristics fall behind as N grows.

## 2. Matched **wall-clock** budget, not matched iterations

Solvers do very different work per iteration (one SA swap vs. an OR-Tools GLS step
vs. a parallel-tempering sweep with periodic O(N²) 2-opt). Equalising *iterations*
would be meaningless. We equalise the only currency a practitioner actually
spends: **wall-clock time**. Each solver gets the same budget per instance; an
OR-Tools run that converges early and stops is reported honestly as a short
runtime. The cost: wall-clock results depend on the host CPU (see §7).

## 3. The TSP → QUBO encoding and penalty weight

We use the standard permutation-matrix formulation with binary variables
`x_{i,p} = 1` iff city *i* is at tour position *p* (N² variables). The objective is

```
min  A · Σ_i (1 − Σ_p x_{i,p})²      # each city used exactly once
   + A · Σ_p (1 − Σ_i x_{i,p})²      # each position filled exactly once
   + B · Σ_{i≠j,p} W_ij x_{i,p} x_{j,p+1}   # tour length (cyclic)
```

The constraint penalty `A` must dominate any distance saving from breaking a
constraint, or the optimiser will "cheat". We default to `A = 10 · max(W)` and
`B = 1`; the tests assert a feasible tour has strictly lower QUBO energy than an
infeasible one. (Tuning `A` is itself a known QAOA pain point — too large and the
energy landscape becomes flat/ill-conditioned for the optimiser.)

## 4. QUBO ↔ Ising conversion (and why the old QAOA was wrong)

QAOA optimises an Ising Hamiltonian over spins `z ∈ {−1,+1}`, so the QUBO must be
converted via `x_i = (1 − z_i)/2`. For a symmetric QUBO `Q` this yields

```
h_i   = −½ · rowsum_i,   J_ij = ½ · Q_ij   (i<j),   const = offset + ¼·ΣQ + ¼·tr(Q)
```

The **original code mapped only the diagonal of Q to single-qubit `Z` terms and
dropped every `ZZ` coupling** — i.e. it optimised a decoupled problem that was not
TSP. The rebuilt `encoding/qubo.py::qubo_to_ising` produces the full `h`, `J`, and
constant, and `tests/test_qubo.py` verifies the Ising energy of any spin string
equals the QUBO energy of the corresponding bit string. `tests/test_qaoa.py`
verifies the assembled Pauli cost Hamiltonian's diagonal matches the QUBO too.

## 5. "Quantum-inspired", named honestly

The multi-replica solver is a **classical** parallel-tempering metaheuristic. It
borrows *structure* from quantum optimisation (a temperature ladder reminiscent of
quantum/thermal annealing; an alternating explore/exploit schedule reminiscent of
QAOA's cost/mixer layers) but runs entirely on a CPU. Calling it "hybrid quantum"
(as the original repo did) would be misleading, so it is `QuantumInspiredSolver`
and the docs say plainly that no qubits are involved.

## 6. QAOA is simulator-only and does not scale

The N²-variable encoding means N² qubits. Statevector simulation is exponential in
qubit count, so even N = 4 (16 qubits) is near the practical ceiling, and N = 5
(25 qubits) is infeasible on a laptop. The QAOA path is therefore a
*correctness demonstration*, not a competitor. It is guarded by `max_qubits` and
clearly labelled.

## 7. Reproducibility caveat

Seeds make instance generation and the heuristic move sequences deterministic. But
the *time-limited* solvers (OR-Tools, and the budget cutoff of SA/quantum-inspired)
depend on CPU speed: a faster machine fits more search into 0.25 s. So the
committed numbers reproduce in **shape and ordering**, not bit-for-bit, across
hardware. `make repro` regenerates everything from scratch.

## 8. VRP is out of scope

The original repo carried placeholder VRP modules (a greedy stub, "future
extension"). Shipping a non-functional VRP that implies a capability is the kind of
overclaim this project exists to avoid, so it was removed. Extending the same fair
methodology to capacitated VRP — real demands, a capacity-penalised QUBO, an
OR-Tools CVRP baseline — is the natural next milestone.

## What a *real* quantum advantage on this problem would require

Being specific about why we don't see one:

1. **Fault-tolerant, large-scale hardware.** TSP's N² encoding needs hundreds of
   logical qubits for interesting N; today's NISQ devices can't hold the circuit
   depth QAOA needs without noise swamping the signal.
2. **Better-than-`p=1..2` QAOA depth**, which the barren-plateau problem makes hard
   to train, plus encodings that need far fewer qubits than the naive N².
3. **Problem structure quantum exploits** — TSP/QUBO has no known structure where
   QAOA provably beats strong classical heuristics. OR-Tools, LKH, and
   concorde-class solvers are extremely strong classical competition.
4. An **honest, matched-budget comparison** — exactly this harness — to detect a
   crossover regime if one ever appears. The framework is the contribution; it is
   ready to register a quantum win the day one is real.
