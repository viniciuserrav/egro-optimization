# EGRO — Echo-Guided Rescue Optimization

Reproducibility package for the paper *"Echo-Guided Rescue Optimization: a modular
gradient-metaheuristic framework inspired by mountain rescue operations."*

EGRO couples L-BFGS-B gradient descent with an interchangeable population-based
inner search. Gradient "skiers" descend to promising basins, and around each
landing an isolated rescue group searches a region whose geometry is dictated by
the descent trajectory itself; a dimension-invariant isolation index then decides
whether a group has converged or has discovered a new region worth a fresh
descent. Two instantiations are provided:

- **EGRO-PSO** — Clerc–Kennedy particle swarm as the inner search;
- **EGRO-CMA** — CMA-ES whose initial covariance is *echo-shaped*, aligning its
  principal search axis with the gradient (echo) direction.

This repository contains the implementation, the raw per-run results behind every
table, and a one-command validation script.

## Quick start

```
pip install -r requirements.txt    # numpy, scipy, opfunu, cma
python validate.py                 # reproduces all headline numbers from stored results
```

`validate.py` uses only the Python standard library and recomputes — directly
from `results/` — the tie-aware Friedman ranks, the win tallies, the EGRO-CMA vs
CMA-ES head-to-head, and the Nemenyi leading/lagging groups.

## Experimental protocol

- Benchmark: **CEC 2017** (via `opfunu`), dimensions **d = 10, 30, 50**.
- 30 independent runs per (function, algorithm); budget **10,000 × d** function
  evaluations.
- Competitors: EGRO-CMA, CMA-ES, L-SHADE, DE, PSO, GWO, WOA.
- **F2 and F5 are excluded** as degenerate in the opfunu CEC 2017 implementation:
  their error at random points is below 1, versus 1e2–1e18 for every other
  function, so all algorithms solve them trivially.

## Headline results (reproduced by `validate.py`)

Mean Friedman rank (lower is better):

| algorithm | d = 10 (25 fns) | d = 30 (27 fns) | d = 50 (27 fns) |
|---|:---:|:---:|:---:|
| L-SHADE      | 1.60 | 1.28 | 1.30 |
| **EGRO-CMA** | **2.02** | **2.43** | **2.44** |
| CMA-ES       | 4.20 | 2.67 | 2.44 |
| DE           | 3.24 | 5.19 | 5.63 |
| GWO / WOA / PSO | bottom group | bottom group | bottom group |

- **d = 10:** Nemenyi leading group = {EGRO-CMA, L-SHADE, DE} (CD = 1.80);
  EGRO-CMA beats standalone CMA-ES on **23/25** functions.
- **d = 30 / 50:** leading group = {EGRO-CMA, CMA-ES, L-SHADE}; EGRO-CMA beats
  CMA-ES on **13/27** and **11/27** (a statistical tie with CMA-ES at d = 50).
- **Inner-search ablation** (EGRO-CMA vs EGRO-PSO under identical framework
  settings): **17–5 (3 ties) / 16–9 (2 ties) / 18–7 (2 ties)** at d = 10 / 30 / 50.

## Layout

```
validate.py                 one-command reproduction of all headline numbers (stdlib only)
requirements.txt            numpy, scipy, opfunu, cma
code/
  egro_cma_competition.py   main competition: all 7 algorithms on CEC 2017, identical FE counting
  egro_pso_vs_cmaes.py      inner-search ablation: EGRO-PSO vs EGRO-CMA, identical framework settings
  consolidate_all.py        analysis: Friedman ranks, Nemenyi groups, ablation, for d = 10/30/50
results/
  CORRECTED_main_LAB_complete.json      raw per-run results, 7 algorithms across dimensions
  CORRECTED_LAB_ablation_complete.json  EGRO-PSO vs EGRO-CMA ablation
  CORRECTED_table_cec2017_d{10,30,50}.tex   the exact LaTeX tables used in the paper
```

## Re-run from scratch

Recompute one function across all seven algorithms (the paper's full 10⁴·d budget;
`--max-fes` is the per-dimension base, so the total budget is `max_fes × dim`):

```
cd code
python egro_cma_competition.py --dim 10 --fns 9 --n-runs 30 --max-fes 10000 --out check.json
```

then compare `F9_d10_EGRO_CMA` against the six competitors in `check.json`.

## Key parameters (as in the paper)

- Framework: `α_in = 0.76`, `σ_dist = 1.5`, `n_p = 15`, `T_diag = 60`,
  `n_skiers = 5`, `N_spawn = 20`, `β_r = 0.5`.
- EGRO-PSO inner search: Clerc–Kennedy `w = 0.729`, `c1 = c2 = 1.494`.
- EGRO-CMA inner search: echo-shaped initial covariance
  `C0 = (σ_long² d̂d̂ᵀ + σ_short²(I − d̂d̂ᵀ)) / λ̄`, step `σ0 = σ_dist·σ_long/√d`.
