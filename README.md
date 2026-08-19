# EGRO — Echo-Guided Rescue Optimization

Reproducibility package for the paper *"Echo-Guided Rescue Optimization: a
gradient-guided hybrid framework for engineering design optimization."*

EGRO couples L-BFGS-B gradient descent with a population-based inner search
(instantiated with PSO and CMA-ES). Gradient "skiers" descend to promising
basins, and around each landing an isolated rescue group searches a region
whose geometry follows from the descent trajectory; a dimension-normalized
isolation index then decides whether a group has converged or has moved beyond
the landing-reference level, in which case its best position seeds a fresh
descent. Two instantiations are provided:

- **EGRO-PSO** — Clerc–Kennedy particle swarm as the inner search;
- **EGRO-CMA** — CMA-ES whose initial covariance is *echo-shaped*, aligning
  its principal search axis with the gradient (echo) direction.

This repository contains the implementation, the raw per-run results behind
every table in the paper, and the scripts that regenerate the tables and
statistics.

## Quick start

```
pip install setuptools==75.8.0 numpy==2.4.4 scipy==1.17.1 opfunu==1.0.1 cma==4.4.4
python code/gen_official_tables.py results .   # tables + ranks + CD diagrams from stored results
python code/engineering_problems.py            # verify the 5 engineering formulations
```

Note: `setuptools` is pinned because `opfunu` imports `pkg_resources`, which
was removed from setuptools >= 81.

## CEC 2017 campaign (official numbering)

- Benchmark: the **29 official CEC 2017 functions F1, F3–F30** via `opfunu`
  1.0.1, at dimensions **d = 10, 30, 50**. `opfunu` renumbers the surviving
  functions consecutively (its `F22017` is the official F3), so
  `code/cec2017_official.py` maps official function *k* to `opfunu` index
  *k−1* for *k* ≥ 3. All indices in `results/cec2017_official_d*.json` and in
  the paper are **official**.
- 30 runs per (function, algorithm), seed = run index; budget **10,000 × d**
  evaluations, counted by a shared wrapper (finite-difference gradient calls
  included).
- Competitors: EGRO-CMA, CMA-ES, L-SHADE, DE, PSO, GWO, WOA. The five
  non-library competitors are study-specific implementations written from the
  original publications.
- Absolute errors below 1e-8 are treated as zero before averaging and ranking
  (CEC convention).

**Known defects of the `opfunu` 1.0.1 instances, disclosed in the paper:**

- official **F18 and F30 return non-finite values at d = 10** and are excluded
  at that dimension (27 valid functions there);
- for a subset of functions (official F10, F12, F16, F17, F20, F22, F26, F27
  at d = 10, similar subsets at d = 30/50) the stated reference optimum lies
  **above** the attainable minimum, so signed errors can be negative. Rankings
  are unaffected (the reference is a per-function constant), but comparisons
  against results produced with the official evaluator should treat these
  functions with caution. Newly generated results store the best decision
  vector per run (`xbest`) to allow external re-evaluation.

### Headline results (regenerate with `gen_official_tables.py`)

Mean tie-aware Friedman rank (lower is better):

| algorithm | d = 10 (27 fns) | d = 30 (29 fns) | d = 50 (29 fns) |
|---|:---:|:---:|:---:|
| L-SHADE      | 1.50 | 1.26 | 1.26 |
| **EGRO-CMA** | **2.24** | **2.41** | **2.55** |
| CMA-ES       | 4.09 | 2.67 | 2.47 |
| DE           | 3.41 | 5.31 | 5.72 |
| WOA          | 5.07 | 4.69 | 4.62 |
| GWO          | 5.52 | 5.34 | 5.10 |
| PSO          | 6.17 | 6.31 | 6.28 |

EGRO-CMA is statistically tied with L-SHADE at every dimension
(Nemenyi, α = 0.05) and beats its own CMA-ES engine on 24/27 functions at
d = 10 (15/29 at d = 30, 12/29 at d = 50).

## Engineering campaign

Five constrained problems in `code/engineering_problems.py` (spring, pressure
vessel, welded beam, speed reducer, and the **10-bar truss**, which assembles
and solves K·u = F per evaluation; vertical displacement limits at the four
free nodes, per the benchmark formulation). `verify_all()` checks each
formulation against its published optimum.

- `code/run_engineering.py` — full 5-problem campaign (30 seeds, budget
  snapshots, best-feasible tracking at 1e-4 tolerance).
- `code/truss_campaign_v2.py` — the truss campaign reported in the paper;
  stores the best design vector and its constraint residual per run
  (`results/engineering_results_truss_v2.json`).
- `code/gen_truss_table.py` — regenerates the paper's truss table.

## Repository layout

- `code/` — algorithms (`egro_cma_competition.py`, `egro_pso_vs_cmaes.py`),
  campaign runners, and analysis scripts (`gen_official_tables.py`,
  `merge_official.py`, `analyze_cec2011.py`).
- `results/` — raw per-run results: `cec2017_official_d{10,30,50}.json`,
  `official_stats.json`, `cec2011_results.json`,
  `engineering_results_truss_v2.json`, plus legacy files from the earlier
  pre-renumbering campaign (kept for provenance; superseded by the official
  files).
- `.github/workflows/` — the CI matrices that produced the CEC campaign.
- `validate.py` — **legacy**: validates the pre-renumbering campaign only;
  use `code/gen_official_tables.py` for the current results.
