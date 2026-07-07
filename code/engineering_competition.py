"""
engineering_competition.py — Run the 7-algorithm competition on classical
constrained engineering design problems (Arora textbook suite: spring, vessel,
welded beam, speed reducer).

Identical algorithms and budgets to egro_cma_competition.py; only the
constraint-aware wrapper changes.  Each algorithm operates on a
*penalty-augmented* fitness F(x) = f(x) + PENALTY * violation(x), where
violation is the sum of max(0, g_i(x)) over all inequality constraints.

The reported statistic for each (problem, algo) is the mean of the
*best feasible objective found at the end of the run*.  If no run ever
visited the feasible region, that (problem, algo) cell is reported
as nan and excluded from the Friedman rank (handled by the analysis
script downstream).

Run:
    python engineering_competition.py --problem AR1_spring --n-runs 30

The `lab_competition_engineering.py` runner fans this out across all four
problems and writes per-problem JSONs in the same shape as the CEC2017
competition artifacts.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Penalty factor: large enough that any positive violation is worse than any
# reasonable feasible objective (objects are O(10) to O(10^4), infeasibilities
# are typically O(1) to O(10^2)).
PENALTY = 1.0e6

# engineering_problems.py lives in the same directory as this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engineering_problems as ep


def penalty_objective(problem):
    """Return a scalar fitness(x) = f(x) + PENALTY * violation(x).  And
    helpers to record (best feasible obj, total budget used)."""
    def obj(x):
        x = np.asarray(x, dtype=float)
        f, cons = problem.evaluate(x)
        v = ep.violation(cons)
        return float(f + PENALTY * v)
    return obj


# ── EGRO-CMA: import the canonical class from the CEC2017 module so that we
# do NOT maintain a second copy of the algorithm.  This is the same class
# the published paper uses; we simply swap the unpenalty objective for a
# penalty-augmented one. ───────────────────────────────────────────────────
import importlib.util as _ilu

egrocma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'egro_cma_competition.py')
if not os.path.exists(egrocma_path):
    # In some directory layouts, the competition script lives in
    # paper_artifacts/code/.  Try that as a fallback.
    egrocma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'paper_artifacts', 'code',
                                'egro_cma_competition.py')
spec = _ilu.spec_from_file_location('egro_cma_competition', egrocma_path)
mod = _ilu.module_from_spec(spec)
sys.modules['egro_cma_competition'] = mod
spec.loader.exec_module(mod)

# Pull out the six competitor classes (EGRO-CMA, CMA-ES, L-SHADE, DE, PSO,
# GWO, WOA) — they don't carry a reference to the original fitness function.
CMAES        = mod.CMAES
LSHADE       = mod.LSHADE
DE           = mod.DE
PSO          = mod.PSO
GWO          = mod.GWO
WOA          = mod.WOA
EGROWithCMA  = mod.EGROWithCMA

ALGORITHMS = {
    'EGRO-CMA':  EGROWithCMA,
    'CMA-ES':    CMAES,
    'L-SHADE':   LSHADE,
    'DE':        DE,
    'PSO':       PSO,
    'GWO':       GWO,
    'WOA':       WOA,
}


# ── Tooling to track best feasible objective separately during the run ─────
class _Evaluator:
    """Wrap a (penalty) fitness so that at every call we also keep a
    side channel with the best FEASIBLE objective encountered so far."""
    def __init__(self, problem):
        self.problem = problem
        self.pen_f   = penalty_objective(problem)
        self.best_feasible_obj = float('inf')
        self.fun_count = 0

    def eval(self, x):
        # x lies in [0, 1]^d (each algorithm runs in unit cube).  We
        # un-map to the actual problem box, evaluate, and return the
        # *penalty-augmented* fitness at the PHYSICAL point x_box.  The
        # penalty must be evaluated at the same point where the constraints
        # are evaluated; otherwise the optimizer is guided by penalties that
        # do not correspond to the actual problem geometry.
        self.fun_count += 1
        xu = np.asarray(x, dtype=float)
        lo, hi = self._bounds()
        # clip into [0,1] to defend against FP drift
        xu = np.clip(xu, 0.0, 1.0)
        x_box = lo + (hi - lo) * xu
        f_raw, cons = self.problem.evaluate(x_box)
        v = ep.violation(cons)
        if v <= 0 and f_raw < self.best_feasible_obj:
            self.best_feasible_obj = f_raw
        return self.pen_f(x_box)

    def _bounds(self):
        b = self.problem.bounds
        lo = np.array([bi[0] for bi in b], dtype=float)
        hi = np.array([bi[1] for bi in b], dtype=float)
        return lo, hi


def run_problem(problem, n_runs=30, max_fes_per_d=10_000, base_seed=0, verbose=True):
    """Run the 7-algorithm competition on one engineering problem."""
    d = problem.n_dim
    max_evals = max_fes_per_d * d
    pen_f     = penalty_objective(problem)

    # Per-algorithm shapes:
    #   { algo: { 'obj_mean': ..., 'std': ..., 'all_objs': [...],
    #             'best_feas_count': ..., 'wall': ..., 'eval_count': ... } }
    raw = {a: [] for a in ALGORITHMS}
    n_feas = {a: 0 for a in ALGORITHMS}
    evals  = {a: [] for a in ALGORITHMS}
    wall   = {a: [] for a in ALGORITHMS}

    if verbose:
        print(f'\n=== {problem.slug} (n={d}, max_fes={max_evals}) ===')
        print('%-12s  %-12s  %-12s  %-8s  %-8s' %
              ('algorithm', 'mean best f', 'std', 'feas', 'avg FE'))

    for run in range(n_runs):
        for algo_name, Cls in ALGORITHMS.items():
            t0 = time.time()
            ev = _Evaluator(problem)

            # The algorithms assume scalar bounds; we operate them on the
            # unit cube [0, 1]^d and un-map inside the evaluator.  See _Evaluator.
            kwargs = dict(func=ev.eval, lb=0.0, ub=1.0, dim=d,
                          max_evals=max_evals, seed=base_seed + run)
            if algo_name == 'EGRO-CMA':
                kwargs['use_echo'] = True
            alg = Cls(**kwargs)
            try:
                _x, _f = alg.optimize()
            except Exception as ex:
                # CMA-ES + clipping can occasionally diverge; record as infeasible
                raw[algo_name].append(float('inf'))
                evals[algo_name].append(getattr(alg, 'eval_count', 0))
                wall[algo_name].append(time.time() - t0)
                continue

            wall[algo_name].append(time.time() - t0)
            evals[algo_name].append(getattr(alg, 'eval_count', 0))
            fstar = ev.best_feasible_obj
            raw[algo_name].append(float(fstar) if np.isfinite(fstar) else float('inf'))
            if np.isfinite(fstar):
                n_feas[algo_name] += 1

    # Compute mean / std for each algorithm.
    summary = {}
    for a in ALGORITHMS:
        objs = np.asarray(raw[a], dtype=float)
        finite = objs[np.isfinite(objs)]
        summary[a] = {
            'mean':  float(np.mean(finite)) if finite.size else float('inf'),
            'std'  : float(np.std (finite)) if finite.size else 0.0,
            'n_feas': int(n_feas[a]),
            'n_runs': n_runs,
            'errors': [float(x) for x in objs],
            'wall'  : [float(x) for x in wall[a]],
            'evals' : [int(x)   for x in evals[a]],
        }

    if verbose:
        for a in ALGORITHMS:
            s = summary[a]
            marker = ('*' if np.isfinite(s['mean']) and
                            s['mean'] <= min((summary[b]['mean']
                                              for b in ALGORITHMS
                                              if np.isfinite(summary[b]['mean'])))
                      else ' ')
            print('%-12s  %s%-12.4e  %-12.2e  %-8d  %-8.0f'
                  % (a, marker, s['mean'], s['std'], s['n_feas'],
                     np.mean(s['evals'])))

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', type=str, default='AR1_spring',
                    help='one of AR1_spring, AR2_vessel, AR3_welded, AR4_reducer')
# Per-problem defaults.  AR1/AR2/AR3 use the standard CEC 10^4 FE/d
    # budget.  AR4 (speed reducer, d=7) also runs at 10^4 FE/d; the larger
    # 3.5*10^4 FE/d budget we tried earlier exceeded the GitHub Actions
    # 90-minute job timeout.
    PER_PROBLEM_FES = {
        'AR1_spring':  10_000,
        'AR2_vessel':  10_000,
        'AR3_welded':  10_000,
        'AR4_reducer': 10_000,
    }
    PER_PROBLEM_RUNS = {
        'AR1_spring':  30,
        'AR2_vessel':  30,
        'AR3_welded':  30,
        'AR4_reducer': 30,
    }
    ap.add_argument('--n-runs',  type=int, default=None,
                    help='overrides the per-problem default')
    ap.add_argument('--max-fes', type=int, default=None,
                    help='FEs per dimension (defaults to per-problem value)')
    ap.add_argument('--out',     type=str, default='')
    args = ap.parse_args()

    if args.n_runs is None:
        args.n_runs = PER_PROBLEM_RUNS[args.problem]
    if args.max_fes is None:
        args.max_fes = PER_PROBLEM_FES[args.problem]

    probs = {p.slug: p for p in ep.all_problems()}
    problem = probs[args.problem]

    out = args.out or f'engineering_{problem.slug}.json'
    summary = run_problem(problem, args.n_runs, args.max_fes)
    summary['__meta__'] = {
        'problem': problem.slug,
        'n_dim':   problem.n_dim,
        'f_opt':   problem.f_opt,
        'budget':  args.max_fes * problem.n_dim,
        'n_runs':  args.n_runs,
    }

    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote', out)


if __name__ == '__main__':
    main()
