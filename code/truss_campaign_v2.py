"""
truss_campaign_v2.py — full 10-bar truss campaign under the corrected
formulation (vertical displacement limits at the four free nodes only).
Fresh output file; does not mix with the earlier all-dof results.
"""
import json
import os

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from engineering_problems import TRUSS10
from egro_cma_competition import EGROWithCMA, CMAES, LSHADE, DE, PSO, GWO, WOA

BUDGETS = [200, 500, 1000, 2000, 5000, 10000]
P, FEAS_TOL, N_SEEDS = 1e6, 1e-4, 30
OUT = os.path.join(os.path.dirname(__file__), '..', 'results',
                   'engineering_results_truss_v2.json')

ALGOS = {
    'EGRO-CMA': lambda f, s: EGROWithCMA(f, 0.0, 1.0, 10, 10_000, seed=s),
    'CMA-ES':   lambda f, s: CMAES(f, 0.0, 1.0, 10, 10_000, seed=s),
    'L-SHADE':  lambda f, s: LSHADE(f, 0.0, 1.0, 10, 10_000, seed=s),
    'DE':       lambda f, s: DE(f, 0.0, 1.0, 10, 10_000, seed=s),
    'PSO':      lambda f, s: PSO(f, 0.0, 1.0, 10, 10_000, seed=s),
    'GWO':      lambda f, s: GWO(f, 0.0, 1.0, 10, 10_000, seed=s),
    'WOA':      lambda f, s: WOA(f, 0.0, 1.0, 10, 10_000, seed=s),
}


class Penalized:
    def __init__(self):
        self.lb = TRUSS10['lb']
        self.span = TRUSS10['ub'] - TRUSS10['lb']
        self.best_feas, self.n = np.inf, 0
        self.snap = {b: None for b in BUDGETS}

    def __call__(self, u):
        x = self.lb + np.clip(np.asarray(u, float), 0.0, 1.0) * self.span
        f = float(TRUSS10['f'](x))
        gs = TRUSS10['g'](x)
        self.n += 1
        if max(gs) <= FEAS_TOL and f < self.best_feas:
            self.best_feas = f
        for b in BUDGETS:
            if self.snap[b] is None and self.n >= b:
                self.snap[b] = (None if not np.isfinite(self.best_feas)
                                else self.best_feas)
        return f + P * sum(max(0.0, g) for g in gs)

    def final(self):
        for b in BUDGETS:
            if self.snap[b] is None:
                self.snap[b] = (None if not np.isfinite(self.best_feas)
                                else self.best_feas)
        return {str(b): self.snap[b] for b in BUDGETS}


res = {}
if os.path.exists(OUT):
    res = json.load(open(OUT))
for name, mk in ALGOS.items():
    for s in range(N_SEEDS):
        key = '%s||%d' % (name, s)
        if key in res:
            continue
        pen = Penalized()
        mk(pen, s).optimize()
        res[key] = {'algo': name, 'run': s, 'budget': pen.final(),
                    'evals': pen.n}
        with open(OUT, 'w') as fh:
            json.dump(res, fh)
    v = [res['%s||%d' % (name, s)]['budget']['10000'] for s in range(N_SEEDS)]
    v = [x for x in v if x is not None]
    print('%－9s done: mean %.1f  min %.1f  feasible %d/30'.replace('－', '-')
          % (name, float(np.mean(v)), float(np.min(v)), len(v)), flush=True)
print('CAMPAIGN COMPLETE ->', OUT)
