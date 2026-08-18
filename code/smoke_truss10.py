"""
smoke_truss10.py — quick validation that the EGRO code runs against the
10-bar truss FEA benchmark.  3 algorithms x 5 seeds x 10k FE (not the full
campaign; run_engineering.py covers that).  Prints best-feasible weights
against the 5060.85 lb literature optimum.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')

from engineering_problems import TRUSS10
from egro_cma_competition import EGROWithCMA, CMAES, PSO

BUDGET, SEEDS, P = 10_000, range(5), 1e6


class Penalized:
    def __init__(self, prob):
        self.p = prob
        self.lb = prob['lb']; self.span = prob['ub'] - prob['lb']
        self.best_feas = np.inf

    def __call__(self, u):
        x = self.lb + np.clip(np.asarray(u, float), 0.0, 1.0) * self.span
        f = float(self.p['f'](x)); gs = self.p['g'](x)
        viol = sum(max(0.0, gi) for gi in gs)
        if max(gs) <= 1e-4 and f < self.best_feas:
            self.best_feas = f
        return f + P * viol


ALGOS = {
    'EGRO-CMA': lambda f, s: EGROWithCMA(f, 0.0, 1.0, 10, BUDGET, seed=s),
    'CMA-ES':   lambda f, s: CMAES(f, 0.0, 1.0, 10, BUDGET, seed=s),
    'PSO':      lambda f, s: PSO(f, 0.0, 1.0, 10, BUDGET, seed=s),
}

print(f"10-bar truss, budget {BUDGET} FE, {len(list(SEEDS))} seeds "
      f"(best known 5060.85 lb)")
for name, mk in ALGOS.items():
    res = []
    for s in SEEDS:
        pen = Penalized(TRUSS10)
        mk(pen, s).optimize()
        res.append(pen.best_feas)
    res = np.array(res)
    print(f"{name:9s} best-feasible weight: mean {res.mean():9.2f}  "
          f"min {res.min():9.2f}  max {res.max():9.2f}")
