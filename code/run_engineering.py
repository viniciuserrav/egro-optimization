"""
run_engineering.py — run the SEVEN paper algorithms on the four standard
constrained mechanical-design problems, reusing the exact algorithm classes
from egro_cma_competition.py.

Each problem is searched in the unit cube [0,1]^n (mapped to the physical box
inside the objective), with a static penalty P*sum(max(0,g_i)) guiding the
search.  We separately track the best *feasible* objective (all g_i <= FEAS_TOL)
and snapshot it at a set of tight evaluation budgets — the paper's engineering
thesis is the small-budget regime, where gradient seeding should discriminate.

30 independent seeds per (problem, algorithm).  Checkpointed: results are
written after every run and completed (problem, algo, seed) keys are skipped
on restart.
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

from engineering_problems import PROBLEMS
from egro_cma_competition import EGROWithCMA, CMAES, LSHADE, DE, PSO, GWO, WOA

BUDGETS    = [200, 500, 1000, 2000, 5000, 10000]
MAX_BUDGET = max(BUDGETS)
P_PENALTY   = float(os.environ.get('P_PENALTY', '1e6'))
PENALTY_POW = float(os.environ.get('PENALTY_POW', '1'))   # 1=linear(kinked), 2=quadratic(smooth)
FEAS_TOL    = 1e-4
N_SEEDS     = 30
_SUFFIX     = '' if PENALTY_POW == 1.0 else '_pow%d' % int(PENALTY_POW)
OUT         = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'engineering_results%s.json' % _SUFFIX)

ALGOS = {
    'EGRO-CMA': lambda f, d, me, s: EGROWithCMA(f, 0.0, 1.0, d, me, seed=s, use_echo=True),
    'CMA-ES':   lambda f, d, me, s: CMAES(f, 0.0, 1.0, d, me, seed=s),
    'L-SHADE':  lambda f, d, me, s: LSHADE(f, 0.0, 1.0, d, me, seed=s),
    'DE':       lambda f, d, me, s: DE(f, 0.0, 1.0, d, me, seed=s),
    'PSO':      lambda f, d, me, s: PSO(f, 0.0, 1.0, d, me, seed=s),
    'GWO':      lambda f, d, me, s: GWO(f, 0.0, 1.0, d, me, seed=s),
    'WOA':      lambda f, d, me, s: WOA(f, 0.0, 1.0, d, me, seed=s),
}


class Penalized:
    """Unit-cube penalized objective with best-feasible tracking + budget snapshots."""
    def __init__(self, prob):
        self.p = prob
        self.lb = prob['lb']; self.span = prob['ub'] - prob['lb']
        self.best_feas = np.inf
        self.n = 0
        self.snap = {b: None for b in BUDGETS}

    def __call__(self, u):
        x = self.lb + np.clip(np.asarray(u, float), 0.0, 1.0) * self.span
        f = float(self.p['f'](x))
        gs = self.p['g'](x)
        viol = sum(max(0.0, gi) ** PENALTY_POW for gi in gs)
        self.n += 1
        if max(gs) <= FEAS_TOL and f < self.best_feas:
            self.best_feas = f
        for b in BUDGETS:
            if self.snap[b] is None and self.n >= b:
                self.snap[b] = self.best_feas
        return f + P_PENALTY * viol

    def finalize(self):
        for b in BUDGETS:
            if self.snap[b] is None:
                self.snap[b] = self.best_feas
        return {str(b): (None if not np.isfinite(self.snap[b]) else self.snap[b])
                for b in BUDGETS}


def load():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh)
    return {}


def save(res):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, OUT)


def main():
    res = load()
    total = len(PROBLEMS) * len(ALGOS) * N_SEEDS
    done = 0
    for pi, prob in enumerate(PROBLEMS):
        for ai, (aname, make) in enumerate(ALGOS.items()):
            for run in range(N_SEEDS):
                key = f"{prob['name']}||{aname}||{run}"
                if key in res:
                    done += 1
                    continue
                seed = 100000 + pi * 10000 + ai * 1000 + run
                obj = Penalized(prob)
                try:
                    alg = make(obj, prob['n'], MAX_BUDGET, seed)
                    alg.optimize()
                except Exception as e:
                    print(f"  ERR {key}: {type(e).__name__}: {e}", flush=True)
                res[key] = dict(problem=prob['name'], algo=aname, run=run,
                                n=prob['n'], f_opt=prob['f_opt'],
                                budget=obj.finalize())
                done += 1
                save(res)
                if done % 20 == 0:
                    print(f"  {done}/{total} runs done", flush=True)
    print(f"DONE: {done}/{total} runs. Saved to {OUT}", flush=True)


if __name__ == '__main__':
    main()
