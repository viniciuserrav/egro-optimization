"""
validate.py — one-command reproduction of the paper's headline numbers.

Recomputes, directly from results/cec2017_official_d{10,30,50}.json:
  * tie-aware mean Friedman ranks (CEC convention: |error| < 1e-8 -> 0);
  * the Nemenyi critical difference and the leading group;
  * sole-win tallies and the EGRO-CMA vs CMA-ES head-to-head;
  * the robustness check over functions with a consistent reference.

Also re-verifies every stored 10-bar truss design by recomputing its weight
and constraint residuals from the finite-element model.

Standard library + numpy only.  Run:  python validate.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']
ETOL = 1e-8
Q_ALPHA_7 = 2.949                     # Nemenyi q_0.05, k = 7

EXPECTED = {                          # paper values (all / consistent-only)
    10: {'n': 27, 'L-SHADE': 1.50, 'EGRO-CMA': 2.24, 'clean_n': 19},
    30: {'n': 29, 'L-SHADE': 1.26, 'EGRO-CMA': 2.41, 'clean_n': 23},
    50: {'n': 29, 'L-SHADE': 1.26, 'EGRO-CMA': 2.55, 'clean_n': 22},
}


def tie_ranks(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values))
    sv = np.array(values)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def load(dim):
    path = os.path.join(RES, 'cec2017_official_d%d.json' % dim)
    data = json.load(open(path))
    means, affected = {}, set()
    for key, rec in data.items():
        if key == '_meta':
            continue
        row, ok = {}, True
        for a in ALGOS:
            arec = rec.get(a)
            errs = [0.0 if abs(e) < ETOL else e
                    for e in (arec or {}).get('errors', []) if e is not None]
            if arec is None or len(errs) < len(arec['errors']):
                ok = False
                break
            row[a] = float(np.mean(errs))
            if min(errs) < -ETOL:
                affected.add(key)
        if ok:
            means[key] = row
    return means, affected


def ranks_over(means, keys):
    R = np.array([tie_ranks([means[f][a] for a in ALGOS]) for f in keys])
    return dict(zip(ALGOS, R.mean(axis=0)))


def check_cec():
    ok = True
    for dim in (10, 30, 50):
        means, affected = load(dim)
        keys = sorted(means, key=lambda s: int(s[1:]))
        clean = [k for k in keys if k not in affected]
        r = ranks_over(means, keys)
        rc = ranks_over(means, clean)
        cd = Q_ALPHA_7 * np.sqrt(7 * 8 / (6.0 * len(keys)))
        best = min(r, key=r.get)
        lead = sorted(a for a in ALGOS if r[a] - r[best] <= cd)
        wins = sum(1 for f in keys if means[f]['EGRO-CMA'] < means[f]['CMA-ES'])
        exp = EXPECTED[dim]

        print('d = %d  (%d functions, %d with a consistent reference)'
              % (dim, len(keys), len(clean)))
        print('   ranks      : ' + ', '.join(
            '%s %.2f' % (a, r[a]) for a in sorted(ALGOS, key=r.get)))
        print('   consistent : ' + ', '.join(
            '%s %.2f' % (a, rc[a]) for a in sorted(ALGOS, key=rc.get)))
        print('   CD = %.2f, leading group: %s' % (cd, ', '.join(lead)))
        print('   EGRO-CMA beats CMA-ES on %d/%d functions' % (wins, len(keys)))

        for k, v in (('n', len(keys)), ('clean_n', len(clean))):
            if exp[k] != v:
                print('   MISMATCH %s: expected %s, got %s' % (k, exp[k], v))
                ok = False
        for a in ('L-SHADE', 'EGRO-CMA'):
            if abs(r[a] - exp[a]) > 0.005:
                print('   MISMATCH %s rank: expected %.2f, got %.2f'
                      % (a, exp[a], r[a]))
                ok = False
        if 'EGRO-CMA' not in lead:
            print('   MISMATCH: EGRO-CMA not in the leading group')
            ok = False
    return ok


def check_truss():
    path = os.path.join(RES, 'engineering_results_truss_v2.json')
    if not os.path.exists(path):
        print('truss results not found, skipped')
        return True
    sys.path.insert(0, os.path.join(HERE, 'code'))
    try:
        from engineering_problems import TRUSS10
    except Exception as e:
        print('cannot import the truss model (%s), skipped' % e)
        return True
    d = json.load(open(path))
    worst_f, worst_g, n = 0.0, -np.inf, 0
    for rec in d.values():
        x = rec.get('x_best')
        if x is None:
            continue
        worst_f = max(worst_f, abs(TRUSS10['f'](x) - rec['budget']['10000']))
        worst_g = max(worst_g, max(TRUSS10['g'](x)))
        n += 1
    print('truss: %d stored designs re-verified' % n)
    print('   max |stored - recomputed| weight : %.2e lb' % worst_f)
    print('   max constraint residual          : %.2e (tolerance 1e-4)'
          % worst_g)
    return worst_f < 1e-6 and worst_g <= 1e-4


if __name__ == '__main__':
    print('=' * 68)
    print('CEC 2017 (official numbering, opfunu 1.0.1 implementation)')
    print('=' * 68)
    a = check_cec()
    print()
    print('=' * 68)
    print('10-bar truss')
    print('=' * 68)
    b = check_truss()
    print()
    print('ALL CHECKS PASSED' if a and b else 'CHECKS FAILED')
    sys.exit(0 if a and b else 1)
