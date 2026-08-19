"""
analyze_cec2011.py — aggregate the CEC2011 real-world results.

Per (problem, algorithm): mean, std, median, best over 30 seeds at the full
10^4*d budget, plus the half-budget snapshot.  Per problem: Wilcoxon rank-sum
(Mann-Whitney U) of EGRO-CMA against every competitor over the 30 independent runs
(two-sided, alpha=0.05).  With only 4 problems a Friedman/Nemenyi CD analysis
is underpowered, so significance is claimed per problem only.
"""
import json, os
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(__file__)
res = json.load(open(os.path.join(HERE, '..', 'results', 'cec2011_results.json')))

ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']
PROBS = ['FM sound-wave estimation', 'Radar polyphase code design',
         'Lennard-Jones cluster (N=10)', 'Lennard-Jones cluster (N=15)']

print("total runs:", len(res), " errors:",
      sum(1 for r in res.values() if 'error' in r))


def vals(prob, algo, frac_key=None):
    out = []
    for r in res.values():
        if r['problem'] == prob and r['algo'] == algo:
            b = r['budget']
            key = frac_key or max(b, key=lambda k: int(k))
            out.append(b[key])
    return np.array(sorted(out, key=lambda _: 0) and out)


ranks_full = {a: [] for a in ALGOS}
for prob in PROBS:
    fb = next(r['f_best'] for r in res.values() if r['problem'] == prob)
    n = next(r['n'] for r in res.values() if r['problem'] == prob)
    print(f"\n=== {prob}  (d={n}, best-known {fb:.6g}) ===")
    stats = {}
    for algo in ALGOS:
        v = vals(prob, algo)
        half = vals(prob, algo, str(int(0.5 * 10000 * n)))
        stats[algo] = v
        print(f"  {algo:9s} mean={v.mean():.6g}  std={v.std():.3g}  "
              f"median={np.median(v):.6g}  best={v.min():.6g}  "
              f"(half-budget mean={half.mean():.6g})  n={len(v)}")
    order = sorted(ALGOS, key=lambda a: stats[a].mean())
    for i, a in enumerate(order):
        ranks_full[a].append(i + 1)
    print(f"  ranking by mean: {' > '.join(order)}")
    # Unpaired two-sided rank-sum tests, EGRO-CMA vs each competitor,
    # Holm-corrected across the six comparisons within this problem.
    # (Runs use independent per-algorithm streams, so the test is unpaired.)
    e = stats['EGRO-CMA']
    raw = {}
    for algo in ALGOS[1:]:
        u, p = mannwhitneyu(e, stats[algo], alternative='two-sided')
        raw[algo] = p
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m_cnt, prev, holm = len(items), 0.0, {}
    for i, (algo, p) in enumerate(items):
        ap = min(1.0, max(prev, (m_cnt - i) * p))
        prev = ap
        holm[algo] = ap
    print("  Rank-sum, EGRO-CMA vs each (Holm-adjusted, * = p<0.05):")
    for algo in ALGOS[1:]:
        direction = '<' if e.mean() < stats[algo].mean() else '>'
        sig = '*' if holm[algo] < 0.05 else ' '
        print(f"    vs {algo}: EGRO {direction} "
              f"(raw p={raw[algo]:.2g}, Holm p={holm[algo]:.2g}){sig}")

print("\n=== mean rank across the 4 problems (by mean) ===")
for a in sorted(ALGOS, key=lambda a: np.mean(ranks_full[a])):
    print(f"  {a:9s} {np.mean(ranks_full[a]):.2f}   per-problem: {ranks_full[a]}")
