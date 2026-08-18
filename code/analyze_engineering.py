import json, os
import numpy as np

HERE = os.path.dirname(__file__)
res = json.load(open(os.path.join(HERE, '..', 'results', 'engineering_results.json')))

ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']
PROBS = ['Tension/compression spring', 'Pressure vessel', 'Welded beam', 'Speed reducer']
BUDGETS = ['200', '500', '1000', '2000', '5000', '10000']
FOPT = {}

print("total runs in file:", len(res))

# aggregate: agg[prob][algo][budget] -> list of feasible values (drop infeasible/None)
def collect(prob, algo, budget):
    vals = []
    feas = 0; tot = 0
    for k, r in res.items():
        if r['problem'] == prob and r['algo'] == algo:
            tot += 1
            v = r['budget'][budget]
            if v is not None:
                vals.append(v); feas += 1
    return np.array(vals), (feas, tot)

for prob in PROBS:
    FOPT[prob] = next(r['f_opt'] for r in res.values() if r['problem'] == prob)

for budget in ['2000', '10000']:
    print("\n" + "=" * 78)
    print(f"BUDGET = {budget} FEs   (mean best-feasible over 30 seeds; * = best per row)")
    print("=" * 78)
    means = {p: {} for p in PROBS}
    for prob in PROBS:
        row = {}
        for algo in ALGOS:
            vals, (feas, tot) = collect(prob, algo, budget)
            row[algo] = (vals.mean() if len(vals) else np.inf, feas, tot)
            means[prob][algo] = vals.mean() if len(vals) else np.inf
        best = min(row, key=lambda a: row[a][0])
        print(f"\n{prob}  (f_opt={FOPT[prob]:.6g})")
        for algo in ALGOS:
            m, feas, tot = row[algo]
            star = '*' if algo == best else ' '
            print(f"  {star}{algo:9s} mean={m:.6g}  feasible {feas}/{tot}")
    # mean rank across the 4 problems
    print(f"\n-- mean rank across {len(PROBS)} problems (budget {budget}) --")
    ranks = {a: [] for a in ALGOS}
    for prob in PROBS:
        order = sorted(ALGOS, key=lambda a: means[prob][a])
        # tie-aware ranking
        vals = [means[prob][a] for a in ALGOS]
        srt = sorted(range(len(ALGOS)), key=lambda i: vals[i])
        rank = {}
        i = 0
        while i < len(srt):
            j = i
            while j + 1 < len(srt) and abs(vals[srt[j+1]] - vals[srt[i]]) < 1e-9:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k2 in range(i, j + 1):
                rank[ALGOS[srt[k2]]] = avg
            i = j + 1
        for a in ALGOS:
            ranks[a].append(rank[a])
    for a in sorted(ALGOS, key=lambda a: np.mean(ranks[a])):
        print(f"  {a:9s} {np.mean(ranks[a]):.2f}")

# detailed stats for the headline budget for table building
print("\n\n### DETAILED (budget 2000) mean / std / best / feasrate ###")
for prob in PROBS:
    print(f"\n{prob} f_opt={FOPT[prob]:.6g}")
    for algo in ALGOS:
        vals, (feas, tot) = collect(prob, algo, '2000')
        if len(vals):
            print(f"  {algo:9s} mean={vals.mean():.6g} std={vals.std():.3g} "
                  f"best={vals.min():.6g} feas={feas}/{tot}")
        else:
            print(f"  {algo:9s} NO FEASIBLE ({feas}/{tot})")
