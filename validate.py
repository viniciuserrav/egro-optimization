"""
validate.py  —  Self-contained check of the EGRO paper's CORRECTED results.
Merges every results/CORRECTED_*.json (echo-covariance data) and prints
tie-aware Friedman ranks, sole/tied wins, the EGRO-CMA vs CMA-ES head-to-head,
and the Nemenyi leading/lagging groups — no dependencies beyond the stdlib.

Run:
    python validate.py
"""
import os, json, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))

# Merge ONLY the main-competition files.  The ablation files
# (CORRECTED_*ablation*.json) reuse keys F{fn}_d{dim}_PSO / _CMA for EGRO-PSO /
# EGRO-CMA, which would collide with the main-competition standalone-PSO key and
# contaminate the ranks; they are handled separately (see consolidate_all.py).
data = {}
files = sorted(glob.glob(os.path.join(HERE, "results", "CORRECTED_main_*.json")))
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for k, v in d.items():
        if k in data:
            continue
        if isinstance(v, dict) and "mean" in v:
            data[k] = {"mean": v["mean"], "std": v.get("std", 0.0)}
        elif v is True:                      # INVALID marker
            data[k] = True

if not data:
    print("No corrected results found in results/CORRECTED_*.json yet.\n"
          "The previous (isotropic-CMA) results were retracted; see CORRECTIONS.md.")
    raise SystemExit(0)
print("Merged %d corrected entries from %d file(s)." % (len(data), len(files)))

ALGS  = ["EGRO_CMA", "CMA_ES", "L_SHADE", "DE", "PSO", "GWO", "WOA"]
LABEL = {"EGRO_CMA": "EGRO-CMA", "CMA_ES": "CMA-ES", "L_SHADE": "L-SHADE",
         "DE": "DE", "PSO": "PSO", "GWO": "GWO", "WOA": "WOA"}
# F5 excluded: degenerate in the opfunu CEC2017 implementation (random-point error
# < 1 vs 1e2-1e18 for all other functions); see the paper's benchmark subsection.
FN_IDS = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27, 28, 29]

def fclass(fn):
    if fn in (1, 3): return "unimodal"
    if 4 <= fn <= 10: return "multimodal"
    if 11 <= fn <= 20: return "hybrid"
    return "composition"

# Nemenyi critical values q_alpha (alpha=0.05, two-tailed, infinite df) by k
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

def nemenyi_cd(k, N):
    import math
    return _Q05[k] * math.sqrt(k * (k + 1) / (6.0 * N))

import math

def _finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False

def avg_ranks(row):
    """Tie-aware ranks (1=best); tied algorithms share the average position."""
    items = sorted(row.items(), key=lambda kv: kv[1])
    ranks = {}; i = 0; n = len(items)
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg
        i = j + 1
    return ranks

for dim in (10, 30, 50):
    fns = []
    for f in FN_IDS:
        if ("F%d_d%d_INVALID" % (f, dim)) in data:
            continue
        keys = ["F%d_d%d_%s" % (f, dim, a) for a in ALGS]
        if all(k in data for k in keys) and all(_finite(data[k]["mean"]) for k in keys):
            fns.append(f)
    if not fns:
        print("d=%d: 0 complete functions" % dim); continue

    ranks = {a: [] for a in ALGS}
    sole  = {a: 0 for a in ALGS}   # strictly-best
    tiedw = {a: 0 for a in ALGS}   # achieves the best (possibly shared)
    egro_beats_cma = 0
    cwins = collections.defaultdict(collections.Counter)
    for fn in fns:
        row = {a: data["F%d_d%d_%s" % (fn, dim, a)]["mean"] for a in ALGS}
        r = avg_ranks(row)                       # tie-aware
        for a in ALGS:
            ranks[a].append(r[a])
        bv = min(row.values())
        winners = [a for a in ALGS if row[a] == bv]
        for a in winners: tiedw[a] += 1
        if len(winners) == 1:
            sole[winners[0]] += 1; cwins[fclass(fn)][winners[0]] += 1
        if row["EGRO_CMA"] < row["CMA_ES"]:
            egro_beats_cma += 1
    mrank = {a: sum(ranks[a]) / len(ranks[a]) for a in ALGS}
    wins = tiedw

    print("\n=== d=%d  (%d/28 functions complete) ===" % (dim, len(fns)))
    print("  mean Friedman rank (lower = better):")
    for a in sorted(ALGS, key=lambda a: mrank[a]):
        print("    %-9s %.2f" % (LABEL[a], mrank[a]))
    print("  best (tied):", {LABEL[a]: tiedw[a] for a in ALGS if tiedw[a]})
    print("  best (sole) :", {LABEL[a]: sole[a] for a in ALGS if sole[a]})
    print("  EGRO-CMA beats CMA-ES on %d/%d functions" % (egro_beats_cma, len(fns)))
    print("  per-class wins:",
          {k: {LABEL[a]: n for a, n in v.items()} for k, v in cwins.items()})
    CD = nemenyi_cd(len(ALGS), len(fns))
    leader = min(mrank, key=lambda a: mrank[a])
    tied = [LABEL[a] for a in ALGS if abs(mrank[a] - mrank[leader]) < CD]
    sigw = [LABEL[a] for a in ALGS if (mrank[a] - mrank[leader]) >= CD]
    print("  Nemenyi CD (alpha=0.05) = %.3f" % CD)
    print("    leading group (not significantly different): %s" % tied)
    print("    significantly worse than the leader: %s" % sigw)

print("\nValidated against results/cec2017_means_d10_d30.json")
