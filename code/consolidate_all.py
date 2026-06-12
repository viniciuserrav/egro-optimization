"""
consolidate_all.py  —  Self-contained analysis of the CORRECTED results.

Reads ONLY the archive's results/ folder (no external/stale paths):
  ../results/CORRECTED_main_*.json        -> main competition (7 algorithms)
  ../results/CORRECTED_*ablation*.json    -> EGRO-PSO vs EGRO-CMA ablation

The ablation files reuse keys F{fn}_d{dim}_PSO / _CMA for EGRO-PSO / EGRO-CMA;
they are loaded into a SEPARATE namespace so they never collide with the
main-competition standalone-PSO key.

Prints, per dimension: tie-aware Friedman ranks, Nemenyi leading group,
EGRO-CMA-vs-standalone-CMA-ES head-to-head, and the EGRO-PSO vs EGRO-CMA ablation.

Run:
    python consolidate_all.py
"""
import os, json, glob, math

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

ALGS = ["EGRO_CMA", "CMA_ES", "L_SHADE", "DE", "PSO", "GWO", "WOA"]
LABEL = {"EGRO_CMA":"EGRO-CMA","CMA_ES":"CMA-ES","L_SHADE":"L-SHADE","DE":"DE",
         "PSO":"PSO","GWO":"GWO","WOA":"WOA"}
# F5 excluded: degenerate in the opfunu CEC2017 implementation (random-point error
# < 1 vs 1e2-1e18 for all other functions); see the paper's benchmark subsection.
FN_IDS = [1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
DIMS = [10, 30, 50]
_Q05 = {7: 2.949}


def _fin(x):
    try: return math.isfinite(float(x))
    except Exception: return False


def avg_ranks(row):
    items = sorted(row.items(), key=lambda kv: kv[1]); r={}; i=0; n=len(items)
    while i < n:
        j = i
        while j+1 < n and items[j+1][1] == items[i][1]: j += 1
        a = ((i+1)+(j+1))/2.0
        for k in range(i, j+1): r[items[k][0]] = a
        i = j+1
    return r


def _mean(v):
    return v["mean"] if isinstance(v, dict) and "mean" in v else None


def main():
    # Main competition: only CORRECTED_main_*.json (excludes the ablation files).
    md = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "CORRECTED_main_*.json"))):
        for k, v in json.load(open(f)).items():
            md.setdefault(k, v)

    print("=== MAIN COMPETITION (corrected echo covariance) ===")
    for dim in DIMS:
        fns = []
        for fn in FN_IDS:
            if ("F%d_d%d_INVALID" % (fn, dim)) in md: continue
            ks = ["F%d_d%d_%s" % (fn, dim, a) for a in ALGS]
            if all(k in md for k in ks) and all(_fin(_mean(md[k])) for k in ks):
                fns.append(fn)
        if not fns:
            print("  d=%d: 0 complete functions" % dim); continue
        ranks = {a: [] for a in ALGS}; ebc = 0
        for fn in fns:
            row = {a: _mean(md["F%d_d%d_%s" % (fn, dim, a)]) for a in ALGS}
            rr = avg_ranks(row)
            for a in ALGS: ranks[a].append(rr[a])
            if row["EGRO_CMA"] < row["CMA_ES"]: ebc += 1
        mr = {a: sum(ranks[a])/len(ranks[a]) for a in ALGS}
        CD = _Q05[7]*math.sqrt(7*8/(6.0*len(fns)))
        leader = min(mr, key=lambda a: mr[a])
        grp = [LABEL[a] for a in ALGS if abs(mr[a]-mr[leader]) < CD]
        print("  d=%2d (%2d fns): %s" % (dim, len(fns),
              "  ".join("%s=%.2f" % (LABEL[a], mr[a]) for a in sorted(ALGS, key=lambda a: mr[a]))))
        print("       Nemenyi CD=%.2f  leading group=%s  |  EGRO-CMA beats CMA-ES %d/%d"
              % (CD, grp, ebc, len(fns)))

    # Ablation in a SEPARATE namespace (keys F{fn}_d{dim}_PSO / _CMA).
    ad = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "CORRECTED_*ablation*.json"))):
        for k, v in json.load(open(f)).items():
            ad.setdefault(k, v)
    print("\n=== ABLATION: EGRO-CMA vs EGRO-PSO (separate namespace) ===")
    for dim in DIMS:
        cw = pw = 0
        for fn in FN_IDS:
            p = _mean(ad.get("F%d_d%d_PSO" % (fn, dim), {}))
            c = _mean(ad.get("F%d_d%d_CMA" % (fn, dim), {}))
            if p is None or c is None or not (_fin(p) and _fin(c)): continue
            if c < p: cw += 1
            elif p < c: pw += 1
        if cw + pw:
            print("  d=%2d: EGRO-CMA wins %d, EGRO-PSO wins %d (n=%d)" % (dim, cw, pw, cw+pw))


if __name__ == "__main__":
    main()
