"""
cec2017_official.py — full CEC 2017 rerun under the OFFICIAL numbering.

Fixes the indexing slip of the original campaign: opfunu numbers the 29
surviving functions consecutively (its F2 is the official F3 Zakharov, its
F29 is the official F30), so the original run dropped official F3 entirely
and mislabeled everything from F4 up.  This runner:

  * runs ALL 29 official functions F1, F3..F30 (mapping to opfunu k-1 for
    k >= 3), including official F6 (Schaffer F7) — exclusions, if any,
    happen at analysis time, not at run time;
  * runs the same 7 algorithms as the paper, imported unchanged from
    egro_cma_competition.py, same 10^4*d budget, 30 paired seeds;
  * stores results keyed by OFFICIAL function number, with the opfunu
    index and the mapping recorded in the metadata block;
  * records the actual evaluations consumed per run (audit request);
  * checkpoints after every completed (function, algorithm, run) and
    resumes cleanly, so the job survives reboots of the lab PC;
  * parallelizes across runs with multiprocessing.

Usage:
    python cec2017_official.py                     # d=10,30,50, all algos
    python cec2017_official.py --dims 10           # one dimension
    python cec2017_official.py --workers 8
    python cec2017_official.py --smoke             # tiny end-to-end check
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import egro_cma_competition as base

OFFICIAL_FNS = [1] + list(range(3, 31))          # 29 functions
ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']
N_RUNS = 30
OUT_TPL = 'cec2017_official_d%d.json'
SMOKE_TPL = 'SMOKE_cec2017_official_d%d.json'
META = {
    'numbering': 'official CEC2017 (F1, F3-F30)',
    'mapping': 'official F1 -> opfunu F12017; official Fk -> opfunu F(k-1)2017 for k>=3',
    'budget': '10^4 * d function evaluations per run',
    'seeds': 'seed = run index (0..29), paired across algorithms',
    'note': 'rerun correcting the indexing slip of the original campaign',
}


def official_to_opfunu(fid: int) -> int:
    return fid if fid == 1 else fid - 1


def make_algo(name, func, lb, ub, dim, budget, seed):
    if name == 'EGRO-CMA':
        return base.EGROWithCMA(func, lb, ub, dim, budget, seed=seed)
    cls = {'CMA-ES': base.CMAES, 'L-SHADE': base.LSHADE, 'DE': base.DE,
           'PSO': base.PSO, 'GWO': base.GWO, 'WOA': base.WOA}[name]
    return cls(func, lb, ub, dim, budget, seed=seed)


def one_run(task):
    """Worker: one (official_fid, algo, run) at dimension dim."""
    fid, algo, run, dim, budget = task
    t0 = time.time()
    try:
        bm = base.load_bm(official_to_opfunu(fid), dim)
        opt = make_algo(algo, bm['func'], bm['lb'], bm['ub'], dim, budget, run)
        xbest, fbest = opt.optimize()
        err = float(fbest) - bm['f_global']
        xb = [float(v) for v in np.asarray(xbest).ravel()]
        if not np.isfinite(err):
            return (fid, algo, run, None, int(opt.eval_count),
                    time.time() - t0, 'non-finite', xb)
        return (fid, algo, run, float(err), int(opt.eval_count),
                time.time() - t0, None, xb)
    except Exception as e:                       # record, never kill the pool
        return fid, algo, run, None, 0, time.time() - t0, \
            f'{type(e).__name__}: {e}', None


def _env_meta():
    """Environment fingerprint stored alongside the results."""
    import platform
    import subprocess
    meta = {'python': platform.python_version(),
            'platform': platform.platform()}
    for mod in ('numpy', 'scipy', 'opfunu', 'cma'):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            meta[mod] = 'unavailable'
    try:
        meta['commit'] = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        meta['commit'] = 'unknown'
    return meta


def load_ckpt(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {'_meta': dict(META)}


def save_ckpt(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def run_dim(dim, fns, algos, n_runs, budget, workers, outdir,
            smoke=False):
    out = os.path.join(outdir, (SMOKE_TPL if smoke else OUT_TPL) % dim)
    data = load_ckpt(out)
    data['_meta'].update(dim=dim, budget_evals=budget, n_runs=n_runs,
                         environment=_env_meta())

    tasks = []
    for fid in fns:
        key = 'F%d' % fid
        rec = data.setdefault(key, {})
        for algo in algos:
            arec = rec.setdefault(algo, {'errors': [None] * n_runs,
                                         'evals': [None] * n_runs,
                                         'flags': [None] * n_runs,
                                         'xbest': [None] * n_runs})
            # pad if a previous smoke run used fewer runs (or older schema)
            arec.setdefault('xbest', [None] * n_runs)
            for fld in ('errors', 'evals', 'flags', 'xbest'):
                while len(arec[fld]) < n_runs:
                    arec[fld].append(None)
            for run in range(n_runs):
                done = (arec['errors'][run] is not None
                        or arec['flags'][run] is not None)
                if not done:
                    tasks.append((fid, algo, run, dim, budget))

    total = len(fns) * len(algos) * n_runs
    print('[d=%d] %d/%d runs pending (%d already done)'
          % (dim, len(tasks), total, total - len(tasks)), flush=True)
    if not tasks:
        return

    t0, done_now = time.time(), 0
    with mp.Pool(workers) as pool:
        for fid, algo, run, err, evals, secs, flag, xb in \
                pool.imap_unordered(one_run, tasks, chunksize=1):
            rec = data['F%d' % fid][algo]
            rec['errors'][run] = err
            rec['evals'][run] = evals
            rec['flags'][run] = flag
            rec['xbest'][run] = xb
            done_now += 1
            save_ckpt(out, data)
            if done_now % 10 == 0 or done_now == len(tasks):
                rate = done_now / max(time.time() - t0, 1e-9)
                eta_h = (len(tasks) - done_now) / max(rate, 1e-9) / 3600.0
                msg = ('[d=%d] %d/%d done  (%.1f runs/min, ETA %.1f h)'
                       % (dim, done_now, len(tasks), rate * 60, eta_h))
                print(msg, flush=True)
                with open(os.path.join(outdir, 'progress_d%d.txt' % dim),
                          'w') as fh:
                    fh.write(msg + '\n')
    # Fail loudly on infrastructure errors: a green job must mean real data.
    # ('non-finite' flags are legitimate benchmark defects, e.g. official
    # F30 at d=10, and do not count as failures.)
    bad = {}
    for fid in fns:
        for algo in algos:
            for f in data['F%d' % fid][algo]['flags']:
                if f and f != 'non-finite':
                    bad[f] = bad.get(f, 0) + 1
    if bad:
        for f, n in sorted(bad.items(), key=lambda x: -x[1])[:3]:
            print('[d=%d] TASK FAILURES %dx: %s' % (dim, n, f), flush=True)
        raise SystemExit('[d=%d] aborting: %d failed tasks indicate a broken '
                         'environment' % (dim, sum(bad.values())))
    print('[d=%d] complete -> %s' % (dim, out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', type=int, nargs='+', default=[10, 30, 50])
    ap.add_argument('--fns', type=int, nargs='+', default=None,
                    help='official function numbers (default: all 29)')
    ap.add_argument('--algos', nargs='+', default=None, choices=ALGOS,
                    help='subset of algorithms (default: all 7)')
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny check: 2 fns x 2 algos x 2 runs, budget 2000')
    args = ap.parse_args()

    if args.smoke:
        print('SMOKE MODE', flush=True)
        run_dim(10, [1, 30], ['EGRO-CMA', 'PSO'], 2, 2000,
                min(args.workers, 4), args.outdir, smoke=True)
        return

    fns = args.fns or OFFICIAL_FNS
    algos = args.algos or ALGOS
    for fid in fns:
        if fid not in OFFICIAL_FNS:
            raise SystemExit('invalid official function number: %d' % fid)
    for dim in args.dims:
        run_dim(dim, fns, algos, N_RUNS, 10_000 * dim,
                args.workers, args.outdir)
    print('ALL DIMENSIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
