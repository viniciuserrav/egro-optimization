"""
egro_pso_vs_cmaes.py  —  EGRO-PSO vs EGRO-CMA on CEC2017.

WHAT IS IDENTICAL (fair comparison):
  - Ellipse construction (skier, sigma_long, sigma_short, mu_k, d_hat)
  - Classification: rho_quad with alpha_in=0.760
  - FE budget per group before classification: T_DIAG * N_PER_GROUP = 900 FEs
  - Final gradient refinement for case 1 (if PSO/CMA improved on x*)
  - Spawn mechanism, restart, max_spawn, n_skiers
  - Initialization: both start from mu_k, both see x* as initial best

WHAT DIFFERS:
  - EGRO-PSO: Clerc-Kennedy PSO (w=0.729, c1=c2=1.494), N_PER_GROUP=15 agents
  - EGRO-CMA: CMA-ES initialized at mu_k with ellipse-shaped covariance,
              population size = N_PER_GROUP, x* seeded as initial best,
              echo direction (d_hat) encoded in initial covariance matrix

Usage:
    python egro_pso_vs_cmaes.py --dim 10
    python egro_pso_vs_cmaes.py --dim 30 --n-runs 30 --max-fes 10000
"""

import argparse, json, os, warnings
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ── Best params from sweep ────────────────────────────────────────────────────
ALPHA_IN   = 0.760
SIGMA_DIST = 1.5
W_EXP, C1_EXP, C2_EXP = 0.729, 1.494, 1.494   # Clerc-Kennedy
W_E, C1_E, C2_E        = 0.4, 2.0, 1.0          # exploitation (PSO only)

# ── Structural constants ──────────────────────────────────────────────────────
N_SKIERS        = 5
N_PER_GROUP     = 15
GRADIENT_MAXFUN = 500
T_DIAG          = 60
# FEs consumed by both PSO and CMA-ES before classification check:
FES_BEFORE_DIAG = T_DIAG * N_PER_GROUP   # = 900
MAX_INNER_ITER  = 300
MAX_SPAWN       = 20
BETA_R, ALPHA_R = 0.5, 0.30
SIGMA_MIN_FRAC  = 0.02
BETA_MPD        = 0.02

ALL_FNS = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]


def load_bm(fid, dim):
    from opfunu.cec_based import cec2017 as cec17
    inst = getattr(cec17, 'F%d2017' % fid)(ndim=dim)
    return {'func': inst.evaluate, 'f_global': inst.f_global,
            'lb': float(inst.lb[0]), 'ub': float(inst.ub[0])}


def compute_mpd(pos):
    n = len(pos)
    if n < 2: return 0.0
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    return float(d[np.triu_indices(n, k=1)].mean())


def ellipse_params(start, xstar, lb, ub):
    dim = len(xstar)
    sr  = ALPHA_R * (ub - lb); sm = SIGMA_MIN_FRAC * (ub - lb)
    t   = float(np.linalg.norm(xstar - start))
    sl  = max(sr / (1.0 + t / sr), sm)
    ss  = BETA_R * sl
    cf  = float(np.sqrt(max(sl**2 - ss**2, 0.0)))
    dh  = (xstar - start) / (t + 1e-30) if t > 1e-10 else np.eye(dim)[0]
    mu  = xstar + cf * dh
    return sl, ss, cf, dh, mu


def rho_quad(gb, mu, dh, sl, ss):
    delta = gb - mu
    dp    = float(np.dot(delta, dh))
    dperp2 = max(float(np.dot(delta, delta)) - dp**2, 0.0)
    return (dp / sl)**2 + dperp2 / ss**2


class _Budget(Exception):
    """Raised when the FE budget is exhausted, to stop any inner loop exactly."""
    pass


def echo_sqrt_cov(sl, ss, dh, sig0, d):
    """
    Matrix square root M of the echo-shaped initial covariance Sigma0 = sig0^2 C0,
    C0 = (sl^2 dd' + ss^2 (I - dd')) / lambda_bar, lambda_bar = (sl^2 + ss^2 (d-1))/d.
    Sampling x = mu + M y, y ~ N(0,I), gives x ~ N(mu, sig0^2 C0).
    Identical to the implementation in egro_cma_competition.py.
    """
    lam = (sl**2 + ss**2 * (d - 1)) / d
    s_par  = sig0 * sl / np.sqrt(lam)
    s_perp = sig0 * ss / np.sqrt(lam)
    P = np.outer(dh, dh)
    return s_par * P + s_perp * (np.eye(d) - P)


# ── Shared base ───────────────────────────────────────────────────────────────
class _EGROBase:
    def __init__(self, func, lb, ub, dim, max_evals, seed=None):
        self.func = func; self.lb = lb; self.ub = ub; self.dim = dim
        self.max_evals = max_evals
        self.sigma_min = SIGMA_MIN_FRAC * (ub - lb)
        self.rng = np.random.default_rng(seed)
        self.eval_count = 0

    def _f(self, x):
        if self.eval_count >= self.max_evals:
            raise _Budget()
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def _skier(self, start):
        remaining = self.max_evals - self.eval_count
        if remaining <= 0:
            return np.clip(start, self.lb, self.ub), float('inf')
        maxfun = min(GRADIENT_MAXFUN, remaining)
        try:
            r = minimize(self._f, np.clip(start, self.lb, self.ub),
                         method='L-BFGS-B',
                         bounds=[(self.lb, self.ub)] * self.dim,
                         options={'maxfun': maxfun, 'maxiter': maxfun})
            return np.clip(r.x, self.lb, self.ub), float(r.fun)
        except _Budget:
            return np.clip(start, self.lb, self.ub), float('inf')

    def _run_group(self, start, xs, fs):
        """Run search inside ellipse. Returns (gb_pos, gb_fit, rho, ellipse_params, case1)."""
        raise NotImplementedError

    def optimize(self):
        queue = list(self.rng.uniform(self.lb, self.ub, (N_SKIERS, self.dim)))
        gbf = np.inf; gbp = queue[0].copy(); n_spawn = 0

        while self.eval_count < self.max_evals:
            if not queue:
                queue.append(self.rng.uniform(self.lb, self.ub, self.dim))

            seed = queue.pop(0)
            xs, fs = self._skier(seed)
            if fs < gbf: gbf, gbp = fs, xs.copy()
            if self.eval_count >= self.max_evals: break

            gb_pos, gb_fit, rho, ell, case1 = self._run_group(seed, xs, fs)
            sl, ss, dh, mu = ell

            if gb_fit < gbf: gbf, gbp = gb_fit, gb_pos.copy()

            if case1:
                # Case 1: inside ellipse — final gradient refinement
                if gb_fit < fs and self.eval_count < self.max_evals:
                    rp, rf = self._skier(gb_pos)
                    if rf < gbf: gbf, gbp = rf, rp.copy()
            else:
                # Case 2: outside ellipse — spawn
                if n_spawn < MAX_SPAWN:
                    queue.append(gb_pos.copy()); n_spawn += 1

        return gbp, gbf


# ── EGRO-PSO ──────────────────────────────────────────────────────────────────
class EGROWithPSO(_EGROBase):
    """EGRO with Clerc-Kennedy PSO inside the ellipse."""

    def _run_group(self, start, xs, fs):
        d = self.dim
        sl, ss, _, dh, mu = ellipse_params(start, xs, self.lb, self.ub)
        sig = SIGMA_DIST * sl / np.sqrt(d)
        gbp, gbf = xs.copy(), fs            # D2 anchor before any evaluation
        case1 = False
        try:
            pos = np.clip(mu + self.rng.normal(0, sig, (N_PER_GROUP, d)),
                          self.lb, self.ub)
            vb  = 2.0 * sl / np.sqrt(d)
            vel = self.rng.uniform(-vb, vb, (N_PER_GROUP, d))
            fit = np.array([self._f(p) for p in pos])
            pb_pos = pos.copy(); pb_fit = fit.copy()
            bi = int(np.argmin(pb_fit))
            if pb_fit[bi] < gbf:
                gbp, gbf = pb_pos[bi].copy(), float(pb_fit[bi])
            mpd0 = compute_mpd(pos)
            classified = False; exploit = False

            for t in range(1, MAX_INNER_ITER + 1):
                if self.eval_count >= self.max_evals: break
                w  = W_E   if exploit else W_EXP
                c1 = C1_E  if exploit else C1_EXP
                c2 = C2_E  if exploit else C2_EXP
                r1 = self.rng.random((N_PER_GROUP, d))
                r2 = self.rng.random((N_PER_GROUP, d))
                vel = (w * vel + c1 * r1 * (pb_pos - pos) + c2 * r2 * (gbp - pos))
                pos = np.clip(pos + vel, self.lb, self.ub)
                fit = np.array([self._f(p) for p in pos])
                imp = fit < pb_fit
                pb_pos[imp] = pos[imp]; pb_fit[imp] = fit[imp]
                bi  = int(np.argmin(pb_fit))
                if pb_fit[bi] < gbf: gbp = pb_pos[bi].copy(); gbf = float(pb_fit[bi])
                if mpd0 > 0 and compute_mpd(pos) < BETA_MPD * mpd0: break
                if t == T_DIAG and not classified:
                    classified = True
                    rho = rho_quad(gbp, mu, dh, sl, ss)
                    if rho >= ALPHA_IN:
                        return gbp, gbf, rho, (sl, ss, dh, mu), False   # case 2
                    else:
                        exploit = True; case1 = True
        except _Budget:
            pass

        rho = rho_quad(gbp, mu, dh, sl, ss)
        return gbp, gbf, rho, (sl, ss, dh, mu), case1


# ── EGRO-CMA ──────────────────────────────────────────────────────────────────
class EGROWithCMA(_EGROBase):
    """
    EGRO with CMA-ES inside the ellipse.

    Echo direction encoded in initial covariance:
      - High variance along d_hat (sigma_long^2)
      - Low variance perpendicular (sigma_short^2)
    x* seeded as initial best candidate.
    Same FE budget before classification as PSO: FES_BEFORE_DIAG = T_DIAG * N_PER_GROUP.
    """

    def _cma_phase(self, mu, M, sig_cma, max_fes, gbp, gbf):
        """
        Run CMA-ES in transformed coordinate y, evaluating f(mu + M y).
        M = echo square-root -> echo-shaped initial covariance; M=None -> isotropic.
        Returns (gb_pos_x, gb_fit, es_sigma).
        """
        import cma as cmalib
        d = self.dim
        pop_min = max(5, N_PER_GROUP)
        opts = cmalib.CMAOptions()
        opts['maxfevals'] = max_fes
        opts['popsize']   = N_PER_GROUP
        opts['verbose']   = -9
        opts['seed']      = int(self.rng.integers(1, 100000))
        # No pycma bounds: search is in y-space; bounds enforced by clipping x in _f.
        if M is None:
            es = cmalib.CMAEvolutionStrategy(np.clip(mu, self.lb, self.ub).tolist(),
                                             sig_cma, opts)
            to_x = lambda y: np.clip(np.asarray(y), self.lb, self.ub)
        else:
            es = cmalib.CMAEvolutionStrategy([0.0]*d, 1.0, opts)   # unit step in y
            to_x = lambda y: np.clip(mu + M @ np.asarray(y), self.lb, self.ub)
        try:
            while not es.stop() and self.eval_count < self.max_evals:
                if self.max_evals - self.eval_count < pop_min:
                    break
                Y  = es.ask()
                Xc = [to_x(y) for y in Y]
                fv = [self._f(x) for x in Xc]
                es.tell(Y, fv)
                bi = int(np.argmin(fv))
                if fv[bi] < gbf:
                    gbp = Xc[bi].copy(); gbf = float(fv[bi])
        except _Budget:
            pass
        return gbp, gbf, es.sigma

    def _run_group(self, start, xs, fs):
        d  = self.dim
        sl, ss, _, dh, mu = ellipse_params(start, xs, self.lb, self.ub)
        sig0 = SIGMA_DIST * sl / np.sqrt(d)

        # Phase A: explore with the ECHO-SHAPED initial covariance (paper's C0).
        M = echo_sqrt_cov(sl, ss, dh, sig0, d)
        gbp, gbf, _ = self._cma_phase(mu, M, 1.0, FES_BEFORE_DIAG, xs.copy(), fs)

        rho = rho_quad(gbp, mu, dh, sl, ss)
        if rho >= ALPHA_IN:
            return gbp, gbf, rho, (sl, ss, dh, mu), False   # case 2: spawn

        # Phase B: local exploitation (tightened isotropic step from the best).
        exploit_fes = (MAX_INNER_ITER - T_DIAG) * N_PER_GROUP
        gbp, gbf, _ = self._cma_phase(gbp, None, sig0 * 0.5, exploit_fes, gbp, gbf)
        rho = rho_quad(gbp, mu, dh, sl, ss)
        return gbp, gbf, rho, (sl, ss, dh, mu), True


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',     type=int, default=10)
    parser.add_argument('--fns',     type=str, default='')
    parser.add_argument('--n-runs',  type=int, default=30)
    parser.add_argument('--max-fes', type=int, default=10_000)
    parser.add_argument('--out',     type=str, default='')
    args = parser.parse_args()

    dim    = args.dim
    n_runs = args.n_runs
    budget = args.max_fes * dim
    fn_ids = ([int(x) for x in args.fns.split(',') if x.strip()]
              if args.fns else ALL_FNS)
    out    = args.out or 'egro_pso_vs_cmaes_d%d.json' % dim

    print('EGRO-PSO vs EGRO-CMA  |  d=%d  budget=%d  n_runs=%d' % (dim, budget, n_runs))
    print('Fairness: same budget, same ellipse, same classification (rho_quad a=%.3f)' % ALPHA_IN)
    print('FEs before classification: %d (T_DIAG=%d x N=%d)' % (FES_BEFORE_DIAG, T_DIAG, N_PER_GROUP))
    print()

    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print('Resuming: %d entries done' % len(results))

    print('%-4s  %-22s  %-22s  %s' % ('Fn', 'EGRO-PSO  mean+-std', 'EGRO-CMA  mean+-std', 'winner'))
    print('-' * 72)

    for fid in fn_ids:
        bm = load_bm(fid, dim)
        kp = 'F%d_d%d_PSO' % (fid, dim)
        kc = 'F%d_d%d_CMA' % (fid, dim)

        # Skip benchmarks that return non-finite values (e.g. F17/F29 at d=10).
        probe = [bm['func'](np.full(dim, c)) for c in (-50.0, 0.0, 50.0)]
        if not (np.all(np.isfinite(probe)) and np.isfinite(bm['f_global'])):
            print('F%-3d  INVALID (non-finite benchmark; excluded)' % fid)
            results['F%d_d%d_INVALID' % (fid, dim)] = True
            with open(out, 'w') as fh: json.dump(results, fh, indent=2)
            continue

        for key, Cls in ((kp, EGROWithPSO), (kc, EGROWithCMA)):
            if key in results: continue
            errs = []; evals = []
            for run in range(n_runs):
                alg = Cls(bm['func'], bm['lb'], bm['ub'], dim, budget, seed=run)
                _, f = alg.optimize()
                e = f - bm['f_global']
                errs.append(max(float(e), 0.0) if np.isfinite(e) else float('inf'))
                evals.append(int(alg.eval_count))
            finite = [e for e in errs if np.isfinite(e)]
            results[key] = {'mean': float(np.mean(finite)) if finite else float('inf'),
                            'std':  float(np.std(finite)) if finite else 0.0,
                            'errors': errs, 'evals': evals,
                            'seeds': list(range(n_runs)), 'budget': budget}
            with open(out, 'w') as fh: json.dump(results, fh, indent=2)

        mp = results[kp]['mean']; sp = results[kp]['std']
        mc = results[kc]['mean']; sc = results[kc]['std']
        win = 'CMA' if mc < mp else ('PSO' if mp < mc else 'tie')
        print('F%-3d  %.3e +- %.2e  %.3e +- %.2e  %s' % (fid, mp, sp, mc, sc, win))

    # Summary
    pso_wins = cma_wins = ties = 0
    for fid in fn_ids:
        kp = 'F%d_d%d_PSO' % (fid, dim); kc = 'F%d_d%d_CMA' % (fid, dim)
        if kp not in results or kc not in results: continue
        mp = results[kp]['mean']; mc = results[kc]['mean']
        if mc < mp:   cma_wins += 1
        elif mp < mc: pso_wins += 1
        else:         ties += 1

    print()
    print('=== SUMMARY d=%d ===' % dim)
    print('PSO wins: %d  |  CMA wins: %d  |  ties: %d' % (pso_wins, cma_wins, ties))

    print('Saved to', out)


if __name__ == '__main__':
    main()
