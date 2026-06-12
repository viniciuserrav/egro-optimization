"""
egro_cma_competition.py  —  EGRO-CMA vs CMA-ES, L-SHADE, DE, PSO on CEC2017.

EGRO-CMA: echo-guided rescue groups with CMA-ES inner search.
  alpha_in=0.760, sigma_dist=1.5, Clerc-Kennedy exploration params,
  CMA-ES initialized at mu_k with ellipse-shaped covariance (echo prior),
  x* seeded as initial best (D2), final gradient refinement for case 1.

Usage:
    python egro_cma_competition.py --dim 10
    python egro_cma_competition.py --dim 30 --n-runs 30 --max-fes 10000
"""

import argparse, json, os, warnings
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ── EGRO-CMA params ───────────────────────────────────────────────────────────
ALPHA_IN   = 0.760
SIGMA_DIST = 1.5
W_E, C1_E, C2_E = 0.4, 2.0, 1.0

# ── Structural constants ──────────────────────────────────────────────────────
N_SKIERS        = 5
N_PER_GROUP     = 15
GRADIENT_MAXFUN = 500
T_DIAG          = 60
FES_BEFORE_DIAG = T_DIAG * N_PER_GROUP   # 900 FEs before classification
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


def ellipse_params(start, xstar, lb, ub):
    dim = len(xstar)
    sr  = ALPHA_R*(ub-lb); sm = SIGMA_MIN_FRAC*(ub-lb)
    t   = float(np.linalg.norm(xstar-start))
    sl  = max(sr/(1.0+t/sr), sm); ss = BETA_R*sl
    cf  = float(np.sqrt(max(sl**2-ss**2, 0.0)))
    dh  = (xstar-start)/(t+1e-30) if t>1e-10 else np.eye(dim)[0]
    mu  = xstar + cf*dh
    return sl, ss, cf, dh, mu


def rho_quad(gb, mu, dh, sl, ss):
    delta = gb-mu; dp = float(np.dot(delta, dh))
    dp2   = max(float(np.dot(delta, delta))-dp**2, 0.0)
    return (dp/sl)**2 + dp2/ss**2


# ── EGRO-CMA ──────────────────────────────────────────────────────────────────
class _Budget(Exception):
    """Raised when the FE budget is exhausted, to stop any inner loop exactly."""
    pass


def echo_sqrt_cov(sl, ss, dh, sig0, d):
    """
    Matrix square root M of the echo-shaped initial covariance Sigma0 = sig0^2 C0,
    with  C0 = (sl^2 dd' + ss^2 (I - dd')) / lambda_bar,  lambda_bar = (sl^2 + ss^2 (d-1))/d.
    Sampling x = mu + M y with y ~ N(0, I) gives x ~ N(mu, sig0^2 C0), i.e. the search
    distribution is elongated along the echo direction d_hat by sl and contracted to ss
    perpendicular.  M is itself the projector combination since dd' and (I-dd') are
    orthogonal idempotents:  sqrt(a dd' + b (I-dd')) = sqrt(a) dd' + sqrt(b) (I-dd').
    """
    lam = (sl**2 + ss**2 * (d - 1)) / d
    s_par  = sig0 * sl / np.sqrt(lam)     # std along d_hat
    s_perp = sig0 * ss / np.sqrt(lam)     # std perpendicular
    P = np.outer(dh, dh)
    return s_par * P + s_perp * (np.eye(d) - P)


class EGROWithCMA:
    # use_echo=True: Phase-A uses the echo-shaped covariance C0 (default, the
    # full EGRO-CMA).  use_echo=False: Phase-A is isotropic (same step sigma0)
    # with EVERYTHING ELSE in the framework identical -- this is the clean
    # echo-prior ablation that isolates C0 from the rest of the framework.
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, use_echo=True):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals
        self.use_echo=use_echo
        self.sigma_min=SIGMA_MIN_FRAC*(ub-lb)
        self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        if self.eval_count >= self.max_evals:
            raise _Budget()
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def _skier(self, start):
        """Gradient descent, capped so it cannot exceed the remaining FE budget."""
        remaining = self.max_evals - self.eval_count
        if remaining <= 0:
            return np.clip(start, self.lb, self.ub), float('inf')
        maxfun = min(GRADIENT_MAXFUN, remaining)
        try:
            r = minimize(self._f, np.clip(start, self.lb, self.ub),
                         method='L-BFGS-B',
                         bounds=[(self.lb,self.ub)]*self.dim,
                         options={'maxfun':maxfun,'maxiter':maxfun})
            return np.clip(r.x, self.lb, self.ub), float(r.fun)
        except _Budget:
            xc = np.clip(start, self.lb, self.ub)
            return xc, float('inf')

    def _run_cmaes(self, mu, M, sig_cma, max_fes, gbp_init, gbf_init):
        """
        Run CMA-ES in a transformed coordinate y, evaluating f(mu + M y).
        If M is the echo square-root (echo_sqrt_cov), the initial search
        distribution in x is the echo-shaped covariance; if M is None it is
        isotropic with step sig_cma (used for local exploitation).
        Returns (gb_pos_x, gb_fit, es_sigma).
        """
        import cma as cmalib
        d = self.dim
        gbp = gbp_init.copy(); gbf = gbf_init
        pop_min = max(5, N_PER_GROUP)

        opts = cmalib.CMAOptions()
        opts['maxfevals'] = max_fes
        opts['popsize']   = N_PER_GROUP
        opts['verbose']   = -9
        opts['seed']      = int(self.rng.integers(1, 100000))
        # No pycma bounds: search is in y-space; bounds are enforced by clipping
        # x = mu + M y inside _f.  This is standard clip bound-handling.

        if M is None:
            x0 = np.clip(mu, self.lb, self.ub)
            es = cmalib.CMAEvolutionStrategy(x0.tolist(), sig_cma, opts)
            to_x = lambda y: np.clip(np.asarray(y), self.lb, self.ub)
        else:
            es = cmalib.CMAEvolutionStrategy([0.0]*d, 1.0, opts)   # y-space, unit step
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

        # Phase A: echo-shaped covariance C0 (use_echo) or isotropic ablation.
        if self.use_echo:
            M = echo_sqrt_cov(sl, ss, dh, sig0, d)
            gbp, gbf, sigma_after = self._run_cmaes(mu, M, 1.0, FES_BEFORE_DIAG, xs, fs)
        else:
            gbp, gbf, sigma_after = self._run_cmaes(mu, None, sig0, FES_BEFORE_DIAG, xs, fs)

        # Classification on the actual x-space best position.
        rho = rho_quad(gbp, mu, dh, sl, ss)
        if rho >= ALPHA_IN:
            return gbp, gbf, (sl, ss, dh, mu), False   # case 2: border -> spawn

        # Phase B: local exploitation (tightened isotropic step from the best).
        exploit_fes = (MAX_INNER_ITER - T_DIAG) * N_PER_GROUP
        gbp, gbf, _ = self._run_cmaes(gbp, None, sig0 * 0.5, exploit_fes, gbp, gbf)
        return gbp, gbf, (sl, ss, dh, mu), True        # case 1: inside

    def optimize(self):
        queue = list(self.rng.uniform(self.lb, self.ub, (N_SKIERS, self.dim)))
        gbf=np.inf; gbp=queue[0].copy(); n_spawn=0

        while self.eval_count < self.max_evals:
            if not queue:
                queue.append(self.rng.uniform(self.lb, self.ub, self.dim))
            seed = queue.pop(0)
            xs, fs = self._skier(seed)
            if fs < gbf: gbf, gbp = fs, xs.copy()
            if self.eval_count >= self.max_evals: break

            gb_pos, gb_fit, ell, case1 = self._run_group(seed, xs, fs)
            if gb_fit < gbf: gbf, gbp = gb_fit, gb_pos.copy()

            if case1:
                if gb_fit < fs and self.eval_count < self.max_evals:
                    rp, rf = self._skier(gb_pos)
                    if rf < gbf: gbf, gbp = rf, rp.copy()
            else:
                if n_spawn < MAX_SPAWN:
                    queue.append(gb_pos.copy()); n_spawn += 1

        return gbp, gbf


# ── Benchmark algorithms ──────────────────────────────────────────────────────
class CMAES:
    def __init__(self, func, lb, ub, dim, max_evals, seed=None):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals; self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        import cma
        x0  = self.rng.uniform(self.lb, self.ub, self.dim)
        sig = 0.3*(self.ub-self.lb)
        opts = cma.CMAOptions()
        opts['bounds']   = [[self.lb]*self.dim, [self.ub]*self.dim]
        opts['maxfevals']= self.max_evals
        opts['verbose']  = -9
        opts['seed']     = int(self.rng.integers(1, 100000))
        es = cma.CMAEvolutionStrategy(x0.tolist(), sig, opts)
        gbf=np.inf; gbp=x0.copy()
        while not es.stop() and self.eval_count < self.max_evals:
            X = es.ask()
            # Stop before a partial generation: tell() requires the full
            # population (>= mu solutions). A <popsize overshoot is acceptable.
            if self.max_evals - self.eval_count < len(X):
                break
            Xc = [np.clip(x, self.lb, self.ub) for x in X]
            fv = [self._f(x) for x in Xc]
            es.tell(X, fv)
            bi = int(np.argmin(fv))
            if fv[bi] < gbf: gbf=float(fv[bi]); gbp=Xc[bi].copy()
        return gbp, gbf


class LSHADE:
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, H=6, N_min=4):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals
        self.N_init=min(18*dim, 360); self.N_min=N_min; self.H=H
        self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        d=self.dim; lb=self.lb; ub=self.ub; rng=self.rng
        N=self.N_init; H=self.H
        M_F=np.full(H,0.5); M_CR=np.full(H,0.5); k=0
        pop=rng.uniform(lb,ub,(N,d)); fit=np.array([self._f(x) for x in pop])
        archive=[]; gbf=fit.min(); gbp=pop[np.argmin(fit)].copy()
        while self.eval_count < self.max_evals:
            N_new=max(self.N_min, round((self.N_min-self.N_init)/self.max_evals*self.eval_count+self.N_init))
            if N_new < N:
                idx=np.argsort(fit)[:N_new]; pop,fit,N=pop[idx],fit[idx],N_new
            S_F=[]; S_CR=[]; dF=[]; new_pop=pop.copy(); new_fit=fit.copy()
            for i in range(N):
                if self.eval_count >= self.max_evals: break
                ri=rng.integers(0,H)
                F=-1
                while F<=0 or F>1: F=float(rng.standard_cauchy()*0.1+M_F[ri]); F=min(F,1.0)
                CR=float(np.clip(rng.normal(M_CR[ri],0.1),0,1))
                p_size=max(2,int(0.11*N)); pbest_idx=np.argsort(fit)[:p_size]
                p=pop[rng.choice(pbest_idx)].copy()
                cands=[j for j in range(N) if j!=i]; r1=rng.choice(cands)
                union=np.vstack([pop,archive]) if archive else pop
                r2=rng.integers(0,len(union))
                v=np.clip(pop[i]+F*(p-pop[i])+F*(pop[r1]-union[r2]),lb,ub)
                j_rand=rng.integers(0,d); mask=rng.random(d)<CR; mask[j_rand]=True
                u=np.where(mask,v,pop[i]); fu=self._f(u)
                if fu<=fit[i]:
                    new_pop[i]=u; new_fit[i]=fu; archive.append(pop[i].copy())
                    S_F.append(F); S_CR.append(CR); dF.append(max(fit[i]-fu,0.0))
                    if fu<gbf: gbf=fu; gbp=u.copy()
            pop,fit=new_pop,new_fit
            if len(archive)>N: rng.shuffle(archive); archive=archive[:N]
            if S_F:
                w=np.array(dF); w=w/w.sum() if w.sum()>0 else np.ones(len(w))/len(w)
                sf=np.array(S_F); scr=np.array(S_CR)
                M_F[k]=float((w*sf**2).sum()/max((w*sf).sum(),1e-30))
                M_CR[k]=float((w*scr).sum()); k=(k+1)%H
        return gbp, gbf


class DE:
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, F=0.8, CR=0.9):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals; self.F=F; self.CR=CR
        self.N=min(10*dim,200); self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        d=self.dim; lb=self.lb; ub=self.ub; N=self.N; rng=self.rng
        pop=rng.uniform(lb,ub,(N,d)); fit=np.array([self._f(x) for x in pop])
        gbf=fit.min(); gbp=pop[np.argmin(fit)].copy()
        while self.eval_count < self.max_evals:
            for i in range(N):
                if self.eval_count >= self.max_evals: break
                idxs=[j for j in range(N) if j!=i]
                a,b,c=pop[rng.choice(idxs,3,replace=False)]
                v=np.clip(a+self.F*(b-c),lb,ub)
                j_rand=rng.integers(0,d); mask=rng.random(d)<self.CR; mask[j_rand]=True
                u=np.where(mask,v,pop[i]); fu=self._f(u)
                if fu<=fit[i]:
                    pop[i]=u; fit[i]=fu
                    if fu<gbf: gbf=fu; gbp=u.copy()
        return gbp, gbf


class PSO:
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, N=40):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals; self.N=N
        self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        d=self.dim; lb=self.lb; ub=self.ub; N=self.N; rng=self.rng
        pos=rng.uniform(lb,ub,(N,d)); vb=0.1*(ub-lb)
        vel=rng.uniform(-vb,vb,(N,d)); fit=np.array([self._f(x) for x in pos])
        pb_pos=pos.copy(); pb_fit=fit.copy()
        bi=np.argmin(pb_fit); gbp=pb_pos[bi].copy(); gbf=float(pb_fit[bi])
        while self.eval_count < self.max_evals:
            r1=rng.random((N,d)); r2=rng.random((N,d))
            vel=(0.729*vel+1.494*r1*(pb_pos-pos)+1.494*r2*(gbp-pos))
            pos=np.clip(pos+vel,lb,ub); fit=np.array([self._f(x) for x in pos])
            imp=fit<pb_fit; pb_pos[imp]=pos[imp]; pb_fit[imp]=fit[imp]
            bi=np.argmin(pb_fit)
            if pb_fit[bi]<gbf: gbf=float(pb_fit[bi]); gbp=pb_pos[bi].copy()
        return gbp, gbf


class GWO:
    """Grey Wolf Optimizer."""
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, N=30):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals; self.N=N
        self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        d=self.dim; lb=self.lb; ub=self.ub; N=self.N; rng=self.rng
        pos=rng.uniform(lb,ub,(N,d)); fit=np.array([self._f(x) for x in pos])
        idx=np.argsort(fit)
        alpha,beta,delta=pos[idx[0]].copy(),pos[idx[1]].copy(),pos[idx[2]].copy()
        fa,fb,fd=fit[idx[0]],fit[idx[1]],fit[idx[2]]
        t=0; max_iter=int((self.max_evals-N)/N)+1
        while self.eval_count < self.max_evals:
            a=2.0-2.0*t/max_iter
            for i in range(N):
                if self.eval_count >= self.max_evals: break
                r1,r2=rng.random(d),rng.random(d)
                A1=2*a*r1-a; C1=2*r2
                X1=alpha-A1*np.abs(C1*alpha-pos[i])
                r1,r2=rng.random(d),rng.random(d)
                A2=2*a*r1-a; C2=2*r2
                X2=beta-A2*np.abs(C2*beta-pos[i])
                r1,r2=rng.random(d),rng.random(d)
                A3=2*a*r1-a; C3=2*r2
                X3=delta-A3*np.abs(C3*delta-pos[i])
                pos[i]=np.clip((X1+X2+X3)/3,lb,ub)
                fit[i]=self._f(pos[i])
                if fit[i]<fa: delta,fd=beta.copy(),fb; beta,fb=alpha.copy(),fa; alpha,fa=pos[i].copy(),fit[i]
                elif fit[i]<fb: delta,fd=beta.copy(),fb; beta,fb=pos[i].copy(),fit[i]
                elif fit[i]<fd: delta,fd=pos[i].copy(),fit[i]
            t+=1
        return alpha, fa


class WOA:
    """Whale Optimization Algorithm."""
    def __init__(self, func, lb, ub, dim, max_evals, seed=None, N=30):
        self.func=func; self.lb=lb; self.ub=ub; self.dim=dim
        self.max_evals=max_evals; self.N=N
        self.rng=np.random.default_rng(seed); self.eval_count=0

    def _f(self, x):
        self.eval_count += 1
        return float(self.func(np.clip(x, self.lb, self.ub)))

    def optimize(self):
        d=self.dim; lb=self.lb; ub=self.ub; N=self.N; rng=self.rng
        pos=rng.uniform(lb,ub,(N,d)); fit=np.array([self._f(x) for x in pos])
        bi=np.argmin(fit); gbp=pos[bi].copy(); gbf=float(fit[bi])
        t=0; max_iter=int((self.max_evals-N)/N)+1
        while self.eval_count < self.max_evals:
            a=2.0-2.0*t/max_iter; a2=-1.0-t/max_iter
            for i in range(N):
                if self.eval_count >= self.max_evals: break
                r=rng.random(); A=2*a*rng.random(d)-a; C=2*rng.random(d)
                b=1.0; l=(a2-1)*rng.random()+1; p=rng.random()
                if p < 0.5:
                    if abs(A).mean() < 1:
                        D=np.abs(C*gbp-pos[i]); pos[i]=np.clip(gbp-A*D,lb,ub)
                    else:
                        rp=pos[rng.integers(0,N)]; D=np.abs(C*rp-pos[i]); pos[i]=np.clip(rp-A*D,lb,ub)
                else:
                    D=np.abs(gbp-pos[i]); pos[i]=np.clip(D*np.exp(b*l)*np.cos(2*np.pi*l)+gbp,lb,ub)
                fit[i]=self._f(pos[i])
                if fit[i]<gbf: gbf=float(fit[i]); gbp=pos[i].copy()
            t+=1
        return gbp, gbf


# ── Runner ────────────────────────────────────────────────────────────────────
ALGORITHMS = {
    'EGRO-CMA': EGROWithCMA,
    'CMA-ES':   CMAES,
    'L-SHADE':  LSHADE,
    'DE':       DE,
    'PSO':      PSO,
    'GWO':      GWO,
    'WOA':      WOA,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',     type=int, default=10)
    parser.add_argument('--fns',     type=str, default='')
    parser.add_argument('--n-runs',  type=int, default=30)
    parser.add_argument('--max-fes', type=int, default=10_000)
    parser.add_argument('--out',     type=str, default='')
    args = parser.parse_args()

    dim=args.dim; n_runs=args.n_runs; budget=args.max_fes*dim
    fn_ids=([int(x) for x in args.fns.split(',') if x.strip()]
            if args.fns else ALL_FNS)
    out=args.out or 'egro_cma_competition_d%d.json' % dim

    print('EGRO-CMA competition  d=%d  budget=%d  n_runs=%d' % (dim, budget, n_runs))
    print('Algorithms:', list(ALGORITHMS))

    results={}
    if os.path.exists(out):
        results=json.load(open(out))
        print('Resuming: %d entries done' % len(results))

    alg_names=list(ALGORITHMS)
    print('\n%-4s ' % 'Fn' + '  '.join(['%-22s'%a for a in alg_names]))
    print('-'*(6+24*len(alg_names)))

    for fid in fn_ids:
        bm=load_bm(fid, dim); row={}

        # Validity probe: some opfunu CEC2017 instances return non-finite values
        # (e.g. F17, F29 at d=10).  Such functions are recorded as invalid and
        # excluded from the analysis rather than silently producing nan/inf.
        probe = [bm['func'](np.full(dim, c)) for c in (-50.0, 0.0, 50.0)]
        if not (np.all(np.isfinite(probe)) and np.isfinite(bm['f_global'])):
            print('F%-3d  INVALID (non-finite benchmark; excluded)' % fid)
            results['F%d_d%d_INVALID' % (fid, dim)] = True
            with open(out,'w') as fh: json.dump(results, fh, indent=2)
            continue

        for name, Cls in ALGORITHMS.items():
            key='F%d_d%d_%s' % (fid, dim, name.replace('-','_'))
            if key not in results:
                errs=[]; evals=[]
                for run in range(n_runs):
                    alg=Cls(bm['func'], bm['lb'], bm['ub'], dim, budget, seed=run)
                    _, f=alg.optimize()
                    e = f - bm['f_global']
                    errs.append(max(float(e), 0.0) if np.isfinite(e) else float('inf'))
                    evals.append(int(getattr(alg, 'eval_count', 0)))
                finite = [e for e in errs if np.isfinite(e)]
                results[key]={'mean':float(np.mean(finite)) if finite else float('inf'),
                              'std': float(np.std(finite)) if finite else 0.0,
                              'errors': errs,
                              'evals': evals,
                              'seeds': list(range(n_runs)),
                              'budget': budget}
                with open(out,'w') as fh: json.dump(results, fh, indent=2)
            row[name]=results[key]['mean']

        if row:
            finite_vals=[v for v in row.values() if np.isfinite(v)]
            best=min(finite_vals) if finite_vals else float('inf')
            cells=['%s%.3e'%('*' if v==best else ' ', v) for v in row.values()]
            print('F%-3d ' % fid + '  '.join(['%-22s'%c for c in cells]))

    # Summary
    wins={a:0 for a in alg_names}
    for fid in fn_ids:
        row={a:results['F%d_d%d_%s'%(fid,dim,a.replace('-','_'))]['mean']
             for a in alg_names
             if 'F%d_d%d_%s'%(fid,dim,a.replace('-','_')) in results}
        if row:
            winner=min(row, key=row.get); wins[winner]+=1

    print('\n=== WINS d=%d ===' % dim)
    for a,w in sorted(wins.items(), key=lambda x:-x[1]):
        print('  %-12s %d/%d' % (a, w, len(fn_ids)))

    print('Saved to', out)

if __name__ == '__main__':
    main()
