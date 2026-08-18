"""
engineering_problems.py — four classical constrained mechanical-design problems
in their STANDARD (Coello / Golinski) formulations, with the widely-cited
best-known optima.  Each problem exposes:

    name, n, lb (n-vector), ub (n-vector),
    f(x)      -> objective (physical variables)
    g(x)      -> list of inequality constraints  g_i(x) <= 0  feasible
    x_opt, f_opt, ref   (canonical optimum + source)

`verify_all()` asserts that each published optimum is FEASIBLE at its published
objective value under these formulations — the check the earlier hand-coded
version failed.  If this passes, the formulations match the literature and the
optima are the community's numbers, not ours to defend.

Discrete variables (pressure-vessel plate thickness on a 0.0625 grid; speed-
reducer integer teeth) are snapped inside f/g, exactly as in the source problems.
"""
import numpy as np

PI = np.pi


# ── 1. Tension/compression spring (Belegundu 1982; Arora; Coello 2000) ──────────
def spring_f(x):
    x1, x2, x3 = x
    return (x3 + 2.0) * x2 * x1**2

def spring_g(x):
    x1, x2, x3 = x
    g1 = 1.0 - (x2**3 * x3) / (71785.0 * x1**4)
    g2 = (4.0*x2**2 - x1*x2) / (12566.0*(x2*x1**3 - x1**4)) + 1.0/(5108.0*x1**2) - 1.0
    g3 = 1.0 - 140.45*x1 / (x2**2 * x3)
    g4 = (x1 + x2)/1.5 - 1.0
    return [g1, g2, g3, g4]

SPRING = dict(
    name='Tension/compression spring', n=3,
    lb=np.array([0.05, 0.25, 2.0]), ub=np.array([2.0, 1.30, 15.0]),
    f=spring_f, g=spring_g,
    x_opt=np.array([0.051689, 0.356718, 11.288966]), f_opt=0.012665,
    ref='Coello 2000')


# ── 2. Pressure vessel (Kannan-Kramer; Coello 2000) ─────────────────────────────
def _snap0625(v):        # plate thicknesses lie on a 0.0625 in grid
    return np.round(v / 0.0625) * 0.0625

def vessel_f(x):
    x1, x2, x3, x4 = x
    x1 = _snap0625(x1); x2 = _snap0625(x2)
    return (0.6224*x1*x3*x4 + 1.7781*x2*x3**2
            + 3.1661*x1**2*x4 + 19.84*x1**2*x3)

def vessel_g(x):
    x1, x2, x3, x4 = x
    x1 = _snap0625(x1); x2 = _snap0625(x2)
    g1 = -x1 + 0.0193*x3
    g2 = -x2 + 0.00954*x3
    g3 = -PI*x3**2*x4 - (4.0/3.0)*PI*x3**3 + 1296000.0
    g4 = x4 - 240.0
    return [g1, g2, g3, g4]

VESSEL = dict(
    name='Pressure vessel', n=4,
    lb=np.array([0.0625, 0.0625, 10.0, 10.0]),
    ub=np.array([6.1875, 6.1875, 200.0, 200.0]),
    f=vessel_f, g=vessel_g,
    x_opt=np.array([0.8125, 0.4375, 42.098446, 176.636596]), f_opt=6059.714335,
    ref='Coello 2000')


# ── 3. Welded beam (Coello 2000) ────────────────────────────────────────────────
_WB = dict(P=6000.0, L=14.0, E=30e6, G=12e6, tmax=13600.0, smax=30000.0, dmax=0.25)

def _welded_terms(x):
    x1, x2, x3, x4 = x
    P, L, E, G = _WB['P'], _WB['L'], _WB['E'], _WB['G']
    Rr  = np.sqrt(x2**2/4.0 + ((x1 + x3)/2.0)**2)
    M   = P*(L + x2/2.0)
    J   = 2.0*(np.sqrt(2.0)*x1*x2*(x2**2/12.0 + ((x1 + x3)/2.0)**2))
    tp  = P/(np.sqrt(2.0)*x1*x2)
    tpp = M*Rr/J
    tau = np.sqrt(tp**2 + 2.0*tp*tpp*(x2/(2.0*Rr)) + tpp**2)
    sig = 6.0*P*L/(x4*x3**2)
    dlt = 4.0*P*L**3/(E*x3**3*x4)
    Pc  = (4.013*E*np.sqrt(x3**2*x4**6/36.0)/L**2)*(1.0 - (x3/(2.0*L))*np.sqrt(E/(4.0*G)))
    return tau, sig, dlt, Pc

def welded_f(x):
    x1, x2, x3, x4 = x
    return 1.10471*x1**2*x2 + 0.04811*x3*x4*(14.0 + x2)

def welded_g(x):
    x1, x2, x3, x4 = x
    tau, sig, dlt, Pc = _welded_terms(x)
    return [tau - _WB['tmax'],
            sig - _WB['smax'],
            x1 - x4,
            0.10471*x1**2 + 0.04811*x3*x4*(14.0 + x2) - 5.0,
            0.125 - x1,
            dlt - _WB['dmax'],
            _WB['P'] - Pc]

WELDED = dict(
    name='Welded beam', n=4,
    lb=np.array([0.1, 0.1, 0.1, 0.1]), ub=np.array([2.0, 10.0, 10.0, 2.0]),
    f=welded_f, g=welded_g,
    x_opt=np.array([0.205730, 3.470489, 9.036624, 0.205730]), f_opt=1.724852,
    ref='Coello 2000')


# ── 4. Speed reducer (Golinski; Coello) ─────────────────────────────────────────
def speed_f(x):
    x1, x2, x4, x5, x6, x7 = x[0], x[1], x[3], x[4], x[5], x[6]
    x3 = np.round(x[2])   # integer teeth
    return (0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934)
            - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3 + x7**3)
            + 0.7854*(x4*x6**2 + x5*x7**2))

def speed_g(x):
    x1, x2, x4, x5, x6, x7 = x[0], x[1], x[3], x[4], x[5], x[6]
    x3 = np.round(x[2])
    return [27.0/(x1*x2**2*x3) - 1.0,
            397.5/(x1*x2**2*x3**2) - 1.0,
            1.93*x4**3/(x2*x3*x6**4) - 1.0,
            1.93*x5**3/(x2*x3*x7**4) - 1.0,
            np.sqrt((745.0*x4/(x2*x3))**2 + 16.9e6)/(110.0*x6**3) - 1.0,
            np.sqrt((745.0*x5/(x2*x3))**2 + 157.5e6)/(85.0*x7**3) - 1.0,
            x2*x3/40.0 - 1.0,
            5.0*x2/x1 - 1.0,
            x1/(12.0*x2) - 1.0,
            (1.5*x6 + 1.9)/x4 - 1.0,
            (1.1*x7 + 1.9)/x5 - 1.0]

SPEED = dict(
    name='Speed reducer', n=7,
    lb=np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
    ub=np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
    f=speed_f, g=speed_g,
    x_opt=np.array([3.5, 0.7, 17.0, 7.3, 7.715320, 3.350215, 5.286654]),
    f_opt=2994.471066, ref='Golinski; Coello')


# ── 5. 10-bar planar truss sizing (real FE analysis in the loop) ────────────────
# Classic cantilever truss (Haug & Arora 1979; standard metaheuristic benchmark).
# Cross-section areas of the 10 members are the design variables; each objective
# evaluation assembles the global stiffness matrix and solves K u = F, so the
# optimizer runs against an actual finite-element simulation.
#   E = 10^4 ksi, rho = 0.1 lb/in^3, allowable stress ±25 ksi,
#   displacement limit ±2 in on every free dof, A in [0.1, 35] in^2,
#   loads: 100 kip downward at nodes 2 and 4 (load case 1).
# Best-known continuous optimum: ~5060.85 lb (widely reported).
_T10_NODES = np.array([[720.0, 360.0], [720.0, 0.0], [360.0, 360.0],
                       [360.0, 0.0], [0.0, 360.0], [0.0, 0.0]])   # in
# members as 0-based node pairs: 5-3, 3-1, 6-4, 4-2, 3-4, 1-2, 5-4, 6-3, 3-2, 4-1
_T10_MEMBERS = [(4, 2), (2, 0), (5, 3), (3, 1), (2, 3),
                (0, 1), (4, 3), (5, 2), (2, 1), (3, 0)]
_T10_E, _T10_RHO = 1.0e4, 0.1          # ksi, lb/in^3
_T10_SIG, _T10_DMAX = 25.0, 2.0        # ksi, in
_T10_NFREE = 8                          # nodes 1-4 free (x,y); nodes 5,6 fixed
_T10_LOAD = np.zeros(_T10_NFREE)
_T10_LOAD[3] = -100.0                   # node 2, y
_T10_LOAD[7] = -100.0                   # node 4, y
_T10_GEOM = []
for _i, _j in _T10_MEMBERS:
    _d = _T10_NODES[_j] - _T10_NODES[_i]
    _L = float(np.hypot(*_d))
    _T10_GEOM.append((_i, _j, _L, _d[0] / _L, _d[1] / _L))


def _t10_fe(A):
    """Assemble and solve the truss; return (free-dof displacements, stresses)."""
    K = np.zeros((12, 12))
    for m, (i, j, L, c, s) in enumerate(_T10_GEOM):
        T = np.array([-c, -s, c, s])
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(idx, idx)] += (_T10_E * A[m] / L) * np.outer(T, T)
    free = list(range(_T10_NFREE))
    u = np.zeros(12)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], _T10_LOAD)
    stress = np.array([_T10_E / L * float(np.dot([-c, -s, c, s],
                                                 u[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]]))
                       for (i, j, L, c, s) in _T10_GEOM])
    return u[free], stress


def truss10_f(x):
    return _T10_RHO * float(sum(a * g[2] for a, g in zip(x, _T10_GEOM)))


def truss10_g(x):
    try:
        u, stress = _t10_fe(np.asarray(x, float))
    except np.linalg.LinAlgError:
        return [1e6] * 18
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(stress))):
        return [1e6] * 18
    return ([abs(sg) / _T10_SIG - 1.0 for sg in stress]
            + [abs(ui) / _T10_DMAX - 1.0 for ui in u])


TRUSS10 = dict(
    name='10-bar truss (FEA)', n=10,
    lb=np.full(10, 0.1), ub=np.full(10, 35.0),
    f=truss10_f, g=truss10_g,
    x_opt=np.array([30.522, 0.100, 23.200, 15.223, 0.100,
                    0.551, 7.457, 21.036, 21.528, 0.100]),
    f_opt=5060.85, ref='Haug-Arora 1979; continuous case-1 best-known')


PROBLEMS = [SPRING, VESSEL, WELDED, SPEED, TRUSS10]


def verify_all(gtol=1e-3, ftol=5e-3):
    """Assert each published optimum is feasible at its published objective."""
    ok = True
    for p in PROBLEMS:
        x = p['x_opt']
        f = p['f'](x)
        gs = p['g'](x)
        maxviol = max(gs)
        frel = abs(f - p['f_opt']) / abs(p['f_opt'])
        feas = maxviol <= gtol
        fmatch = frel <= ftol
        status = 'OK' if (feas and fmatch) else 'FAIL'
        if not (feas and fmatch):
            ok = False
        print(f"[{status}] {p['name']:28s} n={p['n']} "
              f"f(x*)={f:.6g} (ref {p['f_opt']:.6g}, rel {frel:.2e})  "
              f"max g(x*)={maxviol:+.3e} (tol {gtol:.0e})")
    print("ALL PROBLEMS VERIFIED" if ok else "VERIFICATION FAILED — fix formulation")
    return ok


if __name__ == '__main__':
    verify_all()
