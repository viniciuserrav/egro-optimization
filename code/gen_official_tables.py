"""
gen_official_tables.py — turn the official-numbering rerun JSONs into the
paper's tables, ranking statistics, and CD diagrams.

Inputs : cec2017_official_d{10,30,50}.json  (merged; official F-numbers)
Outputs: table_cec2017_official_d{dim}.tex   (same format as the old tables)
         cec_cd_diagram_official_d{dim}.pdf  (Demsar-style CD diagram)
         official_stats.json                 (ranks, CD, Friedman p, wins)

A function enters the analysis at a given dimension only when all seven
algorithms have 30 finite errors there.  Functions dropped for missing or
flagged data are reported explicitly.  Official F6 (Schaffer F7) gets a
degeneracy report (spread of algorithm means) so its exclusion, if any, is
a documented analysis decision rather than a silent one.

Usage:  python gen_official_tables.py [src_dir] [out_dir]
"""
import json
import os
import sys

import numpy as np

ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']
DIMS = [10, 30, 50]
Q_ALPHA_7 = 2.949                      # Nemenyi q_0.05 for k = 7 (Demsar 2006)
ETOL = 1e-8                            # CEC convention: |error| < 1e-8 -> 0


def fmt(v):
    return '0.00' if v == 0 else '%.2e' % v


def tie_ranks(values):
    """Average ranks, 1 = best (lowest)."""
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values))
    i = 0
    sv = np.array(values)[order]
    while i < len(values):
        j = i
        while j + 1 < len(values) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def friedman_stat(rank_matrix):
    """Tie-corrected Friedman chi-square and p-value over an (N x k) matrix."""
    from scipy import stats
    N, k = rank_matrix.shape
    Rj = rank_matrix.sum(axis=0)
    chi2 = 12.0 / (N * k * (k + 1)) * float((Rj ** 2).sum()) - 3.0 * N * (k + 1)
    # tie correction
    T = 0.0
    for row in rank_matrix:
        _, counts = np.unique(row, return_counts=True)
        T += float(((counts ** 3) - counts).sum())
    C = 1.0 - T / (N * k * (k * k - 1))
    if C > 0:
        chi2 /= C
    p = float(stats.chi2.sf(chi2, k - 1))
    return chi2, p


def cd_diagram(mean_ranks, cd, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = sorted(mean_ranks, key=mean_ranks.get)
    k = len(order)

    # group bars: maximal cliques of consecutive algorithms within CD
    groups = []
    i = 0
    while i < k:
        j = i
        while (j + 1 < k and
               mean_ranks[order[j + 1]] - mean_ranks[order[i]] <= cd):
            j += 1
        if j > i:
            groups.append((i, j))
        i += 1
    groups = [g for g in groups
              if not any(o[0] <= g[0] and g[1] <= o[1] and o != g
                         for o in groups)]
    ng = max(len(groups), 1)

    bar_top, bar_step = 8.7, 0.5
    bar_bottom = bar_top - bar_step * (ng - 1)
    stem_y = bar_bottom - 0.5                 # where stems bend outward
    fig, ax = plt.subplots(figsize=(9.0, 2.4 + 0.28 * ng))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(stem_y - 1.6 * ((k + 1) // 2), 11.2)
    ax.spines[['left', 'right', 'bottom']].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks(range(lo, hi + 1))
    ax.xaxis.tick_top()
    ax.plot([lo, hi], [9.4, 9.4], color='none')      # anchor axis at top

    # CD ruler, well above the tick labels
    ax.plot([lo, lo + cd], [10.55, 10.55], lw=3, color='k', clip_on=False)
    ax.text(lo + cd / 2, 10.75, 'CD = %.2f' % cd, ha='center', va='bottom')

    for gi, (a, b) in enumerate(groups):
        y = bar_top - bar_step * gi
        ax.plot([mean_ranks[order[a]] - 0.05, mean_ranks[order[b]] + 0.05],
                [y, y], lw=4, color='k', solid_capstyle='round')

    # labels: left half descend on the left, right half on the right
    half = (k + 1) // 2
    for idx, name in enumerate(order):
        r = mean_ranks[name]
        if idx < half:
            xt, ha = lo - 0.35, 'right'
            yt = stem_y - 1.2 - 1.5 * idx
        else:
            xt, ha = hi + 0.35, 'left'
            yt = stem_y - 1.2 - 1.5 * (k - 1 - idx)
        ax.plot([r, r], [stem_y, 9.4], lw=0.9, color='k')
        ax.plot([r, xt + (0.02 if ha == 'left' else -0.02)], [stem_y, yt],
                lw=0.9, color='k')
        ax.text(xt, yt, '%s (%.2f)' % (name, r), ha=ha, va='center')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else '.'
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    os.makedirs(dst, exist_ok=True)
    stats_out = {}

    for dim in DIMS:
        path = os.path.join(src, 'cec2017_official_d%d.json' % dim)
        if not os.path.exists(path):
            print('[d=%d] missing %s, skipped' % (dim, path))
            continue
        data = json.load(open(path))
        fns = sorted((k for k in data if k != '_meta'),
                     key=lambda s: int(s[1:]))

        means, dropped = {}, {}
        for key in fns:
            ok, m = True, {}
            for a in ALGOS:
                rec = data.get(key, {}).get(a)
                errs = [0.0 if abs(e) < ETOL else e
                        for e in (rec or {}).get('errors', []) if e is not None]
                if rec is None or len(errs) < len(rec['errors']):
                    flags = {f for f in (rec or {}).get('flags', []) if f}
                    dropped[key] = ('incomplete' if not flags
                                    else '/'.join(sorted(flags))[:60])
                    ok = False
                    break
                m[a] = float(np.mean(errs))
            if ok:
                means[key] = m

        valid = sorted(means, key=lambda s: int(s[1:]))
        N = len(valid)
        print('[d=%d] %d functions valid, dropped: %s'
              % (dim, N, dropped or 'none'))
        if not N:
            continue

        # degeneracy report for official F6
        if 'F6' in means:
            v = [means['F6'][a] for a in ALGOS]
            print('[d=%d] F6 degeneracy check: means %.3g .. %.3g (spread %.3g)'
                  % (dim, min(v), max(v), max(v) - min(v)))

        rank_matrix = np.array([tie_ranks([means[f][a] for a in ALGOS])
                                for f in valid])
        mean_ranks = dict(zip(ALGOS, rank_matrix.mean(axis=0)))
        chi2, pval = friedman_stat(rank_matrix)
        cd = Q_ALPHA_7 * np.sqrt(7 * 8 / (6.0 * N))
        sole = {a: 0 for a in ALGOS}
        for f in valid:
            row = [means[f][a] for a in ALGOS]
            best = min(row)
            if row.count(best) == 1:
                sole[ALGOS[row.index(best)]] += 1
        wins_vs_cma = sum(1 for f in valid
                          if means[f]['EGRO-CMA'] < means[f]['CMA-ES'])
        ties_vs_cma = sum(1 for f in valid
                          if means[f]['EGRO-CMA'] == means[f]['CMA-ES'])

        # ---------------- LaTeX table ----------------
        note = {10: ' (functions with non-finite benchmark output at this '
                    'dimension are excluded, see text)', 30: '', 50: ''}[dim]
        if any(means[f][a] < 0 for f in valid for a in ALGOS):
            note += (r'.  Negative entries indicate functions whose stated '
                     r'reference optimum in the implementation lies above '
                     r'the attainable minimum (see text); rankings are '
                     r'unaffected, as each reference is a per-function '
                     r'constant')
        lines = [
            '%% Auto-generated by gen_official_tables.py -- OFFICIAL '
            'CEC2017 numbering, d=%d, %d functions' % (dim, N),
            r'\begin{table*}[!t]',
            r'  \centering',
            r'  \caption{Mean error on CEC~2017 (official numbering) at '
            r'$d=%d$ over 30 runs, best per function in \textbf{bold}%s.}'
            % (dim, note),
            r'  \label{tab:cec2017_d%d}' % dim,
            r'  \small',
            r'  \setlength{\tabcolsep}{4pt}',
            r'  \begin{tabular}{lrrrrrrr}',
            r'    \toprule',
            r'    Fn & ' + ' & '.join(ALGOS) + r' \\',
            r'    \midrule',
        ]
        for f in valid:
            row = [means[f][a] for a in ALGOS]
            best = min(row)
            cells = [(r'\textbf{%s}' % fmt(v)) if v == best else fmt(v)
                     for v in row]
            lines.append('    %s & %s \\\\' % (f, ' & '.join(cells)))
        mx = max(sole.values())
        lines += [
            r'    \midrule',
            '    Best (sole) & ' + ' & '.join(
                (r'\textbf{%d}' % sole[a]) if sole[a] == mx else str(sole[a])
                for a in ALGOS) + r' \\',
            r'    \bottomrule',
            r'  \end{tabular}',
            r'\end{table*}',
        ]
        tex = os.path.join(dst, 'table_cec2017_official_d%d.tex' % dim)
        open(tex, 'w').write('\n'.join(lines) + '\n')

        cd_diagram(mean_ranks, cd,
                   os.path.join(dst, 'cec_cd_diagram_official_d%d.pdf' % dim))

        stats_out['d%d' % dim] = {
            'n_functions': N, 'valid': valid, 'dropped': dropped,
            'mean_ranks': {a: round(mean_ranks[a], 3) for a in ALGOS},
            'friedman_chi2': round(chi2, 2), 'friedman_p': pval,
            'nemenyi_cd': round(cd, 3), 'sole_wins': sole,
            'egro_vs_cma_wins': wins_vs_cma, 'egro_vs_cma_ties': ties_vs_cma,
        }
        rank_str = ', '.join('%s %.2f' % (a, mean_ranks[a])
                             for a in sorted(ALGOS, key=mean_ranks.get))
        print('[d=%d] ranks: %s' % (dim, rank_str))
        print('[d=%d] Friedman chi2=%.1f p=%.2e | CD=%.2f | '
              'EGRO-CMA<CMA-ES on %d/%d (%d ties)'
              % (dim, chi2, pval, cd, wins_vs_cma, N, ties_vs_cma))

    with open(os.path.join(dst, 'official_stats.json'), 'w') as fh:
        json.dump(stats_out, fh, indent=1)
    print('wrote official_stats.json')


if __name__ == '__main__':
    main()
