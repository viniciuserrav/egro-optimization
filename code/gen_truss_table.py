# -*- coding: utf-8 -*-
"""Generate the truss results table (v2: corrected vertical-only limits)."""
import json

import numpy as np

SRC = (r"c:\Vinicius\Papers\EGRO - Metaeuristc\paper_artifacts\results"
       r"\engineering_results_truss_v2.json")
OUT = (r"c:\Vinicius\Papers\EGRO - Metaeuristc"
       r"\Submission_Advances_in_Engineering_Software\image_latex"
       r"\table_truss10.tex")

d = json.load(open(SRC))
ALGOS = ['EGRO-CMA', 'CMA-ES', 'L-SHADE', 'DE', 'PSO', 'GWO', 'WOA']

rows = []
for a in ALGOS:
    v1 = np.array([d['%s||%d' % (a, s)]['budget']['1000'] for s in range(30)])
    v4 = np.array([d['%s||%d' % (a, s)]['budget']['10000'] for s in range(30)])
    rows.append((v4.mean(), a, v1.mean(), v1.std(), v4.std(), v4.min()))
rows.sort()
best1 = min(r[2] for r in rows)
best4 = rows[0][0]
bestmin = min(r[5] for r in rows)


def cell(v, sd=None):
    return ('%.1f' % v) if sd is None else '%.1f {\\scriptsize(%.0f)}' % (v, sd)


L = [
    '%% Auto-generated from engineering_results_truss_v2.json (30 seeds,',
    '%% vertical displacement limits at the four free nodes).',
    '\\begin{table}[!t]',
    '  \\centering',
    '  \\caption{10-bar truss sizing with a finite-element solve per',
    '           evaluation: best \\emph{feasible} weight (lb) over 30 seeds,',
    '           mean (std) at $10^3$ and $10^4$ evaluations and best single',
    '           run, against the best-known continuous optimum of',
    '           $5060.85$~lb.  All runs of all algorithms returned feasible',
    '           designs.}',
    '  \\label{tab:truss10}',
    '  \\small',
    '  \\setlength{\\tabcolsep}{3.5pt}',
    '  \\begin{tabular}{lrrr}',
    '    \\toprule',
    '    Algorithm & Mean @ $10^3$ & Mean @ $10^4$ & Min @ $10^4$ \\\\',
    '    \\midrule',
]
for m4, a, m1, s1, s4, mn in rows:
    c1, c2, c3 = cell(m1, s1), cell(m4, s4), '%.1f' % mn
    if m1 == best1:
        c1 = '\\textbf{%s}' % c1
    if m4 == best4:
        c2 = '\\textbf{%s}' % c2
    if mn == bestmin:
        c3 = '\\textbf{%s}' % c3
    L.append('    %s & %s & %s & %s \\\\' % (a, c1, c2, c3))
L += ['    \\bottomrule', '  \\end{tabular}', '\\end{table}']

open(OUT, 'w').write('\n'.join(L) + '\n')
print('written:', OUT)
for m4, a, m1, s1, s4, mn in rows:
    print('%-9s @1e3 %.1f (%.0f)   @1e4 %.1f (%.1f)   min %.1f'
          % (a, m1, s1, m4, s4, mn))
