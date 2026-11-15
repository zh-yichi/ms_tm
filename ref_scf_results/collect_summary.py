#!/usr/bin/env python
"""Collect results/*/*.json into summary.tsv and problems.txt"""
import glob, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'results_minao')
recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, '*', '*.json')))]
for r in recs:   # lowest internal Hessian eigenvalue of the final stability round
    eigs = r.get('stab_eigs') or []
    last = eigs[-1] if eigs else {}
    vals = [v for k, v in last.items() if k.endswith('_internal') and v is not None]
    r['min_hess_eig'] = min(vals) if vals else None

cols = ['name', 'subdirs', 'charge', 'mult', 'method', 'scf_class', 'nao', 'energy',
        'converged', 'converged_by', 'internal_stable', 'external_stable',
        'stab_rounds', 'min_hess_eig', 's2', 'time_s', 'chkfile', 'logfile', 'note']
def fmt(v):
    if isinstance(v, float): return '%.10f' % v
    if isinstance(v, list): return ';'.join(map(str, v))
    return '' if v is None else str(v)

with open(os.path.join(outdir, 'summary.tsv'), 'w') as f:
    f.write('\t'.join(cols) + '\n')
    for r in recs:
        f.write('\t'.join(fmt(r.get(c)) for c in cols) + '\n')

problems = [r for r in recs if (not r.get('converged')) or r.get('internal_stable') is False
            or any('ERROR' in n for n in r.get('note', []))]
with open(os.path.join(outdir, 'problems.txt'), 'w') as f:
    f.write('# systems that did not converge, errored, or remain unstable\n')
    for r in problems:
        f.write('%s  %s  charge=%s mult=%s  converged=%s  stable=%s  %s\n' % (
            r['name'], r['method'], r.get('charge'), r.get('mult'), r.get('converged'),
            r.get('internal_stable'), '; '.join(r.get('note', []))))

n_all = len(recs)
print('%d records, %d converged, %d problem records -> %s' % (
    n_all, sum(1 for r in recs if r.get('converged')), len(problems),
    os.path.join(outdir, 'problems.txt')))
for r in problems:
    print('  ', r['name'], r['method'], '; '.join(r.get('note', [])))
