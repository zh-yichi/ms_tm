#!/usr/bin/env python
"""Shared driver for the Newton (second-order) SCF runs in the larger basis sets
(tz/run_newton.py, qz/run_newton.py; sfX2C-1e, cc-pVXZ-DK).  Do not run this file directly.

For every system and method (RHF/ROHF, UHF) the starting point is the lowest Newton
solution of the smaller basis, <parent>/<stem>/<stem>.<method>.chk, projected onto the
new basis.  No DIIS anywhere; same procedure as dz/rerun_newton.py:
    Newton SCF -> internal stability analysis -> if unstable, step along the lowest
    Hessian eigenvector in BOTH directions (+v and -v), re-converge both with Newton
    and keep the lower one -> repeat until stable.

Output, per system, in <here>/results_newton/<stem>/ :
    <stem>.<method>.chk     SCF chkfile (kept; it starts the next basis and the CCSD)
    <stem>.<method>.json    result record (with the energy and <S^2> of the smaller basis)
    <stem>.newton.log       pyscf output
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

here_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here_)
sys.path.insert(0, os.path.join(here_, 'dz'))
from pyscf import gto, scf, lib
from scf_lowest import parse_xyz, build_mol, load_json
from rerun_newton import run_guess, new_mf        # Newton + stability following

S2_TOL = 0.1      # <S^2> change relative to the smaller basis that is flagged


def projected_dm(mol, method, chk):
    """Density matrix of the smaller-basis orbitals projected onto the basis of mol."""
    if method == 'uhf':
        return scf.uhf.init_guess_by_chkfile(mol, chk, project=True)
    if mol.spin > 0:
        return scf.rohf.init_guess_by_chkfile(mol, chk, project=True)
    return scf.hf.init_guess_by_chkfile(mol, chk, project=True)


def summarize(outdir, expected, methods):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, '*', '*.json')))]
    cols = ['name', 'charge', 'mult', 'method', 'scf_class', 'basis', 'nao', 'energy', 's2', 'converged',
            'internal_stable', 'stab_rounds', 'parent_basis', 'parent_energy', 'parent_s2', 'parent_guess',
            'time_s', 'note']

    def fmt(v):
        if isinstance(v, float): return '%.10f' % v
        if isinstance(v, list): return ';'.join(map(str, v))
        return '' if v is None else str(v)
    with open(os.path.join(outdir, 'summary.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in recs:
            f.write('\t'.join(fmt(r.get(c)) for c in cols) + '\n')
    have = {(r['name'], r['method']): r for r in recs}
    prob = ['%s  %s  MISSING (not run yet, or no converged chkfile in the smaller basis)' % (s, m)
            for s in expected for m in methods if (s, m) not in have]
    for r in recs:
        why = []
        if not r.get('converged'): why.append('not converged')
        if r.get('converged') and r.get('internal_stable') is not True: why.append('internally unstable')
        why += [n for n in r.get('note', []) if 'ERROR' in n]
        if r.get('stab_rounds'):
            why.append('instability followed %d time(s): the state may differ from the smaller basis' % r['stab_rounds'])
        if r.get('s2') is not None and r.get('parent_s2') is not None and abs(r['s2'] - r['parent_s2']) > S2_TOL:
            why.append('<S^2> %.3f vs %.3f in the smaller basis: possibly a different state' % (r['s2'], r['parent_s2']))
        if r.get('energy') is not None and r.get('parent_energy') is not None and r['energy'] > r['parent_energy']:
            why.append('energy ABOVE the smaller basis by %.3f mEh' % ((r['energy'] - r['parent_energy']) * 1e3))
        u = have.get((r['name'], 'rhf'))
        if r['method'] == 'uhf' and u and None not in (r.get('energy'), u.get('energy')) and r['energy'] > u['energy'] + 1e-6:
            why.append('UHF is %.4f mEh ABOVE RHF/ROHF' % ((r['energy'] - u['energy']) * 1e3))
        if why:
            prob.append('%s  %s  %s' % (r['name'], r['method'], '; '.join(why)))
    with open(os.path.join(outdir, 'problems.txt'), 'w') as f:
        f.write('# missing, not converged, unstable, instability followed, <S^2> changed between basis sets, ...\n')
        f.write('\n'.join(prob) + ('\n' if prob else ''))
    print('%d records (%d expected), %d converged and stable, %d flagged -> %s' % (
        len(recs), len(expected) * len(methods),
        sum(1 for r in recs if r.get('converged') and r.get('internal_stable')), len(prob),
        os.path.join(outdir, 'problems.txt')))
    for p in prob:
        print('  ', p)


def main(basis, here, parent):
    ap = argparse.ArgumentParser(description='Newton SCF (%s, sfX2C-1e) starting from the chkfiles in %s' % (basis, parent))
    ap.add_argument('files', nargs='*', help='xyz files or stems (default: all under --xyzdir)')
    ap.add_argument('--xyzdir', default=os.path.join(here, '..', 'xyzfiles'))
    ap.add_argument('--parent', default=parent,
                    help='results folder of the smaller basis: <stem>/<stem>.<method>.{chk,json} (default: %(default)s)')
    ap.add_argument('--outdir', default=os.path.join(here, 'results_newton'))
    ap.add_argument('--methods', default='rhf,uhf')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--newton-max-cycle', type=int, default=200)
    ap.add_argument('--max-stab', type=int, default=15)
    ap.add_argument('--stab-tol', type=float, default=1e-4)
    ap.add_argument('--external', choices=['on', 'off'], default='off',
                    help='also run the external stability analysis (recorded only)')
    ap.add_argument('--memory', type=int, default=4000, help='MB per calculation')
    ap.add_argument('--force', action='store_true', help='rerun even if the json exists')
    ap.add_argument('--sizes', action='store_true',
                    help='only print "<nao> <xyz file>" for every system, largest first')
    ap.add_argument('--summary', action='store_true', help='only write summary.tsv / problems.txt')
    a = ap.parse_args()

    only = {os.path.splitext(os.path.basename(s))[0] for s in a.files}
    uniq = {}       # identical files appear in several subfolders
    for f in sorted(glob.glob(os.path.join(a.xyzdir, '*', '*.xyz'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not only or stem in only:
            uniq.setdefault(stem, []).append(os.path.abspath(f))
    methods = a.methods.split(',')
    if a.sizes:
        sz = []
        for stem, paths in uniq.items():
            charge, mult, atoms = parse_xyz(paths[0])
            sz.append((gto.M(atom=atoms, basis=basis, charge=charge, spin=mult - 1, verbose=0).nao, paths[0]))
        for n, p in sorted(sz, reverse=True):
            print(n, p)
        return
    if a.summary:
        summarize(a.outdir, sorted(uniq), methods)
        return
    os.makedirs(a.outdir, exist_ok=True)
    print('threads:', lib.num_threads(), ' systems:', len(uniq), ' basis:', basis, ' parent:', a.parent, flush=True)

    for stem, paths in uniq.items():
        sysdir = os.path.join(a.outdir, stem)
        jf = lambda m, d=sysdir: os.path.join(d, '%s.%s.json' % (stem, m))
        chk = lambda m, d=sysdir: os.path.join(d, '%s.%s.chk' % (stem, m))
        todo = [m for m in methods if a.force or not os.path.exists(jf(m))]
        if not todo:
            print('skip (done):', stem, flush=True)
            continue
        pdir = os.path.join(a.parent, stem)
        logf = os.path.join(sysdir, '%s.newton.log' % stem)
        mol = None
        for method in todo:
            t0 = time.time()
            prec = load_json(jf(method, pdir))
            if not (prec and prec.get('converged') and os.path.exists(chk(method, pdir))):
                print('skip (no converged %s result/chkfile in %s): %s' % (method, a.parent, stem), flush=True)
                continue
            try:
                if mol is None:
                    os.makedirs(sysdir, exist_ok=True)
                    mol = build_mol(paths[0], basis, logf, a.memory)      # one log per system
                dm0 = projected_dm(mol, method, chk(method, pdir))
                rec = run_guess(mol, method, 'project:' + chk(method, pdir), dm0, chk(method), logf, a)
            except Exception as ex:
                traceback.print_exc()
                rec = {'converged': False, 'energy': None, 'internal_stable': None, 's2': None,
                       'note': ['ERROR: %r' % ex]}
            rec.pop('guess', None)
            charge, mult, _ = parse_xyz(paths[0])
            rec.update(method=method, name=stem, xyz=paths, charge=charge, mult=mult,
                       subdirs=[os.path.basename(os.path.dirname(p)) for p in paths],
                       basis=basis, x2c=True, solver='newton', start='project:%s' % chk(method, pdir),
                       parent_energy=prec.get('energy'), parent_s2=prec.get('s2'), parent_basis=prec.get('basis'),
                       parent_guess=prec.get('init_guess'), chkfile=chk(method), logfile=logf,
                       conv_tol=a.conv_tol, newton_max_cycle=a.newton_max_cycle, max_stab=a.max_stab,
                       stab_tol=a.stab_tol, time_s=round(time.time() - t0, 1))
            if mol is not None:
                rec.update(nao=mol.nao, nelec=mol.nelec)
            if rec.get('s2') is not None and prec.get('s2') is not None and abs(rec['s2'] - prec['s2']) > S2_TOL:
                rec['note'].append('<S^2> changed from %.3f in %s' % (prec['s2'], prec.get('basis')))
            with open(jf(method), 'w') as f:
                json.dump(rec, f, indent=1)
            print('%-50s %-4s E=%-18s conv=%-5s stab_i=%-5s rounds=%s <S^2>=%-8s (smaller basis %s) %6.0fs %s' % (
                stem[:50], method, rec.get('energy'), rec['converged'], rec.get('internal_stable'),
                rec.get('stab_rounds'), None if rec.get('s2') is None else round(rec['s2'], 4),
                None if prec.get('s2') is None else round(prec['s2'], 4), rec['time_s'], '; '.join(rec['note'])),
                flush=True)
        if mol is not None:
            mol.stdout.close()
