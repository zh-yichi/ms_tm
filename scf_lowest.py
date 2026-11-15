#!/usr/bin/env python
"""
Shared driver for the production SCF runs in dz/, tz/, qz/ (sfX2C-1e, cc-pVXZ-DK).
Each of those folders has a small run_scf.py that calls main() with its basis and
its starting point; do not run this file directly.

Two ways to start the SCF:

  reference=<dz/reference.json>   (dz)
      Reproduce the lowest known solution: RHF/ROHF and UHF are run with the
      initial guess and SCF settings that gave the lowest energy in
      ref_scf_results (see dz/make_reference.py).  UHF entries with guess
      'rhfchk' start from the RHF/ROHF orbitals of this run.  The energy is
      compared with the reference energy.

  parent=<results folder of the smaller basis>   (tz, qz, ...)
      RHF/ROHF and UHF start from the orbitals in
      <parent>/<stem>/<stem>.{rhf,uhf}.chk, projected onto the new basis.

Per method: DIIS (level shift) -> Newton if DIIS fails -> internal stability
analysis, following instabilities and re-converging (same procedure as
ref_scf_results/run_scf.py).  The external stability analysis (recorded only,
never followed) is on in dz and off in tz/qz, where it only costs time (--external).

Output, per system, in <here>/results/<stem>/:
  <stem>.log                 pyscf output
  <stem>.{rhf,uhf}.chk       SCF chkfiles (kept; they start the next basis)
  <stem>.{rhf,uhf}.json      result records
"""
import argparse
import glob
import json
import os
import re
import time
import traceback

import numpy as np
from pyscf import gto, scf, lib
from pyscf.lib import logger

VERBOSE = 4
E_TOL = 1e-5      # Eh; dz energy more than this above the reference is flagged
S2_TOL = 0.1      # <S^2> change relative to the smaller basis that is flagged
SETTINGS = dict(max_cycle=1000, newton_max_cycle=200, max_stab=15, stab_tol=1e-4, level_shift=0.4)


def parse_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    natm = int(lines[0].split()[0])
    hdr = lines[1].split()
    charge, mult = int(hdr[0]), int(hdr[1])
    atoms = []
    for line in lines[2:2 + natm]:
        s = line.split()
        atoms.append((s[0], (float(s[1]), float(s[2]), float(s[3]))))
    return charge, mult, atoms


def build_mol(path, basis, logfile, memory):
    charge, mult, atoms = parse_xyz(path)
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.charge = charge
    mol.spin = mult - 1
    mol.unit = 'Angstrom'
    mol.verbose = VERBOSE
    mol.output = logfile
    mol.max_memory = memory
    mol.build()
    return mol


_EIG_RE = re.compile(r'(\w+): lowest eigs of H = \[([^\]]*)\]')


def _stability_with_eigs(mf, logfile, internal, external):
    """mf.stability plus the lowest Hessian eigenvalues pyscf prints to the log."""
    mf.stdout.flush()
    pos = os.path.getsize(logfile)
    mo_i, mo_e, stable_i, stable_e = mf.stability(
        internal=internal, external=external, return_status=True)
    mf.stdout.flush()
    eigs = {}
    with open(logfile) as f:
        f.seek(pos)
        for m in _EIG_RE.finditer(f.read()):
            vals = [float(x) for x in m.group(2).split()]
            eigs[m.group(1)] = min(vals) if vals else None
    return mo_i, mo_e, stable_i, stable_e, eigs


def dm_from_rhf_chk(chk):
    """UHF (alpha, beta) density matrices from RHF/ROHF orbitals in the same basis."""
    mo = np.asarray(lib.chkfile.load(chk, 'scf/mo_coeff'))
    occ = np.asarray(lib.chkfile.load(chk, 'scf/mo_occ'))
    if mo.ndim != 2:
        raise ValueError('%s is not an RHF/ROHF chkfile' % chk)
    dma = (mo[:, occ > 0]).dot(mo[:, occ > 0].T)     # doubly + singly occupied
    dmb = (mo[:, occ > 1]).dot(mo[:, occ > 1].T)     # doubly occupied
    return np.array([dma, dmb])


def run_one(mol, method, basis, chkfile, logfile, st, conv_tol, start, external=True):
    """start = ('guess', name) | ('rhfchk', chk in the same basis) | ('project', chk in a smaller basis).
    st: dict with max_cycle, newton_max_cycle, max_stab, stab_tol, level_shift."""
    rec = {'method': method, 'converged': False, 'converged_by': None,
           'internal_stable': None, 'external_stable': None,
           'stab_rounds': 0, 'stab_eigs': [], 'energy': None, 's2': None, 'note': []}
    log = logger.new_logger(mol, VERBOSE)
    log.note('\n' + '=' * 70 + '\n  %s   %s   charge=%d  2S=%d  basis=%s  sfX2C-1e   start: %s %s\n' % (
        method.upper(), os.path.basename(chkfile), mol.charge, mol.spin, basis, start[0], start[1])
        + '=' * 70)

    mf = (scf.RHF(mol) if method == 'rhf' else scf.UHF(mol)).x2c()   # RHF -> ROHF if mol.spin > 0
    is_rohf = (method == 'rhf' and mol.spin > 0)
    mf.chkfile = chkfile
    mf.level_shift = st['level_shift']
    mf.max_cycle = st['max_cycle']
    mf.conv_tol = conv_tol
    mf.diis_space = 12

    kind, arg = start
    dm0 = None
    if kind == 'project':
        # orbitals of the smaller basis projected onto this one
        if method == 'uhf':
            dm0 = scf.uhf.init_guess_by_chkfile(mol, arg, project=True)
        elif is_rohf:
            dm0 = scf.rohf.init_guess_by_chkfile(mol, arg, project=True)
        else:
            dm0 = scf.hf.init_guess_by_chkfile(mol, arg, project=True)
    elif method == 'uhf':
        dm0 = dm_from_rhf_chk(arg) if kind == 'rhfchk' else np.asarray(mf.get_init_guess(key=arg))
        if mol.spin == 0:
            # break alpha/beta symmetry so that singlets can relax to a spin-polarized solution
            dm0 = np.array([dm0[0] * 1.05, dm0[1] * 0.95])
    else:
        mf.init_guess = arg

    def converge(mf_, dm):
        mf_.kernel(dm0=dm)
        if mf_.converged:
            return mf_, 'diis'
        log.note('DIIS SCF did not converge -> second-order (Newton) SCF')
        dm = mf_.make_rdm1()
        nmf = mf_.newton()
        nmf.chkfile = chkfile
        nmf.max_cycle = st['newton_max_cycle']
        nmf.conv_tol = conv_tol
        nmf.kernel(dm0=dm)
        mf_.mo_coeff, mf_.mo_energy, mf_.mo_occ = nmf.mo_coeff, nmf.mo_energy, nmf.mo_occ
        mf_.e_tot, mf_.converged = nmf.e_tot, nmf.converged
        return mf_, ('newton' if nmf.converged else 'none')

    mf, how = converge(mf, dm0)
    rec['converged_by'] = how
    if not mf.converged:
        rec['note'].append('SCF NOT CONVERGED (diis + newton)')

    # ---- stability analysis; follow internal instabilities ----
    stable_i = stable_e = None
    external = external and not is_rohf     # rohf_external is not implemented in pyscf
    for it in range(st['max_stab'] + 1):
        if not mf.converged:
            break
        log.note('--- stability analysis, round %d ---', it)
        mo_i, mo_e, stable_i, stable_e, eigs = _stability_with_eigs(
            mf, logfile, internal=True, external=external)
        if not external:
            stable_e = None
        rec['stab_eigs'].append(eigs)
        min_eig = min([v for k, v in eigs.items() if k.endswith('_internal') and v is not None]
                      or [None], key=lambda x: x if x is not None else 0)
        if (not stable_i) and min_eig is not None and min_eig > -st['stab_tol']:
            log.note('lowest internal Hessian eigenvalue %.3e > -stab_tol (%.1e): treated as stable',
                     min_eig, st['stab_tol'])
            rec['note'].append('internal eig %.1e within stab_tol, treated as stable' % min_eig)
            stable_i = True
        if stable_i or it == st['max_stab']:
            break
        rec['stab_rounds'] += 1
        log.note('internal instability found -> following the lowest Hessian eigenvector and re-converging')
        e_old = mf.e_tot
        dm = mf.make_rdm1(mo_i, mf.mo_occ)
        mf, how = converge(mf, dm)
        rec['converged_by'] = how
        if not mf.converged:
            rec['note'].append('SCF NOT CONVERGED after following instability')
        elif mf.e_tot > e_old - 1e-8:
            log.note('following the instability did not lower the energy (%.10f -> %.10f): zero mode, stop',
                     e_old, mf.e_tot)
            rec['note'].append('instability following did not lower E; zero mode, treated as stable')
            stable_i = True
            break

    rec['converged'] = bool(mf.converged)
    rec['internal_stable'] = None if stable_i is None else bool(stable_i)
    rec['external_stable'] = None if stable_e is None else bool(stable_e)
    if rec['converged'] and rec['internal_stable'] is False:
        rec['note'].append('still internally UNSTABLE after %d rounds' % st['max_stab'])
    if rec['converged'] and rec['external_stable'] is False:
        rec['note'].append('external instability (%s)' % ('RHF->UHF or real->complex' if method == 'rhf'
                                                          else 'UHF->GHF or real->complex'))
    if rec['converged']:
        rec['energy'] = float(mf.e_tot)
        if method == 'uhf' or is_rohf:
            rec['s2'] = float(mf.spin_square()[0])
        # make sure the chk holds the final orbitals
        mf.dump_chk({'e_tot': mf.e_tot, 'mo_energy': mf.mo_energy,
                     'mo_coeff': mf.mo_coeff, 'mo_occ': mf.mo_occ})
    rec.update(scf_class=type(mf).__name__, nao=mol.nao, nelec=mol.nelec)
    log.note('RESULT %s  E=%s  converged=%s (%s)  internal_stable=%s  external_stable=%s  <S^2>=%s',
             method.upper(), rec['energy'], rec['converged'], rec['converged_by'],
             rec['internal_stable'], rec['external_stable'], rec['s2'])
    return rec


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def summarize(outdir, expected):
    """summary.tsv and problems.txt in outdir.  expected: list of stems that should be there."""
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, '*', '*.json')))]
    cols = ['name', 'charge', 'mult', 'method', 'scf_class', 'basis', 'nao', 'energy', 's2',
            'converged', 'converged_by', 'internal_stable', 'external_stable', 'stab_rounds',
            'start', 'ref_energy', 'dE_ref_mEh', 'parent_energy', 'parent_s2', 'time_s', 'note']

    def fmt(v):
        if isinstance(v, float): return '%.10f' % v
        if isinstance(v, list): return ';'.join(map(str, v))
        return '' if v is None else str(v)
    with open(os.path.join(outdir, 'summary.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in recs:
            f.write('\t'.join(fmt(r.get(c)) for c in cols) + '\n')

    prob = []
    have = {(r['name'], r['method']) for r in recs}
    for stem in expected:
        for m in ('rhf', 'uhf'):
            if (stem, m) not in have:
                prob.append('%s  %s  MISSING (not run yet, or its starting chkfile is missing)' % (stem, m))
    for r in recs:
        why = []
        if not r.get('converged'): why.append('not converged')
        if r.get('internal_stable') is False: why.append('internally unstable')
        why += [n for n in r.get('note', []) if 'ERROR' in n]
        if r.get('dE_ref_mEh') is not None and r['dE_ref_mEh'] > E_TOL * 1e3:
            why.append('%.4f mEh ABOVE the reference energy' % r['dE_ref_mEh'])
        if r.get('s2') is not None and r.get('parent_s2') is not None and abs(r['s2'] - r['parent_s2']) > S2_TOL:
            why.append('<S^2> %.3f vs %.3f in the smaller basis: possibly a different state' % (r['s2'], r['parent_s2']))
        if why:
            prob.append('%s  %s  %s' % (r['name'], r['method'], '; '.join(why)))
    with open(os.path.join(outdir, 'problems.txt'), 'w') as f:
        f.write('# missing, not converged, unstable, above the dz reference, or <S^2> changed between basis sets\n')
        f.write('\n'.join(prob) + ('\n' if prob else ''))
    low = [r for r in recs if r.get('dE_ref_mEh') is not None and r['dE_ref_mEh'] < -E_TOL * 1e3]
    print('%d records (%d expected), %d converged, %d problems -> %s' % (
        len(recs), 2 * len(expected), sum(1 for r in recs if r.get('converged')), len(prob),
        os.path.join(outdir, 'problems.txt')))
    for p in prob:
        print('  ', p)
    for r in low:
        print('   note: %s %s is %.4f mEh BELOW the reference' % (r['name'], r['method'], -r['dE_ref_mEh']))


def main(basis, here, reference=None, parent=None, external=True):
    assert (reference is None) != (parent is None)
    ap = argparse.ArgumentParser(description='SCF (%s, sfX2C-1e) %s' % (
        basis, 'reproducing ' + reference if reference else 'starting from ' + parent))
    ap.add_argument('files', nargs='*', help='xyz files (default: all under --xyzdir)')
    ap.add_argument('--xyzdir', default=os.path.join(here, '..', 'xyzfiles'))
    ap.add_argument('--outdir', default=os.path.join(here, 'results'))
    ap.add_argument('--methods', default='rhf,uhf')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--memory', type=int, default=4000, help='MB per calculation')
    ap.add_argument('--force', action='store_true', help='rerun even if the json exists')
    ap.add_argument('--external', choices=['on', 'off'], default='on' if external else 'off',
                    help='also run the external stability analysis (RHF->UHF, UHF->GHF, real->complex; '
                         'recorded only, never followed).  default: %(default)s')
    ap.add_argument('--sizes', action='store_true',
                    help='only print "<nao> <xyz file>" for every system, largest first')
    ap.add_argument('--summary', action='store_true', help='only write summary.tsv / problems.txt')
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(args.xyzdir, '*', '*.xyz')))
    uniq = {}       # identical files appear in several subfolders
    for f in files:
        uniq.setdefault(os.path.splitext(os.path.basename(f))[0], []).append(os.path.abspath(f))
    if args.sizes:
        sz = []
        for stem, paths in uniq.items():
            charge, mult, atoms = parse_xyz(paths[0])
            sz.append((gto.M(atom=atoms, basis=basis, charge=charge, spin=mult - 1, verbose=0).nao, paths[0]))
        for n, p in sorted(sz, reverse=True):
            print(n, p)
        return
    if args.summary:
        summarize(args.outdir, sorted(uniq))
        return
    ref = json.load(open(reference)) if reference else None
    methods = args.methods.split(',')
    os.makedirs(args.outdir, exist_ok=True)

    print('threads:', lib.num_threads(), ' systems:', len(uniq), ' basis:', basis, flush=True)
    for stem, paths in uniq.items():
        sysdir = os.path.join(args.outdir, stem)
        jf = lambda m, d=sysdir: os.path.join(d, '%s.%s.json' % (stem, m))
        chk = lambda m, d=sysdir: os.path.join(d, '%s.%s.chk' % (stem, m))
        todo = [m for m in methods if args.force or not os.path.exists(jf(m))]
        if not todo:
            print('skip (done):', stem, flush=True)
            continue
        if ref is not None and stem not in ref:
            print('skip (not in the reference table):', stem, flush=True)
            continue
        os.makedirs(sysdir, exist_ok=True)
        logf = os.path.join(sysdir, '%s.log' % stem)
        charge, mult, _ = parse_xyz(paths[0])
        mol = None
        for method in todo:
            t0 = time.time()
            extra, st, start = {}, dict(SETTINGS), None
            # ---- where to start from ----
            if ref is not None:
                e = ref[stem][method]
                st = {k: e[k] for k in SETTINGS}
                extra = dict(ref_energy=e['energy'], ref_s2=e['s2'], ref_source=e['source'])
                if e['init_guess'] == 'rhfchk':
                    own = load_json(jf('rhf'))
                    if not (own and own.get('converged') and os.path.exists(chk('rhf'))):
                        print('skip (needs the converged RHF/ROHF of this run first): %s %s' % (stem, method), flush=True)
                        continue
                    start = ('rhfchk', chk('rhf'))
                    extra.update(init_rhf_energy=own['energy'])
                else:
                    start = ('guess', e['init_guess'])
            else:
                pdir = os.path.join(parent, stem)
                prec = load_json(jf(method, pdir))
                if not (prec and prec.get('converged') and os.path.exists(chk(method, pdir))):
                    print('skip (no converged %s result/chkfile in %s): %s' % (method, parent, stem), flush=True)
                    continue
                start = ('project', chk(method, pdir))
                extra = dict(parent_energy=prec['energy'], parent_s2=prec.get('s2'), parent_basis=prec.get('basis'))
            # ---- run ----
            try:
                if mol is None:
                    mol = build_mol(paths[0], basis, logf, args.memory)   # one log per system
                rec = run_one(mol, method, basis, chk(method), logf, st, args.conv_tol, start,
                              external=(args.external == 'on'))
            except Exception as ex:
                rec = {'method': method, 'converged': False, 'energy': None,
                       'internal_stable': None, 'note': ['ERROR: %r' % ex]}
                traceback.print_exc()
            if rec.get('energy') is not None and 'ref_energy' in extra:
                d = (rec['energy'] - extra['ref_energy']) * 1e3
                extra['dE_ref_mEh'] = d
                if abs(d) > E_TOL * 1e3:
                    rec['note'].append('%+.4f mEh relative to the reference energy' % d)
            rec.update(extra)
            rec.update(name=stem, xyz=paths, charge=charge, mult=mult,
                       subdirs=[os.path.basename(os.path.dirname(p)) for p in paths],
                       chkfile=chk(method), logfile=logf, basis=basis, x2c=True,
                       start='%s:%s' % start, conv_tol=args.conv_tol, time_s=round(time.time() - t0, 1), **st)
            with open(jf(method), 'w') as f:
                json.dump(rec, f, indent=1)
            print('%-50s %-4s E=%-18s conv=%-5s by=%-6s stab_i=%-5s <S^2>=%-8s %6.0fs %s' % (
                stem[:50], method, rec.get('energy'), rec['converged'], rec.get('converged_by'),
                rec.get('internal_stable'), None if rec.get('s2') is None else round(rec['s2'], 4),
                rec['time_s'], '; '.join(rec['note'])), flush=True)
        if mol is not None:
            mol.stdout.close()
