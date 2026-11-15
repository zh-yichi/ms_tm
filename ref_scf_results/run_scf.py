#!/usr/bin/env python
"""
Run (R/RO)HF and UHF with sfX2C-1e scalar relativity and cc-pVDZ-DK on the
transition-metal complexes in ../xyzfiles.

Per system:
  <outdir>/<stem>/<stem>.log          pyscf output (verbose=4) for RHF and UHF
  <outdir>/<stem>/<stem>.rhf.chk      RHF/ROHF chkfile
  <outdir>/<stem>/<stem>.uhf.chk      UHF chkfile
  <outdir>/<stem>/<stem>.{rhf,uhf}.json   result record

Per method:
  * DIIS SCF (level_shift = 0.4, max_cycle = 200) from the --init-guess density
    (default minao)
  * if not converged -> second-order (Newton) SCF from the DIIS density
  * second-order (orbital-Hessian) stability analysis:
      internal  : RHF->RHF, ROHF->ROHF, UHF->UHF   (followed if unstable)
      external  : RHF->UHF / real->complex, UHF->GHF / real->complex (recorded only;
                  not available for ROHF in pyscf)
    If internally unstable, the lowest Hessian eigenvector is followed and the
    SCF re-converged, up to --max-stab rounds.  Lowest Hessian eigenvalues are
    stored in the json ('stab_eigs').

RHF on an open-shell system (multiplicity > 1) is ROHF (pyscf behaviour).

Usage:
  python run_scf.py                       # all xyz files under ../xyzfiles (deduplicated)
  python run_scf.py a.xyz b.xyz           # only these files
  python run_scf.py --methods rhf         # only one method
  python run_scf.py --init-guess atom --outdir results_atom   # other initial guess
  python run_scf.py --methods uhf --outdir results_rhfchk \\
      --rhf-chk-dirs results_minao results_atom results_huckel \\
      --guess-table guess_comparison.tsv   # UHF from the lowest RHF/ROHF chkfile; RHF is
                                           # recomputed with its best guess if the chk is missing
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

BASIS = 'cc-pvdz-dk'
VERBOSE = 4


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


def build_mol(path, logfile, memory):
    charge, mult, atoms = parse_xyz(path)
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = BASIS
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
    """Run mf.stability and also harvest the lowest Hessian eigenvalues that
    pyscf prints to the log (verbose>=4)."""
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


def find_lowest_rhf(stem, dirs, tie=1e-6):
    """Among <dir>/<stem>/<stem>.rhf.json for dir in dirs, pick the converged
    RHF/ROHF record with the lowest energy.  Returns (chkfile, record, why):
    chkfile is None if it is missing (a record within `tie` Eh of the lowest
    that has a chkfile is used instead when there is one); record is then the
    lowest record (or None if there is none) and `why` says what is missing."""
    cands = []
    for d in dirs:
        jf = os.path.join(d, stem, '%s.rhf.json' % stem)
        if not os.path.exists(jf):
            continue
        r = json.load(open(jf))
        if r.get('converged') and r.get('energy') is not None:
            cands.append((r['energy'], os.path.join(d, stem, '%s.rhf.chk' % stem), r))
    if not cands:
        return None, None, 'no converged rhf record in %s' % ' '.join(dirs)
    cands.sort(key=lambda c: c[0])
    for e, chk, r in cands:
        if e - cands[0][0] < tie and os.path.exists(chk):
            return chk, r, ''
    return None, cands[0][2], 'chkfile of the lowest rhf solution is missing: %s' % cands[0][1]


def read_guess_table(path, default='minao'):
    """{(system, 4-char hash): best RHF initial guess} from guess_comparison.tsv
    (whitespace-separated, columns system / hash / best_RHF).  'All' -> default."""
    with open(path) as f:
        lines = [l.split() for l in f if l.strip()]
    hdr = lines[0]
    i_s, i_h, i_b = hdr.index('system'), hdr.index('hash'), hdr.index('best_RHF')
    return {(l[i_s], l[i_h]): (default if l[i_b].lower() == 'all' else l[i_b]) for l in lines[1:]}


def dm_from_rhf_chk(chk):
    """UHF (alpha, beta) density matrices from the orbitals of an RHF/ROHF chkfile."""
    mo = np.asarray(lib.chkfile.load(chk, 'scf/mo_coeff'))
    occ = np.asarray(lib.chkfile.load(chk, 'scf/mo_occ'))
    if mo.ndim != 2:
        raise ValueError('%s is not an RHF/ROHF chkfile' % chk)
    dma = (mo[:, occ > 0]).dot(mo[:, occ > 0].T)     # doubly + singly occupied
    dmb = (mo[:, occ > 1]).dot(mo[:, occ > 1].T)     # doubly occupied
    return np.array([dma, dmb])


def run_one(mol, method, chkfile, logfile, args, init_chk=None):
    """Return (mf, record).  init_chk: RHF/ROHF chkfile used as the UHF starting point."""
    rec = {'method': method, 'converged': False, 'converged_by': None,
           'internal_stable': None, 'external_stable': None,
           'stab_rounds': 0, 'stab_eigs': [], 'energy': None, 's2': None,
           'note': []}
    log = logger.new_logger(mol, VERBOSE)
    log.note('\n' + '=' * 70 + '\n  %s   %s   charge=%d  2S=%d  basis=%s  sfX2C-1e\n' % (
        method.upper(), os.path.basename(chkfile), mol.charge, mol.spin, BASIS) + '=' * 70)

    if method == 'rhf':
        mf = scf.RHF(mol).x2c()      # ROHF automatically if mol.spin > 0
    elif method == 'uhf':
        mf = scf.UHF(mol).x2c()
    else:
        raise ValueError(method)
    is_rohf = (method == 'rhf' and mol.spin > 0)

    mf.chkfile = chkfile
    mf.level_shift = args.level_shift
    mf.max_cycle = args.max_cycle
    mf.conv_tol = args.conv_tol
    mf.diis_space = 12
    if init_chk is None:
        mf.init_guess = args.init_guess     # used by kernel() when dm0 is None (RHF/ROHF)

    dm0 = None
    if method == 'uhf':
        # break alpha/beta symmetry in the initial guess so that singlets can
        # relax to a spin-polarized solution if one exists
        if init_chk is not None:
            log.note('UHF initial guess from RHF/ROHF chkfile %s', init_chk)
            dm0 = dm_from_rhf_chk(init_chk)
        else:
            dm0 = np.asarray(mf.get_init_guess(key=args.init_guess))
        if mol.spin == 0:
            dm0 = np.array([dm0[0] * 1.05, dm0[1] * 0.95])

    def converge(mf_, dm):
        mf_.kernel(dm0=dm)
        if mf_.converged:
            return mf_, 'diis'
        log.note('DIIS SCF did not converge -> second-order (Newton) SCF')
        dm = mf_.make_rdm1()
        nmf = mf_.newton()
        nmf.chkfile = chkfile
        nmf.max_cycle = args.newton_max_cycle
        nmf.conv_tol = args.conv_tol
        nmf.kernel(dm0=dm)
        mf_.mo_coeff, mf_.mo_energy, mf_.mo_occ = nmf.mo_coeff, nmf.mo_energy, nmf.mo_occ
        mf_.e_tot, mf_.converged = nmf.e_tot, nmf.converged
        return mf_, ('newton' if nmf.converged else 'none')

    mf, how = converge(mf, dm0)
    rec['converged_by'] = how
    if not mf.converged:
        rec['note'].append('SCF NOT CONVERGED (diis + newton)')

    # ---- second-order stability analysis; follow internal instabilities ----
    stable_i = stable_e = None
    external = not is_rohf          # rohf_external is not implemented in pyscf
    for it in range(args.max_stab + 1):
        if not mf.converged:
            break
        log.note('--- stability analysis, round %d ---', it)
        mo_i, mo_e, stable_i, stable_e, eigs = _stability_with_eigs(
            mf, logfile, internal=True, external=external)
        rec['stab_eigs'].append(eigs)
        min_eig = min([v for k, v in eigs.items() if k.endswith('_internal') and v is not None]
                      or [None], key=lambda x: x if x is not None else 0)
        if (not stable_i) and min_eig is not None and min_eig > -args.stab_tol:
            log.note('lowest internal Hessian eigenvalue %.3e > -stab_tol (%.1e): treated as stable',
                     min_eig, args.stab_tol)
            rec['note'].append('internal eig %.1e within stab_tol, treated as stable' % min_eig)
            stable_i = True
        if stable_i or it == args.max_stab:
            break
        rec['stab_rounds'] += 1
        log.note('internal instability found -> following the lowest Hessian '
                 'eigenvector and re-converging')
        e_old = mf.e_tot
        dm = mf.make_rdm1(mo_i, mf.mo_occ)
        mf, how = converge(mf, dm)
        rec['converged_by'] = how
        if not mf.converged:
            rec['note'].append('SCF NOT CONVERGED after following instability')
        elif mf.e_tot > e_old - 1e-8:
            # following the "instability" did not lower the energy: a zero mode
            # (e.g. rotation among degenerate orbitals), not a real instability
            log.note('following the instability did not lower the energy (%.10f -> %.10f): '
                     'zero mode, stop', e_old, mf.e_tot)
            rec['note'].append('instability following did not lower E; zero mode, treated as stable')
            stable_i = True
            break

    rec['converged'] = bool(mf.converged)
    rec['internal_stable'] = None if stable_i is None else bool(stable_i)
    rec['external_stable'] = None if stable_e is None else bool(stable_e)
    if rec['converged'] and rec['internal_stable'] is False:
        rec['note'].append('still internally UNSTABLE after %d rounds' % args.max_stab)
    if rec['converged'] and rec['external_stable'] is False:
        rec['note'].append('external instability (%s)' %
                           ('RHF->UHF or real->complex' if method == 'rhf'
                            else 'UHF->GHF or real->complex'))
    if rec['converged']:
        rec['energy'] = float(mf.e_tot)
        if method == 'uhf' or is_rohf:
            rec['s2'] = float(mf.spin_square()[0])
        # make sure the chk holds the final orbitals
        mf.dump_chk({'e_tot': mf.e_tot, 'mo_energy': mf.mo_energy,
                     'mo_coeff': mf.mo_coeff, 'mo_occ': mf.mo_occ})
    log.note('RESULT %s  E=%s  converged=%s (%s)  internal_stable=%s  external_stable=%s  <S^2>=%s',
             method.upper(), rec['energy'], rec['converged'], rec['converged_by'],
             rec['internal_stable'], rec['external_stable'], rec['s2'])
    return mf, rec


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='xyz files (default: all under --xyzdir)')
    ap.add_argument('--xyzdir', default=os.path.join(here, '..', 'xyzfiles'))
    ap.add_argument('--outdir', default=os.path.join(here, 'results_minao'))
    ap.add_argument('--methods', default='rhf,uhf')
    ap.add_argument('--init-guess', default='minao',
                    help="pyscf initial guess: minao (default), atom, huckel, 1e, vsap, ...")
    ap.add_argument('--rhf-chk-dirs', nargs='+', metavar='DIR',
                    help='UHF only: start from the chkfile of the lowest RHF/ROHF solution '
                         'found in these result folders (overrides --init-guess); put another '
                         'option between this list and the xyz files')
    ap.add_argument('--guess-table', metavar='TSV',
                    help='with --rhf-chk-dirs: if the RHF chkfile is missing, recompute RHF/ROHF '
                         'with the best initial guess listed in this table (guess_comparison.tsv, '
                         'column best_RHF; "All" -> minao) and start the UHF from that')
    ap.add_argument('--level-shift', type=float, default=0.4)
    ap.add_argument('--max-cycle', type=int, default=200)
    ap.add_argument('--max-stab', type=int, default=3)
    ap.add_argument('--newton-max-cycle', type=int, default=60)
    ap.add_argument('--stab-tol', type=float, default=1e-5,
                    help='internal Hessian eigenvalue above -stab_tol counts as stable')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--memory', type=int, default=4000, help='MB per calculation')
    ap.add_argument('--force', action='store_true', help='rerun even if json exists')
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(args.xyzdir, '*', '*.xyz')))
    # de-duplicate by file name (identical files appear in several subfolders)
    uniq = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        uniq.setdefault(stem, []).append(os.path.abspath(f))
    methods = args.methods.split(',')
    if args.rhf_chk_dirs:
        if methods != ['uhf']:
            ap.error('--rhf-chk-dirs needs --methods uhf')
        args.init_guess = 'rhfchk'
    guess_table = read_guess_table(args.guess_table) if args.guess_table else {}
    os.makedirs(args.outdir, exist_ok=True)

    print('threads:', lib.num_threads(), ' systems:', len(uniq), flush=True)
    for stem, paths in uniq.items():
        sysdir = os.path.join(args.outdir, stem)
        init_chk = init_rec = rhf_guess = None
        if args.rhf_chk_dirs:
            init_chk, init_rec, why = find_lowest_rhf(stem, args.rhf_chk_dirs)
            if init_chk is None:
                # an RHF recomputed by an earlier run of this job?
                init_chk, own, _ = find_lowest_rhf(stem, [args.outdir])
                if init_chk is not None:
                    init_rec = own
            if init_chk is None:
                rhf_guess = guess_table.get((stem.split('-')[0], stem.split('-')[-1][:4]))
                if rhf_guess is None:
                    print('skip (%s): %s' % (why, stem), flush=True)
                    continue
        os.makedirs(sysdir, exist_ok=True)
        todo = [m for m in methods
                if args.force or not os.path.exists(os.path.join(sysdir, '%s.%s.json' % (stem, m)))]
        if not todo:
            print('skip (done):', stem, flush=True)
            continue
        logf = os.path.join(sysdir, '%s.log' % stem)
        charge, mult, _ = parse_xyz(paths[0])
        try:
            mol = build_mol(paths[0], logf, args.memory)   # one log per system
        except Exception as e:
            traceback.print_exc()
            for method in todo:
                rec = {'method': method, 'converged': False, 'energy': None,
                       'note': ['ERROR building mol: %r' % e], 'name': stem, 'xyz': paths,
                       'charge': charge, 'mult': mult}
                json.dump(rec, open(os.path.join(sysdir, '%s.%s.json' % (stem, method)), 'w'), indent=1)
            continue
        if rhf_guess is not None and todo:
            # RHF chkfile not available: recompute RHF/ROHF with the best guess of the table
            t0 = time.time()
            expected = init_rec['energy'] if init_rec else None
            rargs = argparse.Namespace(**vars(args))
            rargs.init_guess = rhf_guess
            rchk = os.path.join(sysdir, '%s.rhf.chk' % stem)
            try:
                mf, rec = run_one(mol, 'rhf', rchk, logf, rargs)
                rec.update(scf_class=type(mf).__name__, nao=mol.nao, nelec=mol.nelec)
            except Exception as e:
                rec = {'method': 'rhf', 'converged': False, 'energy': None,
                       'internal_stable': None, 'note': ['ERROR: %r' % e]}
                traceback.print_exc()
            rec['note'].insert(0, 'recomputed (chkfile missing) with init guess %s' % rhf_guess)
            if rec['converged'] and expected is not None and abs(rec['energy'] - expected) > 1e-6:
                rec['note'].append('differs from the lowest known RHF by %+.3f mEh'
                                   % ((rec['energy'] - expected) * 1e3))
            rec.update(name=stem, xyz=paths, charge=charge, mult=mult,
                       subdirs=[os.path.basename(os.path.dirname(p)) for p in paths],
                       chkfile=rchk, logfile=logf, basis=BASIS, x2c=True,
                       level_shift=args.level_shift, init_guess=rhf_guess,
                       max_cycle=args.max_cycle, newton_max_cycle=args.newton_max_cycle,
                       max_stab=args.max_stab, stab_tol=args.stab_tol,
                       lowest_known_rhf_energy=expected, time_s=round(time.time() - t0, 1))
            with open(os.path.join(sysdir, '%s.rhf.json' % stem), 'w') as f:
                json.dump(rec, f, indent=1)
            print('%-50s %-4s E=%-18s conv=%-5s by=%-6s stab_i=%-5s stab_e=%-5s %5.0fs %s' % (
                stem[:50], 'rhf', rec.get('energy'), rec['converged'],
                rec.get('converged_by'), rec.get('internal_stable'),
                rec.get('external_stable'), rec['time_s'], '; '.join(rec['note'])), flush=True)
            if not rec['converged']:
                print('skip (recomputed rhf did not converge): %s' % stem, flush=True)
                mol.stdout.close()
                continue
            init_chk, init_rec = rchk, rec
        for method in todo:
            t0 = time.time()
            chkf = os.path.join(sysdir, '%s.%s.chk' % (stem, method))
            try:
                mf, rec = run_one(mol, method, chkf, logf, args, init_chk)
                if init_chk is not None:
                    rec.update(init_chk=init_chk, init_rhf_energy=init_rec['energy'],
                               init_rhf_guess=init_rec.get('init_guess'))
                rec.update(scf_class=type(mf).__name__, nao=mol.nao, nelec=mol.nelec)
            except Exception as e:
                rec = {'method': method, 'converged': False, 'energy': None,
                       'internal_stable': None, 'note': ['ERROR: %r' % e]}
                traceback.print_exc()
            rec.update(name=stem, xyz=paths, charge=charge, mult=mult,
                       subdirs=[os.path.basename(os.path.dirname(p)) for p in paths],
                       chkfile=chkf, logfile=logf, basis=BASIS, x2c=True,
                       level_shift=args.level_shift, init_guess=args.init_guess,
                       max_cycle=args.max_cycle, newton_max_cycle=args.newton_max_cycle,
                       max_stab=args.max_stab, stab_tol=args.stab_tol, time_s=round(time.time() - t0, 1))
            with open(os.path.join(sysdir, '%s.%s.json' % (stem, method)), 'w') as f:
                json.dump(rec, f, indent=1)
            print('%-50s %-4s E=%-18s conv=%-5s by=%-6s stab_i=%-5s stab_e=%-5s %5.0fs %s' % (
                stem[:50], method, rec.get('energy'), rec['converged'],
                rec.get('converged_by'), rec.get('internal_stable'),
                rec.get('external_stable'), rec['time_s'], '; '.join(rec['note'])), flush=True)
        mol.stdout.close()


if __name__ == '__main__':
    main()
