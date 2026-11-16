#!/usr/bin/env python
"""CCSD (sfX2C-1e / cc-pVDZ-DK) on top of the lowest SCF solutions, saving the amplitudes.

For every system the SCF orbitals are read from <scfdir>/<stem>/<stem>.{rhf,uhf}.chk
(default scfdir: ./results_newton, the lowest Newton solutions of rerun_newton.py):

    rhf, closed shell   RHF  orbitals -> RCCSD
    rhf, open shell     ROHF orbitals -> UCCSD on the ROHF determinant (ROHF-CCSD)
    uhf                 UHF  orbitals -> UCCSD

The SCF is not rerun; the energy of the determinant is recomputed and compared with the
SCF energy as a check.  Frozen core (--frozen): 'chemcore' (default; pyscf's chemical
core, e.g. 1s2s2p for Sc-Zn and Al-Cl, 1s for B-F and Mg, nothing for H, Be), 'none'
(all electrons), or a number of lowest orbitals.

Output, per system, in <outdir>/<stem>/   (default outdir: ./ccsd):
    <stem>.<method>.ccsd.h5     amplitudes and the orbitals they refer to (*.h5 is not tracked by git)
    <stem>.<method>.ccsd.json   energies, convergence, T1 diagnostic, largest amplitudes
    <stem>.ccsd.log             pyscf output

Datasets of the h5 file (o, v = active occupied / virtual orbitals, in the order of mo_coeff
after removing the frozen ones):
    RCCSD   t1 (o,v)   t2 (o,o,v,v)
    UCCSD   t1a (oa,va)  t1b (ob,vb)   t2aa (oa,oa,va,va)  t2ab (oa,ob,va,vb)  t2bb (ob,ob,vb,vb)
    both    mo_coeff, mo_occ (for UCCSD: [alpha, beta]), frozen (indices of the frozen orbitals),
            e_hf, e_corr, e_tot, converged
  read back with e.g.   t1 = h5py.File(f)['t1'][()]   or   pyscf.lib.chkfile.load(f, 't2ab')

  python run_ccsd.py                 # all systems (finished ones are skipped)
  python run_ccsd.py a.xyz b.xyz     # only these (stems or xyz files)
  python run_ccsd.py --methods uhf   # only one reference
  python run_ccsd.py --summary       # write <outdir>/summary.tsv and print the problems
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import h5py
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from pyscf import scf, cc, lib
from pyscf.data import elements
from pyscf.lib import logger
from scf_lowest import parse_xyz, build_mol, load_json, VERBOSE

BASIS = 'cc-pvdz-dk'
E_CHECK = 1e-6      # Eh; allowed difference between the recomputed and the stored SCF energy


def load_scf(mol, method, chk):
    """SCF object (x2c) with the orbitals of the chkfile; ROHF is returned as a UHF object
    holding the ROHF determinant.  Returns (mf, kind, stored SCF energy)."""
    mo = np.asarray(lib.chkfile.load(chk, 'scf/mo_coeff'))
    occ = np.asarray(lib.chkfile.load(chk, 'scf/mo_occ'))
    e_scf = float(lib.chkfile.load(chk, 'scf/e_tot'))
    if method == 'uhf':
        if mo.ndim != 3:
            raise ValueError('%s is not a UHF chkfile' % chk)
        mf, kind = scf.UHF(mol).x2c(), 'UCCSD'
    elif mol.spin == 0:
        if mo.ndim != 2:
            raise ValueError('%s is not an RHF chkfile' % chk)
        mf, kind = scf.RHF(mol).x2c(), 'RCCSD'
    else:
        if mo.ndim != 2:
            raise ValueError('%s is not an ROHF chkfile' % chk)
        mf, kind = scf.UHF(mol).x2c(), 'ROHF-UCCSD'
        mo = np.array([mo, mo])
        occ = np.array([(occ > 0).astype(float), (occ > 1).astype(float)])
    mf.mo_coeff, mf.mo_occ = mo, occ
    dm = mf.make_rdm1(mo, occ)
    fock = mf.get_fock(dm=dm)
    mf.e_tot = float(mf.energy_tot(dm=dm))
    # orbital energies = diagonal of the Fock matrix (only used for the MP2 start and denominators)
    if mo.ndim == 3:
        mf.mo_energy = np.array([np.einsum('pi,pq,qi->i', mo[s], fock[s], mo[s]) for s in (0, 1)])
    else:
        mf.mo_energy = np.einsum('pi,pq,qi->i', mo, fock, mo)
    mf.converged = True
    return mf, kind, e_scf


def n_frozen(mol, arg):
    if arg == 'none':
        return 0
    if arg == 'chemcore':
        return int(elements.chemcore(mol))
    return int(arg)


def run_one(mol, method, chk, h5f, a):
    log = logger.new_logger(mol, VERBOSE)
    mf, kind, e_scf = load_scf(mol, method, chk)
    nfrz = n_frozen(mol, a.frozen)
    log.note('\n' + '=' * 70 + '\n  %s on %s   charge=%d  2S=%d  basis=%s  sfX2C-1e   frozen orbitals: %d\n'
             % (kind, os.path.basename(chk), mol.charge, mol.spin, BASIS, nfrz) + '=' * 70)
    rec = {'method': method, 'cc_type': kind, 'converged': False, 'n_frozen': nfrz, 'note': [],
           'e_scf_chk': e_scf, 'e_hf': mf.e_tot, 'e_corr': None, 'e_tot': None, 'nao': mol.nao, 'nelec': mol.nelec}
    if abs(mf.e_tot - e_scf) > E_CHECK:
        rec['note'].append('ERROR: recomputed SCF energy %.10f differs from the chkfile %.10f' % (mf.e_tot, e_scf))
        return rec
    if method == 'uhf' or mol.spin > 0:
        rec['s2_ref'] = float(mf.spin_square()[0])

    mycc = cc.CCSD(mf, frozen=nfrz if nfrz else None)      # RCCSD for RHF, UCCSD for UHF objects
    mycc.max_cycle = a.max_cycle
    mycc.conv_tol = a.conv_tol
    mycc.conv_tol_normt = a.conv_tol_normt
    mycc.diis_space = a.diis_space
    mycc.level_shift = a.level_shift
    mycc.max_memory = mol.max_memory
    mycc.kernel()
    if not mycc.converged:
        rec['note'].append('CCSD NOT CONVERGED in %d cycles (amplitudes of the last cycle saved)' % a.max_cycle)

    t1, t2 = mycc.t1, mycc.t2
    frozen = np.arange(nfrz)
    data = dict(mo_coeff=np.asarray(mf.mo_coeff), mo_occ=np.asarray(mf.mo_occ), frozen=frozen,
                e_hf=mf.e_tot, e_corr=float(mycc.e_corr), e_tot=float(mycc.e_tot), converged=bool(mycc.converged))
    if kind == 'RCCSD':
        data.update(t1=t1, t2=t2)
        amps1, amps2 = [t1], [t2]
        nocc_act = (mol.nelectron // 2 - nfrz,)
    else:
        data.update(t1a=t1[0], t1b=t1[1], t2aa=t2[0], t2ab=t2[1], t2bb=t2[2])
        amps1, amps2 = list(t1), list(t2)
        nocc_act = (mol.nelec[0] - nfrz, mol.nelec[1] - nfrz)
    tmp = h5f + '.tmp'
    with h5py.File(tmp, 'w') as f:
        for k, v in data.items():
            f[k] = v
        f.attrs.update(cc_type=kind, basis=BASIS, x2c='sfX2C-1e', scf_chkfile=os.path.abspath(chk),
                       charge=mol.charge, spin=mol.spin)
    os.replace(tmp, h5f)

    nel = sum(nocc_act) * (2 if kind == 'RCCSD' else 1)
    t1norm2 = sum(float(np.vdot(x, x)) for x in amps1) * (2 if kind == 'RCCSD' else 1)
    rec.update(converged=bool(mycc.converged), e_corr=float(mycc.e_corr), e_tot=float(mycc.e_tot),
               e_mp2=float(getattr(mycc, 'emp2', np.nan)), cycles=int(getattr(mycc, 'cycles', -1)),
               t1_diag=float(np.sqrt(t1norm2 / nel)) if nel else None,
               max_t1=max(float(abs(x).max()) if x.size else 0. for x in amps1),
               max_t2=max(float(abs(x).max()) if x.size else 0. for x in amps2),
               nocc_active=list(nocc_act), nmo=int(np.asarray(mf.mo_coeff).shape[-1]), amplitudes=h5f)
    log.note('RESULT %s  E_HF=%.10f  E_corr=%.10f  E_tot=%.10f  converged=%s  T1 diag=%.4f  max|t2|=%.4f',
             kind, mf.e_tot, rec['e_corr'], rec['e_tot'], rec['converged'], rec['t1_diag'] or 0, rec['max_t2'])
    return rec


def summarize(outdir, expected, methods):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, '*', '*.ccsd.json')))]
    cols = ['name', 'charge', 'mult', 'method', 'cc_type', 'nao', 'n_frozen', 'e_hf', 'e_corr', 'e_tot',
            'converged', 'cycles', 't1_diag', 'max_t1', 'max_t2', 's2_ref', 'scf_guess', 'time_s', 'note']

    def fmt(v):
        if isinstance(v, float): return '%.10f' % v
        if isinstance(v, list): return ';'.join(map(str, v))
        return '' if v is None else str(v)
    with open(os.path.join(outdir, 'summary.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in recs:
            f.write('\t'.join(fmt(r.get(c)) for c in cols) + '\n')
    have = {(r['name'], r['method']): r for r in recs}
    prob = ['%s  %s  MISSING' % (s, m) for s in expected for m in methods if (s, m) not in have]
    for r in recs:
        why = [n for n in r.get('note', [])]
        if r.get('max_t2') is not None and r['max_t2'] > 0.15:
            why.append('large amplitudes: max|t2| = %.3f' % r['max_t2'])
        if why:
            prob.append('%s  %s  %s' % (r['name'], r['method'], '; '.join(why)))
    print('%d records (%d expected), %d converged -> %s' % (
        len(recs), len(expected) * len(methods), sum(1 for r in recs if r.get('converged')),
        os.path.join(outdir, 'summary.tsv')))
    for p in prob:
        print('  ', p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('systems', nargs='*', help='stems or xyz files (default: all under --xyzdir)')
    ap.add_argument('--xyzdir', default=os.path.join(here, '..', 'xyzfiles'))
    ap.add_argument('--scfdir', default=os.path.join(here, 'results_newton'),
                    help='folder with <stem>/<stem>.{rhf,uhf}.chk (default: %(default)s)')
    ap.add_argument('--outdir', default=os.path.join(here, 'ccsd'))
    ap.add_argument('--methods', default='rhf,uhf', help='SCF references (default: %(default)s)')
    ap.add_argument('--frozen', default='chemcore', help="'chemcore' (default), 'none', or a number of orbitals")
    ap.add_argument('--max-cycle', type=int, default=200)
    ap.add_argument('--conv-tol', type=float, default=1e-7, help='CCSD energy convergence (Eh)')
    ap.add_argument('--conv-tol-normt', type=float, default=1e-6, help='CCSD amplitude convergence')
    ap.add_argument('--diis-space', type=int, default=10)
    ap.add_argument('--level-shift', type=float, default=0.0, help='CCSD level shift (helps hard cases)')
    ap.add_argument('--memory', type=int, default=4000, help='MB per calculation')
    ap.add_argument('--force', action='store_true', help='rerun even if the json exists')
    ap.add_argument('--summary', action='store_true', help='only write summary.tsv and print the problems')
    a = ap.parse_args()

    only = {os.path.splitext(os.path.basename(s))[0] for s in a.systems}
    uniq = {}       # identical files appear in several subfolders
    for f in sorted(glob.glob(os.path.join(a.xyzdir, '*', '*.xyz'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not only or stem in only:
            uniq.setdefault(stem, []).append(os.path.abspath(f))
    methods = a.methods.split(',')
    if a.summary:
        summarize(a.outdir, sorted(uniq), methods)
        return
    os.makedirs(a.outdir, exist_ok=True)
    print('threads:', lib.num_threads(), ' systems:', len(uniq), ' frozen:', a.frozen, flush=True)

    for stem, paths in uniq.items():
        sysdir = os.path.join(a.outdir, stem)
        jf = lambda m: os.path.join(sysdir, '%s.%s.ccsd.json' % (stem, m))
        todo = [m for m in methods if a.force or not os.path.exists(jf(m))]
        if not todo:
            print('skip (done):', stem, flush=True)
            continue
        os.makedirs(sysdir, exist_ok=True)
        logf = os.path.join(sysdir, '%s.ccsd.log' % stem)
        mol = None
        for method in todo:
            t0 = time.time()
            chk = os.path.join(a.scfdir, stem, '%s.%s.chk' % (stem, method))
            if not os.path.exists(chk):
                print('skip (no SCF chkfile %s): %s %s' % (chk, stem, method), flush=True)
                continue
            srec = load_json(os.path.join(a.scfdir, stem, '%s.%s.json' % (stem, method))) or {}
            try:
                if mol is None:
                    mol = build_mol(paths[0], BASIS, logf, a.memory)     # one log per system
                rec = run_one(mol, method, chk, os.path.join(sysdir, '%s.%s.ccsd.h5' % (stem, method)), a)
            except Exception as ex:
                traceback.print_exc()
                rec = {'method': method, 'converged': False, 'e_corr': None, 'e_tot': None,
                       'note': ['ERROR: %r' % ex]}
            charge, mult, _ = parse_xyz(paths[0])
            rec.update(name=stem, xyz=paths, charge=charge, mult=mult, basis=BASIS, x2c=True,
                       scf_chkfile=chk, scf_guess=srec.get('init_guess'), scf_energy=srec.get('energy'),
                       frozen=a.frozen, conv_tol=a.conv_tol, conv_tol_normt=a.conv_tol_normt,
                       max_cycle=a.max_cycle, level_shift=a.level_shift, logfile=logf,
                       time_s=round(time.time() - t0, 1))
            with open(jf(method), 'w') as f:
                json.dump(rec, f, indent=1)
            f_ = lambda v, p='%.8f': 'None' if v is None else p % v
            print('%-40s %-4s %-10s E_corr=%-13s E_tot=%-17s conv=%-5s T1=%-7s max|t2|=%-7s %6.0fs %s' % (
                stem[:40], method, rec.get('cc_type'), f_(rec.get('e_corr')), f_(rec.get('e_tot')),
                rec['converged'], f_(rec.get('t1_diag'), '%.4f'), f_(rec.get('max_t2'), '%.4f'),
                rec['time_s'], '; '.join(rec['note'])), flush=True)
        if mol is not None:
            mol.stdout.close()


if __name__ == '__main__':
    main()
