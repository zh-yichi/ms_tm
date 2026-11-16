#!/usr/bin/env python
"""Second-order (Newton) SCF search over several initial guesses, for ALL systems
(sfX2C-1e / cc-pVDZ-DK).  No DIIS anywhere.

For every system and method (RHF/ROHF, UHF), one calculation per initial guess (--guesses):
    minao, atom, huckel, ...   pyscf initial guesses
    rhfchk  (UHF only)         RHF/ROHF orbitals of the same system (the lowest one found
                               by this script, else the one in ./results)
    current                    orbitals of the present ./results solution (if there is one)
Each calculation:
    Newton SCF -> internal stability analysis -> if unstable, step along the lowest
    Hessian eigenvector in BOTH directions (+v and -v), re-converge both with Newton
    and keep the lower one -> repeat until stable.

Output in results_newton/<stem>/ :
    <stem>.<method>.<guess>.chk   orbitals of EVERY guess (all are kept)
    <stem>.<method>.chk           copy of the lowest one (stable solutions are preferred);
                                  same name as in ./results, so tz can start from this folder
    <stem>.<method>.json          record of the lowest one + energy, <S^2>, chkfile of every guess
    <stem>.newton.log             pyscf output of all calculations

  python rerun_newton.py                  # all systems (finished ones are skipped)
  python rerun_newton.py a.xyz b.xyz      # only these systems (stems or xyz files)
  python rerun_newton.py --above-only     # only records of ./results above the reference energy
  python rerun_newton.py --list           # only print "<xyz file> <methods>" of what would run
  python rerun_newton.py --replace        # also copy a solution that is lower than the
                                          # present one over ./results/<stem>/ (json + chk)
  python rerun_newton.py --summary        # table of what was found
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from pyscf import scf, lib
from pyscf.lib import logger
from scf_lowest import (parse_xyz, build_mol, _stability_with_eigs, dm_from_rhf_chk,
                        load_json, VERBOSE, E_TOL)

BASIS = 'cc-pvdz-dk'
TIE = 1e-7      # Eh; energies closer than this are the same solution


def new_mf(mol, method):
    return (scf.RHF(mol) if method == 'rhf' else scf.UHF(mol)).x2c()   # RHF -> ROHF if mol.spin > 0


def newton(mf, chk, conv_tol, max_cycle, dm0=None, mo=None, occ=None):
    """Newton SCF from a density matrix or from orbitals.  Returns a state dict."""
    nmf = mf.newton()
    nmf.chkfile = chk
    nmf.max_cycle = max_cycle
    nmf.conv_tol = conv_tol
    if mo is not None:
        nmf.kernel(np.asarray(mo), np.asarray(occ))
    else:
        nmf.kernel(dm0=dm0)
    return dict(mo_coeff=nmf.mo_coeff, mo_energy=nmf.mo_energy, mo_occ=nmf.mo_occ,
                e_tot=float(nmf.e_tot), converged=bool(nmf.converged))


def set_state(mf, s):
    mf.mo_coeff, mf.mo_energy, mf.mo_occ = s['mo_coeff'], s['mo_energy'], s['mo_occ']
    mf.e_tot, mf.converged = s['e_tot'], s['converged']
    return mf


def opposite_step(mo0, mo1, ovlp):
    """mo1 = mo0.U (pyscf's step along +v).  Returns mo0.U^T, the step along -v."""
    mo0, mo1 = np.asarray(mo0), np.asarray(mo1)
    if mo0.ndim == 3:
        return np.array([opposite_step(a, b, ovlp) for a, b in zip(mo0, mo1)])
    u = mo0.T.dot(ovlp).dot(mo1)
    return mo0.dot(u.T)


def start_density(mol, mf, method, guess, rhf_chk, cur_chk):
    """Initial density matrix for this guess, or None if it is not available."""
    if guess == 'rhfchk':
        if method != 'uhf' or not (rhf_chk and os.path.exists(rhf_chk)):
            return None
        dm0 = dm_from_rhf_chk(rhf_chk)
    elif guess == 'current':
        if not (cur_chk and os.path.exists(cur_chk)):
            return None
        return mf.from_chk(cur_chk)
    else:
        dm0 = np.asarray(mf.get_init_guess(key=guess))
    if method == 'uhf' and mol.spin == 0:
        # break alpha/beta symmetry so that singlets can relax to a spin-polarized solution
        dm0 = np.array([dm0[0] * 1.05, dm0[1] * 0.95])
    return dm0


def run_guess(mol, method, guess, dm0, chk, logfile, a):
    """Newton SCF + stability following (both directions) from one initial density."""
    log = logger.new_logger(mol, VERBOSE)
    log.note('\n' + '=' * 70 + '\n  %s   %s   charge=%d  2S=%d  basis=%s  sfX2C-1e   NEWTON   guess: %s\n'
             % (method.upper(), os.path.basename(chk), mol.charge, mol.spin, BASIS, guess) + '=' * 70)
    is_rohf = (method == 'rhf' and mol.spin > 0)
    external = a.external == 'on' and not is_rohf
    rec = {'guess': guess, 'converged': False, 'internal_stable': None, 'external_stable': None,
           'stab_rounds': 0, 'stab_eigs': [], 'energy': None, 's2': None, 'note': []}
    mf = new_mf(mol, method)
    mf.chkfile = chk
    mf.conv_tol = a.conv_tol
    ovlp = mf.get_ovlp()
    state = newton(mf, chk, a.conv_tol, a.newton_max_cycle, dm0=dm0)
    set_state(mf, state)
    if not state['converged']:
        rec['note'].append('Newton SCF NOT CONVERGED')

    stable_i = stable_e = None
    for it in range(a.max_stab + 1):
        if not mf.converged:
            break
        log.note('--- stability analysis, round %d ---', it)
        mo_i, mo_e, stable_i, stable_e, eigs = _stability_with_eigs(mf, logfile, True, external)
        if not external:
            stable_e = None
        rec['stab_eigs'].append(eigs)
        vals = [v for k, v in eigs.items() if k.endswith('_internal') and v is not None]
        if (not stable_i) and vals and min(vals) > -a.stab_tol:
            rec['note'].append('internal eig %.1e within stab_tol, treated as stable' % min(vals))
            stable_i = True
        if stable_i or it == a.max_stab:
            break
        rec['stab_rounds'] += 1
        e_old, tried = state['e_tot'], []
        for sign, mo in (('+v', mo_i), ('-v', opposite_step(state['mo_coeff'], mo_i, ovlp))):
            log.note('internal instability -> step along %s and Newton SCF', sign)
            try:
                s = newton(mf, chk, a.conv_tol, a.newton_max_cycle, mo=mo, occ=state['mo_occ'])
            except Exception as ex:
                log.note('step %s failed: %r', sign, ex)
                continue
            log.note('step %s: E = %.10f  converged = %s', sign, s['e_tot'], s['converged'])
            if s['converged']:
                tried.append(s)
        set_state(mf, state)
        if not tried:
            rec['note'].append('Newton SCF NOT CONVERGED after following the instability; kept the unstable solution')
            break
        best = min(tried, key=lambda s: s['e_tot'])
        if best['e_tot'] > e_old - 1e-8:
            log.note('neither direction lowered the energy (%.10f): zero mode, stop', e_old)
            rec['note'].append('instability following did not lower E; zero mode, treated as stable')
            stable_i = True
            break
        state = best
        set_state(mf, state)

    rec['converged'] = bool(mf.converged)
    rec['internal_stable'] = None if stable_i is None else bool(stable_i)
    rec['external_stable'] = None if stable_e is None else bool(stable_e)
    if rec['converged'] and rec['internal_stable'] is False:
        rec['note'].append('still internally UNSTABLE')
    if rec['converged'] and rec['external_stable'] is False:
        rec['note'].append('external instability (%s)' % ('RHF->UHF or real->complex' if method == 'rhf'
                                                          else 'UHF->GHF or real->complex'))
    if rec['converged']:
        rec['energy'] = float(mf.e_tot)
        if method == 'uhf' or is_rohf:
            rec['s2'] = float(mf.spin_square()[0])
        mf.dump_chk({'e_tot': mf.e_tot, 'mo_energy': mf.mo_energy,
                     'mo_coeff': mf.mo_coeff, 'mo_occ': mf.mo_occ})
    rec['scf_class'] = type(mf).__name__
    log.note('RESULT %s NEWTON guess=%s  E=%s  converged=%s  internal_stable=%s  <S^2>=%s',
             method.upper(), guess, rec['energy'], rec['converged'], rec['internal_stable'], rec['s2'])
    return rec


def find_targets(a, only, methods):
    """{stem: (xyz paths, {method: {'old': present record or None, 'ref': reference entry or None}})}"""
    ref = load_json(a.reference) or {}
    uniq = {}       # identical files appear in several subfolders
    for f in sorted(glob.glob(os.path.join(a.xyzdir, '*', '*.xyz'))):
        uniq.setdefault(os.path.splitext(os.path.basename(f))[0], []).append(os.path.abspath(f))
    out = {}
    for stem, paths in uniq.items():
        if only and stem not in only:
            continue
        todo = {}
        for m in methods:
            old = load_json(os.path.join(a.resdir, stem, '%s.%s.json' % (stem, m)))
            if a.above_only and not only:
                d = (old or {}).get('dE_ref_mEh')
                if d is None or d <= a.tol_mEh:
                    continue
            todo[m] = {'old': old or {}, 'ref': ref.get(stem, {}).get(m) or {}}
        if todo:
            out[stem] = (paths, todo)
    return out


def summary(outdir):
    f3 = lambda v: 'n/a' if v is None else '%.3f' % v
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, '*', '*.json')))]
    print('%-22s %-4s %-8s %16s %10s %10s %7s %6s  %s' % (
        'system', 'meth', 'guess', 'E', 'dE_ref/mEh', 'dE_old/mEh', '<S^2>', 'stable', 'all guesses: E - E_lowest (mEh)'))
    for r in rows:
        f = lambda v, p='%.4f': '' if v is None else p % v
        cand = '  '.join('%s:%s' % (c['guess'], 'fail' if c['energy'] is None else '%.3f' % ((c['energy'] - r['energy']) * 1e3))
                         for c in r['candidates']) if r.get('energy') is not None else 'nothing converged'
        print('%-22s %-4s %-8s %16s %10s %10s %7s %6s  %s' % (
            r['name'][:22], r['method'], r.get('init_guess'), f(r.get('energy'), '%.8f'), f(r.get('dE_ref_mEh')),
            f(r.get('dE_prev_mEh')), f(r.get('s2'), '%.3f'), r.get('internal_stable'), cand))
    d = [r['dE_ref_mEh'] for r in rows if r.get('dE_ref_mEh') is not None]
    t = E_TOL * 1e3
    print('%d records: %d not converged, vs the reference energy: %d same, %d below, %d above, %d without reference' % (
        len(rows), sum(1 for r in rows if r.get('energy') is None), sum(1 for x in d if abs(x) <= t),
        sum(1 for x in d if x < -t), sum(1 for x in d if x > t), len(rows) - len(d)))
    for r in rows:
        if r.get('dE_ref_mEh') is not None and abs(r['dE_ref_mEh']) > t:
            print('   %-50s %-4s %+10.4f mEh  <S^2> %s (reference %s)' % (r['name'][:50], r['method'], r['dE_ref_mEh'],
                  f3(r.get('s2')), f3(r.get('ref_s2'))))
    lo = {(r['name'], r['method']): r.get('energy') for r in rows}
    for (n_, m), e in sorted(lo.items()):
        er = lo.get((n_, 'rhf'))
        if m == 'uhf' and e is not None and er is not None and e > er + 1e-6:
            print('   %-50s UHF is %.4f mEh ABOVE RHF/ROHF' % (n_[:50], (e - er) * 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('systems', nargs='*', help='stems or xyz files (default: all under --xyzdir)')
    ap.add_argument('--above-only', action='store_true',
                    help='only the records of --resdir that are more than --tol-mEh above the reference')
    ap.add_argument('--reference', default=os.path.join(here, 'reference.json'))
    ap.add_argument('--xyzdir', default=os.path.join(here, '..', 'xyzfiles'))
    ap.add_argument('--resdir', default=os.path.join(here, 'results'), help='present dz results')
    ap.add_argument('--outdir', default=os.path.join(here, 'results_newton'))
    ap.add_argument('--methods', default='rhf,uhf')
    ap.add_argument('--guesses', default='minao,atom,huckel,rhfchk,current')
    ap.add_argument('--tol-mEh', type=float, default=E_TOL * 1e3,
                    help='threshold of --above-only (default %(default)s)')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--newton-max-cycle', type=int, default=200)
    ap.add_argument('--max-stab', type=int, default=15)
    ap.add_argument('--stab-tol', type=float, default=1e-4)
    ap.add_argument('--external', choices=['on', 'off'], default='off',
                    help='also run the external stability analysis (recorded only)')
    ap.add_argument('--memory', type=int, default=4000, help='MB per calculation')
    ap.add_argument('--force', action='store_true', help='rerun even if the json exists')
    ap.add_argument('--replace', action='store_true',
                    help='copy solutions lower than the present one over <resdir>/<stem>/ (json and chk)')
    ap.add_argument('--list', action='store_true', help='only print "<xyz file> <methods>" of what would run')
    ap.add_argument('--summary', action='store_true', help='only print the table of results_newton')
    a = ap.parse_args()
    if a.summary:
        summary(a.outdir)
        return

    only = {os.path.splitext(os.path.basename(s))[0] for s in a.systems}
    targets = find_targets(a, only, a.methods.split(','))
    if a.list:
        for s, (paths, todo) in targets.items():
            print(paths[0], ','.join(sorted(todo)))
        return
    print('threads:', lib.num_threads(), ' targets:', sum(len(v[1]) for v in targets.values()),
          'records in', len(targets), 'systems', flush=True)

    for stem, (paths, todo) in targets.items():
        sysdir = os.path.join(a.outdir, stem)
        jf = lambda m, d=sysdir: os.path.join(d, '%s.%s.json' % (stem, m))
        chk = lambda m, d=sysdir: os.path.join(d, '%s.%s.chk' % (stem, m))
        olddir = os.path.join(a.resdir, stem)
        gchk = lambda m, g, d=sysdir: os.path.join(d, '%s.%s.%s.chk' % (stem, m, g))
        # done = json written by this version (it lists the chkfile of every guess)
        done = lambda m: (load_json(jf(m)) or {}).get('all_chk_saved') is True
        methods = [m for m in ('rhf', 'uhf') if m in todo and (a.force or not done(m))]
        if not methods:
            print('skip (done):', stem, flush=True)
            continue
        os.makedirs(sysdir, exist_ok=True)
        logf = os.path.join(sysdir, '%s.newton.log' % stem)
        mol = build_mol(paths[0], BASIS, logf, a.memory)
        for method in methods:          # rhf first: its orbitals are the 'rhfchk' guess of uhf
            t0, old, ref = time.time(), todo[method]['old'], todo[method]['ref']
            own_rhf = load_json(jf('rhf'))
            rhf_chk = chk('rhf') if (own_rhf and own_rhf.get('converged')) else chk('rhf', olddir)
            cands, best = [], None
            for guess in a.guesses.split(','):
                if guess == 'rhfchk' and method == 'rhf':
                    continue
                tmp = gchk(method, guess)
                try:
                    dm0 = start_density(mol, new_mf(mol, method), method, guess, rhf_chk, chk(method, olddir))
                    if dm0 is None:
                        print('  %-22s %-4s %-8s not available, skipped' % (stem[:22], method, guess), flush=True)
                        continue
                    rec = run_guess(mol, method, guess, dm0, tmp, logf, a)
                except Exception as ex:
                    traceback.print_exc()
                    rec = {'guess': guess, 'converged': False, 'energy': None, 'internal_stable': None,
                           's2': None, 'note': ['ERROR: %r' % ex]}
                rec['chkfile'] = tmp if os.path.exists(tmp) else None
                cands.append(rec)
                print('  %-22s %-4s %-8s E=%-18s conv=%-5s stab_i=%-5s <S^2>=%-7s %s' % (
                    stem[:22], method, guess, rec['energy'], rec['converged'], rec['internal_stable'],
                    None if rec.get('s2') is None else round(rec['s2'], 4), '; '.join(rec['note'])), flush=True)
                # lowest converged solution; an internally stable one beats an unstable one
                key = lambda r: (r['internal_stable'] is not True, r['energy'])
                if rec['energy'] is not None and (best is None or key(rec) < key(best)
                                                  and not (key(rec)[0] == key(best)[0]
                                                           and rec['energy'] > best['energy'] - TIE)):
                    best = rec
            out = dict(best) if best else {'converged': False, 'energy': None, 'internal_stable': None,
                                           'note': ['no guess converged']}
            out['note'] = list(out.get('note', []))
            out.update(method=method, name=stem, init_guess=out.pop('guess', None), solver='newton',
                       all_chk_saved=True, lowest_chkfile=(best or {}).get('chkfile'),
                       start='newton:%s' % (best['guess'] if best else None),
                       candidates=[{k: c.get(k) for k in ('guess', 'energy', 's2', 'converged', 'internal_stable',
                                                          'stab_rounds', 'note', 'chkfile')} for c in cands],
                       ref_energy=ref.get('energy'), ref_s2=ref.get('s2'), prev_energy=old.get('energy'),
                       prev_s2=old.get('s2'), xyz=paths, charge=mol.charge, mult=mol.spin + 1,
                       subdirs=[os.path.basename(os.path.dirname(p)) for p in paths], basis=BASIS, x2c=True, nao=mol.nao, nelec=mol.nelec,
                       chkfile=chk(method), logfile=logf, conv_tol=a.conv_tol,
                       newton_max_cycle=a.newton_max_cycle, max_stab=a.max_stab, stab_tol=a.stab_tol,
                       time_s=round(time.time() - t0, 1))
            if best:
                shutil.copy(best['chkfile'], chk(method))
                for k, e0 in (('dE_ref_mEh', ref.get('energy')), ('dE_prev_mEh', old.get('energy'))):
                    out[k] = None if e0 is None else (best['energy'] - e0) * 1e3
                if out['dE_ref_mEh'] is not None and abs(out['dE_ref_mEh']) > E_TOL * 1e3:
                    out['note'].append('%+.4f mEh relative to the reference energy' % out['dE_ref_mEh'])
            with open(jf(method), 'w') as f:
                json.dump(out, f, indent=1)
            fm = lambda v: 'n/a' if v is None else '%+.4f' % v
            print('%-22s %-4s LOWEST: %s  E=%s  vs reference %s mEh, vs present dz result %s mEh  (%.0fs)' % (
                stem[:22], method, out['init_guess'], out['energy'], fm(out.get('dE_ref_mEh')),
                fm(out.get('dE_prev_mEh')), out['time_s']), flush=True)
            if a.replace and best and out.get('dE_prev_mEh') is not None and out['dE_prev_mEh'] < -E_TOL * 1e3 \
                    and best['internal_stable']:
                rep = dict(out, chkfile=chk(method, olddir))
                shutil.copy(chk(method), chk(method, olddir))
                with open(jf(method, olddir), 'w') as f:
                    json.dump(rep, f, indent=1)
                print('   -> replaced %s' % jf(method, olddir), flush=True)
        mol.stdout.close()


if __name__ == '__main__':
    main()
