#!/usr/bin/env python
"""Build dz/reference.json: for every system, the recipe that gave the lowest
RHF/ROHF and UHF solution (sfX2C-1e, cc-pVDZ-DK) in ../ref_scf_results.

Sources: results_minao, results_atom, results_huckel (three initial guesses) and
results_rhfchk (RHF/ROHF recomputed with its best guess; UHF started from it).
Only converged, internally stable records are used.

  RHF/ROHF  the results_rhfchk record (its guess, uniform settings) unless another
            source is lower by more than TIE.
  UHF       the lowest record; within TIE the first of atom, huckel, minao, rhfchk.
            'rhfchk' means: start from the RHF/ROHF orbitals of the same system.

Each entry stores the initial guess, the SCF settings of that run, and the
reference energy and <S^2>.   Usage: python make_reference.py
"""
import glob, json, os

here = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(here, '..', 'ref_scf_results')
TIE = 1e-6      # Eh
UNIFORM = dict(max_cycle=1000, newton_max_cycle=200, max_stab=15, stab_tol=1e-4)
ORIGINAL = dict(max_cycle=200, newton_max_cycle=60, max_stab=3, stab_tol=1e-5)   # first minao run
ORDER = {'rhf': ['rhfchk', 'atom', 'huckel', 'minao'], 'uhf': ['atom', 'huckel', 'minao', 'rhfchk']}

R = {}
for src in ('minao', 'atom', 'huckel', 'rhfchk'):
    for f in glob.glob(os.path.join(REF, 'results_%s' % src, '*', '*.json')):
        r = json.load(open(f))
        if r.get('converged') and r.get('energy') is not None and r.get('internal_stable') is True:
            R[(r['name'], r['method'], src)] = r


def settings(r, src):
    if r.get('max_cycle') is not None:
        s = {k: r[k] for k in UNIFORM}
    elif r.get('rerun_settings'):
        s = dict(r['rerun_settings'])
    else:
        s = dict(ORIGINAL)
    s['level_shift'] = r.get('level_shift', 0.4)
    return s


out = {}
for name in sorted({k[0] for k in R}):
    ent = {}
    for m in ('rhf', 'uhf'):
        cand = [(src, R[(name, m, src)]) for src in ORDER[m] if (name, m, src) in R]
        lo = min(r['energy'] for _, r in cand)
        src, r = next((s, r) for s, r in cand if r['energy'] - lo < TIE)
        guess = r['init_guess']                 # minao / atom / huckel / rhfchk
        ent[m] = dict(init_guess=guess, source='results_' + src, energy=r['energy'], s2=r.get('s2'),
                      note=r.get('note', []), **settings(r, src))
        ent['charge'], ent['mult'] = r['charge'], r['mult']
    out[name] = ent

with open(os.path.join(here, 'reference.json'), 'w') as f:
    json.dump(out, f, indent=1, sort_keys=True)
import collections
for m in ('rhf', 'uhf'):
    print(m, dict(collections.Counter(e[m]['init_guess'] for e in out.values())),
          'original-settings:', sum(e[m]['max_cycle'] == 200 for e in out.values()))
print('%d systems -> reference.json' % len(out))
