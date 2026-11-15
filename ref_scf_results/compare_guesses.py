#!/usr/bin/env python
"""Compare the SCF solutions obtained from different initial guesses.

Reads results_<guess>/*/*.json and writes
  guess_comparison.md    one row per system: lowest RHF/ROHF and UHF energy, which
                         guess(es) reached it, and how far above it each guess ended (mEh)
  guess_comparison.tsv   (space-aligned columns) energy of the first guess (reference, Eh) and the
                         other guesses relative to it (mEh), and the lowest guess

Usage: python compare_guesses.py [guess ...]      (default: minao atom huckel)
"""
import glob, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
GUESSES = sys.argv[1:] or ['minao', 'atom', 'huckel']
SAME = 1e-4      # Eh; guesses within 0.1 mEh of the lowest count as the same solution (statistics, d_ columns)
AGREE = 1e-6     # Eh; label is 'All' if all guesses agree within this (numerical noise),
                 # otherwise the single guess with the strictly lowest energy

R = {}
for g in GUESSES:
    for f in glob.glob(os.path.join(here, 'results_%s' % g, '*', '*.json')):
        r = json.load(open(f))
        R[(r['name'], r['method'], g)] = r
# extra UHF-only source: UHF started from the lowest RHF/ROHF orbitals (results_rhfchk), labelled 'RHF'
UG = list(GUESSES)
for f in glob.glob(os.path.join(here, 'results_rhfchk', '*', '*.uhf.json')):
    r = json.load(open(f))
    R[(r['name'], 'uhf', 'RHF')] = r
    if 'RHF' not in UG:
        UG.append('RHF')
names = sorted({k[0] for k in R},
               key=lambda n: (n.split('-')[0].lower(), R[(n, 'rhf', GUESSES[0])]['charge']))

def usable(r):
    return r.get('converged') and r.get('energy') is not None and r.get('internal_stable') is True

rows = []
for n in names:
    row = {'system': n.split('-')[0], 'hash': n.split('-')[1]}
    for m in ('rhf', 'uhf'):
        GL = GUESSES if m == 'rhf' else UG
        recs = {g: R.get((n, m, g)) for g in GL}
        ok = {g: r for g, r in recs.items() if r and usable(r)}
        row['charge'], row['mult'] = recs[GUESSES[0]]['charge'], recs[GUESSES[0]]['mult']
        lo = min(r['energy'] for r in ok.values())
        best = [g for g in GL if g in ok and ok[g]['energy'] - lo < SAME]
        spread = max(r['energy'] for r in ok.values()) - lo
        label = 'All' if len(ok) == len(GL) and spread < AGREE else min(ok, key=lambda g: ok[g]['energy'])
        row[m] = {'lowest': lo, 'best': best, 'label': label,
                  'd': {g: (ok[g]['energy'] - lo) * 1e3 if g in ok else None for g in GL},
                  'e': {g: recs[g]['energy'] if recs[g] else None for g in GL},
                  's2': {g: recs[g].get('s2') if recs[g] else None for g in GL},
                  'ext': ok[best[0]].get('external_stable')}
    row['rhf_type'] = 'ROHF' if row['mult'] > 1 else 'RHF'
    rows.append(row)

fd = lambda x: 'n/a' if x is None else ('0' if x < SAME * 1e3 else '%.2f' % x)

with open(os.path.join(here, 'guess_comparison.md'), 'w') as f:
    f.write('SCF solutions from different initial guesses (sfX2C-1e, cc-pVDZ-DK).  '
            'E = lowest energy over the guesses (Eh); d_<guess> = energy of that guess above '
            'the lowest (mEh, 0 = within %.1f mEh); best = guess with the strictly lowest energy, or All if all guesses agree within %.3f mEh.\n\n' % (SAME * 1e3, AGREE * 1e3))
    H = ['system', 'hash', 'charge', 'mult', 'rhf_type', 'E_RHF'] + ['d_' + g for g in GUESSES] + ['best_RHF', 'E_UHF'] \
        + ['d_' + g for g in UG] + ['best_UHF', 'S2_UHF']
    f.write('| ' + ' | '.join(H) + ' |\n|' + '---|' * len(H) + '\n')
    for r in rows:
        a, b = r['rhf'], r['uhf']
        f.write('| ' + ' | '.join(
            [r['system'], r['hash'][:4], str(r['charge']), str(r['mult']), r['rhf_type'], '%.8f' % a['lowest']]
            + [fd(a['d'][g]) for g in GUESSES] + [a['label'], '%.8f' % b['lowest']]
            + [fd(b['d'][g]) for g in UG] + [b['label'], '%.4f' % b['s2'][min(b['d'], key=lambda g: b['d'][g] if b['d'][g] is not None else 9e9)]]) + ' |\n')

with open(os.path.join(here, 'guess_comparison.tsv'), 'w') as f:
    # the first guess is the reference: its absolute energy (Eh), then the energy of
    # every other guess relative to it (mEh; negative = lower)
    ref = GUESSES[0]
    fm = lambda d: ('%.4f' % (d * 1e3)).replace('-0.0000', '0.0000')
    H = ['system', 'hash', 'rhf_type']
    for m in ('RHF', 'UHF'):
        others = (GUESSES if m == 'RHF' else UG)[1:]
        H += ['E_%s_%s' % (m, ref)] + ['dE_%s_%s_mEh' % (m, g) for g in others] \
             + ['best_%s' % m]
    T = [H]
    for r in rows:
        x = [r['system'], r['hash'][:4], r['rhf_type']]
        for m in ('rhf', 'uhf'):
            a = r[m]
            others = (GUESSES if m == 'rhf' else UG)[1:]
            e0 = a['e'][ref]
            x += ['%.10f' % e0] \
                 + ['n/a' if a['e'][g] is None else fm(a['e'][g] - e0) for g in others] \
                 + [a['label']]
        T.append(x)
    # fixed-width columns (two spaces apart); numeric columns right-aligned
    w = [max(len(x[i]) for x in T) for i in range(len(H))]
    num = [h.startswith(('E_', 'dE_')) for h in H]
    for x in T:
        f.write('  '.join(x[i].rjust(w[i]) if num[i] else x[i].ljust(w[i])
                          for i in range(len(H))).rstrip() + '\n')

print('%d systems, guesses: %s' % (len(rows), ' '.join(GUESSES)))
for m in ('rhf', 'uhf'):
    GL = GUESSES if m == 'rhf' else UG
    print('%s: all guesses agree for %d systems' % (m.upper(), sum(len(r[m]['best']) == len(GL) for r in rows)))
    for g in GL:
        d = [r[m]['d'][g] for r in rows if r[m]['d'][g] is not None]
        print('   %-7s lowest in %2d   above by >0.1 mEh: %2d  >1: %2d  >10: %2d  max %.1f mEh' % (
            g, sum(g in r[m]['best'] for r in rows), sum(x >= 0.1 for x in d),
            sum(x > 1 for x in d), sum(x > 10 for x in d), max(d)))
