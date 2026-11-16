#!/usr/bin/env python
"""Compare the Newton (second-order) SCF solutions obtained from different initial guesses.

Reads results_newton/*/*.json (written by rerun_newton.py) and writes
  guess_comparison.md    one row per system: lowest RHF/ROHF and UHF energy, how far above
                         it each guess ended (mEh), and the guess to use to reproduce it
  guess_comparison.tsv   (space-aligned columns) energy of the first guess (Eh), the other
                         guesses relative to it (mEh), and the guess to use

Guesses: minao, atom, huckel, and for UHF also 'RHF' (= rhfchk: UHF started from the
RHF/ROHF orbitals of the same system).  The guess 'current' (orbitals of the earlier DIIS
result in ./results) cannot be reproduced from scratch and is left out unless it is named
on the command line; records whose stored lowest solution was only reached from it are
listed at the end.

Usage: python compare_guesses.py [guess ...]      (default: minao atom huckel rhfchk)
"""
import glob, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
GUESSES = sys.argv[1:] or ['minao', 'atom', 'huckel', 'rhfchk']
NAME = {'rhfchk': 'RHF'}                    # column / label names
SAME = 1e-5      # Eh; guesses within 0.01 mEh of the lowest reached the same solution
AGREE = 1e-6     # Eh; 'All' if all guesses agree within this (numerical noise)

R = {}
for f in glob.glob(os.path.join(here, 'results_newton', '*', '*.json')):
    r = json.load(open(f))
    R[(r['name'], r['method'])] = r
names = sorted({k[0] for k in R}, key=lambda n: (n.split('-')[0].lower(), R[(n, 'rhf')]['charge']))
GL = {'rhf': [g for g in GUESSES if g != 'rhfchk'], 'uhf': list(GUESSES)}

def usable(c):
    return c and c.get('converged') and c.get('energy') is not None and c.get('internal_stable') is True

rows, warn = [], []
for n in names:
    row = {'system': n.split('-')[0], 'hash': n.split('-')[1]}
    for m in ('rhf', 'uhf'):
        r = R[(n, m)]
        row['charge'], row['mult'] = r['charge'], r['mult']
        cand = {c['guess']: c for c in r['candidates']}
        ok = {g: cand[g] for g in GL[m] if usable(cand.get(g))}
        lo = min(c['energy'] for c in ok.values())
        best = [g for g in GL[m] if g in ok and ok[g]['energy'] - lo < SAME]
        spread = max(c['energy'] for c in ok.values()) - lo
        # 'All', or the first guess (in the order given) that reaches the lowest solution
        label = 'All' if len(ok) == len(GL[m]) and spread < AGREE else NAME.get(best[0], best[0])
        row[m] = {'lowest': lo, 'best': best, 'label': label, 's2': ok[best[0]].get('s2'),
                  'd': {g: (ok[g]['energy'] - lo) * 1e3 if g in ok else None for g in GL[m]},
                  'e': {g: cand[g]['energy'] if g in cand else None for g in GL[m]}}
        if r.get('energy') is not None and r['energy'] < lo - SAME:
            warn.append('%s %s: the stored lowest solution (guess %s, E=%.8f) is %.4f mEh BELOW every guess of this table'
                        % (n, m, r['init_guess'], r['energy'], (lo - r['energy']) * 1e3))
        if r.get('ref_energy') is not None:
            row[m]['dref'] = (lo - r['ref_energy']) * 1e3
    row['rhf_type'] = 'ROHF' if row['mult'] > 1 else 'RHF'
    rows.append(row)

fd = lambda x: 'n/a' if x is None else ('0' if x < SAME * 1e3 else '%.2f' % x)
col = lambda g: NAME.get(g, g)

with open(os.path.join(here, 'guess_comparison.md'), 'w') as f:
    f.write('Newton (second-order) SCF solutions from different initial guesses (sfX2C-1e, cc-pVDZ-DK; '
            'internal instabilities followed).  E = lowest energy over the guesses (Eh); d_<guess> = energy of '
            'that guess above the lowest (mEh, 0 = within %.2f mEh); RHF = UHF started from the RHF/ROHF orbitals; '
            'best = All if all guesses agree within %.3f mEh, otherwise the first guess in the order of the '
            'columns that reaches the lowest solution.\n\n' % (SAME * 1e3, AGREE * 1e3))
    H = ['system', 'hash', 'charge', 'mult', 'rhf_type', 'E_RHF'] + ['d_' + col(g) for g in GL['rhf']] + ['best_RHF', 'E_UHF'] \
        + ['d_' + col(g) for g in GL['uhf']] + ['best_UHF', 'S2_UHF']
    f.write('| ' + ' | '.join(H) + ' |\n|' + '---|' * len(H) + '\n')
    for r in rows:
        a, b = r['rhf'], r['uhf']
        f.write('| ' + ' | '.join(
            [r['system'], r['hash'][:4], str(r['charge']), str(r['mult']), r['rhf_type'], '%.8f' % a['lowest']]
            + [fd(a['d'][g]) for g in GL['rhf']] + [a['label'], '%.8f' % b['lowest']]
            + [fd(b['d'][g]) for g in GL['uhf']] + [b['label'], '%.4f' % b['s2']]) + ' |\n')

with open(os.path.join(here, 'guess_comparison.tsv'), 'w') as f:
    # the first guess is the reference: its absolute energy (Eh), then the energy of
    # every other guess relative to it (mEh; negative = lower)
    ref = GUESSES[0]
    fm = lambda d: ('%.4f' % (d * 1e3)).replace('-0.0000', '0.0000')
    H = ['system', 'hash', 'rhf_type']
    for m in ('rhf', 'uhf'):
        H += ['E_%s_%s' % (m.upper(), ref)] + ['dE_%s_%s_mEh' % (m.upper(), col(g)) for g in GL[m][1:]] + ['best_%s' % m.upper()]
    T = [H]
    for r in rows:
        x = [r['system'], r['hash'][:4], r['rhf_type']]
        for m in ('rhf', 'uhf'):
            a = r[m]
            e0 = a['e'][ref]
            x += ['%.10f' % e0] + ['n/a' if a['e'][g] is None else fm(a['e'][g] - e0) for g in GL[m][1:]] + [a['label']]
        T.append(x)
    # fixed-width columns (two spaces apart); numeric columns right-aligned
    w = [max(len(x[i]) for x in T) for i in range(len(H))]
    num = [h.startswith(('E_', 'dE_')) for h in H]
    for x in T:
        f.write('  '.join(x[i].rjust(w[i]) if num[i] else x[i].ljust(w[i]) for i in range(len(H))).rstrip() + '\n')

print('%d systems, guesses: %s' % (len(rows), ' '.join(GUESSES)))
for m in ('rhf', 'uhf'):
    print('%s: all guesses agree for %d systems' % (m.upper(), sum(len(r[m]['best']) == len(GL[m]) for r in rows)))
    for g in GL[m]:
        d = [r[m]['d'][g] for r in rows if r[m]['d'][g] is not None]
        print('   %-7s lowest in %2d   above by >0.1 mEh: %2d  >1: %2d  >10: %2d  max %.1f mEh' % (
            col(g), sum(g in r[m]['best'] for r in rows), sum(x >= 0.1 for x in d),
            sum(x > 1 for x in d), sum(x > 10 for x in d), max(d)))
    d = [r[m]['dref'] for r in rows if 'dref' in r[m]]
    print('   vs the earlier DIIS reference (reference.json): %d same, %d lower, %d higher (max +%.2f mEh)' % (
        sum(abs(x) <= SAME * 1e3 for x in d), sum(x < -SAME * 1e3 for x in d), sum(x > SAME * 1e3 for x in d), max(max(d), 0)))
for w_ in warn:
    print('WARNING', w_)
