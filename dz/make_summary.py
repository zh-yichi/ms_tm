#!/usr/bin/env python
"""Summary table of the dz (sfX2C-1e / cc-pVDZ-DK) results: SCF from different initial
guesses and CCSD on the lowest SCF solutions.

Reads results_newton/*/*.json (rerun_newton.py; second-order SCF only) and
ccsd/*/*.ccsd.json (run_ccsd.py) and writes
  summary_table.md    markdown table
  summary_table.tsv   the same with space-aligned columns and more digits in the relative energies
  summary_table.csv   the same as the tsv, comma separated, for a spreadsheet (Google Sheets)
Absolute SCF energies are given to 1e-8 Eh and CCSD energies to 1e-6 Eh.

The header has two rows: the method (RHF / ROHF, UHF, CCSD) and below it the initial guess.
SCF, for RHF/ROHF and for UHF:
  minao           absolute energy from the minao guess (Eh)
  atom, huckel, RHF   energy from the other guesses relative to minao (mEh; negative = lower).
                  'RHF' (UHF only) = UHF started from the RHF/ROHF orbitals of the system.
  best            'All' if all guesses agree, otherwise the first guess (in column order)
                  that reaches the lowest solution
CCSD, frozen core (pyscf chemcore), total energy (Eh) on the lowest solution of each kind:
  on RHF/ROHF     RCCSD (closed shell) or UCCSD on the ROHF determinant (open shell)
  on UHF          UCCSD on the lowest UHF solution
'*' and 'n/a' are explained in the notes at the bottom of both files.  summary_table.md uses
an html table, because a markdown table cannot have a header cell spanning several columns.

Usage: python make_summary.py
"""
import glob, json, os

here = os.path.dirname(os.path.abspath(__file__))
GUESSES = ['minao', 'atom', 'huckel', 'rhfchk']
NAME = {'rhfchk': 'RHF'}
REPO = 'https://github.com/zh-yichi/ms_tm'      # given at the bottom of the tables
SAME = 1e-5      # Eh; guesses within 0.01 mEh reached the same solution
AGREE = 1e-6     # Eh; 'All' if all guesses agree within this
GL = {'rhf': GUESSES[:3], 'uhf': GUESSES}
col = lambda g: NAME.get(g, g)

def load(pattern, key):
    out = {}
    for f in glob.glob(os.path.join(here, pattern)):
        r = json.load(open(f))
        out[(r['name'], r['method'])] = r
    return out

S, C = load('results_newton/*/*.json', 0), load('ccsd/*/*.ccsd.json', 0)
names = sorted({k[0] for k in S}, key=lambda n: (n.split('-')[0].lower(), S[(n, 'rhf')]['charge']))
usable = lambda c: c and c.get('converged') and c.get('energy') is not None and c.get('internal_stable') is True

rows, warn, off = [], [], []
for n in names:
    row = {'system': n.split('-')[0], 'hash': n.split('-')[1][:4]}
    for m in ('rhf', 'uhf'):
        r = S[(n, m)]
        row['charge'], row['mult'] = r['charge'], r['mult']
        cand = {c['guess']: c for c in r['candidates']}
        ok = {g: cand[g] for g in GL[m] if usable(cand.get(g))}
        for g in GL[m]:
            if g not in ok:
                warn.append('%s %s: guess %s is missing, not converged or unstable' % (n, m, g))
        lo = min(c['energy'] for c in ok.values())
        best = [g for g in GL[m] if g in ok and ok[g]['energy'] - lo < SAME]
        spread = max(c['energy'] for c in ok.values()) - lo
        # minao is the reference of the relative energies; if it converged to an unstable
        # solution its energy is still shown (marked '*') but it cannot be the 'best' guess
        c0 = cand.get('minao') or {}
        e0 = c0.get('energy') if c0.get('converged') else None
        cc = C.get((n, m), {})
        row[m] = {'e0': e0, 'e0_ok': 'minao' in ok, 'lowest': lo, 's2': ok[best[0]].get('s2'),
                  'label': 'All' if len(ok) == len(GL[m]) and spread < AGREE else col(best[0]),
                  'd': {g: (ok[g]['energy'] - e0) * 1e3 if g in ok and e0 is not None else None for g in GL[m][1:]},
                  'cc': cc.get('e_tot'), 'cc_conv': bool(cc.get('converged')), 'cc_type': cc.get('cc_type')}
        if not cc:
            warn.append('%s %s: no CCSD record' % (n, m))
        elif not cc.get('converged'):
            warn.append('%s %s: CCSD not converged (%s)' % (n, m, '; '.join(cc.get('note', []))))
        # the CCSD reference is the stored lowest chkfile; it should be the lowest of the table
        if cc.get('e_hf') is not None and abs(cc['e_hf'] - lo) > SAME:
            off.append('%s (%s), %s: %.2f mEh' % (row['system'], row['hash'], 'UHF' if m == 'uhf' else 'RHF/ROHF', (cc['e_hf'] - lo) * 1e3))
            warn.append('%s %s: the CCSD reference (E_SCF=%.8f, guess %s) is %+.4f mEh from the lowest of the '
                        'guesses in the table' % (n, m, cc['e_hf'], r.get('init_guess'), (cc['e_hf'] - lo) * 1e3))
    row['rhf_type'] = 'ROHF' if row['mult'] > 1 else 'RHF'
    rows.append(row)

def cells(r, nd, ne=8, nc=6):       # digits: relative (mEh), SCF (Eh), CCSD (Eh)
    fe = lambda x: 'n/a' if x is None else '%.*f' % (ne, x)
    fd = lambda x: 'n/a' if x is None else ('%.*f' % (nd, x)).replace('-0.' + '0' * nd, '0.' + '0' * nd)
    fc = lambda a: 'n/a' if a['cc'] is None else '%.*f' % (nc, a['cc']) + ('' if a['cc_conv'] else '*')
    x = [r['system'], r['hash'], str(r['charge']), str(r['mult']), r['rhf_type']]
    for m in ('rhf', 'uhf'):
        a = r[m]
        x += [fe(a['e0']) + ('' if a['e0_ok'] or a['e0'] is None else '*')] + [fd(a['d'][g]) for g in GL[m][1:]] + [a['label']]
    return x + ['%.4f' % r['uhf']['s2'], fc(r['rhf']), fc(r['uhf'])]

sysname = lambda r: '%s (%s)' % (r['system'], r['hash'])
FOOT = [
    "*   on a CCSD energy: the CCSD did not converge within the maximum number of cycles; "
    "the energy of the last cycle is given and should not be used as a converged value"
    + ''.join(' [%s: %s]' % (lab, ', '.join(sysname(r) for r in rows if r[m]['cc'] is not None and not r[m]['cc_conv']))
              for m, lab in (('rhf', 'on RHF/ROHF'), ('uhf', 'on UHF'))
              if any(r[m]['cc'] is not None and not r[m]['cc_conv'] for r in rows)) + ".",
    "*   on a minao SCF energy: the SCF from the minao guess converged to an internally unstable "
    "solution (a saddle point, not a minimum); its energy is still the reference of the relative energies "
    "of that row, but minao is not the guess to use"
    + ''.join(' [%s: %s]' % (m.upper() if m == 'uhf' else 'RHF/ROHF', ', '.join(sysname(r) for r in rows if r[m]['e0'] is not None and not r[m]['e0_ok']))
              for m in ('rhf', 'uhf') if any(r[m]['e0'] is not None and not r[m]['e0_ok'] for r in rows)) + ".",
    "n/a no energy is available: the calculation failed or was not run (for CCSD: the amplitudes diverged "
    "and the run stopped with an error; for an SCF guess: not converged, unstable or missing)"
    + ''.join(' [%s: %s]' % (lab, ', '.join(sysname(r) for r in rows if r[m]['cc'] is None))
              for m, lab in (('rhf', 'on RHF/ROHF'), ('uhf', 'on UHF')) if any(r[m]['cc'] is None for r in rows)) + ".",
]

GENERAL = [
    "SCF energies: the minao column is the absolute energy (Eh) obtained from the minao initial guess. "
    "The columns of the other initial guesses (atom, huckel, and RHF = UHF started from the RHF/ROHF "
    "orbitals) are differences from the minao energy of the same method, E(guess) - E(minao), in mEh; "
    "a negative value means that guess gave a lower energy than minao.",
    "CCSD energies: total energies (Eh) of CCSD calculations that use the lowest SCF solution as the "
    "reference: the lowest RHF/ROHF solution for 'on RHF/ROHF' and the lowest UHF solution for 'on UHF' "
    "(the guess named in the 'best' column), not necessarily the minao solution."
    + (" Exception: the CCSD reference is a lower SCF solution than any guess of this table reached "
       "(it was obtained from the orbitals of an earlier calculation) for " + '; '.join(off)
       + " relative to the lowest guess in the table." if off else ""),
]
GENERAL.append("Data and scripts (geometries, SCF and CCSD inputs and outputs; this table is written by "
               "dz/make_summary.py): " + REPO)

# two header rows: the method on the first, the initial guess (or quantity) on the second
LEAD = ['system', 'hash', 'charge', 'mult', 'rhf_type']
GROUPS = [('RHF / ROHF', ['minao (Eh)'] + [col(g) + ' (mEh)' for g in GL['rhf'][1:]] + ['best']),
          ('UHF', ['minao (Eh)'] + [col(g) + ' (mEh)' for g in GL['uhf'][1:]] + ['best', '<S^2>']),
          ('CCSD (Eh)', ['on RHF/ROHF', 'on UHF'])]

with open(os.path.join(here, 'summary_table.md'), 'w') as f:
    f.write('sfX2C-1e / cc-pVDZ-DK.  SCF (columns RHF / ROHF and UHF): second-order (Newton) solver, internal '
            'instabilities followed.  minao = energy from the minao guess (Eh); the other guesses = energy from '
            'that guess relative to minao (mEh, negative = lower); RHF under UHF = UHF started from the RHF/ROHF '
            'orbitals; best = All if all guesses agree within %.3f mEh, otherwise the first guess in column order '
            'that reaches the lowest solution; &lt;S^2&gt; of the lowest UHF.  CCSD (frozen core, pyscf chemcore), '
            'total energies (Eh): on RHF/ROHF = RCCSD on the lowest RHF, or UCCSD on the lowest ROHF determinant '
            'for open shells; on UHF = UCCSD on the lowest UHF.  See the notes at the bottom for * and n/a.\n\n'
            % (AGREE * 1e3))
    esc = lambda t: t.replace('<', '&lt;').replace('>', '&gt;')
    # html table: markdown tables cannot have a header cell that spans several columns
    f.write('<table>\n<thead>\n<tr>' + ''.join('<th rowspan="2">%s</th>' % h for h in LEAD)
            + ''.join('<th colspan="%d">%s</th>' % (len(sub), g) for g, sub in GROUPS) + '</tr>\n<tr>'
            + ''.join('<th>%s</th>' % esc(h) for g, sub in GROUPS for h in sub) + '</tr>\n</thead>\n<tbody>\n')
    for r in rows:
        x = cells(r, 2)
        f.write('<tr>' + ''.join('<td%s>%s</td>' % ('' if i < len(LEAD) else ' align="right"', v)
                                 for i, v in enumerate(x)) + '</tr>\n')
    f.write('</tbody>\n</table>\n')
    f.write('\nNotes:\n\n' + ''.join('- %s\n' % t for t in GENERAL) + ''.join('- `%s` %s\n' % (t[:3].strip(), t[4:]) for t in FOOT))

with open(os.path.join(here, 'summary_table.tsv'), 'w') as f:
    H2 = LEAD + [h for g, sub in GROUPS for h in sub]
    T = [H2] + [cells(r, 4) for r in rows]
    w = [max(len(x[i]) for x in T) for i in range(len(H2))]
    num = [i >= len(LEAD) and not h.startswith('best') for i, h in enumerate(H2)]
    line = lambda x: '  '.join(x[i].rjust(w[i]) if num[i] else x[i].ljust(w[i]) for i in range(len(H2))).rstrip() + '\n'
    # first header row: the method name over the first column of its group, dashes over the rest
    top, i = ' ' * (sum(w[:len(LEAD)]) + 2 * len(LEAD)), len(LEAD)
    for g, sub in GROUPS:
        width = sum(w[i:i + len(sub)]) + 2 * (len(sub) - 1)
        top += (' ' + g + ' ').center(width, '-') + '  '
        i += len(sub)
    f.write(top.rstrip() + '\n' + line(H2))
    for x in T[1:]:
        f.write(line(x))
    f.write(''.join('# %s\n' % t for t in GENERAL + FOOT))

# csv for a spreadsheet (Google Sheets: File > Import > Upload).  Two header rows as in the
# other files; csv has no merged cells, so the method stands over the first column of its group.
import csv
with open(os.path.join(here, 'summary_table.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow([''] * len(LEAD) + [c for g, sub in GROUPS for c in [g] + [''] * (len(sub) - 1)])
    wr.writerow(LEAD + [h for g, sub in GROUPS for h in sub])
    for r in rows:
        x = cells(r, 4)
        # written as a text formula: a spreadsheet would otherwise read hashes such as
        # '83e3' or '0607' as the numbers 83000 and 607
        x[1] = '="%s"' % x[1]
        wr.writerow(x)
    wr.writerow([])
    wr.writerow(['Notes:'])
    for t in GENERAL + FOOT:
        wr.writerow([' '.join(t.split())])

print('%d systems -> summary_table.md, summary_table.tsv, summary_table.csv' % len(rows))
for m in ('rhf', 'uhf'):
    print('CCSD on %s: %d converged, %d not converged, %d without energy' % (
        m.upper(), sum(r[m]['cc_conv'] for r in rows), sum(r[m]['cc'] is not None and not r[m]['cc_conv'] for r in rows),
        sum(r[m]['cc'] is None for r in rows)))
for w_ in warn:
    print('NOTE', w_)
