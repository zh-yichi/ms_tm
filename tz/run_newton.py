#!/usr/bin/env python
"""Newton (second-order) RHF/ROHF and UHF, sfX2C-1e / cc-pVTZ-DK, for the transition-metal
complexes in ../xyzfiles.  Starts from the lowest Newton solutions of the dz basis,
../dz/results_newton/<stem>/<stem>.{rhf,uhf}.chk, projected onto this basis.
Results, including the SCF chkfiles, go to ./results_newton/<stem>/.

  python run_newton.py                 # all systems (finished ones are skipped)
  python run_newton.py a.xyz b.xyz     # only these
  python run_newton.py --summary       # write results_newton/summary.tsv and problems.txt
  python run_newton.py --parent DIR    # start from another results folder (e.g. ../dz/results_newton)
  python run_newton.py -h              # other options
The driver itself is ../scf_newton.py.
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from scf_newton import main

main(basis='cc-pvtz-dk', here=here, parent=os.path.join(here, '..', 'dz', 'results_newton'))
