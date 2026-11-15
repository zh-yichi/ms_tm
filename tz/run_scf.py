#!/usr/bin/env python
"""RHF/ROHF and UHF, sfX2C-1e / cc-pVTZ-DK, for the transition-metal complexes in ../xyzfiles.
Starts from the dz chkfiles in ../dz/results (orbitals projected onto this basis).
Results, including the SCF chkfiles, go to ./results/<stem>/.

  python run_scf.py                 # all systems (finished ones are skipped)
  python run_scf.py a.xyz b.xyz     # only these
  python run_scf.py --summary       # write results/summary.tsv and results/problems.txt
  python run_scf.py -h              # other options
The driver itself is ../scf_lowest.py.
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))
from scf_lowest import main

main(basis='cc-pvtz-dk', here=here, parent=os.path.join(here, '..', 'dz', 'results'), external=False)
