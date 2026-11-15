#!/usr/bin/env python
"""RHF/ROHF and UHF, sfX2C-1e / cc-pVDZ-DK, for the transition-metal complexes in ../xyzfiles.
Reproduces the lowest known RHF/ROHF and UHF solutions listed in reference.json
(initial guess and SCF settings per system; built by make_reference.py).
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

main(basis='cc-pvdz-dk', here=here, reference=os.path.join(here, 'reference.json'))
