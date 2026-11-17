#!/bin/bash
#SBATCH --job-name=tm_ccsd_dz_stalled
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_ccsd_stalled.out
#SBATCH --error=slurm_ccsd_stalled.err

# ---------------------------------------------------------------------------
# Rerun the dz CCSD calculations whose energy was converged but whose amplitudes
# stalled just above the old threshold (1e-6):
#     BHSc (ROHF, UHF), HOSc (ROHF), ClHTi (UHF), H3NiP-1dd... (UHF)
# run_ccsd.py now uses conv_tol = 1e-6 Eh and conv_tol_normt = 1e-5 and reruns
# only the records that are not converged, so the converged partner of each
# system is left untouched.
# A system counts as fine when its UHF-based UCCSD is converged; a
# non-converged ROHF-based one is only reported.
#
#   sbatch rerun_stalled_ccsd.sbatch
# ---------------------------------------------------------------------------
STEMS=(
    BHSc-16a0f729f45cb6f38a35d0a8f357f0ab73cdff4d
    HOSc-bb7e1b908facb6b5ce0fb6e0ca15efd6b4c13614
    ClHTi-757e5b56ea7b8049f8d4c01209001f5b5fc0a8c9
    H3NiP-1dda542f4c9c2b4506f6b6e76816b087824305b9
)
EXTRA=""        # extra options for run_ccsd.py

source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs ccsd

NW=${#STEMS[@]}
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
NT=$(( NCPU / NW )); [ "$NT" -lt 1 ] && NT=1
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export MEM_MB=$(( ${SLURM_MEM_PER_NODE:-16000} / NW * 8 / 10 )) EXTRA

echo "${#STEMS[@]} systems, $NT threads and $MEM_MB MB each   start: $(date)"
printf '%s\n' "${STEMS[@]}" | xargs -P "$NW" -I{} bash -c \
    'python run_ccsd.py --memory "$MEM_MB" $EXTRA "$1" > "logs/$1.ccsd.out" 2>&1' _ {}

OMP_NUM_THREADS=1 python run_ccsd.py --summary > /dev/null      # refresh ccsd/summary.tsv

python - "${STEMS[@]}" <<'EOF'
import json, os, sys
bad = 0
for stem in sys.argv[1:]:
    rec = {}
    for m in ('rhf', 'uhf'):
        f = os.path.join('ccsd', stem, '%s.%s.ccsd.json' % (stem, m))
        rec[m] = json.load(open(f)) if os.path.exists(f) else {}
    def show(m):
        r = rec[m]
        if not r:
            return '%s: missing' % m
        e = r.get('e_tot')
        return '%s (%s): %s  cycles=%s  E_tot=%s' % (m, r.get('cc_type', '?'),
               'converged' if r.get('converged') else 'NOT CONVERGED', r.get('cycles'),
               '%.8f' % e if e is not None else 'None')
    ok = bool(rec['uhf'].get('converged'))
    bad += not ok
    note = '' if rec['rhf'].get('converged') or not ok else '   (RHF/ROHF-based not converged: accepted)'
    print('%-8s %s\n         %s\n         %s%s' % ('OK' if ok else 'PROBLEM', stem, show('rhf'), show('uhf'), note))
print('\n%d of %d systems have a converged UHF-based CCSD' % (len(sys.argv) - 1 - bad, len(sys.argv) - 1))
sys.exit(1 if bad else 0)
EOF
echo "done: $(date)"
