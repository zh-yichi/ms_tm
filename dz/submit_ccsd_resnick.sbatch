#!/bin/bash
#SBATCH --job-name=tm_ccsd_dz
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_ccsd.out
#SBATCH --error=slurm_ccsd.err

# ---------------------------------------------------------------------------
# CCSD (sfX2C-1e / cc-pVDZ-DK) on the lowest RHF/ROHF and UHF solutions in
# ./results_newton, saving the amplitudes (see run_ccsd.py).
# Output: ./ccsd/<stem>/<stem>.<method>.ccsd.{h5,json}   (*.h5 is not tracked by git)
# Finished calculations (json present) are skipped, so the job can simply be
# resubmitted if it hits the time limit.
#
#   sbatch submit_ccsd_resnick.sbatch
# ---------------------------------------------------------------------------
NW=8            # systems at the same time
EXTRA=""        # e.g. "--frozen none" or "--methods uhf"

source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK cores, ${SLURM_MEM_PER_NODE}MB RAM"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs ccsd

NT=$(( SLURM_CPUS_PER_TASK / NW ))
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export MEM_MB=$(( SLURM_MEM_PER_NODE / NW * 8 / 10 )) EXTRA

# largest systems first
mapfile -t FILES < <(OMP_NUM_THREADS=1 python run_scf.py --sizes | awk '{print $2}')
echo "${#FILES[@]} systems, $NW at a time, $NT threads and $MEM_MB MB each   start: $(date)"
printf '%s\n' "${FILES[@]}" | xargs -P "$NW" -I{} bash -c \
    'python run_ccsd.py --memory "$MEM_MB" $EXTRA "$1" > "logs/$(basename "$1" .xyz).ccsd.out" 2>&1' _ {}

echo "== skipped:"
grep -h '^skip (' logs/*.ccsd.out | grep -v '^skip (done)' || echo "none"
python run_ccsd.py --summary $EXTRA
echo "done: $(date)"
