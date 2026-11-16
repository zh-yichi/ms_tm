#!/bin/bash
#SBATCH --job-name=tm_scf_dz_newton
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_newton.out
#SBATCH --error=slurm_newton.err

# ---------------------------------------------------------------------------
# Newton-only SCF search over several initial guesses for the dz records that
# are above the reference energy (see rerun_newton.py).  Only the lowest
# solution per (system, method) is kept, in ./results_newton/<stem>/.
# Finished records are skipped, so the job can be resubmitted.
# Add --replace to EXTRA to copy the lower solutions over ./results/<stem>/.
#
#   sbatch submit_newton_resnick.sbatch
# ---------------------------------------------------------------------------
NW=8            # systems at the same time
EXTRA=""        # e.g. "--replace"

source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK cores, ${SLURM_MEM_PER_NODE}MB RAM"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_newton

NT=$(( SLURM_CPUS_PER_TASK / NW ))
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export MEM_MB=$(( SLURM_MEM_PER_NODE / NW * 8 / 10 )) EXTRA

mapfile -t FILES < <(OMP_NUM_THREADS=1 python rerun_newton.py --list | awk '{print $1}')
echo "${#FILES[@]} systems above the reference, $NW at a time, $NT threads and $MEM_MB MB each   start: $(date)"
printf '%s\n' "${FILES[@]}" | xargs -P "$NW" -I{} bash -c \
    'python rerun_newton.py --memory "$MEM_MB" $EXTRA "$1" > "logs/$(basename "$1" .xyz).newton.out" 2>&1' _ {}

python rerun_newton.py --summary
echo "done: $(date)"
