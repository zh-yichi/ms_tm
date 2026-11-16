#!/bin/bash
#SBATCH --job-name=tm_newton_qz
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # charged per core-hour
#SBATCH --mem=220G                # 32 cores' share of a 56-core/386 GB node (6.9 GB/core)
#SBATCH --time=72:00:00
#SBATCH --output=slurm_newton_qz.out
#SBATCH --error=slurm_newton_qz.err

# ---------------------------------------------------------------------------
# Newton (second-order) RHF/ROHF + UHF, sfX2C-1e / cc-pVQZ-DK, all transition-metal complexes,
# starting from the tz chkfiles in ../tz/results_newton.
# Needs ../tz/results_newton/<stem>/<stem>.{rhf,uhf}.{chk,json} on this cluster (run tz first).
# Results and SCF chkfiles: ./results_newton/<stem>/   (*.chk is not tracked by git)
# Finished calculations (json present) are skipped, so the job can simply be
# resubmitted if it hits the time limit.
#
#   sbatch submit_resnick.sbatch
# ---------------------------------------------------------------------------
NW_SMALL=8      # systems at the same time when the integrals fit in memory at this share
NW_BIG=2        # systems at the same time for the larger ones (more memory and threads each)
XYZDIR=../xyzfiles

# --- activate conda env ---
source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf

# --- PATH ---
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK cores, ${SLURM_MEM_PER_NODE}MB RAM"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_newton

# run_tier NW file...: NW systems at a time, each with cpus/NW threads and an 80%
# share of the memory; xargs hands the next system to whichever slot becomes free
run_tier() {
  local nw=$1; shift
  [ $# -eq 0 ] && return
  local nt=$(( SLURM_CPUS_PER_TASK / nw ))
  export OMP_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt MKL_NUM_THREADS=$nt NUMEXPR_NUM_THREADS=$nt
  export MEM_MB=$(( SLURM_MEM_PER_NODE / nw * 8 / 10 ))
  echo "$(date)  $# systems, $nw at a time, $nt threads and $MEM_MB MB each"
  printf '%s\n' "$@" | xargs -P "$nw" -I{} bash -c \
      'python run_newton.py --memory "$MEM_MB" "$1" > "logs/$(basename "$1" .xyz).newton.out" 2>&1' _ {}
}

# pyscf keeps the two-electron integrals in memory (much faster SCF, Newton and
# stability steps) if nao^4/1e6 MB fits in 95% of max_memory.  Systems that fit
# with the NW_SMALL share run NW_SMALL at a time; the others run NW_BIG at a time.
# Both lists are sorted largest first.
LIMIT=$(( SLURM_MEM_PER_NODE / NW_SMALL * 8 / 10 ))
SIZES=$(OMP_NUM_THREADS=1 python run_newton.py --sizes)        # "<nao> <xyz file>"
mapfile -t BIG   < <(awk -v m="$LIMIT" '$1^4/1e6 + 500 >= 0.95*m {print $2}' <<< "$SIZES")
mapfile -t SMALL < <(awk -v m="$LIMIT" '$1^4/1e6 + 500 <  0.95*m {print $2}' <<< "$SIZES")
echo "$(( ${#BIG[@]} + ${#SMALL[@]} )) unique systems   start: $(date)"

run_tier "$NW_BIG" "${BIG[@]}"
run_tier "$NW_SMALL" "${SMALL[@]}"

echo "== skipped:"
grep -h '^skip (' logs/*.newton.out | grep -v '^skip (done)' || echo "none"
python run_newton.py --summary
echo "done: $(date)"
