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
# Newton-only SCF for ALL systems, one calculation per initial guess
# (minao, atom, huckel, rhfchk, current); see rerun_newton.py.
# ./results_newton/<stem>/ keeps the chkfile of every guess and a copy of the
# lowest one as <stem>.<method>.chk  (*.chk is not tracked by git).
# Finished records are skipped, so the job can be resubmitted if it hits the
# time limit.  Add --replace to EXTRA to copy solutions lower than the present
# ones over ./results/<stem>/.
#
#   sbatch submit_newton_resnick.sbatch
# ---------------------------------------------------------------------------
NW_SMALL=16     # systems at the same time when the integrals fit in memory at this share
NW_BIG=4        # systems at the same time for the larger ones
EXTRA=""        # e.g. "--replace" or "--above-only"

source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK cores, ${SLURM_MEM_PER_NODE}MB RAM"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_newton

export EXTRA

# same two tiers as submit_resnick.sbatch: NW systems at a time, each with
# cpus/NW threads and an 80% share of the memory
run_tier() {
  local nw=$1; shift
  [ $# -eq 0 ] && return
  local nt=$(( SLURM_CPUS_PER_TASK / nw ))
  export OMP_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt MKL_NUM_THREADS=$nt NUMEXPR_NUM_THREADS=$nt
  export MEM_MB=$(( SLURM_MEM_PER_NODE / nw * 8 / 10 ))
  echo "$(date)  $# systems, $nw at a time, $nt threads and $MEM_MB MB each"
  printf '%s\n' "$@" | xargs -P "$nw" -I{} bash -c \
      'python rerun_newton.py --memory "$MEM_MB" $EXTRA "$1" > "logs/$(basename "$1" .xyz).newton.out" 2>&1' _ {}
}

LIMIT=$(( SLURM_MEM_PER_NODE / NW_SMALL * 8 / 10 ))
WANT=$(OMP_NUM_THREADS=1 python rerun_newton.py --list $EXTRA | awk '{print $1}' | xargs -n1 basename)
SIZES=$(OMP_NUM_THREADS=1 python run_scf.py --sizes | grep -F -f <(echo "$WANT"))    # "<nao> <xyz file>", largest first
mapfile -t BIG   < <(awk -v m="$LIMIT" '$1^4/1e6 + 500 >= 0.95*m {print $2}' <<< "$SIZES")
mapfile -t SMALL < <(awk -v m="$LIMIT" '$1^4/1e6 + 500 <  0.95*m {print $2}' <<< "$SIZES")
echo "$(( ${#BIG[@]} + ${#SMALL[@]} )) systems   start: $(date)"

run_tier "$NW_BIG" "${BIG[@]}"
run_tier "$NW_SMALL" "${SMALL[@]}"

python rerun_newton.py --summary
echo "done: $(date)"
