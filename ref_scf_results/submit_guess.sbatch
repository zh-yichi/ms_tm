#!/bin/bash
#SBATCH --job-name=tm_scf_guess
#SBATCH --partition=gpuMI100x8
#SBATCH --account=becu-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64         # half the node (128 cores total)
#SBATCH --mem=900g
#SBATCH --gpus-per-node=0          # CPU-only
#SBATCH --constraint="scratch"
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH -e slurm_guess.err
#SBATCH -o slurm_guess.out

# ---------------------------------------------------------------------------
# Repeat RHF/ROHF + UHF (sfX2C-1e, cc-pVDZ-DK) for all transition-metal
# complexes from initial guesses other than the default minao, to look for
# lower SCF solutions.  One output folder per guess: results_<guess>/
# (same layout as results_minao/).  The guesses run one after another; within
# a guess the systems are split round-robin over NW workers x NT threads.
# Finished records (json present) are skipped, so the job can be resubmitted.
#
#   sbatch submit_guess.sbatch                    # atom and huckel
#   GUESSES="atom" sbatch submit_guess.sbatch     # choose the guesses
# ---------------------------------------------------------------------------
GUESSES=${GUESSES:-"atom huckel"}
NW=8                               # number of concurrent pyscf workers
NT=$(( SLURM_CPUS_PER_TASK / NW )) # threads per worker
MEM_MB=$(( 900000 / NW * 8 / 10 )) # pyscf max_memory per worker (MB), 80% share
XYZDIR=../xyzfiles                 # relative to this directory

# SCF settings: the generous ones from rerun_problems.sh, so that a separate
# rerun pass should not be needed
MAX_CYCLE=1000        # DIIS iterations
NEWTON_MAX_CYCLE=200  # Newton iterations in the fallback
MAX_STAB=15           # stability-following rounds
STAB_TOL=1e-4         # |Hessian eigenvalue| below this is a zero mode, not an instability

module purge
module list

export OMP_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT
export OPENBLAS_NUM_THREADS=$NT
export TMPDIR=/work/hdd/bdka/yzhang65/tmp

source /projects/bdka/yzhang65/software/miniconda3/etc/profile.d/conda.sh
conda activate pyscf

export PYTHONPATH=/u/yzhang65/myprojects/software/pyscf:/u/yzhang65/myprojects/software/trot:$PYTHONPATH
export PYSCF_EXT_PATH=/u/yzhang65/myprojects/software/pyscf-forge
export PYSCF_TMPDIR=/work/hdd/bdka/yzhang65/tmp

unset LD_LIBRARY_PATH
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME="cpu"

echo "node: $(hostname)"
echo "guesses: $GUESSES   workers: $NW  threads/worker: $NT  max_memory/worker: ${MEM_MB} MB"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# unique xyz files (identical files appear in several subfolders)
mapfile -t FILES < <(ls "$XYZDIR"/*/*.xyz | awk -F/ '{print $NF, $0}' | sort -u -k1,1 | awk '{print $2}')
echo "${#FILES[@]} unique systems"

for g in $GUESSES; do
  OUTDIR=results_$g
  mkdir -p "$OUTDIR"
  echo "== init guess: $g -> $OUTDIR   start: $(date)"
  for ((w=0; w<NW; w++)); do
    sub=()
    for ((i=w; i<${#FILES[@]}; i+=NW)); do sub+=("${FILES[$i]}"); done
    python run_scf.py --init-guess "$g" --outdir "$OUTDIR" --memory "$MEM_MB" \
        --max-cycle $MAX_CYCLE --newton-max-cycle $NEWTON_MAX_CYCLE \
        --max-stab $MAX_STAB --stab-tol $STAB_TOL \
        "${sub[@]}" > "logs/${g}_worker$w.out" 2>&1 &
  done
  wait
  python collect_summary.py "$OUTDIR"
done
echo "done: $(date)"
