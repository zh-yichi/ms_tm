#!/bin/bash
#SBATCH --job-name=tm_scf_guess
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # fits every CPU node type (32/56/64 cores per node)
#SBATCH --mem=64G                  # 2 GB/core; well under the 192 GB of a standard node
#SBATCH --time=04:00:00
#SBATCH --output=slurm_guess.out
#SBATCH --error=slurm_guess.err

# ---------------------------------------------------------------------------
# Caltech Resnick HPC version of submit_guess.sbatch.
#
# Repeat RHF/ROHF + UHF (sfX2C-1e, cc-pVDZ-DK) for all transition-metal
# complexes from initial guesses other than the default minao, to look for
# lower SCF solutions.  One output folder per guess: results_<guess>/
# (same layout as results_minao/).  The guesses run one after another; within
# a guess the systems are split round-robin over NW workers x NT threads.
# Finished records (json present) are skipped, so the job can be resubmitted.
#
#   sbatch submit_guess_resnick.sbatch                  # atom and huckel
#   GUESSES="atom" sbatch submit_guess_resnick.sbatch   # choose the guesses
#
# The systems are small (<= 155 basis functions), so many few-thread workers
# use the cores better than one 32-thread calculation at a time.
# ---------------------------------------------------------------------------
GUESSES=${GUESSES:-"atom huckel"}
NW=8                                        # number of concurrent pyscf workers
NT=$(( SLURM_CPUS_PER_TASK / NW ))          # threads per worker (NW*NT = cpus-per-task)
MEM_MB=$(( SLURM_MEM_PER_NODE / NW * 8 / 10 ))   # pyscf max_memory per worker (MB), 80% share
XYZDIR=../xyzfiles                          # relative to this directory

# SCF settings: the generous ones from rerun_problems.sh, so that a separate
# rerun pass should not be needed
MAX_CYCLE=1000        # DIIS iterations
NEWTON_MAX_CYCLE=200  # Newton iterations in the fallback
MAX_STAB=15           # stability-following rounds
STAB_TOL=1e-4         # |Hessian eigenvalue| below this is a zero mode, not an instability

# --- activate conda env ---
source /resnick/groups/sharmagroup/yichi/miniconda3/etc/profile.d/conda.sh
conda activate pyscf

# --- PATH ---
export PYTHONPATH=/resnick/groups/sharmagroup/yichi/pyscf:/resnick/groups/sharmagroup/yichi/afqmc:$PYTHONPATH
export PYSCF_EXT_PATH=/resnick/groups/sharmagroup/yichi/pyscf-forge

# --- threads: NT per worker, not the whole allocation ---
export OMP_NUM_THREADS=$NT
export OPENBLAS_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT
export NUMEXPR_NUM_THREADS=$NT
export VECLIB_MAXIMUM_THREADS=$NT

echo "Running on $(hostname) with $SLURM_CPUS_PER_TASK cores, ${SLURM_MEM_PER_NODE}MB RAM"
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
