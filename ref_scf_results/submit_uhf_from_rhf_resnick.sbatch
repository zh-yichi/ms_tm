#!/bin/bash
#SBATCH --job-name=tm_uhf_rhfchk
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # fits every CPU node type (32/56/64 cores per node)
#SBATCH --mem=64G                  # 2 GB/core; well under the 192 GB of a standard node
#SBATCH --time=08:00:00
#SBATCH --output=slurm_rhfchk.out
#SBATCH --error=slurm_rhfchk.err

# ---------------------------------------------------------------------------
# UHF only (sfX2C-1e, cc-pVDZ-DK) for all transition-metal complexes, started
# from the orbitals of the lowest RHF/ROHF solution found so far.
#
# For each system run_scf.py looks at <dir>/<stem>/<stem>.rhf.json in every
# folder of RHF_DIRS, takes the converged record with the lowest energy and
# reads the matching <stem>.rhf.chk.  alpha density = doubly + singly occupied
# orbitals, beta density = doubly occupied orbitals.  Results go to OUTDIR
# (same layout as the other result folders; json has init_chk, init_rhf_energy).
#
# The .chk files are not in git.  If the chkfile of the lowest RHF/ROHF is not
# on this machine, RHF/ROHF is recomputed first with the best initial guess for
# that system from GUESS_TABLE (column best_RHF; "All" -> minao), written to
# OUTDIR/<stem>/<stem>.rhf.{chk,json}, and the UHF starts from that.  The rhf
# json notes it if the recomputed energy differs from the lowest known one.
# Finished UHF records (json present) are skipped, so the job can be resubmitted.
#
#   sbatch submit_uhf_from_rhf_resnick.sbatch
# ---------------------------------------------------------------------------
RHF_DIRS=${RHF_DIRS:-"results_minao results_atom results_huckel"}
OUTDIR=${OUTDIR:-results_rhfchk}
GUESS_TABLE=${GUESS_TABLE:-guess_comparison.tsv}
NW=8                                        # number of concurrent pyscf workers
NT=$(( SLURM_CPUS_PER_TASK / NW ))          # threads per worker (NW*NT = cpus-per-task)
MEM_MB=$(( SLURM_MEM_PER_NODE / NW * 8 / 10 ))   # pyscf max_memory per worker (MB), 80% share
XYZDIR=../xyzfiles                          # relative to this directory

# SCF settings: same as submit_guess_resnick.sbatch
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
echo "rhf folders: $RHF_DIRS -> $OUTDIR   workers: $NW  threads/worker: $NT  max_memory/worker: ${MEM_MB} MB"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

# unique xyz files (identical files appear in several subfolders)
mapfile -t FILES < <(ls "$XYZDIR"/*/*.xyz | awk -F/ '{print $NF, $0}' | sort -u -k1,1 | awk '{print $2}')
echo "${#FILES[@]} unique systems   start: $(date)"

for ((w=0; w<NW; w++)); do
  sub=()
  for ((i=w; i<${#FILES[@]}; i+=NW)); do sub+=("${FILES[$i]}"); done
  python run_scf.py --methods uhf --rhf-chk-dirs $RHF_DIRS --guess-table "$GUESS_TABLE" --outdir "$OUTDIR" --memory "$MEM_MB" \
      --max-cycle $MAX_CYCLE --newton-max-cycle $NEWTON_MAX_CYCLE \
      --max-stab $MAX_STAB --stab-tol $STAB_TOL \
      "${sub[@]}" > "logs/rhfchk_worker$w.out" 2>&1 &
done
wait

echo "== RHF/ROHF recomputed because the chkfile was missing:"
grep -h 'recomputed (chkfile missing)' logs/rhfchk_worker*.out || echo "none"
echo "== skipped systems:"
grep -h '^skip (' logs/rhfchk_worker*.out | grep -v '^skip (done)' || echo "none"
python collect_summary.py "$OUTDIR"
echo "done: $(date)"
