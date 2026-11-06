#!/bin/bash
#SBATCH --job-name=tm_scf
#SBATCH --partition=gpuMI100x8
#SBATCH --account=becu-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64         # half the node (128 cores total)
#SBATCH --mem=900g                 # ~half of the ~1.8-1.9 TB usable
#SBATCH --gpus-per-node=0          # CPU-only
#SBATCH --constraint="scratch"
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH -e slurm.err
#SBATCH -o slurm.out

# ---------------------------------------------------------------------------
# RHF/ROHF + UHF (sfX2C-1e, cc-pVDZ-DK) for all transition-metal complexes.
# The 75 unique xyz files are split round-robin over NW independent workers,
# each running run_scf.py with NT OpenMP threads (NW*NT = cpus-per-task).
# Finished systems (json present) are skipped, so the job can be resubmitted.
# ---------------------------------------------------------------------------
NW=8                               # number of concurrent pyscf workers
NT=$(( SLURM_CPUS_PER_TASK / NW )) # threads per worker
MEM_MB=$(( 900000 / NW * 8 / 10 )) # pyscf max_memory per worker (MB), 80% share
XYZDIR=../xyzfiles                 # relative to this directory
OUTDIR=results

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

# NOTE!!!! THIS IS AN AMD MACHINE!!
echo "node: $(hostname)"
free -g
echo "workers: $NW  threads/worker: $NT  max_memory/worker: ${MEM_MB} MB"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUTDIR" logs

# unique xyz files (identical files appear in both top-pct and top-t1)
mapfile -t FILES < <(ls "$XYZDIR"/*/*.xyz | awk -F/ '{print $NF, $0}' | sort -u -k1,1 | awk '{print $2}')
echo "${#FILES[@]} unique systems"

for ((w=0; w<NW; w++)); do
  sub=()
  for ((i=w; i<${#FILES[@]}; i+=NW)); do sub+=("${FILES[$i]}"); done
  python run_scf.py --outdir "$OUTDIR" --memory "$MEM_MB" "${sub[@]}" > "logs/worker$w.out" 2>&1 &
done
wait

python collect_summary.py "$OUTDIR" | tee scf_summary.out
echo "done: $(date)"
