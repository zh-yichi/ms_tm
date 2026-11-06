#!/bin/bash
#SBATCH --job-name=tm_scf_rerun
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bdka-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --gpus-per-node=0
#SBATCH --constraint="scratch"
#SBATCH --no-requeue
#SBATCH -t 6:00:00
#SBATCH -e slurm_rerun.err
#SBATCH -o slurm_rerun.out

module purge
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
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
cd "$SLURM_SUBMIT_DIR"
./rerun_problems.sh results results_rerun
echo "done: $(date)"
