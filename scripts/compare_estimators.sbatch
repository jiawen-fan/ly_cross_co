#!/bin/bash
#SBATCH --job-name=lya-co-compare
#SBATCH --partition=standard-s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=compare-%j.log
set -euo pipefail

# Submit from the repository root with your scientific Python environment active.
# Four allocated CPUs are a ceiling, not a promise of four-core utilization.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

srun python -m pytest -q
srun python scripts/compare_estimators.py \
    --nmesh 10 --output "results/comparison-${SLURM_JOB_ID}-synthetic"
srun python scripts/compare_estimators.py \
    --nmesh 10 --window cosine --r-max 80 \
    --output "results/comparison-${SLURM_JOB_ID}-synthetic-taper"

# Pass the existing CO npz and Ly-alpha delta_F HDF5 as two sbatch arguments.
if [[ $# -eq 2 ]]; then
    srun python scripts/compare_estimators.py \
        --co "$1" --lya "$2" --nmesh 10 --boxsize 250 250 250 --redshift 2.6 \
        --output "results/comparison-${SLURM_JOB_ID}-astrid"
    srun python scripts/compare_estimators.py \
        --co "$1" --lya "$2" --nmesh 10 --boxsize 250 250 250 --redshift 2.6 \
        --window cosine --r-max 80 \
        --output "results/comparison-${SLURM_JOB_ID}-astrid-taper"
elif [[ $# -ne 0 ]]; then
    echo 'Supply either zero arguments or both CO.npz and LyAlpha.hdf5' >&2
    exit 2
fi
