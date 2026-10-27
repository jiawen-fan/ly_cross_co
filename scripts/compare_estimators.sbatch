#!/bin/bash
#SBATCH --job-name=lya-co-compare
#SBATCH --account=pbreysse_lya_forest_co_0001
#SBATCH --partition=htc
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

# Use supplied maps by default; two arguments override these paths.
CO_FILE="/users/jiawenfan/work/lya_CO/limlam_cartesian_v2_simplified/co_li_cartesian.npz"
LYA_FILE="/projects/pbreysse/lya_forest_CO/lya_storage/Lyman-alpha-ASTRID_z2.5_unzipped/Lyman-alpha-ASTRID_z2.5/map_ASTRID_true_0.25_z2.5.hdf5"
if [[ $# -eq 2 ]]; then
    CO_FILE="$1"
    LYA_FILE="$2"
elif [[ $# -ne 0 ]]; then
    echo 'Supply zero arguments for default maps, or both CO.npz and LyAlpha.hdf5' >&2
    exit 2
fi
echo "CO input: $CO_FILE"
echo "Ly-alpha input: $LYA_FILE"
echo 'Requested redshift label: 2.6. Default Ly-alpha filename says z2.5; verify snapshot metadata before physical interpretation.'
# --redshift records the requested label; it does not transform either map.
srun python scripts/compare_estimators.py \
    --co "$CO_FILE" --lya "$LYA_FILE" --nmesh 10 --boxsize 250 250 250 --redshift 2.6 \
    --output "results/comparison-${SLURM_JOB_ID}-astrid"
srun python scripts/compare_estimators.py \
    --co "$CO_FILE" --lya "$LYA_FILE" --nmesh 10 --boxsize 250 250 250 --redshift 2.6 \
    --window cosine --r-max 80 \
    --output "results/comparison-${SLURM_JOB_ID}-astrid-taper"
