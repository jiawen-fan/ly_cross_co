#!/bin/bash
#SBATCH --job-name=lya-recon-compare
#SBATCH --account=pbreysse_lya_forest_co_0001
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=reconstruction-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${MEAN_FLUX:?Set MEAN_FLUX to the normalization used by the ASTRID true map}"
DACHSHUND=${DACHSHUND:-/users/jiawenfan/work/lya_CO/dachshund/dachshund.exe}
CO=${CO:-/users/jiawenfan/work/lya_CO/limlam_cartesian_v2_simplified/co_li_cartesian.npz}
ASTRID=${ASTRID:-/projects/pbreysse/lya_forest_CO/lya_storage/Lyman-alpha-ASTRID_z2.5_unzipped/Lyman-alpha-ASTRID_z2.5}
noise_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    noise_args=(--noise-seed "$((10000 + SLURM_ARRAY_TASK_ID))")
fi
srun python -u scripts/reconstruction_compare.py \
    --co "$CO" --lya-map "$ASTRID/map_ASTRID_true_0.25_z2.5.hdf5" \
    --spectra "$ASTRID/spectra_ASTRID_noiseless_z2.5_1000_voxels.hdf5" \
    --dachshund "$DACHSHUND" --mean-flux "$MEAN_FLUX" \
    --lya-power-source "${LYA_POWER_SOURCE:-full-map}" --lya-map-field "${LYA_MAP_FIELD:-delta}" \
    --nmesh 250 --sightlines 1000 --stride 5 --kmax 1 --bins 40 \
    --co-samples 20000 --rmax 80 --dr .1 "${noise_args[@]}" \
    --output "results/reconstruction-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-reference}"
