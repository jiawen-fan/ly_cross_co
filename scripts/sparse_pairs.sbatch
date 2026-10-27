#!/bin/bash
#SBATCH --job-name=lya-co-sparse
#SBATCH --account=pbreysse_lya_forest_co_0001
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sparse-%j.log
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun python -m pytest -q
srun python -u scripts/sparse_pairs.py \
 --co /users/jiawenfan/work/lya_CO/limlam_cartesian_v2_simplified/co_li_cartesian.npz \
 --lya /projects/pbreysse/lya_forest_CO/lya_storage/Lyman-alpha-ASTRID_z2.5_unzipped/Lyman-alpha-ASTRID_z2.5/map_ASTRID_true_0.25_z2.5.hdf5 \
 --nmesh 250 --sightlines 1000 --kmax 1.0 --rmax 80 --seed 42 \
 --output "results/sparse-${SLURM_JOB_ID}"
