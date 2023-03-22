#!/usr/bin/env bash
export BASE=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/
export IMAGE=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/containers/s.img
export OUTPUT=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/CIFAR_10_multiswag_300_005/
export WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/wandb_cache
export WANDB_OUTPUT=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/
export SLURM_PROJECT="${SLURM_PROJECT:-SNIC2022-22-448}"
export SLURM_GPU="${SLURM_GPU:-T4}"
mkdir -p ${WANDB_CACHE_DIR}
mkdir -p ${OUTPUT}
sbatch <<EOT
#!/bin/bash

#SBATCH -A ${SLURM_PROJECT} -p alvis
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gpus-per-node=${SLURM_GPU}:1
#SBATCH -t 7-00:00:00

echo "################################################################################"
echo "Running slurm job"
echo "################################################################################"
set -e

cd ${HOME}/stochastic_ensembles/stochastic-ensembles/CIFAR/
singularity exec --cleanenv --no-home --env PYTHONNOUSERSITE=1 --nv ${IMAGE} python3 -u ./run_eval.py --output_dir ${OUTPUT} $@
exit 0
EOT
