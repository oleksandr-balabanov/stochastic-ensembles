#!/usr/bin/env bash
export IMAGE=/mimer/NOBACKUP/groups/snic2022-22-448/uncertainty/containers/s.img
export OUTPUT=/mimer/NOBACKUP/groups/snic2022-22-448/CVPR_revision/toy_multiswag_2000_0005/
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
# #SBATCH -C MEM1536

echo "################################################################################"
echo "Running slurm job"
echo "################################################################################"
set -e

cd ${HOME}/stochastic_ensembles/stochastic-ensembles/toy
singularity exec --cleanenv --no-home --env PYTHONNOUSERSITE=1 --nv ${IMAGE} python3 -u ./run_eval.py --output_dir ${OUTPUT} $@
exit 0
EOT

# ./start_toy.sh --method multiswag --num_swag_samples 20 --domain out --case 1 --compute_save_softmax_probs True