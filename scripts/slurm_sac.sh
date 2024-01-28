#!/bin/bash

#SBATCH --job-name=sac
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/Variational-Preference-Learning
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes
#SBATCH --array=0-2

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=online
export WANDB_PROJECT=sac-runs-final-seeds

source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

# env=$1

# python experiments/run_sac.py \
#     --env_name $env \
#     --eval_episodes 100 \
#     --seed 0 &
#     --save_video True

envs=("maze2d-pointmass-fixed-v0" "maze2d-pointmass-v0" "maze2d-fourrooms-fixed-v0" "maze2d-fourrooms-v0" "kitchen-v0" "kitchen-fixed-v0")

for env in "${envs[@]}"
do
    python experiments/run_sac.py \
    --env_name $env \
    --eval_episodes 100 \
    --seed $SLURM_ARRAY_TASK_ID &
    # --save_video True
done