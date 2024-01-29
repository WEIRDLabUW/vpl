#!/bin/bash

#SBATCH --job-name=sac
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/Variational-Preference-Learning
#SBATCH --export=all
#SBATCH --output=slurm_iql/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm_iql/%j-err.txt    # where STDERR goes
#SBATCH --array=0-1

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=online
export WANDB_PROJECT='iql-franka_kitchen'
export WANDB_GROUP='iql-franka_kitchen'

source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

env=$1
export MUJOCO_GL=egl

python experiments/run_iql.py \
        --env_name $env \
        --eval_interval 10000 \
        --eval_episodes 10 \
        --log_interval 10000 \
        --seed $SLURM_ARRAY_TASK_ID \
        --save_video True
