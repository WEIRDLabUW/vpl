#!/bin/bash

#SBATCH --job-name=multimodal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=8:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/Variational-Preference-Learning
#SBATCH --export=all
#SBATCH --output=slurm_multimodal/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm_multimodal/%j-err.txt    # where STDERR goes
#SBATCH --array=0-2

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=online
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

env="maze2d-twogoals-multimodal-v0"

# ## multimodal env testing

# WANDB_PROJECT=multimodal python experiments/run_iql.py \
#         --config=configs/maze_config.py \
#         --env_name $env \
#         --save_video True \
#         --seed $SLURM_ARRAY_TASK_ID \
#         --eval_interval 100000 --eval_episodes 10 \
#         --append_goal True

## testing VAE + multimodal
model_type="VAE"
dataset_path=./pref_datasets/$env/relabelled_queries_num5000_q1_s16

export WANDB_PROJECT=multimodal-rewards
# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$dataset_path \
#     --model_type=$model_type \
#     --logging.output_dir="logs" \
#     --seed $SLURM_ARRAY_TASK_ID \
#     --learned_prior=True \
#     --use_annealing=True \
#     --n_epochs=500 --early_stop=True --patience=10

ckpt_dir="logs/$env/$model_type/$WANDB_PROJECT/s$SLURM_ARRAY_TASK_ID"

# WANDB_PROJECT=multimodal python experiments/run_iql.py \
#         --config=configs/maze_config.py \
#         --env_name $env \
#         --save_video True \
#         --seed $SLURM_ARRAY_TASK_ID \
#         --eval_interval 100000 --eval_episodes 10 \
#         --use_reward_model True \
#         --model_type $model_type \
#         --ckpt $ckpt_dir --append_goal True --fix_mode 0

# WANDB_PROJECT=multimodal python experiments/run_iql.py \
#         --config=configs/maze_config.py \
#         --env_name $env \
#         --save_video True \
#         --seed $SLURM_ARRAY_TASK_ID \
#         --eval_interval 100000 --eval_episodes 10 \
#         --use_reward_model True \
#         --model_type $model_type \
#         --ckpt $ckpt_dir --append_goal True

# WANDB_PROJECT=multimodal python experiments/run_iql.py \
#         --config=configs/maze_config.py \
#         --env_name $env \
#         --save_video True \
#         --seed $SLURM_ARRAY_TASK_ID \
#         --eval_interval 100000 --eval_episodes 10 \
#         --use_reward_model True \
#         --model_type $model_type \
#         --ckpt $ckpt_dir --append_goal True --fix_mode 1

WANDB_PROJECT=multimodal python experiments/run_iql.py \
        --config=configs/maze_config.py \
        --env_name $env \
        --save_video True \
        --seed $SLURM_ARRAY_TASK_ID \
        --eval_interval 100000 --eval_episodes 10 \
        --use_reward_model True \
        --model_type $model_type \
        --ckpt $ckpt_dir
