#!/bin/bash

#SBATCH --job-name=maze_hidden
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/Variational-Preference-Learning
#SBATCH --export=all
#SBATCH --output=maze_hidden/%j-out.txt   # where STDOUT goes
#SBATCH --error=maze_hidden/%j-err.txt    # where STDERR goes
#SBATCH --array=0-4

HOME_DIR="/gscratch/weirdlab/sriyash/vpl"
export WANDB_MODE=online
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

env="maze2d-hidden-v0"
model_type="VAE"
dataset_path=data/test_maze_dataset

export WANDB_PROJECT=maze_hidden_rewards
export WANDB_GROUP=$model_type
comment="vae-hidden"
python pref_learn/train.py \
        --comment=$comment \
        --env=$env \
        --dataset_path=$dataset_path \
        --model_type=$model_type \
        --logging.output_dir="maze_hidden" \
        --seed $SLURM_ARRAY_TASK_ID \
        --learned_prior=True \
        --use_annealing=True \
        --n_epochs=500 --latent_dim 16 --batch_size 256 --hidden_dim 256

ckpt_dir="maze_hidden/$env/$model_type/$comment/s$SLURM_ARRAY_TASK_ID"

export WANDB_PROJECT=maze_hidden_policy
export WANDB_GROUP=$model_type

python experiments/run_iql.py \
        --config=configs/maze_config.py \
        --env_name $env \
        --save_video True \
        --seed $SLURM_ARRAY_TASK_ID \
        --eval_interval 10000 --eval_episodes 10 \
        --model_type $model_type --preference_dataset_path $dataset_path \
        --vae_norm "max" --comp_size 1000 \
        --batch_size 1024 --use_reward_model=True --ckpt $ckpt_dir
