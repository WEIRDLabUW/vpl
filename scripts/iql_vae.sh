#!/bin/bash

#SBATCH --job-name=debug
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
#SBATCH --output=debug/%j-out.txt   # where STDOUT goes
#SBATCH --error=debug/%j-err.txt    # where STDERR goes
#SBATCH --array=0-2

HOME_DIR="/gscratch/weirdlab/sriyash/vpl"
export WANDB_MODE=online
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

env="maze2d-twogoals-multimodal-v0"
model_type="VAE"
dataset_path=./pref_datasets/maze2d-twogoals-multimodal-v0/relabelled_queries_num5000_q1_s16

export WANDB_PROJECT=release_debug
export WANDB_GROUP=$model_type
comment="vae"
python pref_learn/train.py \
        --comment=$comment \
        --env=$env \
        --dataset_path=$dataset_path \
        --model_type=$model_type \
        --logging.output_dir="logs" \
        --seed 0 \
        --learned_prior=True \
        --use_annealing=True \
        --n_epochs=500 --debug_plots True

ckpt_dir="logs/$env/$model_type/$comment/s0"

export WANDB_PROJECT=release_debug
export WANDB_GROUP=$model_type

python experiments/run_iql.py \
        --config=configs/maze_config.py \
        --env_name $env \
        --save_video True \
        --seed 0 \
        --eval_interval 100000 --eval_episodes 10 \
        --model_type $model_type --preference_dataset_path $dataset_path \
        --vae_norm "none" --comp_size 1000 \
        --batch_size 256 --use_reward_model=True --ckpt $ckpt_dir --vae_sampling True
