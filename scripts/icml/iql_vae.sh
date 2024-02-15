#!/bin/bash

#SBATCH --job-name=icml_vae
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sriyash@cs.washington.edu

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/sriyash/Variational-Preference-Learning
#SBATCH --export=all
#SBATCH --output=slurm_icml/vae-%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm_icml/vae-%j-err.txt    # where STDERR goes
#SBATCH --array=0-2


env="maze2d-twogoals-multimodal-v0"
model_type=VAE

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=offline
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

export WANDB_PROJECT=feb_debug
dataset_path=./pref_datasets/$env/relabelled_queries_num10000_q1_s16

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$dataset_path \
#     --model_type=$model_type \
#     --logging.output_dir="feb_debug" \
#     --seed $SLURM_ARRAY_TASK_ID \
#     --learned_prior=True \
#     --use_annealing=True \
#     --n_epochs=500 --early_stop=True


ckpt_dir="feb_debug/$env/$model_type/$WANDB_PROJECT/s$SLURM_ARRAY_TASK_ID"

export WANDB_PROJECT=feb_debug_policies

python experiments/run_iql.py \
        --env_name $env \
        --eval_interval 100000 \
        --eval_episodes 10 \
        --log_interval 10000 \
        --seed $SLURM_ARRAY_TASK_ID \
        --save_video True \
        --model_type $model_type \
        --ckpt $ckpt_dir \
        --debug=True \
        --use_reward_model=True