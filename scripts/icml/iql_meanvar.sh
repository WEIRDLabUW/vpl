#!/bin/bash

#SBATCH --job-name=icml_meanvar
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
#SBATCH --output=slurm_icml/mean_var-%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm_icml/mean_var-%j-err.txt    # where STDERR goes
#SBATCH --array=0-4


env="medium-twogoals-v0"
model_type=MeanVar

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=online
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

export WANDB_PROJECT=imcl_feb_reward_model
dataset_path=./pref_datasets/medium-twogoals-v0/relabelled_queries_num10000_q1_s32

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$dataset_path \
#     --model_type=$model_type \
#     --logging.output_dir="icml_feb_logs" \
#     --seed $SLURM_ARRAY_TASK_ID \
#     --learned_prior=True \
#     --use_annealing=True \
#     --reward_scaling=1.0 \
#     --n_epochs=500 --early_stop=True --patience=10


ckpt_dir="icml_feb_logs/$env/$model_type/$WANDB_PROJECT/s$SLURM_ARRAY_TASK_ID"

export WANDB_PROJECT=icml_feb_policies_exp

python experiments/run_iql.py \
        --env_name $env \
        --eval_interval 10000 \
        --eval_episodes 10 \
        --log_interval 10000 \
        --seed $SLURM_ARRAY_TASK_ID \
        --save_video True \
        --model_type $model_type \
        --ckpt $ckpt_dir \
        --debug=True \
        --use_reward_model=True