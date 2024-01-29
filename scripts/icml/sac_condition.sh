#!/bin/bash

#SBATCH --job-name=fourrooms_icml
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
#SBATCH --output=slurm_icml2/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm_icml2/%j-err.txt    # where STDERR goes
#SBATCH --array=0-3


env=$1
model_type=$2

HOME_DIR="/gscratch/weirdlab/sriyash/Variational-Preference-Learning"
export WANDB_MODE=online
source ${HOME}/.bashrc
conda activate offline
cd $HOME_DIR

export WANDB_PROJECT=icml_$env\_reward_learning_conditioned
dataset_path=icml_datasets/$env/relabelled_queries_num10000_q1_s32

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=$model_type \
    --logging.output_dir="logs" \
    --seed $SLURM_ARRAY_TASK_ID \
    --learned_prior=True \
    --use_annealing=True \


ckpt_dir="logs/$env/MLP/$WANDB_PROJECT/s$SLURM_ARRAY_TASK_ID"

export WANDB_PROJECT=icml_$env\_policy_conditioned

python experiments/run_sac_conditioned.py \
        --env_name $env \
        --eval_interval 50000 \
        --eval_episodes 100 \
        --log_interval 10000 \
        --seed $SLURM_ARRAY_TASK_ID \
        --save_video True \
        --model_type $model_type \
        --ckpt $ckpt_dir \
        --debug=True