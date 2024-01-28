#!/bin/sh

env="maze2d-fourrooms-v0"
path="/home/max/Distributional-Preference-Learning/vpl/pref_datasets/$env/relabelled_queries_num10000_q1_s16"

export WANDB_PROJECT='pointmass-reward-models'

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=MLP \
    --logging.output_dir="logs_$env" \
    --seed 0

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=Categorical \
    --logging.output_dir="logs_$env" \
    --seed 0

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=MeanVar \
    --logging.output_dir="logs_$env" \
    --seed 0

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=VAE \
    --learned_prior=True \
    --logging.output_dir="logs_$env" \
    --seed 0