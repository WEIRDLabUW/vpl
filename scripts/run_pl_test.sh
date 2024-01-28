#!/bin/sh

env="maze2d-pointmass-v0"
path="/home/max/Distributional-Preference-Learning/vpl/pref_datasets/$env/relabelled_queries_num10000_q1_s32"
model_type="VAE"

export WANDB_PROJECT='vpl-test'

python pref_learn/train.py \
    --comment="test" \
    --env=$env \
    --dataset_path=$path \
    --model_type=$model_type \
    --logging.output_dir="logs" \
    --seed 0