#!/bin/sh

env="maze2d-twogoals-multimodal-v0"
path="./pref_datasets/$env/relabelled_queries_num10000_q1_s16"

export WANDB_PROJECT='post_icml_reward'

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=MLP \
#     --logging.output_dir="icml_final_logs_$env" \
#     --seed 0 \
#     --batch_size 256

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=Categorical \
#     --logging.output_dir="icml_final_logs_$env" \
#     --seed 0 --batch_size 256

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=MeanVar \
#     --logging.output_dir="icml_final_logs_$env" \
#     --seed 0 --batch_size 256

python pref_learn/train.py \
    --comment=$WANDB_PROJECT \
    --env=$env \
    --dataset_path=$path \
    --model_type=VAE \
    --learned_prior=True \
    --logging.output_dir="post_icml_logs/$env" \
    --seed 0 \
    --use_annealing True \
    --batch_size 256 --early_stop True --patience 10

