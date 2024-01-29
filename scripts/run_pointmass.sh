#!/bin/sh

env="maze2d-pointmass-v0"
path="/home/max/Distributional-Preference-Learning/vpl/pref_datasets/$env/relabelled_queries_num10000_q1_s32"

export WANDB_PROJECT='pointmass-reward-models-final-debug'

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=MLP \
#     --logging.output_dir="logs/$env_$WANDB_PROJECT" \
#     --seed 0

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=Categorical \
#     --logging.output_dir="logs/$env_$WANDB_PROJECT" \
#     --seed 0

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=MeanVar \
#     --logging.output_dir="logs/$env_$WANDB_PROJECT" \
#     --seed 0

# python pref_learn/train.py \
#     --comment=$WANDB_PROJECT \
#     --env=$env \
#     --dataset_path=$path \
#     --model_type=VAE \
#     --learned_prior=True \
#     --logging.output_dir="logs/$env_$WANDB_PROJECT" \
#     --seed 0 \
#     --use_annealing=True \

for model_type in "VAE"
do
    ckpt="/home/max/Distributional-Preference-Learning/vpl/logs/pointmass-reward-models-final/maze2d-pointmass-v0/VAE/pointmass-reward-models-final/s0"
    python experiments/run_sac_conditioned.py \
        --env_name $env \
        --eval_interval 50000 \
        --eval_episodes 100 \
        --log_interval 10000 \
        --seed 0 \
        --save_video True \
        --model_type $model_type \
        --ckpt $ckpt \
        --debug=False \
        # --fixed_mode=False
done