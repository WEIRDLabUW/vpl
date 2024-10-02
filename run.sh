#! /bin/sh

python -m pref_learn.create_dataset --num_query=5000 --env="maze2d-twogoals-multimodal-v0" --query_len=1 --set_len=8

env="maze2d-twogoals-multimodal-v0"
# model_type="VAEClassifier"
model_type=$1
vae_norm=$2
dataset_path=./pref_datasets/maze2d-twogoals-multimodal-v0/relabelled_queries_num5000_q1_s16

for seed in 0 1 2; do
export WANDB_PROJECT=vpl
export WANDB_GROUP=$model_type
comment=$WANDB_PROJECT-$WANDB_GROUP
python pref_learn/train.py \
        --comment=$comment \
        --env=$env \
        --dataset_path=$dataset_path \
        --model_type=$model_type \
        --logging.output_dir="logs" \
        --seed $seed \
        --learned_prior=True \
        --use_annealing=True \
        --n_epochs=500 --debug_plots True

ckpt_dir="logs/$env/$model_type/$comment/s$seed"

export WANDB_PROJECT=vpl_iql
export WANDB_GROUP=$model_type
python experiments/run_iql.py \
        --config=configs/maze_config.py \
        --env_name $env \
        --save_video True \
        --seed $seed \
        --eval_interval 100000 --eval_episodes 10 \
        --model_type $model_type --preference_dataset_path $dataset_path \
        --vae_norm "$vae_norm" --comp_size 1000 \
        --batch_size 256 --use_reward_model=True --ckpt $ckpt_dir \
        --vae_sampling True
done
