#! /bin/sh

# Run the SAC program
export WANDB_PROJECT='iql-test'
export WANDB_GROUP='iql-test'

envs=("kitchen-v0" "kitchen-fixed-v0")

for env in "${envs[@]}"
do
    python experiments/run_iql.py --env_name $env --max_steps 1000 --eval_interval 500 --eval_episodes 2 --log_interval 100 --seed 0 --save_video True
done
