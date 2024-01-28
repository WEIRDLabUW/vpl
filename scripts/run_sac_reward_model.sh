#! /bin/sh

# Run the SAC program
export WANDB_PROJECT='sac-rm-test'
export WANDB_GROUP='sac-rm-test'

# envs=("maze2d-pointmass-fixed-v0" "maze2d-pointmass-v0" "maze2d-fourrooms-fixed-v0" "maze2d-fourrooms-v0" "kitchen-v0" "kitchen-fixed-v0")
envs=("maze2d-fourrooms-v0")
ckpt="logs_maze2d-fourrooms-v0/maze2d-fourrooms-v0/MLP/pointmass-reward-models/s0"

for env in "${envs[@]}"
do
    # sbatch scripts/slurm_sac.sh $env
    # python experiments/run_sac.py --env_name $env --max_steps 1000 --start_steps 100 --eval_interval 500 --eval_episodes 2 --log_interval 100 --seed 0 --save_video True
    python experiments/run_sac_reward_model.py \
            --env_name $env \
            --eval_interval 50000 \
            --eval_episodes 100 \
            --log_interval 10000 \
            --seed 0 \
            --save_video True \
            --model_type MLP \
            --ckpt $ckpt
done
