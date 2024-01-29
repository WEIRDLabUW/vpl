#! /bin/sh

# Run the SAC program
export WANDB_PROJECT='iql-test-fr'
export WANDB_GROUP='iql-test-fr'

envs=("kitchen-v0" "kitchen-fixed-v0")
export MUJOCO_GL=egl

for env in "${envs[@]}"
do
    # python experiments/run_iql.py \
    #         --env_name $env \
    #         --eval_interval 10000 \
    #         --eval_episodes 10 \
    #         --log_interval 10000 \
    #         --seed 0 \
    #         --save_video True > slurm/$env.log 2>&1 &
    sbatch scripts/slurm_iql.sh $env
done
