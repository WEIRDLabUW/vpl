#!/bin/bash

envs=("maze2d-fourrooms-v0" "maze2d-fourrooms2-v0")
models=("MLP" "VAE")

for env in "${envs[@]}"
do
    for model in "${models[@]}"
    do
        sbatch scripts/icml/sac_reward.sh $env $model
    done
done

envs=("maze2d-fourrooms-v0" "maze2d-fourrooms2-v0" "maze2d-pointmass-v0")
models=("MLP" "VAE")

for env in "${envs[@]}"
do
    for model in "${models[@]}"
    do
        sbatch scripts/icml/sac_condition.sh $env $model
    done
done