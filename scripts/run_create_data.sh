#!/bin/sh

envs=("maze2d-pointmass-fixed-v0" "maze2d-pointmass-v0" "maze2d-fourrooms-fixed-v0" "maze2d-fourrooms-v0" "kitchen-v0" "kitchen-fixed-v0")
sample_from_envs=(True False)
query=(1 10)
set_len=(8 32)
num_query=(10000 500)

for env in "${envs[@]}"
do
    for sample_from_env in "${sample_from_envs[@]}"
    do
        for q in "${query[@]}"
        do
            for s in "${set_len[@]}"
            do
                for n in "${num_query[@]}"
                do
                    python -m pref_learn.create_dataset --num_query $n --query_len $q --env=$env --sample_from_env=$sample_from_env --set_len=$s
                done
            done
        done
    done
done