#!/usr/bin/env bash
# Simple BASH script to run an example experiment

set -e

LAYERS="[] [64] [64,64] [64,64,64]"
SEEDS="0 42 1337"

for layers in $LAYERS
do
for seed in $SEEDS
do
    python -m mlflow_hydra.experiment \
        input.data_file=./data/wines-data.csv \
        input.run_name=\"layers:$layers\" \
        train.model.layers=$layers \
        train.random_seed=$seed
done
done
