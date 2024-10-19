#!/bin/bash
expname="5v6-naht-poam"
logdir="~/naht_results/5v6/poam"
dt=$(date '+%d-%m-%Y-%H-%M-%S')

# Create log directory if it does not exist
mkdir -p "$logdir"

python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=112358