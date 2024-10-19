#!/bin/bash
expname="5v6-mappo-agent"
logdir="~/naht_results/5v6/mappo"
dt=$(date '+%d-%m-%Y-%H-%M-%S')

# Create log directory if it does not exist
mkdir -p "$logdir"

python src/main.py --env-config=sc2 --config=default/default_5v6 --alg-config=sc2/mappo with env_args.map_name=5m_vs_6m --seed=112358
