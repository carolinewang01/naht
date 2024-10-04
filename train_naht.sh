#!/bin/bash
expname="5v6-naht-poam"
logdir="~/naht_results/5v6/poam"
dt=$(date '+%d-%m-%Y-%H-%M-%S')

# Create log directory if it does not exist
mkdir -p "$logdir"

nohup python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=112358 > "${logdir}/${expname}_seed=112358_${dt}.out" 2>&1 &
nohup python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=1285842 > "${logdir}/${expname}_seed=1285842_${dt}.out" 2>&1 & 
nohup python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=78590 > "${logdir}/${expname}_seed=78590_${dt}.out" 2>&1 &
nohup python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=38410 > "${logdir}/${expname}_seed=38410_${dt}.out" 2>&1 &
nohup python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=93718 > "${logdir}/${expname}_seed=93718_${dt}.out" 2>&1 &
