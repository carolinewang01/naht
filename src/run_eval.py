#!/scratch/cluster/<uname>/conda_envs/oil/bin/python

import os
import subprocess
import argparse

from utils.nk_eval_utils import ENV_CONFIGS, wait_for_result, cleanup_temp_config
from utils.str2bool import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="e.g. 5v6")
    parser.add_argument("--log_folder", type=str, help="e.g. /scratch/cluster/.../open_eval_results/5v6/open_eval/qmix-vs-qix ")
    parser.add_argument("--expt_label", type=str, help="e.g. qmix-seed=1_qmix-seed=1_n-1")
    parser.add_argument("--dest_config_path", type=str, help="e.g. src/config/temp/default.yaml")
    parser.add_argument("--eval_seed", type=int, default=394820)
    parser.add_argument("--debug", type=str2bool, default=False)
    return parser.parse_args()
        
def perform_eval(env_nickname, dest_config_path, eval_seed, debug):
    ''' implement the logic to evaluate algorithm i against algorithm j
    '''
    env_config_name = ENV_CONFIGS[env_nickname]["env_config"]
    env_args = ENV_CONFIGS[env_nickname]["env_args"]
    dest_config_name = dest_config_path.replace('src/config/', '').replace('.yaml', '')
    exec = [f"python src/main.py",
            f"--seed={eval_seed}",
            f"--env-config={env_config_name}", 
            f"--config={dest_config_name}",
            f"--alg-config=open_dummy",
            "with",
            *[f"env_args.{k}={str(v)}" for k, v in env_args.items()],
            ]
    exec = " ".join(exec)
    if debug: 
        print("RUN_EVAL.PY: exec=", exec)
        print("RUN_EVAL.PY: dest_config_path=", dest_config_path)
        return
    else:
        print("RUN_EVAL.PY: exec=", exec)
        print("RUN_EVAL.PY: dest_config_path=", dest_config_path)
        subprocess.call(exec, shell=True) # this will wait for the result
    cleanup_temp_config(dest_config_path)


if __name__ == "__main__":
    '''this script is meant to be run by a condor node'''
    args = parse_args()

    env_nickname = args.env
    eval_seed = args.eval_seed
    log_folder = args.log_folder
    expt_label = args.expt_label
    debug = args.debug
    
    # get the config path for the algorithm
    dest_config_path = args.dest_config_path
    if not os.path.exists(dest_config_path):
        raise ValueError(f"Config path {dest_config_path} does not exist")
    perform_eval(env_nickname=env_nickname, 
                 dest_config_path=dest_config_path, 
                 eval_seed=eval_seed, 
                 debug=debug
                 )

