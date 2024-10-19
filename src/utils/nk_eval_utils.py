import copy
import os
import shutil
import random
import time
import json
import yaml

from utils.load_utils import get_expt_paths
from main import recursive_dict_update


ENV_CONFIGS = {
    # format: env_nickname: (env full name, env config name)
        "3m": {
            "full_name": "3m", 
            "env_config": "sc2",
            "env_args": {"map_name": "3m"}
        },
        "8m": {
            "full_name": "8m", 
            "env_config": "sc2",
            "env_args": {"map_name": "8m"}
        },
        "5v6": {
            "full_name": "5m_vs_6m", 
            "env_config": "sc2",
            "env_args": {"map_name": "5m_vs_6m"}
        }, 
        "8v9": {
            "full_name": "8m_vs_9m", 
            "env_config": "sc2",
            "env_args": {"map_name": "8m_vs_9m"}
        },
        "10v11": {
            "full_name": "10m_vs_11m", 
            "env_config": "sc2",
            "env_args": {"map_name": "10m_vs_11m"}
        },
        "3sv5z": {
            "full_name": "3s_vs_5z", 
            "env_config": "sc2",
            "env_args": {"map_name": "3s_vs_5z"}
        },
        "mpe-pp": {
            "full_name": "mpe-pp",
            "env_config": "gymma",
            "env_args": {"key": "mpe:PredatorPrey-v0",
                         "time_limit": 100, 
                         "pretrained_wrapper": "PretrainedTag"
            }
        }

    }


def get_runs(algorithm: str, expt_path: str, expt_basenames: list):
    '''
    algorithm: string name of algorithm
    expt_basenames: list of valid experiment names to match
    '''
    results_path = os.path.join(expt_path, algorithm)
    assert os.path.exists(results_path), f"{results_path} does not exist"
    runs = []
    for nm in expt_basenames:
        # match all experiments with the given basename
        ret = get_expt_paths(base_folder=results_path, subfolder="models", expt_regex=nm)
        runs = [*runs, *ret]
    return runs

def all_pairs(
        algo_i: str, algo_j: str,
        algo_i_runs: list, algo_j_runs: list, 
        existing_seed_pairs: list=[],
                 ):
    '''
    returns all possible pairs between algo_i_runs and algo_j_runs
    this function doesn't care if the seeds of algo i and algo j match or not
    finally, doesn't sort the pairs 
    '''
    
    run_pairs = []
    # get paths for existing seed pairs
    algo_i = algo_i.replace("-", "_")
    algo_j = algo_j.replace("-", "_")
    for (algo1_nm, algo1_seed), (algo2_nm, algo2_seed) in existing_seed_pairs:
        if algo_i != algo1_nm:
            print(f"Warning: algo_i {algo_i} does not match algo1_nm {algo1_nm}")
        if algo_j != algo2_nm:
            print(f"Warning: algo_j {algo_j} does not match algo2_nm {algo2_nm}")
            
        for algo_i_path in algo_i_runs:
            if algo1_seed in algo_i_path:
                matched_i_path = algo_i_path
                break
        for algo_j_path in algo_j_runs:
            if algo2_seed in algo_j_path:
                matched_j_path = algo_j_path
                break
        run_pairs.append((matched_i_path, matched_j_path))

    ### generate all possible pairs pairs for algo i and algo j ###
    assert len(algo_i_runs) >= 1 and len(algo_j_runs) >= 1
    for algo_i_run in algo_i_runs:
        for algo_j_run in algo_j_runs: 
            run_pair = (algo_i_run, algo_j_run)
            if run_pair not in run_pairs:
                run_pairs.append(run_pair)
    return run_pairs


def random_pairs(
        algo_i: str, algo_j: str,
        algo_i_runs: list, algo_j_runs: list, 
        num_pairs: int=3,
        existing_seed_pairs: list=[],
        match_selfplay_seeds: bool=True
                 ):
    '''
    match_selfplay: if true, then for algo_i == algo_j, the pairs will be sampled 
    with matching seeds. If false, then for algo_i == algo_j, the pairs will be
    sampled with mismatched seeds.
    '''
    
    run_pairs = []
    # get paths for existing seed pairs
    algo_i = algo_i.replace("-", "_")
    algo_j = algo_j.replace("-", "_")
    for (algo1_nm, algo1_seed), (algo2_nm, algo2_seed) in existing_seed_pairs:
        if algo_i != algo1_nm:
            print(f"Warning: algo_i {algo_i} does not match algo1_nm {algo1_nm}")
        if algo_j != algo2_nm:
            print(f"Warning: algo_j {algo_j} does not match algo2_nm {algo2_nm}")
            
        for algo_i_path in algo_i_runs:
            if algo1_seed in algo_i_path:
                matched_i_path = algo_i_path
                break
        for algo_j_path in algo_j_runs:
            if algo2_seed in algo_j_path:
                matched_j_path = algo_j_path
                break
        if algo_i == algo_j:
            if match_selfplay_seeds: # only add the pair if the paths are the same
                if matched_i_path == matched_j_path:
                    run_pairs.append((matched_i_path, matched_j_path))
            else: # add the pair if the paths are different
                if matched_i_path != matched_j_path:
                    run_pairs.append((matched_i_path, matched_j_path))
        else:
            run_pairs.append((matched_i_path, matched_j_path))

    ### generate NEW pairs for algo i and algo j ###
    assert len(algo_i_runs) >= num_pairs and len(algo_j_runs) >= num_pairs
    algo_i_runs, algo_j_runs = copy.deepcopy(algo_i_runs), copy.deepcopy(algo_j_runs)

    while len(run_pairs) < num_pairs:
        if algo_i == algo_j:
            if  match_selfplay_seeds: # sort runs to ensure that the same seed pairs are sampled
                algo_i_runs.sort()
                algo_j_runs.sort()
            else: # resort until all pairs are mismatched seeds
                random.shuffle(algo_i_runs)
                # handle the case where the same seed is sampled for both algo_i and algo_j
                while any([algo_i_runs[k] == algo_j_runs[k] for k in range(len(algo_i_runs))]):
                    random.shuffle(algo_j_runs)
        else:
            random.shuffle(algo_i_runs)
            random.shuffle(algo_j_runs)

        new_run_pairs = list(zip(algo_i_runs[:num_pairs], algo_j_runs[:num_pairs]))
        # only add new pairs
        for run_pair in new_run_pairs:
            
            if run_pair not in run_pairs :
                run_pairs.append(run_pair)

    return run_pairs

def get_seed_pairs(log_folder):
    """Get seed pairs for each experiment"""
    eval_paths = os.listdir(os.path.join(log_folder, "sacred"))
    seed_pairs = []
    for p in eval_paths: 
        algo_i_seed = p.split("_")[1]
        algo_j_seed = p.split("_")[2]
        algo_i_nm, algo_i_seed = algo_i_seed.split("-seed=")
        algo_j_nm, algo_j_seed = algo_j_seed.split("-seed=")
        algo_i_nm = algo_i_nm.replace("-", "_")
        algo_j_nm = algo_j_nm.replace("-", "_")
        seed_pairs.append(((algo_i_nm, algo_i_seed), 
                           (algo_j_nm, algo_j_seed)))
    unique_seed_pairs =  list(set(seed_pairs))
    return unique_seed_pairs

def cleanup_temp_config(dest_config_path):
    "delete the temporary config file"
    # if there is only 1 file in the directory of the config path, delete the directory
    print("DEST CONFIG PATH IS ", dest_config_path)
    if len(os.listdir(os.path.dirname(dest_config_path))) == 1:
        shutil.rmtree(os.path.dirname(dest_config_path))
    else: # delete the file only
        os.remove(dest_config_path)

def is_result_written(log_folder, expt_label):
    # Check if the result was written by sacred
    paths = get_expt_paths(base_folder=log_folder, subfolder="sacred", expt_regex=expt_label)
    if len(paths) == 0:
        return False
    if len(paths) > 1: 
        print(f"Warning: multiple experiments with label {expt_label} found in {log_folder}")
    return os.path.exists(os.path.join(paths[-1], "1","info.json"))

def wait_for_result(log_folder, expt_label, max_wait_seconds=60, debug=False):
    if debug:
        return
    # Wait for the result to be written by sacred
    start = time.time()
    while not is_result_written(log_folder, expt_label):
        time.sleep(5)
        wait_time = time.time() - start
        print("Wait time: ", wait_time)
        if wait_time > max_wait_seconds:
            raise TimeoutError(f"Result for {expt_label} not written after {max_wait_seconds} seconds")

def get_necessary_agent_args(model_path):
    '''agent specific args will be fully read in later, but some are needed 
    in the default config file to built the runners, controllers, 
    replay buffers, etc.'''
    args = {}
    config_path = f"{model_path.replace('models', 'sacred')}/1/config.json"
    saved_args = json.load(open(config_path, "r"))
    args['hidden_dim'] = saved_args['hidden_dim'] 
    args['agent'] = saved_args['agent']
    return args

def write_temp_config(env_nickname, 
                      results_path,
                      src_config_path, 
                      dest_config_path,
                      k, num_agents, 
                      algo_i_path, algo_j_path, 
                      algo_i_specific_args, algo_j_specific_args,
                      load_step_type="best"
                      ):
    with open(src_config_path) as f:
        conf = yaml.load(f)
    conf = recursive_dict_update(primary=conf, secondary=algo_i_specific_args, precedence="secondary")
    conf['env'] = ENV_CONFIGS[env_nickname]["env_config"]
    conf['local_results_path'] = results_path
    conf['test_verbose'] = False
    conf['log_discounted_return'] = False

    # do not modify, for consistency across experiments
    conf['test_nepisode'] = 128 
    conf['eval_mode'] = "open"
    conf['n_uncontrolled'] = k

    conf['trained_agents'] = {
        'agent_0': {
            'agent_loader': 'poam_eval_agent_loader' if algo_i_specific_args['agent'] == "rnn_poam" else "rnn_eval_agent_loader",
            'agent_path': algo_i_path,
            'load_step': f'{load_step_type}',
            'n_agents_to_populate': num_agents,
    }}

    conf['uncontrolled_agents'] = {
        'agent_0': {
            'agent_loader': 'poam_eval_agent_loader' if algo_j_specific_args["agent"] == "rnn_poam" else "rnn_eval_agent_loader",
            'agent_path': algo_j_path,
            'load_step': f'{load_step_type}',
            'n_agents_to_populate': num_agents,
    }}
    
    # get alg i seed from alg i's experiment name
    run_i_name = os.path.basename(algo_i_path)
    assert "_baseline_" in run_i_name, "algo_i_path must be a baseline run"
    alg_i_nm, alg_i_seed = run_i_name.split("_baseline_")
    alg_i_nm = alg_i_nm.replace("_", "-")
    alg_i_seed = alg_i_seed.split("_")[0].split("=")[-1]
    
    run_j_name = os.path.basename(algo_j_path)
    assert "_baseline_" in run_j_name, "algo_j_path must be a baseline run"
    alg_j_nm, alg_j_seed = run_j_name.split("_baseline_")
    alg_j_nm = alg_j_nm.replace("_", "-")
    alg_j_seed = alg_j_seed.split("_")[0].split("=")[-1]    

    conf['label'] = f"{alg_i_nm}-seed={alg_i_seed}_{alg_j_nm}-seed={alg_j_seed}_n-{k}"

    # make directory for path:
    os.makedirs(os.path.dirname(dest_config_path), exist_ok=True)
    with open(dest_config_path, "w") as f:
        yaml.dump(conf, f)
    
    return conf['label']