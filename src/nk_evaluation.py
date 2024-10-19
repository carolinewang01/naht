import datetime
import os
import random
import shutil
import yaml 

from run_eval import perform_eval
from utils.cluster_helpers import submit_to_condor
from utils.nk_eval_utils import get_runs, random_pairs, all_pairs, get_seed_pairs, \
    is_result_written, get_necessary_agent_args, write_temp_config
from utils.load_utils import get_expt_paths
                            
USER_INFO = yaml.load(open(os.path.join(os.path.dirname(__file__), "config", "user_info.yaml"), "r"), Loader=yaml.FullLoader) 

def cross_eval(expt_path: str, 
               expt_basenames: list,
               env_nickname: str, 
               num_agents: int, 
               algorithms: list,
               src_config_path: str,
               dest_config_folder: str,
               dest_results_name: str,
               skip_existing: bool = True,
               num_seed_pairs: int = 3, 
               eval_seed: int = 394820,
               load_step_type: str = "best",
               match_selfplay_seeds: bool=True, # whether to match seeds for self play evals or not
               selfplay_ony: bool=False, # whether to only evaluate self play agents (alg vs alg)
               use_condor: bool = False,
               debug: bool=False):
    # Create the open_eval folder if it doesn't exist
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    
    for i in range(len(algorithms)):
        for j in range(i, len(algorithms)):
            if i != j and selfplay_ony: continue
            algo_i_path = algorithms[i]
            algo_j_path = algorithms[j]
            
            # sort alphabetically based on algo name 
            algo_i_path, algo_j_path = sorted([algo_i_path, algo_j_path], key=lambda x: x.split(os.path.sep)[-1])
            algo_i_dir, algo_i = os.path.split(algo_i_path)
            algo_j_dir, algo_j = os.path.split(algo_j_path)

            print(f"\nCHECKING {algo_i} VS {algo_j}")

            # Create log folder based on algo names, alphabetically sorted
            log_folder = os.path.join(open_eval_path, f"{algo_i}-vs-{algo_j}")
            os.makedirs(log_folder, exist_ok=True)
            
            # Get all seeds of algo i and j
            runs_i = get_runs(algo_i, os.path.join(expt_path, algo_i_dir), expt_basenames)
            runs_j = get_runs(algo_j, os.path.join(expt_path, algo_j_dir), expt_basenames)

            # Get previously evaluated seed pairs and sample new if needed
            if skip_existing and os.path.exists(os.path.join(log_folder, "sacred")):
                existing_seed_pairs = get_seed_pairs(log_folder)
                print("Existing seed pairs:", existing_seed_pairs)
                print("Runs i", runs_i)
                print("Log folder: ", log_folder)
                run_pairs = random_pairs(
                        algo_i, algo_j,
                        runs_i, runs_j, 
                        num_pairs=num_seed_pairs,
                        existing_seed_pairs=existing_seed_pairs,
                        match_selfplay_seeds=match_selfplay_seeds 
                        )
                print("Generated run pairs: ", run_pairs)
                print("Number generated run pairs: ", len(run_pairs))
            else: 
                run_pairs = random_pairs(algo_i, algo_j, 
                                         runs_i, runs_j,
                                         num_pairs=num_seed_pairs,
                                         match_selfplay_seeds=match_selfplay_seeds)

            # Iterate over the paired seed lists to perform eval
            for m, (algo_i_path, algo_j_path) in enumerate(run_pairs):
                algo_i_specific_args = get_necessary_agent_args(algo_i_path)
                algo_j_specific_args = get_necessary_agent_args(algo_j_path)
                
                for k in range(1, num_agents):
                    # Rewrite config as needed
                    dest_config_path = os.path.join(f'{dest_config_folder}', 
                                f'{env_nickname}_{algo_i}-vs-{algo_j}_n-{k}_runpair{m}.yaml')
                    
                    expt_label = write_temp_config(
                                    env_nickname, 
                                    results_path=log_folder, 
                                    src_config_path=src_config_path,
                                    dest_config_path=dest_config_path,
                                    k=k, num_agents=num_agents, 
                                    algo_i_path=algo_i_path, 
                                    algo_j_path=algo_j_path, 
                                    algo_i_specific_args=algo_i_specific_args,
                                    algo_j_specific_args=algo_j_specific_args,
                                    load_step_type=load_step_type
                                    )
                    # Check if the algorithms have already been evaluated
                    if skip_existing and is_result_written(log_folder, expt_label):
                        print(f"Skipping existing evaluation for {algo_i} vs {algo_j}, seed pair {m}, k={k}")
                        if m > num_seed_pairs - 1: 
                            res_paths = get_expt_paths(base_folder=log_folder, subfolder="sacred", expt_regex=expt_label)
                            # remove paths in extra paths (are folders)
                            print(f"EXTRA SEED PAIR DISCOVERED. REMOVING EXTRA FILES FROM {res_paths}")
                            for res_path in res_paths:
                                shutil.rmtree(res_path)
                        continue
                    else:
                        print(f"Performing evaluation for {algo_i} vs {algo_j}, seed pair {m}, k={k}")
                        if use_condor:
                            perform_eval_condor(env_nickname, 
                                                dest_config_path, 
                                                log_folder=log_folder, 
                                                expt_label=expt_label,
                                                condor_log_folder=open_eval_path,
                                                eval_seed=eval_seed, 
                                                debug=debug
                                                )
                        else: # directly perform the eval
                            perform_eval(env_nickname, dest_config_path,
                                         eval_seed=eval_seed,
                                         debug=debug)
    return       
             
def target_set_eval(expt_path: str, 
               expt_basenames: list,
               env_nickname: str, 
               num_agents: int, 
               algs_to_eval: list,
               target_algs: list,
               algs_to_eval_seeds: list,
               target_algs_seeds: list,
               src_config_path: str,
               dest_config_folder: str,
               dest_results_name: str,
               skip_existing: bool = True,
               eval_seed: int = 394820,
               load_step_type: str = "best",
               use_condor: bool = False,
               debug: bool=False):
    # Create the open_eval folder if it doesn't exist
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    
    for i in range(len(algs_to_eval)):
        for j in range(len(target_algs)):
            algo_to_eval_path = algs_to_eval[i]
            algo_target_path = target_algs[j]
            
            algo_to_eval_dir, algo_to_eval = os.path.split(algo_to_eval_path)
            algo_target_dir, algo_target = os.path.split(algo_target_path)
            print(f"\nCHECKING {algo_to_eval} VS {algo_target}")

            # Create log folder based on algo names, alphabetically sorted
            log_folder = os.path.join(open_eval_path, f"{algo_to_eval}-vs-{algo_target}")
            os.makedirs(log_folder, exist_ok=True)
        
            # Filter by the seeds that are provided
            eval_expt_basenames_seeds = [f"{expt_basename}_seed={seed}" for expt_basename in expt_basenames for seed in algs_to_eval_seeds]
            target_expt_basenames_seeds = [f"{expt_basename}_seed={seed}" for expt_basename in expt_basenames for seed in target_algs_seeds]

            # Get all relevant seeds for the algorithms 
            runs_to_eval = get_runs(algo_to_eval, os.path.join(expt_path, algo_to_eval_dir), 
                                    expt_basenames=eval_expt_basenames_seeds)
            runs_target = get_runs(algo_target, os.path.join(expt_path, algo_target_dir), 
                                   expt_basenames=target_expt_basenames_seeds)

            # Get previously evaluated seed pairs and sample new if needed
            if skip_existing and os.path.exists(os.path.join(log_folder, "sacred")):
                existing_seed_pairs = get_seed_pairs(log_folder)
                print("Existing seed pairs:", existing_seed_pairs)
                print("Log folder: ", log_folder)
                run_pairs = all_pairs(
                        algo_to_eval, algo_target,
                        runs_to_eval, runs_target, 
                        existing_seed_pairs=existing_seed_pairs
                        ) 
            else: 
                run_pairs = all_pairs(algo_to_eval, algo_target, 
                                      runs_to_eval, runs_target)

            print("All pairs: ", run_pairs)
            # Iterate over the paired seed lists to perform eval
            for m, (algo_to_eval_path, algo_target_path) in enumerate(run_pairs):
                # assumption is that the run-specific args are either 
                # the same for agent i and j or don't matter
                eval_specific_args = get_necessary_agent_args(algo_to_eval_path)
                target_specific_args = get_necessary_agent_args(algo_target_path)
                for k in range(1, num_agents):
                    # Rewrite config as needed
                    dest_config_path = os.path.join(f'{dest_config_folder}', 
                                f'{env_nickname}_{algo_to_eval}-vs-{algo_target}_n-{k}_runpair{m}.yaml')
                    
                    expt_label = write_temp_config(
                                    env_nickname, 
                                    results_path=log_folder, 
                                    src_config_path=src_config_path,
                                    dest_config_path=dest_config_path,
                                    k=k, num_agents=num_agents, 
                                    algo_i_path=algo_to_eval_path, 
                                    algo_j_path=algo_target_path, 
                                    algo_i_specific_args=eval_specific_args,
                                    algo_j_specific_args=target_specific_args,
                                    load_step_type=load_step_type
                                    )
                    # Check if the algorithms have already been evaluated
                    if skip_existing and is_result_written(log_folder, expt_label):
                        print(f"Skipping existing evaluation for {algo_to_eval} vs {algo_target}, seed pair {m}, k={k}")
                        continue
                    else:
                        print(f"Performing evaluation for {algo_to_eval} vs {algo_target}, seed pair {m}, k={k}")
                        if use_condor:
                            perform_eval_condor(env_nickname, 
                                                dest_config_path, 
                                                log_folder=log_folder, 
                                                expt_label=expt_label,
                                                condor_log_folder=open_eval_path,
                                                eval_seed=eval_seed, 
                                                debug=debug
                                                )
                            
                        else: # directly perform the eval
                            perform_eval(env_nickname, dest_config_path,
                                         eval_seed=eval_seed,
                                         debug=debug)
    return                    

def perform_eval_condor(env_nickname, dest_config_path, 
                        log_folder, expt_label, condor_log_folder,
                        eval_seed, debug):
    '''Logic to evaluate algorithm i against algorithm j is wrapped into a script
    and submitted to condor.
    '''
    exec_cmd = "src/run_eval.py"
    expt_params = {
        "env": env_nickname,
        "dest_config_path": dest_config_path,
        "log_folder": log_folder,
        "expt_label": expt_label,
        "eval_seed": eval_seed,
        "debug": debug
    }
    if debug: 
        print("NK_EVALUATION.PY: exec_cmd=", exec_cmd)
        print("NK_EVALUATION.PY: expt_params=", expt_params)
        return
    else:
        # make condor log folder if it doesn't exit
        os.makedirs(condor_log_folder, exist_ok=True)
        submit_to_condor(env_id=env_nickname, 
                         exec_cmd=exec_cmd, 
                         results_dir=condor_log_folder,
                         job_name=expt_label, 
                         expt_params=expt_params, 
                         user_email=USER_INFO["email"],
                         sleep_time=5, 
                         print_without_submit=False # print sub file
                         )        
        
if __name__ == "__main__":
    '''
    Inputs: 
        - Domain
        - Number total agents
        - List of algorithms to be evaluated against each other
        - Path to input algorithms
        - Number of run pairings to evaluate 
        
    Outputs:
        - Output + Source monitoring via Sacred: 

    Procedure: N-K eval for each pair of algorithms, where N is the number of agents and k is the number of uncontrolled agents.
        - For  I in algo list: 
            ○ For j in algo list: 
                § Create log folder based on algo names, alphabetically sorted.
                § If I, j or j, I has been evaluated already, continue. 
                § Get all seeds of algo I and j (there should be the same number). To prevent quadratic complexity in seeds, pair the seeds at random. 
                § Iterating over the paired seed lists:
                    □ For k=1, …, n-1, do: 
                        ® Rewrite config as needed
                        ® Eval algorithm I against algorithm j
                        ® (Results will be written by sacred)
                        ® Check that result was written by sacred.
                § Write result to log folder. 
        
    Does order matter? i.e. does algo pair (i, j) differ from (j, i)?
        - No, since we randomize the order of agents in the team and iterate from 1, cdots, n-1
    Does k refer to the number of agents generated by algo i or algo j? 
        - k refers to the number of uncontrolled agents, generated by algorithm j (2nd algo)
    '''
    # set seed 
    random.seed(0)
    EVAL_SEED = 394820

    DEBUG = False # toggle this to debug the script without running the evaluation
    USE_CONDOR = True # toggle this to run the evaluation on condor
    base_path = USER_INFO['base_results_path']
    
    expt_dir = "5v6"
    num_agents = 5
    env_nickname = "5v6"

    # expt_dir = "8v9" 
    # num_agents = 8
    # env_nickname = "8v9"

    # expt_dir = "10v11"
    # num_agents = 10
    # env_nickname = "10v11"

    # expt_dir = "3sv5z"
    # num_agents = 3
    # env_nickname = "3sv5z"

    # expt_dir = "mpe-pp/ts=100_shape=0.01"
    # num_agents = 3
    # env_nickname = "mpe-pp"


    algorithms = [
        "vdn", 
        "qmix",
        "iql",
        "mappo",
        "ippo",
        # "open_train/ippo-pqvmq_aht",
        # "open_train/poam-pqvmq_aht",
        # "open_train/ippo-pqvmq_open",
        # "open_train/poam-pqvmq_open",
        # "open_train/poam-pqvmq_open",
        ]

    # cross_eval(expt_path=os.path.join(base_path, expt_dir),
    #            expt_basenames=["baseline"], 
    #            env_nickname=env_nickname,
    #            num_agents=num_agents, 
    #            src_config_path="src/config/open/open_eval_default.yaml",
    #            dest_config_folder=f"src/config/temp/temp_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/",
    #         #    dest_results_name="selfplay_mismatched_eval_best",
    #            dest_results_name="open_eval_best",
    #            algorithms=algorithms, 
    #            num_seed_pairs=5,
    #            eval_seed=EVAL_SEED,
    #            load_step_type="best",
    #            match_selfplay_seeds=True,
    #         #    match_selfplay_seeds=False, # set to False for selfplay mismatched eval best
    #            selfplay_ony=False, # set to True for selfplay mismatched eval best
    #            use_condor=USE_CONDOR,
    #            debug=DEBUG)

    target_set_eval(expt_path=os.path.join(base_path, expt_dir),
                    expt_basenames=["baseline"], 
                    env_nickname=env_nickname,
                    num_agents=num_agents, 
                    algs_to_eval=[
                        "open_train/ippo-pqvmq_open", 
                        "open_train/poam-pqvmq_open",
                        "open_train/poam-pqvmq_aht", # for naht vs aht comparison only
                        # "open_train/ippo-qmq-3trainseeds", 
                        # "open_train/poam-qmq-3trainseeds",
                                  ],
                    target_algs=["vdn", "qmix", "iql", "mappo", "ippo"],
                    # target_algs=["vdn", "ippo"], # for ood alt train/test split
                    algs_to_eval_seeds=["112358", "1285842", "78590", "38410", "93718"], 
                    # target_algs_seeds=["1285842", "78590", "38410", "93718"],# not eval on 112358 because that's the training set
                    target_algs_seeds=["112358"], # for in-distribution eval
                    # target_algs_seeds=["112358", "1285842", "78590", "38410", "93718"], # for alt train/test split 
                    src_config_path="src/config/open/open_eval_default.yaml",
                    dest_config_folder=f"src/config/temp/temp_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/",
                    # dest_results_name="ood_generalization",
                    # dest_results_name="in_distr_eval",
                    # dest_results_name="ood_gen_vp",
                    eval_seed=EVAL_SEED,
                    load_step_type="best",
                    use_condor=USE_CONDOR,
                    debug=DEBUG
                    )
