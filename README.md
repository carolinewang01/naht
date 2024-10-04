# N Agent Ad Hoc Teamwork

This codebase was built on top of the [EPyMARL](https://github.com/uoe-agents/epymarl) codebase. 
Please see that codebase for instructions to install this codebase, SMAC, and the MPE environment. 

The following additions were made by us: 
- Custom implementation of IPPO, MAPPO, and POAM
- Minor modifications to agent architectures to enable fair comparisons between the newly added methods and existing methods
- Minor modifications to the orders in which config files are loaded (values in the alg configs override default values)
- Support for training / evaluation with uncontrolled teammates 

# Getting Started 
To get started, please first modify `src/config/user_info.yaml` with your preferred base path for results.  
By default, results will be written to the folder `~/naht_results`.
All results will be written relative to this directory.

# Instructions to Run CMARL Experiments
Bash scripts showing how to train agents are given in the top-level folder of this codebase.
To run a particular algorithm on a particular task, the following changes to the listed files are necessary: 
- `train_agent.sh`:
    - Change the `--alg-config` argument to the name of the appropriate algorithm config file 
found under `config/algs`. 

# Instructions to Run NAHT Experiments
An example bash script corresponding to running NAHT experiments for each task may be found in the top-level folder of this codebase, with the naming format `example_train_naht.sh`.
To run NAHT experiments, a set of teammates to train with must be provided. The teammates should be policy checkpoints that were generated by 
this codebase. During training time, the information to reload the policies is automatically read out of the configs stored by Sacred.
Note that the teammate policies are assumed to be stored relative to the `base_results_path` directory specified in `src/config/user_info.yaml`.


Towards this, please make the following changes to the configs: 
- `train_naht.sh`: 
    - Change the `--alg-config` argument to the name of the appropriate algorithm config file 
- `src/config/default/default_<task>.yaml`: 
    - Update the checkpoint paths for each desired teammate type (under `unseen_agents`). Paths should be relative to 
    the `base_results_path` specified in `src/config/user_info.yaml`
    - To run an NAHT experiment where the number of teammates are sampled, set `n_unseen: null`. To run an AHT experiment where 
    only a single agent is sampled, set `n_unseen: <max_agents> - 1`.


# Instructions to Run NAHT Evaluations
The evaluation code may be run from `src/nk_evaluation.py`.
The current evaluation is parallelized using a Condor cluster; if a condor cluster is not available, the global 
variable `USE_CONDOR` may be set to `False`.
The task that the evaluation should be run on must be specified under the `ifmain` block.
The OOD evaluations may be run using the `target_set_eval()` function.

# Visualization Instructions 
 All code to generate plots in the paper may be found at `src/notebooks/paper_results.ipynb`.
 Code to generate the cross play tables in the Appendix, and the plots showing the change in performance 
 as the number of agents changes may be found at `src/notebooks/summarize_nk_eval.ipynb`
 Paths may need to be modified to point at the results. 
