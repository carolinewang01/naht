# N Agent Ad Hoc Teamwork


This is the code release for the paper, ["N-Agent Ad Hoc Teamwork"](https://arxiv.org/abs/2404.10740), published at NeurIPS 2024. If you find the code or paper useful for your work, please cite: 

```bibtex
@inproceedings{wang2024naht,
    title={N-Agent Ad Hoc Teamwork},
    author={Wang, Caroline and Rahman, Arrasy and Durugkar, Ishan and Liebman, Elad and Stone, Peter},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024}
}
```
This codebase was built on top of the [EPyMARL](https://github.com/uoe-agents/epymarl) codebase.

The following additions and modifications were made by us: 
- Custom implementation of IPPO, MAPPO, and POAM
- Support for training/evaluating MARL/AHT algorithms in an open environment (i.e. with uncontrolled teammates). 
- Minor modifications to agent architectures to enable fair comparisons between the newly added methods and existing methods
- Minor modifications to the orders in which config files are loaded (values in the alg configs override default values)

# Getting Started 
This section covers installation instructions, configuring repo-wide user variables, and downloading uncontrolled agent policies.

## Installation
1. We recommend creating a conda environment. As of Nov. 2024, these installation instructions were verified with Python 3.10 and PyTorch 2.5. Note that `torch_scatter` is not required to reproduce the experiments in our paper, but is necessary to import the PAC method (inherited from ePyMARL).

```
conda create -n <my_env> python=3.10
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

2. Install environment and repository package requirements via 
```
pip install -r requirements.txt
pip install -r env_requirement.txt
```
The `env_requirements.txt` will install the following environments, used in our experiments: 
- [SMAC](https://github.com/oxwhirl/smac)
- Our fork of [MPE](https://github.com/carolinewang01/multiagent-particle-envs)
- Our fork of [Matrix Games](https://github.com/carolinewang01/matrix-games)

2. Install StarCraft2 
Please see the instructions in the [SMAC](https://github.com/oxwhirl/smac) codebase for instructions to install the StarCraft II game. 

## Configuring Results Directories

Configuring results directories: modify `src/config/user_info.yaml` with your preferred base path for results, and preferred directory where uncontrolled agent policies will be stored. 
By default, results will be written to the folder `./naht_results`, and uncontrolled agents should be stored at `./uncntrl_agents`.

## Uncontrolled Agent Policies

To run the NAHT experiments, you will need uncontrolled agent policies. You can either generate them using this codebase (see Instructions to Run CMARL Experiments) or you can download the agents used in the paper from this Google Drive [link](https://drive.google.com/file/d/1lpYEgYdRaj7u1rWInCXe3HzpxwJpaYsq/view?usp=sharing).


# Instructions to Run CMARL Experiments
*TLDR*: 
To train a MAPPO agent on the StarCraft `5v6` task, run: 

```
python src/main.py --env-config=sc2 --config=default/default_5v6 --alg-config=sc2/mappo with env_args.map_name=5m_vs_6m --seed=112358
```

To train a MAPPO agent on the MPE `predator-prey` task, run: 

```
python src/main.py --config=default/default_mpe_pp --alg-config=mpe/mappo --env-config=mpe with env_args.pretrained_wrapper="PretrainedTag" env_args.time_limit=100 env_args.key="mpe:PredatorPrey-v0" --seed=112358
```

*Details*: 
An example bash script showing how to train MARL agents on the `5v6` domain may be found in the top-level folder of this codebase, called `train_agent.sh`.
To run a particular algorithm on a particular task, the following changes to the listed files are necessary : 
- `train_agent.sh`:
    - Change the `--env-config` to one of the supported environment configs under `src/config/envs/`.
    - Change the `--config` to the default config corresponding to the selected environment and task. Supported env/task combinations and corresponding hyperparameters may be found under `src/config/default`.
    - Change the `--alg-config` argument to the name of the appropriate algorithm config file, found under `config/algs/<env_name>`. The hyperparameters used for the paper are the values contained within these configs.  

# Instructions to Run NAHT Experiments

*TLDR*: 
To train POAM on the StarCraft `5v6` task, run the following commmand: 

```
python src/main.py --env-config=sc2 --config=open/open_train_5v6 --alg-config=sc2/poam with env_args.map_name=5m_vs_6m --seed=112358
```

To train POAM on the MPE `predator-prey` task, run the following command: 

```
python src/main.py --config=open/open_train_pp --alg-config=mpe/poam --env-config=mpe with env_args.pretrained_wrapper="PretrainedTag" env_args.time_limit=100 env_args.key="mpe:PredatorPrey-v0" --seed=112358
```

*Details*:
To run NAHT experiments, a set of teammates to train with must be provided. 
The teammates should be policy checkpoints that were generated by this codebase. 
During training time, the information to reload the policies is automatically read out of the configs stored by Sacred.
Note that the teammate policies are assumed to be stored relative in the `base_uncntrl_agents` directory specified in `src/config/user_info.yaml`. See "Getting Started" for links to download the agents used in this paper. 




An example bash script corresponding to running NAHT experiments on the `5v6` domain may be found in the top-level folder of this codebase, named `train_naht.sh`.
To run a particular algorithm on a particular task, the following changes to the listed files are necessary : 
- `train_naht.sh`: 
    - Change the `--env-config` to one of the supported environment configs under `src/config/envs/`.
    - Change the `--config` to the config corresponding to the selected environment and task. Supported env/task combinations and corresponding hyperparameters *for the NAHT experiments* may be found under `src/config/open`.
    - Change the `--alg-config` argument to the name of the appropriate algorithm config file, found under `config/algs/<env_name>`. The hyperparameters used for the paper are the values contained within these configs.  
- `src/config/open/open_algs_<task>.yaml`: 
    - Check that the checkpoint paths for each desired teammate type are correct (under `uncntrl_agents`). Paths should be relative to the `base_uncntrl_agents` specified in `src/config/user_info.yaml`
    - To run an NAHT experiment where the number of teammates are sampled, set `n_uncontrolled: null`. To run an AHT experiment where only a single agent is sampled, set `n_uncontrolled: <max_agents> - 1`.
    - If running POAM, set `agent_loader` to `poam_train_agent_loader`. Else, set `agent_loader` to `rnn_train_agent_loader`

### A Note on Available NAHT Training Configs
Within the `src/config/open` directory, there are configs with format,  `open_train_*.yaml` and `open_algs_*.yaml`.
The first set of configs generated the results in the main paper, and correspond to a training teammate set consisting of *a single seed* of all types of training teammates. 
The second set of configs correspond to the results presented in Section A.5.3. of the Appendix (Generalization to Unseen Teammate Types), where POAM/IPPO are trained on MAPPO, QMIX, IQL, and tested on IPPO and VDN. 


# Instructions to Run NAHT Evaluations
The evaluation code may be run from `src/nk_evaluation.py`.
The current evaluation is parallelized using a Condor cluster; if a condor cluster is not available, the global variable `USE_CONDOR` may be set to `False`.
The task that the evaluation should be run on must be specified under the `ifmain` block.
The OOD evaluations may be run using the `target_set_eval()` function.

# Visualization
 All code to generate plots in the paper may be found at `src/notebooks/paper_results.ipynb`.
 Please contact the authors to get access to the data used to generate the results. 
