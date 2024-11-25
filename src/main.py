import datetime
import numpy as np
import os
import collections
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("epymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.main
def my_main(_run, _config, _log):
    print("RUNNING MAIN")
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def set_arg(config_dict, params, arg_name, arg_type):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            arg_name = _v.split("=")[0].replace("--", "")
            arg_value = _v.split("=")[1]
            config_dict[arg_name] = arg_type(arg_value)
            del params[_i]
            return config_dict

def recursive_dict_update(primary, secondary, 
                          precedence="primary", 
                          non_overridable=[]):
    '''update dict primary with items in secondary recursively. 
    if key present in both d and u, value specified by precendence takes precedence,
    unless key is in non_overridable, in which case primary value takes precedence. 
    '''
    assert precedence in ["primary", "secondary"]
    for k, v in secondary.items():
        if isinstance(v, collections.abc.Mapping):
            primary[k] = recursive_dict_update(primary.get(k, {}), v, precedence=precedence,
                                               non_overridable=non_overridable)
        else:
            if precedence == "primary": # non overridable doesn't even matter
                if k not in primary.keys():
                    primary[k] = v        
            elif precedence == "secondary":
                if k not in non_overridable:
                    primary[k] = v         
    return primary


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def add_base_results_path(base_results_path, new_path):
    '''Checks whether base results path is already in new_path. If not, adds it.'''
    if base_results_path not in new_path:
        new_path = os.path.join(base_results_path, new_path)
    return new_path

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml and user_info.yaml
    user_info = yaml.load(open(os.path.join(os.path.dirname(__file__), "config", "user_info.yaml"), "r"), Loader=yaml.FullLoader) 
    config_dict = _get_config(params, "--config", "")

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--alg-config", "algs")

    # update env_config and alg_config with values in config dict
    # if a config value is present in both config dicts, precedence is specified by precedence argument
    # keys in non_overridable are not overridable by env_config or alg_config
    config_dict = recursive_dict_update(primary=config_dict, secondary=env_config, precedence='primary')
    config_dict = recursive_dict_update(primary=config_dict, secondary=alg_config, precedence='secondary', 
                                        non_overridable=config_dict.get("non_overridable", [])
    )

    # overwrite seed and map name
    config_dict = set_arg(config_dict, params, "--seed", int)
    namelist = [config_dict['name'], config_dict['label'], f"seed={config_dict['seed']}", datetime.datetime.now().strftime("%m-%d-%H-%M-%S")]
    # namelist = [name.replace("_", "-") for name in namelist if name is not None]
    config_dict["expt_logname"]  = "_".join(namelist) 

    # add base results path to local results path and checkpoint paths if necessary
    config_dict["base_results_path"] = user_info["base_results_path"]
    config_dict["base_uncntrl_path"] = user_info["base_uncntrl_path"]

    config_dict["local_results_path"] = add_base_results_path(user_info["base_results_path"], config_dict["local_results_path"])
    ckpt_path = config_dict.get("checkpoint_path", "")
    config_dict["checkpoint_path"] = add_base_results_path(user_info["base_results_path"], config_dict["checkpoint_path"]) if ckpt_path != "" else ""

    # add config to sacred
    ex.add_config(config_dict)

    file_obs_path = os.path.join(config_dict["local_results_path"], 
                                 "sacred",  
                                 config_dict["expt_logname"])
    logger.info(f"Saving to FileStorageObserver in {file_obs_path}.")

    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)