import os
import re
import json


def find_model_path(checkpoint_path, load_step, logger=None):
    timesteps = []
    other_ckpts = []
    if not os.path.isdir(checkpoint_path):
        if logger:
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(checkpoint_path)
            )
        else: 
            print("Checkpoint directory {} doesn't exist".format(checkpoint_path))
        return

    # Go through all files in args.checkpoint_path
    for name in os.listdir(checkpoint_path):
        full_name = os.path.join(checkpoint_path, name)
        # Check if they are dirs
        if os.path.isdir(full_name):
            if name.isdigit(): 
                timesteps.append(int(name))
            else:
                other_ckpts.append(name)

    ckpt_ts = None
    if load_step == "best": 
        # a ckpt with best in name was saved
        if len(other_ckpts) > 0 and 'best' in other_ckpts[0]:
            ckpt_to_load = other_ckpts[0]
            with open(os.path.join(checkpoint_path, ckpt_to_load, "best_info.json"), 'r') as f:
                best_info = json.load(f)
            ckpt_ts = int(best_info['best_ts'])
        # a best ckpt wasn't saved, estimate from test returns
        else: 
            # attempt loading best checkpoint from sacred file
            try:
                ckpt_to_load = estimate_best_checkpoint(checkpoint_path, timesteps)
            except ValueError as e:
                print(e)
                print("Defaulting to last saved ckpt")
                ckpt_to_load = max(timesteps)

    elif load_step == "last": 
        ckpt_to_load = max(timesteps)
    elif load_step.isdigit(): # load_step is a timestep
        ckpt_to_load = int(load_step)
    else: 
        raise NotImplementedError(f"load_step {load_step} not implemented")
    
    model_path = os.path.join(checkpoint_path, str(ckpt_to_load))
    if ckpt_ts is None:
        ckpt_ts = ckpt_to_load
    return model_path, ckpt_ts


def estimate_best_checkpoint(checkpoint_path, ckpt_timesteps):
    sacred_path = checkpoint_path.replace("models", "sacred")
    sacred_path = os.path.join(sacred_path, "1", "info.json")
    with open(sacred_path , 'r') as f:
        data = json.load(f)

    returns = data['test_return_mean']
    return_ts = data['test_return_mean_T']

    best_return = -1000000
    best_save_ts = None
    for save_t in ckpt_timesteps: 
        diff_ts = [abs(save_t - return_t) for return_t in return_ts]
        # get the index of the closest ts
        idx = diff_ts.index(min(diff_ts))
        # if the closest ts is too far away, skip
        if diff_ts[idx] > 100000:
            continue
        if returns[idx]['value'] > best_return:
            best_return = returns[idx]['value']
            best_save_ts = save_t
    if best_save_ts is None:
        raise ValueError("No best checkpoint found")

    return best_save_ts


def glob_re(pattern, strings):
    '''Given a list of strings, returns those that contain the regex pattern'''
    return filter(re.compile(pattern).search, strings)

def get_expt_paths(base_folder, subfolder, expt_regex):
    '''returns a list of paths to experiments'''
    basepath = os.path.join(base_folder, subfolder)
    try:
        models_expts = os.listdir(basepath)
    except FileNotFoundError:
        print(f"WARNING: {basepath} not found")
        return []
    models_expts = [os.path.join(basepath, f) for f in glob_re(expt_regex, models_expts)]
    return models_expts