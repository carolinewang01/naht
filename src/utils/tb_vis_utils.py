import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

from utils.read_log_data import combine_tb_logs

def gen_expt_dict(expt_paths: dict, 
                  base_path:str,
                 ):
    '''
    task_names should be a dict of form, {"5v6": expt_regex}
    base_path should be the path to the folder that the tb logs for the expt are contained in 
    expt_name should be the experiment name as you want it in the legend
    '''
    experiments = {}
    for expt_display_name, (expt_basename, folder_name) in expt_paths.items():
        results = glob.glob(os.path.join(base_path, folder_name, "tb_logs", expt_basename))
        experiments[expt_display_name] = results
    return experiments

def gen_log_data(experiments:dict, stat_name=None,
                 smooth=False, smooth_window=5, 
                 ts_round_base=5e+4):
    '''experiments has structure: 
    {exp_name: [filepath1, filepath2, ...]}
    the folder name should correspond to an algo name
    '''
    exp_dfs = {}
    for exp_name, exp_files_list in experiments.items():
        if stat_name is None:
            if "rmappo" in exp_name.lower():
                stat_name = "eval_win_rate"
            else: 
                print("Warning: unmatched exp name, default will be used")
                stat_name = "test_battle_won_mean"            

        res_dict = combine_tb_logs(exp_files_list, stat_name, ts_round_base=ts_round_base)
        if res_dict["ts"].shape[0] == 0: # read failed
            print(f"warning: read failed for {exp_name}")
        #     res_dict = combine_tb_logs(exp_files_list, backup_stat_name, ts_round_base=5e+4)
            
        # apply smoothing and trim to same length
        log_lens = []
        for k, v in res_dict.items():
            # apply smoothing
            if smooth:
                res_dict[k] = uniform_filter1d(v, size=smooth_window)
            log_lens.append(len(res_dict[k]))
        min_log_len = min(log_lens)
        for k, v in res_dict.items():
            res_dict[k] = v[:min_log_len]
            
        # convert to dataframe
        df = pd.DataFrame.from_dict(res_dict, orient='columns')
        df = df.melt(id_vars="ts", var_name="runs", value_name=stat_name)
        
        exp_dfs[exp_name] = df

    return exp_dfs
