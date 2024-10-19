import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import pytablewriter
import dataframe_image as dfi
from rliable import metrics
from rliable import library as rly

matplotlib.rc('font', size=14)
matplotlib.rc('axes', titlesize=16)

RLY_AGG_IQM_FUNC = lambda x: np.array([
    metrics.aggregate_iqm(x),
])
RLY_BOOTSTRAP_CI_REPS = 50000

def glob_re(pattern, strings):
    '''Given a list of strings, returns those that contain the regex pattern'''
    return filter(re.compile(pattern).search, strings)

def generate_summary(eval_paths,
                     remove_duplicates=False):
    '''Assumption: eval path is a json file'''
    eval_path_res = {
        "test_return_mean": {},
        "test_return_std": {},
        "test_ep_length_mean": {},
        "seed_paths": {} # record seeds involved in computation
    }

    for p in eval_paths: 
        seedi, seedj, n = get_seed_pair(p)
        seed_pair_str = f"seedi={seedi}_seedj={seedj}"
        full_eval_name = f"{seed_pair_str}_n-{n}"
        n_agents_str = f"n-{n}"
        for k, stats_dict in eval_path_res.items():
            if n_agents_str not in stats_dict:
                stats_dict[n_agents_str] = []    
                    
        if full_eval_name in eval_path_res["seed_paths"][n_agents_str]:
            if remove_duplicates:
                shutil.rmtree(os.path.dirname(os.path.dirname(p)))
                print(f"Warning: removing duplicate eval {p}")
            else:
                print("Warning: duplicate eval detected: ", p)
            continue
        else:
            eval_path_res["seed_paths"][n_agents_str].append(full_eval_name)
            
        with open(p) as f:
            eval_info = json.load(f)
            mean = eval_info["test_return_mean"][0]["value"]
            std = eval_info["test_return_std"][0]["value"]
            ep_len=  eval_info["test_ep_length_mean"][0]

            eval_path_res["test_return_mean"][n_agents_str].append(mean)
            eval_path_res["test_return_std"][n_agents_str].append(std)
            eval_path_res["test_ep_length_mean"][n_agents_str].append(ep_len)
    
    returns = []
    for seed_pair_str in eval_path_res["test_return_mean"]:
        returns.extend(eval_path_res["test_return_mean"][seed_pair_str])

    # compute iqm and bootstrapped ci, treating n-k as the tasks 
    # unpack eval_path_res means into a dict with format {alg_name: a matrix with shape (n_runs, n_tasks )} for rliable
    score_dict = {"curr_alg": np.array([score_list for n_agents, score_list in eval_path_res["test_return_mean"].items()]).T}
    agg_scores, agg_score_cis = rly.get_interval_estimates(
    score_dict, RLY_AGG_IQM_FUNC, 
    reps=RLY_BOOTSTRAP_CI_REPS
    )
    summary = {
        "iqm":  agg_scores["curr_alg"][0],
        "ci": agg_score_cis["curr_alg"].flatten(),
    }

    return summary, returns

def check_seed_pair_equal(eval_name):
    '''Given an eval name, check if the seeds of algo1 and algo2 are the same'''
    # last seed is the eval seed
    seed_pair = re.findall(r"seed=(\d+)", eval_name)[:2]
    if seed_pair[0] == seed_pair[1]:
        return True
    return False

def get_seed_pair(eval_name):
    '''Given an eval name, find seeds in the eval name'''
    # last seed is the eval seed
    seed_pair = re.findall(r"seed=(\d+)", eval_name)[:2]
    n = re.findall(r"n-(\d+)", eval_name)[0]
    return seed_pair[0], seed_pair[1], n

def get_ood_gen_scores(log_path, 
                       algs_to_eval, target_algs, 
                       n_expected_evals, 
                       remove_empty_evals=True,
                       remove_duplicates=True
                       ):
    '''Returns a dictionary of self-play means and std errors for each algorithm'''
    ood_gen_scores_means = {}
    ood_gen_scores_stds = {}
    for algo_to_eval in algs_to_eval:
        ood_gen_scores_means[algo_to_eval] = {target_alg: {} for target_alg in target_algs}
        ood_gen_scores_stds[algo_to_eval] = {target_alg: {} for target_alg in target_algs}
        for target_alg in target_algs:
            eval_folder = os.path.join(log_path, f"{algo_to_eval}-vs-{target_alg}")
            eval_names = os.listdir(os.path.join(eval_folder, "sacred"))
            eval_paths = [os.path.join(eval_folder, "sacred",  nm, "1", "info.json") for nm in eval_names]
            
            if remove_empty_evals:
                empty_eval_paths = [p for p in eval_paths if not os.path.exists(p)]
                if empty_eval_paths != []: print("Warning: removing empty eval paths", empty_eval_paths)
                [shutil.rmtree(os.path.dirname(os.path.dirname(p))) for p in empty_eval_paths]
                eval_paths = [p for p in eval_paths if p not in empty_eval_paths]
                
            if len(eval_paths) != n_expected_evals:
                print(f"Warning: expected {n_expected_evals} eval paths but got {len(eval_paths)} for {algo_to_eval} vs {target_alg}")
                print("Eval paths are", eval_paths)

            summary, _ = generate_summary(eval_paths, remove_duplicates=remove_duplicates)
            ood_gen_scores_means[algo_to_eval][target_alg] = summary["iqm"]
            ood_gen_scores_stds[algo_to_eval][target_alg] = summary["ci"]
    return ood_gen_scores_means, ood_gen_scores_stds

def get_selfplay_scores(log_path, algorithms, n_expected_evals, 
                        remove_empty_evals=True,
                        remove_duplicates=False,
                        require_matched_seeds=False):
    '''Returns a dictionary of self-play scores for each algorithm'''
    self_play_score_means = {}
    self_play_score_stds = {}
    for algo in algorithms:
        self_play_score_means[algo] = {}
        self_play_score_stds[algo] = {}

        eval_folder = os.path.join(log_path, f"{algo}-vs-{algo}")
        eval_names = os.listdir(os.path.join(eval_folder, "sacred"))
        eval_paths = [os.path.join(eval_folder, "sacred",  nm, "1", "info.json") for nm in eval_names]

        if require_matched_seeds: # only keep eval paths where seeds are the same
            eval_names = [nm for nm in eval_names if check_seed_pair_equal(nm)]
        
        eval_paths = [os.path.join(eval_folder, "sacred",  nm, "1", "info.json") for nm in eval_names]

        if remove_empty_evals:
            empty_eval_paths = [p for p in eval_paths if not os.path.exists(p)]
            if empty_eval_paths != []: print("Warning: removing empty eval paths", empty_eval_paths)
            [shutil.rmtree(os.path.dirname(os.path.dirname(p))) for p in empty_eval_paths]
            eval_paths = [p for p in eval_paths if p not in empty_eval_paths]
            
        if len(eval_paths) != n_expected_evals:
            print(f"Warning: expected {n_expected_evals} eval paths but got {len(eval_paths)} for {algo} vs {algo}")
            print("Eval paths are", eval_paths)

        summary, _ = generate_summary(eval_paths, remove_duplicates=remove_duplicates)
        self_play_score_means[algo] = summary["mean"]
        self_play_score_stds[algo] = summary["std_error"]
    return self_play_score_means, self_play_score_stds

def vis_cross_play_matrix(log_path, 
                          algorithms:dict, 
                          eval_algs, 
                          n_expected_evals,
                          vis_table_colors=False, 
                          rm_empty_evals=True,
                          rm_duplicates=True,
                          save=False,
                          savedir=None, 
                          savename="", 
                          verbose=False
                          ):
    '''
    Generates and visualizes table of cross-play results.
    Rows of table are algorithms, columns are algorithms.
    Only the upper diagonal entries are filled in.
    
    rm_empty_evals: whether to remove empty evals (defined by evals not having an info.json file)
    '''
    # initialize pandas dataframe with algorithms as rows and columns
    display_names = algorithms.values()
    df = pd.DataFrame(columns=display_names, index=display_names)
    df = df.fillna("-") # with "-"s rather than NaNs
    
    for algo_i, algo_i_displaynm in algorithms.items():
        for algo_j, algo_j_displaynm in algorithms.items():
            # for each algorithm, load the eval info
            algo1, algo2 = sorted([algo_i, algo_j])
            eval_folder = os.path.join(log_path, f"{algo1}-vs-{algo2}")
            eval_names = os.listdir(os.path.join(eval_folder, "sacred"))

            if algo_i == algo_j:
                # only keep eval paths where seeds are the same
                eval_names = [nm for nm in eval_names if check_seed_pair_equal(nm)]

            eval_paths = [os.path.join(eval_folder, "sacred",  nm, "1", "info.json") for nm in eval_names]
            
            # filter or remove empty eval paths
            empty_eval_paths = [p for p in eval_paths if not os.path.exists(p)]
            if rm_empty_evals:
                if empty_eval_paths != []: print("Warning: removing empty eval paths", empty_eval_paths)
                [shutil.rmtree(os.path.dirname(os.path.dirname(p))) for p in empty_eval_paths]
            eval_paths = [p for p in eval_paths if p not in empty_eval_paths]
            if len(eval_paths) != n_expected_evals and verbose:
                print(f"Warning: expected {n_expected_evals} eval paths but got {len(eval_paths)} for {algo_i} vs {algo_j}. Limiting to {n_expected_evals} evals.")
                print("Eval paths are", eval_paths)
                if len(eval_paths) > n_expected_evals: 
                    eval_paths = eval_paths[:n_expected_evals]

            # summarize
            summary, _ = generate_summary(eval_paths, remove_duplicates=rm_duplicates)
            mean = summary["mean"]
            std_error = summary["std_error"]
            # fill in entries of df in format (display_str, mean), where the 2nd entry is for styling later
            df.at[algo_i_displaynm, algo_j_displaynm] = (f"{mean:.3f} +/- ({std_error:.3f})", mean)
            df.at[algo_j_displaynm, algo_i_displaynm] = (f"{mean:.3f} +/- ({std_error:.3f})", mean)
                
    # each entry of table is in format (display_str, mean) so we need to do data formatting first: 
    mean_only_df = df.applymap(lambda x: float(x[1]))
    self_play_scores = pd.Series(np.diag(df), index=[df.index])
    self_play_scores = self_play_scores.apply(lambda x: x[1]).to_numpy()
    # set diagonal to nan for temp df to exclude diag from summary computation
    np.fill_diagonal(mean_only_df.values, np.nan)

    # add a XP column
    mean_only_df['mean'] = mean_only_df[eval_algs].mean(axis=1).apply(lambda x: f"{x:.3f}")
    mean_only_df['std'] = mean_only_df[eval_algs].std(axis=1).apply(lambda x: f"{x:.3f}")
    mean_only_df['mean_plus_std'] = mean_only_df.agg(lambda x: f"{x['mean']} +/- ({x['std']})", axis=1)

 
    df["mean xp \nscore (+/-std)"] = mean_only_df.agg(lambda x: (x['mean_plus_std'], float(x['mean'])), axis=1)

    if vis_table_colors:
        styler = style_table(df)
    else: # return plain table
        styler = df.style.format(lambda x: x[0])
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = os.path.join(savedir, savename)
        dfi.export(styler, savepath, table_conversion='matplotlib')

    return styler    

def style_table(df):  
    '''Style table with color gradient'''
    df_display = df.applymap(lambda x: x[0])
    df_stylevalues = df.applymap(lambda x: x[1])
    styler = df_display.style.background_gradient(cmap='viridis', gmap=df_stylevalues, axis=None)
    return styler


def plot_sp_by_xp(df, algorithms:list, 
                  xp_col_name:str, task:str,
                  legend_loc=(1.0, -0.25)
                ):
    '''Plot self play by cross play score. Df is the output of vis_cross_play_matrix.'''
    sp_xp_dict = {}
    for algo in algorithms:
        sp_xp_dict[algo] = {
            "sp": float(df.data.loc[algo, algo].split(" +/- ")[0]),
            "xp": float(df.data.loc[algo, xp_col_name].split(" +/- ")[0])
        }
    sp_xp_dict

    # plot SP by XP score
    fig, ax = plt.subplots()
    for algo in algorithms:
        sp_score = sp_xp_dict[algo]["sp"]
        xp_score = sp_xp_dict[algo]["xp"]
        ax.scatter(sp_score, xp_score, label=algo)
        # 
        ax.annotate(algo, (sp_score + 0.2, xp_score - 0.1),
                    )

    # set axis limits
    min_sp = min([sp_xp_dict[algo]["sp"] for algo in algorithms])
    max_sp = max([sp_xp_dict[algo]["sp"] for algo in algorithms])
    min_xp = min([sp_xp_dict[algo]["xp"] for algo in algorithms])
    max_xp = max([sp_xp_dict[algo]["xp"] for algo in algorithms])
    ax.set_xlim(min(min_sp, min_xp) - 0.5, max(max_sp, max_xp) + 0.5)
    ax.set_ylim(min(min_sp, min_xp) - 0.5, max(max_sp, max_xp) + 0.5)
    
    # plot line x=y
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color="gray", label="x=y")

    ax.set_xlabel("Self-play score")
    ax.set_ylabel("Cross-play score")
    # legend with two columns and box around it
    ax.set_title(task)
    ax.legend(bbox_to_anchor=legend_loc, 
              borderaxespad=0.,
              ncol=3, 
              frameon=True)
    plt.show()


def check_name_match(algo_vs_algo, target_algs):
    '''
    Given an eval_path, check if any of target_algs is in the log_path
    '''
    algo1, algo2 = algo_vs_algo.split("-vs-")
    if any(target_alg == algo1 for target_alg in target_algs) and any(target_alg == algo2 for target_alg in target_algs):
        return True
    return False

def vis_nk_curve(log_path, algo:str, target_algs,
                  n, title, save=False, savedir=None): 
    '''
    Visualizes the n vs k curve for a given algorithm.
    Assumption log_path is a path to a folder containing 
    "<log_path>/<algo>-vs-*/sacred/eval_<algo1>-seed=<seed1>_<algo2>-seed=<seed2>_n-<k>_seed=394820_<datetime>/1/info.json"
    '''
    algo_log_paths = os.listdir(log_path)
    algo_log_paths = [os.path.join(log_path, p) for p in algo_log_paths if check_name_match(p, target_algs + [algo])]

    plt.figure(figsize=(6, 5.5))
    results = {}
    for a_path in algo_log_paths:
        eval_names = os.listdir(os.path.join(a_path, "sacred"))
        label = algo_vs_algo = os.path.basename(a_path)
        xs, ys, stds = [], [], []
        
        for k in range(1, n):
            if algo_vs_algo == f"{algo}-vs-{algo}":
                # only keep eval paths where seeds are the same
                eval_names = [nm for nm in eval_names if check_seed_pair_equal(nm)]
                
            names = [nm for nm in eval_names if f"_n-{k}_" in nm]
            

            paths = [os.path.join(a_path, "sacred",  nm, "1", "info.json") for nm in names]
            summary, _ = generate_summary(paths)
            xs.append(k)
            ys.append(summary["mean"])
            stds.append(summary["std_error"])

        color = None
        if f"{algo}-vs-{algo}" == algo_vs_algo:
            color = "black"
        # algo in 2nd position -- flip algo and results to 1st position
        elif f"-vs-{algo}" in algo_vs_algo and f"-vs-{algo}_ns" not in algo_vs_algo:
            algo1, algo2 = algo_vs_algo.split("-vs-")
            label = f"{algo2}-vs-{algo1}"
            # reverse ys, stds
            ys = ys[::-1]
            stds = stds[::-1]
            color = None

        results[label] = {
            "xs": xs,
            "ys": ys,
            "stds": stds,
            "color": color,
        }

    # visualize results
    for label, res in results.items():
        if f"{algo}-vs-" not in label: 
            continue
        # compute performance with 0 uncontrolled agents
        res["xs"].insert(0, 0)
        res["ys"].insert(0, np.mean(results[f"{algo}-vs-{algo}"]["ys"]))
        res["stds"].insert(0, np.std(results[f"{algo}-vs-{algo}"]["ys"]))

        # compute performance with all uncontrolled agents
        uncontrolled_algo = label.split("-vs-")[1]
        res["xs"].append(n)
        res["ys"].append(np.mean(results[f"{uncontrolled_algo}-vs-{uncontrolled_algo}"]["ys"]))
        res["stds"].append(np.std(results[f"{uncontrolled_algo}-vs-{uncontrolled_algo}"]["ys"]))

        # generate line plot where the x axis is k and the y axis is the mean return and std is the error bar
        plt.errorbar(res["xs"], res["ys"], 
                     yerr=res["stds"], fmt='-o', 
                     label=label, 
                     color=res["color"]
                     )

    plt.xlabel("k uncontrolled Agents", fontsize=14)
    plt.ylabel("Mean Test Return", fontsize=14)
    # modify xticks and yticks to have fontsize 14 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), 
               fontsize=12, ncol=3)
    plt.tight_layout()
    plt.title(f"{title}, Evaluated Algorithm={algo.upper().replace('_', '-')}", fontsize=14)
    if save: 
        # make directory
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{algo}_nk_curves.pdf"))
    plt.show()

def plot_ood_gen_scores(ood_res:dict, exp_namemap, 
                        plot_suptitle="OOD Evaluation", plot_subplot_titles=True,
                        save=False, savedir=None, savename=None):
    '''
    ood_res is  dict with structure {task_name: (ood_gen_means, ood_gen_stds)}
    where ood_gen_means is a dict with structure {alg_to_eval: {target_alg: [ood_gen_scores]}}
    and correspondingly for ood_gen_stds
    '''
    fig, axes = plt.subplots(1, len(ood_res), 
                             figsize=(5*len(ood_res), 5) if len(ood_res) > 1 else (7.5, 6.5) # mult figs
                            #  figsize=(10, 10) # single fig
                             )
    bar_width = 0.4
    opacity = 0.8
    
    for i, (task, (ood_gen_means, ood_gen_cis)) in enumerate(ood_res.items()):
        # Extract target algorithms and sort them for consistent plotting
        target_algs = sorted({target_alg for algs_to_eval in ood_gen_means.values() for target_alg in algs_to_eval})
        # Extract alg_to_eval names
        alg_to_eval_names = sorted(ood_gen_means.keys())

        # Organize data for plotting
        scores = {alg: [] for alg in target_algs}
        for target_alg in target_algs:
            for alg_name in alg_to_eval_names:
                scores[target_alg].append(ood_gen_means.get(alg_name, {}).get(target_alg, 0))

        # Plotting
        n_groups = len(target_algs)
        index = np.arange(n_groups)

        for j, alg_name in enumerate(alg_to_eval_names):
            ax = axes[i] if len(ood_res) > 1 else axes
            y_values = [scores[alg][j] for alg in target_algs]
            ax.bar(index + j * bar_width, y_values, bar_width,
                    alpha=opacity, label=exp_namemap[alg_name])
            # add 95% CIs            
            cis = np.array([ood_gen_cis.get(alg_name, {}).get(alg, 0) for alg in target_algs]).T
            y_errs_upper = cis[0] - y_values
            y_errs_lower = y_values - cis[1]
            ax.errorbar(x=index + j * bar_width, y=y_values, 
                        yerr=(y_errs_lower, y_errs_upper),
                        fmt='none', capsize=5, color='black')

        if plot_subplot_titles:
            ax.set_title(task)
        ax.set_xlabel('Target Algorithms')
        # set tick labels
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(target_algs, rotation=0)

    leftmost_ax = axes[0] if len(ood_res) > 1 else axes
    rightmost_ax = axes[-1] if len(ood_res) > 1 else axes
    leftmost_ax.set_ylabel('Mean Test Return')
    rightmost_ax.legend(
        # loc='upper right', 
        loc="lower right", 
        ncol=1)
        # bbox_to_anchor=(1.0, -0.25),
    
    if plot_suptitle:
        plt.suptitle(plot_suptitle, x=0.6, fontsize=22)
    plt.tight_layout()
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
    plt.show()  

def plot_mm_selfplay_scores(mm_selfplay_res:dict, 
                            save=False, savedir=None, savename=None):
    '''
    ood_res is  dict with structure {task_name: (ood_gen_means, ood_gen_stds)}
    where ood_gen_means is a dict with structure {alg_to_eval: {target_alg: [ood_gen_scores]}}
    and correspondingly for ood_gen_stds
    '''
    fig, axes = plt.subplots(1, len(mm_selfplay_res), figsize=(5*len(mm_selfplay_res), 5))
    bar_width = 0.4
    opacity = 0.8
    
    for i, (task, mm_dict) in enumerate(mm_selfplay_res.items()):
        # extract algs under consideration and sort for consistent plotting
        eval_algs = sorted(mm_dict["matched"]["mean"].keys())
        # Extract alg_to_eval names
        names_to_compare = sorted(mm_dict.keys())

        # Organize data for plotting
        scores = {alg: [] for alg in eval_algs}
        for name in names_to_compare:
            for alg in eval_algs:
                scores[alg].append(mm_dict[name]["mean"][alg])
 
        # Plotting
        n_groups = len(eval_algs)
        index = np.arange(n_groups)

        for j, name in enumerate(names_to_compare):
            axes[i].bar(index + j * bar_width, 
                        [scores[alg][j] for alg in eval_algs], 
                        bar_width,
                        alpha=opacity, label=name)
            # add standard errors 
            axes[i].errorbar(index + j * bar_width, 
                             [scores[alg][j] for alg in eval_algs], 
                             yerr=[mm_dict[name]["std"][alg] for alg in eval_algs], 
                             fmt='none', capsize=5, color='black')

        axes[i].set_title(task)
        axes[i].set_xlabel('Algorithms')
        # set tick labels
        axes[i].set_xticks(index + bar_width / 2)
        axes[i].set_xticklabels(eval_algs, rotation=0)

    axes[0].set_ylabel('Mean Test Return')
    axes[-1].legend(loc="lower right", ncol=1)
    
    plt.suptitle(f'Matched vs Mismatched Seed Self Play Evaluation')
    plt.tight_layout()            
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
    plt.show()  