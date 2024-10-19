import copy
import json
import os
import re
import shutil

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pytablewriter
import dataframe_image as dfi

# apply seaborne theme globally
sns.set_theme()

FONTSIZE = 20 # 16
TICK_FONTSIZE = 16
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE, labelsize=FONTSIZE)
matplotlib.rc('xtick', labelsize=TICK_FONTSIZE)
matplotlib.rc('ytick', labelsize=TICK_FONTSIZE)
matplotlib.rc('legend', fontsize=FONTSIZE)


def glob_re(pattern, strings):
    '''Given a list of strings, returns those that contain the regex pattern'''
    return filter(re.compile(pattern).search, strings)

def compute_figure_size(num_subfigs): 
    if num_subfigs == 1: 
        figsize = (6.5, 6.0)
    elif num_subfigs ==4: 
        figsize = (5.0*num_subfigs, 5.2)
    else:
        figsize = (5.0*num_subfigs, 6.0)
    return figsize       

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
        eval_name = f"{seed_pair_str}_n-{n}"
        for k, stats_dict in eval_path_res.items():
            if seed_pair_str not in stats_dict:
                stats_dict[seed_pair_str] = []
                            
        if eval_name in eval_path_res["seed_paths"][seed_pair_str]:
            if remove_duplicates:
                shutil.rmtree(os.path.dirname(os.path.dirname(p)))
                print(f"Warning: removing duplicate eval {p}")
            else:
                print("Warning: duplicate eval detected: ", p)
            continue
        else:
            eval_path_res["seed_paths"][seed_pair_str].append(eval_name)
            
        with open(p) as f:
            eval_info = json.load(f)
            mean = eval_info["test_return_mean"][0]["value"]
            std = eval_info["test_return_std"][0]["value"]
            ep_len=  eval_info["test_ep_length_mean"][0]

            eval_path_res["test_return_mean"][seed_pair_str].append(mean)
            eval_path_res["test_return_std"][seed_pair_str].append(std)
            eval_path_res["test_ep_length_mean"][seed_pair_str].append(ep_len)
    # seed_pair_means = [np.mean(eval_path_res["test_return_mean"][seed_pair_str]) for seed_pair_str in eval_path_res["test_return_mean"]]
    # seed_pair_std_errors = np.std(seed_pair_means) / np.sqrt(n_trials)

    # compute standard error of the return mean from the std dev
    # unpack all return means into a single list
    returns = []
    for seed_pair_str in eval_path_res["test_return_mean"]:
        returns.extend(eval_path_res["test_return_mean"][seed_pair_str])
    n_samples = len(returns)
    std_errors = np.std(returns) / np.sqrt(n_samples)

    summary = {
        "mean": np.mean(returns),
        "ci": 1.96 * std_errors
    }
    # return summary stats and raw eval path results
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


def plot_single_exp(ax, exp_dfs, baselines=None,
                    stat_name=None, 
                    xlabel=None, ylabel=None,
                    yaxis_lims=None, xaxis_lims=None, 
                    plot_title=None):
    baseline_palette = list(sns.color_palette("Oranges", as_cmap=False, n_colors=7).as_hex())
    open_palette = list(sns.color_palette("mako", as_cmap=False, n_colors=12).as_hex())
    default_palette = sns.color_palette("colorblind", as_cmap=True)

    for i, (exp_name, exp_df) in enumerate(exp_dfs.items()):
        if "baseline" in exp_name: 
            color = baseline_palette.pop()
            baseline_palette.pop()
        elif "open" in exp_name: 
            color = open_palette[i*2 + 3]
        else: 
            color = default_palette[i]
        g = sns.lineplot(data=exp_df, 
                     x="ts", y="test_battle_won_mean" if stat_name is None else stat_name,
                     errorbar=('ci', 95), # "sd", 
                     ax=ax, label=exp_name, 
                     color=color,
                     legend=False
                    )
    g.set(xlabel=xlabel, ylabel=ylabel, title=plot_title, ylim=yaxis_lims, xlim=xaxis_lims)
    if baselines is not None:
        color_list = ["darkslategray", "slateblue", "darkslateblue", "indigo", "darkviolet"]
        # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        linestyle_list = [
             ('densely dashed',        (0, (5, 1))),
             ('loosely dotted',        (0, (1, 10))),
             ('dotted',                (0, (1, 1))),
             ('loosely dashed',        (0, (5, 10))),
             ('long dash with offset', (5, (10, 3))),
              ]
        for i, (baseline_nm, baseline_data) in enumerate(baselines.items()):
            bs_mean, bs_std = baseline_data
            ax.axhline(y=bs_mean, xmin=0.01, xmax=0.99, 
                       color=color_list[0], # figure_colors[i+1], 
                       linestyle= linestyle_list.pop(-1)[1], 
                       label=baseline_nm
                       )

def plot_learning_curves(exp_dfs:dict, 
                     savename:str, 
                     plot_title:str=None, 
                     stat_name:str=None,
                     legend=True,
                     legend_cols=4,
                     legend_loc=(1.0, -0.25),
                     baselines:dict=None, 
                     xaxis_lims=None,
                     yaxis_lims=None,
                     save=False, 
                     savedir="figures/"):
    '''
    baselines: a dict with format {exp_name: [mean, std]}. will be plotted as horizontal line
    '''
    _, axis = plt.subplots(1, 1, figsize=(7, 5)) # figsize argument

    plot_single_exp(axis, exp_dfs, baselines,
                    stat_name=stat_name, 
                    xlabel="Timesteps", 
                    ylabel=stat_name.replace("_", " ").title() if stat_name is not None else "Mean Test Return",
                    yaxis_lims=yaxis_lims, xaxis_lims=xaxis_lims, 
                    plot_title=plot_title,
                    )

    if legend:
        plt.legend(bbox_to_anchor=legend_loc, 
                   borderaxespad=0.,
                   ncol=legend_cols,
                  )
    else:
        leg = axis.get_legend()
        leg.remove()
    
    if save:
        if not os.path.exists(savedir):
            os.mkdir("figures")            
        savepath = os.path.join(savedir, savename + ".pdf")
        
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_learning_curves_all(exp_dict:dict, 
                     savename:str, 
                     plot_suptitle:str=None, 
                     show_subplot_task_name=True,
                     stat_name:str=None,
                     legend=True,
                     legend_cols=4,
                     legend_loc=(1.0, -0.25),
                     baselines:dict=None, 
                     xaxis_lims=None,
                     yaxis_lims=None,
                     save=False, 
                     savedir="figures/"):
    '''
    exp_dict: a dict with format {task: {exp_name: exp_df}}
    baselines: a dict with format {task: {exp_name: [mean, std]}}. 
    will be plotted as horizontal line
    '''
    ntasks =len(exp_dict)
    fig, axes = plt.subplots(1, ntasks, 
                             figsize=(6*ntasks, 4.3), 
                            squeeze=True) # figsize argument
    for i, (task, exp_df) in enumerate(exp_dict.items()):
        task = task.split("/")[0]
        ylabel = stat_name.replace("_", " ").title() if stat_name is not None else "Mean Test Return"
        plot_single_exp(axes[i] if ntasks > 1 else axes, 
                        exp_df, 
                        baselines[task],
                        stat_name=stat_name, 
                        xlabel="Timesteps", 
                        ylabel=ylabel if i == 0 else None,
                        yaxis_lims=yaxis_lims, xaxis_lims=xaxis_lims, 
                        plot_title=task if show_subplot_task_name else None)

    if legend:
        plt.legend(borderaxespad=0., ncol=legend_cols)
    if plot_suptitle:
        plt.suptitle(plot_suptitle)

    if save:
        if not os.path.exists(savedir):
            os.mkdir("figures")            
        savepath = os.path.join(savedir, savename + ".pdf")
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def get_ood_gen_scores(log_path, 
                       algs_to_eval, target_algs, 
                       n_expected_evals, 
                       remove_empty_evals=True,
                       remove_duplicates=True
                       ):
    '''Returns a dictionary of self-play scores for each algorithm'''
    ood_gen_scores_means = {}
    ood_gen_scores_cis = {}
    for algo_to_eval in algs_to_eval:
        ood_gen_scores_means[algo_to_eval] = {target_alg: {} for target_alg in target_algs}
        ood_gen_scores_cis[algo_to_eval] = {target_alg: {} for target_alg in target_algs}
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
            ood_gen_scores_means[algo_to_eval][target_alg] = summary["mean"]
            ood_gen_scores_cis[algo_to_eval][target_alg] = summary["ci"]

    return ood_gen_scores_means, ood_gen_scores_cis

def get_selfplay_scores(log_path, algorithms, n_expected_evals, 
                        remove_empty_evals=True,
                        remove_duplicates=False,
                        require_matched_seeds=False):
    '''Returns a dictionary of self-play scores for each algorithm'''
    self_play_score_means = {}
    self_play_score_cis = {}
    for algo in algorithms:
        self_play_score_means[algo] = {}
        self_play_score_cis[algo] = {}

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
        self_play_score_cis[algo] = summary["ci"]
    return self_play_score_means, self_play_score_cis

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
    df = pd.DataFrame(columns=list(display_names) + ["mean xp \nscore"], 
                      index=display_names)
    df = df.fillna("-") # with "-"s rather than NaNs
    
    
    for algo_i, algo_i_displaynm in algorithms.items():
        row_returns_all = [] # accumulate ALL underlying row returns for computing CI for cross-play
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
            summary, rets = generate_summary(eval_paths, remove_duplicates=rm_duplicates)
            mean = summary["mean"]
            ci = summary["ci"]
            if algo_i != algo_j and algo_j in eval_algs: # xp score should not include self play performance or performance of methods not in the eval algs 
                row_returns_all.extend(rets)

            # fill in entries of df in format (display_str, mean), where the 2nd entry is for styling later
            df.at[algo_i_displaynm, algo_j_displaynm] = (f"{mean:.3f} +/- ({ci:.3f})", mean)
            df.at[algo_j_displaynm, algo_i_displaynm] = (f"{mean:.3f} +/- ({ci:.3f})", mean)

        # fill in xp score for each algo    
        row_mean = np.mean(row_returns_all)
        row_ci = 1.96 * np.std(row_returns_all) / np.sqrt(len(row_returns_all))
        df.at[algo_i_displaynm, "mean xp \nscore"] = (f"{row_mean:.3f} +/- ({row_ci:.3f})", row_mean)
    
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

def check_name_match(algo_vs_algo, alg_to_eval, target_algs):
    '''
    Given algo_vs_algo, check if any of target_algs is in the log_path, 
    while also requiring that alg_to_eval is in the log_path.
    Also return True if the algo_vs_algo is a self-play comparison for 
    either alg_to_eval or any of the target_algs.
    '''
    if algo_vs_algo in ["condor_logs", "slurm_logs"]: 
        return False
    algo1, algo2 = algo_vs_algo.split("-vs-")

    check_algo_is_targ = lambda algo: any(target_alg == algo for target_alg in target_algs + [alg_to_eval])
    if  check_algo_is_targ(algo1) and check_algo_is_targ(algo2) and \
        (algo1 == alg_to_eval or algo2 == alg_to_eval):
        return True
    elif algo1 == algo2 and (algo1 == alg_to_eval or algo1 in target_algs):
        return True
    return False

def get_nk_summary(log_dir:str, eval_alg:str, target_algs:list, n_total:int):
    '''
    Get the mean and CI of the n-k curve for a given algorithm against a set of target algorithms, 
    where where n_total is the total team size and k is the number of uncontrolled agents. 
    '''
    all_log_paths = os.listdir(log_dir)
    algo_log_paths = [os.path.join(log_dir, p) for p in all_log_paths if (check_name_match(p, eval_alg, target_algs))]
    
    nk_xp_summary, nk_sp_summary = {}, {}
    for a_path in algo_log_paths:
        eval_names = os.listdir(os.path.join(a_path, "sacred"))
        label = algo_vs_algo = os.path.basename(a_path)
        xs, ys, cis = [], [], []
        
        for k in range(1, n_total):
            if algo_vs_algo == f"{eval_alg}-vs-{eval_alg}":
                # only keep eval paths where seeds are the same
                eval_names = [nm for nm in eval_names if check_seed_pair_equal(nm)]
                
            names = [nm for nm in eval_names if f"_n-{k}_" in nm]
            

            paths = [os.path.join(a_path, "sacred",  nm, "1", "info.json") for nm in names]
            summary, _ = generate_summary(paths)
            xs.append(k)
            ys.append(summary["mean"])
            cis.append(summary["ci"])

        color = None
        algo1, algo2 = algo_vs_algo.split("-vs-")
        # if f"{eval_alg}-vs-{eval_alg}" == algo_vs_algo:
        if algo1 == algo2:
            color = "black"
            summary_dict = nk_sp_summary        
        # algo in 2nd position -- flip algo and results to 1st position
        elif f"-vs-{eval_alg}" in algo_vs_algo and f"-vs-{eval_alg}_ns" not in algo_vs_algo:
            
            label = f"{algo2}-vs-{algo1}"
            # reverse ys, cis
            ys = ys[::-1]
            cis = cis[::-1]
            color = None
            summary_dict = nk_xp_summary
        
        summary_dict[label] = {
            "xs": xs,
            "ys": ys,
            "cis": cis,
            "color": color,
        }

    # add performance with 0 uncontrolled agents and all uncontrolled agents
    nk_summary = copy.deepcopy(nk_xp_summary)
    for label, res in nk_summary.items():
        # compute performance with 0 uncontrolled agents
        res["xs"].insert(0, 0)
        res["ys"].insert(0, np.mean(nk_sp_summary[f"{eval_alg}-vs-{eval_alg}"]["ys"]))
        res["cis"].insert(0, np.std(nk_sp_summary[f"{eval_alg}-vs-{eval_alg}"]["ys"]))

        # compute performance with all uncontrolled agents
        uncontrolled_algo = label.split("-vs-")[1]
        res["xs"].append(n_total)
        res["ys"].append(np.mean(nk_sp_summary[f"{uncontrolled_algo}-vs-{uncontrolled_algo}"]["ys"]))
        res["cis"].append(np.std(nk_sp_summary[f"{uncontrolled_algo}-vs-{uncontrolled_algo}"]["ys"]))

    return nk_summary
    

def vis_nk_curve_single(log_dir:str, ax:plt.Axes,
                               eval_algs:dict, target_algs:dict,
                               n_total:int, task:str, 
                               plot_n_controlled=True
                               ): 
    '''
    Visualizes the *mean* n-k curve for the algorithms in eval_algs over the set of target algorithms, 
    where n is the total number of agents and k is the number of uncontrolled. 
    
    Args: 
        log_dir (str): path to a folder containing 
                      "<log_path>/<algo>-vs-*/sacred/eval_<algo1>-seed=<seed1>_<algo2>-seed=<seed2>_n-<k>_seed=394820_<datetime>/1/info.json"
    '''
    for eval_alg, eval_displayname in eval_algs.items():
        nk_summary = get_nk_summary(log_dir=log_dir, 
                                    eval_alg=eval_alg, 
                                    target_algs=list(target_algs.keys()), 
                                    n_total=n_total
                                    )
        for label, res in nk_summary.items():
            assert label != f"{eval_alg}-vs-{eval_alg}", "Self play performance should not be included in curve."
        
        # compute mean and 95% CI over the labels
        xs = list(range(0, n_total+1))
        nk_ret_mean = np.mean([res["ys"] for res in nk_summary.values()], axis=0)
        assert nk_ret_mean.shape[0] == n_total+1, f"nk_ret_mean shape: {nk_ret_mean.shape}"
        nk_ret_ci = 1.96 * np.std([res["ys"] for res in nk_summary.values()], axis=0) / np.sqrt(len(nk_summary))
        assert nk_ret_ci.shape[0] == n_total+1, f"nk_ret_ci shape: {nk_ret_ci.shape}"

        # generate line plot where the x axis is k and the y axis is the mean return and the error bar is the 95% CI
        if plot_n_controlled: # xaxis means the number of controlled
            nk_ret_mean = nk_ret_mean[::-1]
            nk_ret_ci = nk_ret_ci[::-1]
            xlabel = "N Controlled Agents"
        else: 
            xlabel = "k uncontrolled Agents"

        # plot mean and CI for all values of N up but not including n_total
        ax.errorbar(xs[:-1], nk_ret_mean[:-1], 
                yerr=nk_ret_ci[:-1], fmt='-o', 
                label=eval_displayname, 
                )
        # get color that errorbar was plotted with 
        color = ax.lines[-1].get_color()
        # plot horizontal line corresponding to performance of all controlled agents
        ax.axhline(y=nk_ret_mean[-1], 
                   color=color,
                   linestyle='--', 
                   label=f"{eval_displayname} (self-play)")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{task}")

def vis_nk_curve_all(tasks:dict, base_path:str,
                     eval_algs:dict, target_algs:dict, 
                     plot_n_controlled=True,
                     save=False, savedir=None, savename=None):
    """
    Visualizes the *mean* nk curve for all tasks.

    Args:
        tasks (dict): Dictionary with structure {task_name: n_agents}.
        eval_algs (dict): Dictionary with structure {eval_alg: display_name}.
        target_algs (dict): Dictionary with structure {target_alg: display_name}.
        plot_n_controlled (bool): If True, the x-axis will reflect the number of controlled agents.
                                  Else, the x-axis will reflect the number of uncontrolled agents.
        save (bool): If True, the plot will be saved to the specified directory.
        savedir (str): Directory where the plot will be saved if save is True.
    """
    num_subfigs = len(tasks)
    fig, axes = plt.subplots(1, num_subfigs, 
                             figsize=(6.0*num_subfigs, 4.5) if num_subfigs > 1 else (8.0, 7.5) # mult figs
                             )
    for i, (task, n_agents) in enumerate(tasks.items()):
        vis_nk_curve_single(
                    log_dir=os.path.join(base_path, task, "open_eval_best"), 
                    ax=axes[i],
                    eval_algs=eval_algs, 
                    target_algs=target_algs,
                    n_total=n_agents, task=task.split("/")[0],
                    plot_n_controlled=plot_n_controlled, 
                    )
        if i == 0:
            axes[i].set_ylabel("Mean Test Return")

    # formatting
    eval_alg_str = ", ".join([e.upper() for e in target_algs.values()])
    # place legend underneath the entire figure, with 4 columns
    plt.legend(bbox_to_anchor=(-0.35, -0.2),
               ncol=4               
               )
    plt.suptitle(f"Varying Number of Controlled Agents", y=1.1)

    plt.tight_layout()
    if save: 
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
    plt.show()

def plot_ood_gen_scores(ood_res:dict, 
                        exp_namemap:dict, 
                        in_distr_baseline:dict=None,
                        plot_suptitle="OOD Evaluation", plot_subplot_titles=True,
                        save=False, savedir=None, savename=None):
    '''
    ood_res is  dict with structure {task_name: (ood_gen_means, ood_gen_cis)}
    where ood_gen_means is a dict with structure {alg_to_eval: {target_alg: [ood_gen_scores]}}
    and correspondingly for ood_gen_cis
    '''
    num_subfigs = len(ood_res)
    _, axes = plt.subplots(1, num_subfigs, figsize=compute_figure_size(num_subfigs)
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
        id_scores = {alg: [] for alg in target_algs}
        for target_alg in target_algs:
            for alg_name in alg_to_eval_names:
                scores[target_alg].append(ood_gen_means.get(alg_name, {}).get(target_alg, 0))
                if in_distr_baseline:
                    id_scores[target_alg].append(in_distr_baseline[task].get(alg_name, {}).get(target_alg, 0))
        # Plotting
        n_groups = len(target_algs)
        index = np.arange(n_groups)

        for j, eval_alg_name in enumerate(alg_to_eval_names):
            ax = axes[i] if len(ood_res) > 1 else axes
            # plot bars 
            ax.bar(index + j * bar_width, [scores[alg][j] for alg in target_algs], bar_width,
                    alpha=opacity, label=exp_namemap[eval_alg_name])
            
            # add standard errors 
            ax.errorbar(index + j * bar_width, [scores[alg][j] for alg in target_algs], 
                        yerr=[ood_gen_cis.get(eval_alg_name, {}).get(alg, 0) for alg in target_algs], 
                        fmt='none', capsize=5, color='black')
            
            # plot in-distr baselines as markers on top of each bar 
            if in_distr_baseline:
                ax.scatter(index + j * bar_width, [id_scores[alg][j] for alg in target_algs], 
                            color='red', marker='*', s=100, zorder=10,
                            label='In-Distr. Baseline' if j == 1 else None)

        if plot_subplot_titles:
            ax.set_title(task)
        ax.set_xlabel('Target Algorithms')
        # set tick labels
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(target_algs, rotation=0)

    leftmost_ax = axes[0] if len(ood_res) > 1 else axes
    rightmost_ax = axes[-1] if len(ood_res) > 1 else axes
    leftmost_ax.set_ylabel('Mean Test Return')
    
    if num_subfigs == 1:
        rightmost_ax.legend(
            ncol=3, bbox_to_anchor=(1.0, -0.15), 
            fontsize=13
            )
    else:
        rightmost_ax.legend(
            loc="lower right", ncol=1, 
            )

    if plot_suptitle:
        plt.suptitle(plot_suptitle)

    plt.tight_layout()    
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
    
    plt.show()  

def plot_mm_selfplay_scores(mm_selfplay_res:dict, 
                            save=False, savedir=None, savename=None):
    '''
    ood_res is  dict with structure {task_name: (ood_gen_means, ood_gen_cis)}
    where ood_gen_means is a dict with structure {alg_to_eval: {target_alg: [ood_gen_scores]}}
    and correspondingly for ood_gen_cis
    '''
    num_subfigs = len(mm_selfplay_res)
    _, axes = plt.subplots(1, len(mm_selfplay_res), figsize=compute_figure_size(num_subfigs))
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
            # add 95% CIs
            axes[i].errorbar(index + j * bar_width, 
                             [scores[alg][j] for alg in eval_algs], 
                             yerr=[mm_dict[name]["ci"][alg] for alg in eval_algs], 
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