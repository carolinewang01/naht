import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch as th
from scipy.ndimage.filters import uniform_filter1d

# visualize how well the encoder decoder can recover the target action using a heatmap 
from src.components.action_selectors import SoftPoliciesSelector

matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

def load_data(expt_datapath, old_format: bool):
    ts_list = os.listdir(expt_datapath) # list of timesteps as strings
    # sort ts_list by integer ordering
    ts_list = sorted(ts_list, key=lambda x: int(x))
    ts_pathlist = [os.path.join(expt_datapath, ts) for ts in ts_list]

    data = {}
    for i, ts in enumerate(ts_list):
        ts_path = ts_pathlist[i]
        archive_list = os.listdir(ts_path)
        archive_pathlist = [os.path.join(ts_path, archive) for archive in archive_list]
        data[ts] = {}
        for i, archive_path in enumerate(archive_pathlist):
            if old_format:
                t_name = archive_list[i]
                data[ts][t_name.replace('.npy', '')] = np.load(archive_path)
            # loading in a npz file
            else:
                archive_data = np.load(archive_path)
                tensor_namelist = archive_data.files
                for t_name in tensor_namelist:
                    data[ts][t_name] = archive_data[t_name]
    print(data.keys())
    return data, ts_list

def vis_within_episode_performance(data, ts_list, 
                                   save=False, savedir=None, task=None,
                                   round_ts=1e6, smooth=False, smooth_window=5
                                    ):
    round_name_dict = {1e6: ['million', 'm'],  
                    1e3: ['thousand', 'k']}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # visualize within-episode accuracy
    colors = cm.viridis(np.linspace(0, 1, len(ts_list)))

    for i, train_ts in enumerate(ts_list):
        mse_list = []
        batch_size, ep_ts, n_agents = data[train_ts]['obs_targ'].shape[:3]
        mask = data[train_ts]['mask']
        
        # copy mask to shape (batch_size, ep_ts, n_agents, n_agents - 1, 1)
        mask = np.reshape(mask, newshape=(batch_size, ep_ts, n_agents, 1, 1))
        mask = np.repeat(mask, repeats=n_agents - 1, axis=-2)

        for ts in range(ep_ts):
            obs = data[train_ts]['obs_targ'][:, ts] # shape (batch_size, 1, num_agents, num_agents - 1, feat_dim)
            decoded_obs = data[train_ts]['decoded_obs'][:, ts]
            masked_mse = np.mean((obs - decoded_obs)**2 * mask[:, ts])
            mse_list.append(masked_mse)
        # plot the mse over time
        # round train ts to the nearest round_ts 
        train_ts_rounded = round(int(train_ts) / round_ts, 1)

        if smooth: 
            mse_list = uniform_filter1d(mse_list, size=smooth_window)
        axes[0].plot(mse_list, 
                     label=f"train ts={train_ts_rounded}", 
                     color=colors[i])
    # print("Last train ts mse_list: ", mse_list)
    axes[0].set_title(f"Average MSE Over Episode", fontsize=20)
    axes[0].set_xlabel("Episode Timestep", fontsize=20)
    axes[0].set_ylabel("Average MSE", fontsize=20)

    ####### visualize within-episode action probabilities
    action_selector = SoftPoliciesSelector(args=None)
    colors = cm.viridis(np.linspace(0, 1, len(ts_list)))

    for i, train_ts in enumerate(ts_list):
        log_prob_list = []
        batch_size, ep_ts, n_agents = data[train_ts]['act_targ_onehot'].shape[:3]

        # shape (batch_size, ts, num_agents, num_agents - 1, feat_dim)
        act_targ = th.tensor(data[train_ts]['act_targ_onehot'])
        act_logits = th.tensor(data[train_ts]['decoded_act_logits'])
        mask = th.tensor(data[train_ts]['mask'])
        # copy mask to shape (batch_size, ep_ts, n_agents, n_agents - 1, 1)
        mask = mask.unsqueeze(-1).unsqueeze(-1)

        for ts in range(ep_ts):
            act_label = th.argmax(act_targ[:, ts], dim=-1)
            decoded_act_logits = act_logits[:, ts]
            log_prob, _ = action_selector.eval_action(agent_inputs=decoded_act_logits,
                                                      actions=act_label)    
            log_prob = log_prob.unsqueeze(-1) * mask[:, ts] # .unsqueeze(-1)
            log_prob_list.append(np.exp(log_prob.mean().item()))

        # each train ts is plotted as a curve on the graph. I want the curves to be colored in a gradient, with lighter colors corresponding to earlier timesteps
        train_ts_rounded = int(round(int(train_ts) / round_ts, 1))
        if smooth: 
            log_prob_list = uniform_filter1d(log_prob_list, size=smooth_window)
        axes[1].plot(log_prob_list, 
                     label=f"{train_ts_rounded}{round_name_dict[round_ts][1]}", 
                     color=colors[i])

    axes[1].set_title(f"Average Prob Over Episode", fontsize=20)
    axes[1].set_xlabel("Episode Timestep", fontsize=20)
    axes[1].set_ylabel("Average Prob", fontsize=20)

    # increase spacing between subplots 
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.0, -0.3), 
               ncol=5,
               title=f"training timesteps (rounded to nearest {round_name_dict[round_ts][0]})",
               title_fontsize=18, fontsize=18)

    if save: 
        # make directory
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = os.path.join(savedir, f"{task}_poam-within-episode-loss.pdf")
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def plot_obs_heatmap(data, train_ts_list, 
                     obs_x_size:int,
                     batch_idx=0, agent_idx=0, modelled_agent_idx=0,
                     ep_ts_list=[20]):
    '''
    Visualize how well the encoder decoder can recover the target observation using a heatmap 
    Heatmap is will be obs_x_size x -1
    '''
    for train_ts in train_ts_list:
        for ts in ep_ts_list:
            obs = data[train_ts]['obs_targ'][batch_idx, ts, agent_idx, modelled_agent_idx] # shape (batch_size, ts, num_agents, num_agents - 1, feat_dim)
            decoded_obs = data[train_ts]['decoded_obs'][batch_idx, ts, agent_idx, modelled_agent_idx]
            print("MASK: ", data[train_ts]['mask'][batch_idx, ts, agent_idx])
            obs = obs / np.max(obs)
            decoded_obs = decoded_obs / np.max(decoded_obs)
            obs = obs.reshape(obs_x_size, -1)
            decoded_obs = decoded_obs.reshape(obs_x_size, -1)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            sns.heatmap(obs, ax=ax[0], vmin=0, vmax=1)
            sns.heatmap(decoded_obs, ax=ax[1], vmin=0, vmax=1)
            ax[0].set_title('Target Observation')
            ax[1].set_title('Decoded Observation')
            plt.suptitle(f"Train TS: {train_ts} | Episode TS: {ts}")
            plt.show()