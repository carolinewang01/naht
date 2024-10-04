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
from src.utils.vis_utils import check_name_match, check_seed_pair_equal, generate_summary

matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


def reshape_mask(mask, n_agents):
    '''reshape mask similarly to encoder-decoder targets'''
    bs, ep_ts = mask.shape[:2]
    mask_inputs_all = []
    for agent_idx in range(n_agents):
        mask_input = np.concatenate([mask[:, :, :agent_idx],
                                     mask[:, :, agent_idx+1:]], axis=2)
        mask_inputs_all.append(mask_input)
    mask_inputs_all = np.stack(mask_inputs_all, axis=2).reshape(bs, ep_ts, n_agents, n_agents - 1, 1)
    return mask_inputs_all


def vis_ed_loss(data, ts_list, 
                ax, loss_type, 
                smooth, smooth_window, 
                round_ts, round_name_dict,
                vis_mode, 
                ylims=None,
                ):
    assert loss_type in ["obs", "act"]
    if loss_type == "obs": 
        loss_name = "MSE"
    else: 
        loss_name = "Prob"
        action_selector = SoftPoliciesSelector(args=None)

    colors = cm.viridis(np.linspace(0, 1, len(ts_list)))
    for i, train_ts in enumerate(ts_list):
        stats_list = []
        batch_size, ep_ts, n_agents = data[train_ts]['obs_targ'].shape[:3]
        mask = data[train_ts]['mask']
        
        if loss_type == "act":
            act_targ = th.tensor(data[train_ts]['act_targ_onehot'])
            act_logits = th.tensor(data[train_ts]['decoded_act_logits'])
            mask = th.tensor(mask)

        if vis_mode == "all": 
            # copy mask to shape (batch_size, ep_ts, n_agents, n_agents - 1, 1)
            mask = np.reshape(mask, newshape=(batch_size, ep_ts, n_agents, 1, 1))
            mask = np.repeat(mask, repeats=n_agents - 1, axis=-2)
        elif vis_mode == "uncontrolled":
            mask = 1. - reshape_mask(mask, n_agents)
        elif vis_mode == "controlled": 
            mask = reshape_mask(mask, n_agents)

        for ts in range(ep_ts):
            if loss_type == "obs":
                obs = data[train_ts]['obs_targ'][:, ts] # shape (batch_size, 1, num_agents, num_agents - 1, feat_dim)
                decoded_obs = data[train_ts]['decoded_obs'][:, ts]
                masked_mse = (obs - decoded_obs)**2 * mask[:, ts]
                stats_list.append(np.mean(masked_mse))
            else: 
                act_label = th.argmax(act_targ[:, ts], dim=-1)
                decoded_act_logits = act_logits[:, ts]
                log_prob, _ = action_selector.eval_action(agent_inputs=decoded_act_logits,
                                                          actions=act_label)    
                log_prob = log_prob.unsqueeze(-1) * mask[:, ts]
                stats_list.append(np.exp(log_prob.mean().item()))

        # round train ts to the nearest round_ts 
        train_ts_rounded = int(round(int(train_ts) / round_ts, 1))
        # apply smoothing
        if smooth: 
            stats_list = uniform_filter1d(stats_list, size=smooth_window)
        # plot stat over time
        ax.plot(stats_list, 
                label=f"{train_ts_rounded}{round_name_dict[round_ts][1]}",
                color=colors[i])
    ax.set_title(f"Average {loss_name} Over Episode", fontsize=16)
    ax.set_xlabel("Episode Timestep", fontsize=16)
    ax.set_ylabel(f"Average {loss_name}", fontsize=16)
    if ylims is not None:
        print("Setting ylims")
        ax.set_ylim(*ylims)


def vis_within_episode_performance_rebuttal(data, ts_list, 
                                   save=False, savedir=None, task=None,
                                   round_ts=1e6, smooth=False, smooth_window=5, 
                                   vis_mode='all'):
    assert vis_mode in ['all', 'uncontrolled', 'controlled']
    round_name_dict = {1e6: ['million', 'm'],  
                    1e3: ['thousand', 'k']}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    vis_ed_loss(data, ts_list, 
                ax=axes[0], loss_type="obs", 
                smooth=smooth, smooth_window=smooth_window, 
                round_ts=round_ts, round_name_dict=round_name_dict, 
                vis_mode=vis_mode, 
                # ylims=(0, 0.5)
                )
    vis_ed_loss(data, ts_list, 
                ax=axes[1], loss_type="act", 
                smooth=smooth, smooth_window=smooth_window, 
                round_ts=round_ts, round_name_dict=round_name_dict, 
                vis_mode=vis_mode, 
                ylims=(0.35, 0.65)
                )

    # increase spacing between subplots 
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.7, -0.3), 
               ncol=5,
               title=f"train ts (rounded to nearest {round_name_dict[round_ts][0]})",
               title_fontsize=14, fontsize=14)

    if save: 
        # make directory
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = os.path.join(savedir, f"{task}_poam-within-episode-loss-{vis_mode}.pdf")
        print(f"Saving to {savepath}")
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
