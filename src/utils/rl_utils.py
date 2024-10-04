import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def get_noop_act(args, device):
    '''No-op action is used by IPPO learner to perform death masking.
    Assumption is that if agent is has only no-op actions after some timestep in ep, it is dead.
    '''
    if args.env == 'sc2':
        no_op_tensor = th.zeros((args.n_actions,), dtype=th.float32, device=device)
        no_op_tensor[0] = 1
        invalid_tensor = th.zeros((args.n_actions,), dtype=th.float32, device=device)
        return [no_op_tensor, invalid_tensor]
    elif args.env in ['stag_hunt', 'stag_hunt2']:
        no_op_tensor = th.zeros((args.n_actions,), dtype=th.float32, device=device)
        no_op_tensor[4] = 1 # when agents are frozen, they can only take the stay-action
        return [no_op_tensor]
    elif "mpe" in args.env_args['key']:
        no_op_tensor = th.zeros((args.n_actions,), dtype=th.float32, device=device)
        return [no_op_tensor]
    elif 'matrixgames' in args.env_args['key']: 
        # use invalid action because no-op actions don't exist
        no_op_tensor = th.ones((args.n_actions,), dtype=th.float32, device=device) * -1.0
        return [no_op_tensor]
    else:
        raise Exception("ERROR: get_noop_act() not implemented for env: ", args.env)

def compute_team_terminated_mask(terminated_batch:th.Tensor, 
                                 n_rollout_threads: int,
                                 ep_limit:int, device:str):
    '''Compute terminated mask based off when last agent dies, using the inbuilt team term mask.
    It is guaranteed that each episode has only index where terminated = 1
    *Found to perform worse than the default mask, so this is unused.
    '''
    # Find ts where terminated = 1
    max_ep_t = (terminated_batch == 1.).nonzero()[:, 1]
    terminated_mask = th.ones((n_rollout_threads, ep_limit, 1), 
                              dtype=th.float16, device=device)
    for j in range(n_rollout_threads):
        terminated_mask[j, max_ep_t[j]:] = 0.

    return terminated_mask

def compute_per_agent_terminated_mask(terminated_batch:th.Tensor, avail_actions_batch:th.Tensor, 
                                      n_rollout_threads: int,
                                      no_op_tensor_list:list, ep_limit:int, device:str):
    '''Compute per-agent terminated mask (when each agent dies) based off avail_actions.

    Args:
        terminated_batch: Shape (n_rollout_threads, ep_limit, 1). Note that for each episode, 
                          terminated_batch should only be 1 for a single timestep. 
                          Note that dimension called n_rollout_threads is equivalent to a dimension over episodes, 
                          as the data is structured such that each thread corresponds to an episode.
        avail_actions_batch: Shape (n_rollout_threads, ep_limit, n_agents, n_actions). 
                            Records the available actions for each agent at each timestep.
        n_rollout_threads: Number of episodes collected in parallel. 
        no_op_tensor_list: List of action representations that correspond to an agent being inactive or dead. 
        ep_limit: Maximum number of timesteps in an episode.
        device: cuda/cpu where the tensors should reside.
    Returns:
        term_mask_all: Per-agent terminated mask, for all agents. Shape (n_rollout_threads, ep_limit, n_agents).
                       To compute this, we explicitly compute the last timestep before either the agent takes a no-op action 
                       or the episode terminates (as computed from the terminated_batch) argument. 
    '''
    # compute indices where terminated=1 (only 1 per episode)
    # this should correspond to the last timestep of each episode
    max_ep_t = (terminated_batch == 1.).nonzero()[:, 1] 
    n_agents = avail_actions_batch.shape[2]
    # there can be multiple types of no-op actions, so we need to check all of them
    not_noop = th.ones((n_rollout_threads, ep_limit, n_agents), 
                       dtype=th.float32, device=device)
    
    for no_op_tensor in no_op_tensor_list:
        not_noop_current = (avail_actions_batch != no_op_tensor).any(dim=-1)
        not_noop = th.logical_and(not_noop, not_noop_current)

    term_mask_all = []
    # computation can likely be batched so that we don't need to loop over agents, but this is easier for now
    for i in range(n_agents): 
        # for agent i, find thread and ts idxes where action is not noop
        not_noop_idxs = not_noop[:, :, i].nonzero(as_tuple=True)
        thread_idxs, ts = not_noop_idxs

        # explicitly compute max timestep where agent i is not taking a no-op action
        max_agent_t = []    
        for j in range(n_rollout_threads):
            # filter data to consider all timestep indices corresponding to the current episode j
            episode_j_ts = ts.index_select(dim=0, index=(thread_idxs == j).nonzero()[:, 0])
            # take the max over all timesteps in episode j where agent i is not taking a no-op action
            # to compute the max timestep where agent i is not taking a no-op action
            max_noop_ts = episode_j_ts.max().item()
            assert max_noop_ts <= ep_limit
            max_agent_t.append(max_noop_ts)
        
        # initialize termination mask for agent i
        term_mask_agent = th.ones((n_rollout_threads, ep_limit, 1), 
                                   dtype=th.float32, device=device)
        
        # set term_mask_agent to 0 for all timesteps after the agent has taken a no-op action 
        # OR after the episode has terminated, whichever comes first 
        for j in range(n_rollout_threads):
            term_mask_agent[j, min(max_agent_t[j], max_ep_t[j]):] = 0.
        term_mask_all.append(term_mask_agent)

    term_mask_all = th.cat(term_mask_all, dim=2)
    return term_mask_all
