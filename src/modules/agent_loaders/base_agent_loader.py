import torch as th
import torch.nn.functional as F
from modules.agents import *


class BaseAgentLoader():
    def __init__(self, args, scheme, 
                 n_agents,
                 obs_last_action, obs_agent_id, 
                 obs_team_composition=False
                 ):
        self.args = args
        self.scheme = scheme
        # passed explicitly in case we want to override args
        self.obs_last_action = obs_last_action
        self.obs_agent_id = obs_agent_id
        self.obs_team_composition = obs_team_composition
        self.n_agents = n_agents

    def _build_inputs(self, batch, t=None, agent_idx=None):
        bs = batch.batch_size
        ret_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        agent_slice = slice(None) if agent_idx is None else slice(agent_idx, agent_idx+1)

        inputs = []
        inputs.append(batch["obs"][:, ts, agent_slice])  # batch['obs'] has shape (n_threads, ts, n_agents, obs_shape)
        if self.obs_last_action:
            if t is None: # build last act for all timesteps
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0], agent_slice]), 
                                       batch["actions_onehot"][:, slice(0, ret_t-1), agent_slice]],
                                       dim=1
                                       )
                inputs.append(last_act_all) 
            else: # get last act for single timestep only 
                last_act = th.zeros_like(batch["actions_onehot"][:, ts, agent_slice]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t), agent_slice]
                inputs.append(last_act)
        if self.obs_agent_id:
            if agent_idx is None: # build agent id for all agents
                agent_id_onehot = th.eye(self.n_agents, device=batch.device)
                agent_id_onehot = agent_id_onehot.expand(bs, ret_t, self.n_agents, -1)
                inputs.append(agent_id_onehot)
            else: # get agent id for single agent only 
                agent_id_onehot = F.one_hot(th.tensor(agent_idx, device=batch.device), 
                                        num_classes=self.n_agents)
                agent_id_onehot = agent_id_onehot.expand(bs, ret_t, 1, -1)
                inputs.append(agent_id_onehot)
        if self.obs_team_composition:
            trainable_agents_mask = batch["trainable_agents"][:, ts, agent_slice] # shape (bs, 1, n_agents, 1)
            trainable_agents_feat = trainable_agents_mask.squeeze(-1).unsqueeze(-2).tile(1, 1, self.n_agents, 1) # shape (bs, 1, n_agents, n_agents)
            # sum across last dimension and divide by n_agents
            trainable_agents_feat = th.sum(trainable_agents_feat, dim=-1, keepdim=True) / self.n_agents   
            inputs.append(trainable_agents_feat)
        inputs = th.cat(inputs, dim=-1)
        return inputs
    
    def predict(self, ep_batch, agent_idx, 
                t_ep, t_env, bs
                ):
        '''Should return agent outs, chosen actions, and hidden state'''
        raise NotImplementedError
        
    def cuda(self):
        try:
            self.policy.cuda()
        except Exception as e:
            print(e)
            print("Device is: ", self.args.device)

    def parameters(self):
        return self.policy.parameters()
    
    def init_hidden(self, batch_size):
        return self.policy.init_hidden(batch_size)

    def save_models(self, path):
        th.save(self.policy.state_dict(), "{}/agent.th".format(path))

    def _get_input_shape(self, scheme, args):
        '''Pass args as parameter to allow greater flexibility.'''
        input_shape = scheme["obs"]["vshape"]
        if self.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.obs_agent_id:
            input_shape += args.n_agents
        if self.obs_team_composition:
            input_shape += 1 # args.n_agents
        return input_shape
