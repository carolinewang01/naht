from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

    def select_actions(self, ep_batch,
                       t_ep, t_env, bs=slice(None), 
                       test_mode=False):
        # Only select actions for the selected batch elements in bs
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        avail_actions = ep_batch["avail_actions"][bs, ts] # why isn't there an extra timestep here? 
        agent_outputs, hidden_states = self.forward(ep_batch[bs], 
                                                    t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs, 
                                                            avail_actions, t_env, test_mode=test_mode)
        return chosen_actions, hidden_states

    def forward(self, ep_batch, t=None, test_mode=False):
        if t is None:
            ts = slice(None)
            ret_batch_dim = (ep_batch.batch_size, ep_batch.max_seq_length, self.n_agents)
        else: 
            ts = slice(t, t+1)
            ret_batch_dim = (ep_batch.batch_size, 1, self.n_agents)

        # forward pass
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, hidden_states = self.agent(agent_inputs, ep_batch["actor_hidden_states"][:, ts])
        agent_outs = agent_outs.view(*ret_batch_dim, -1)
        hidden_states = hidden_states.view(*ret_batch_dim, -1)
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                avail_actions = ep_batch["avail_actions"][:, ts]
                agent_outs[avail_actions == 0] = -1e10
                
        return agent_outs, hidden_states

    def init_hidden(self, batch_size):
        return self.agent.init_hidden(batch_size)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t=None):
        '''
        Assumes homogenous agents with flat observations.
        Other MACs might want to e.g. delegate building inputs to each agent
        If t=None, then returns inputs for all timesteps.
        '''
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts])  # b1av
        if self.args.obs_last_action:
            if t is None: # build last act for all timesteps
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                       batch["actions_onehot"][:, slice(0, max_t-1)]],
                                       dim=1
                                       )
                inputs.append(last_act_all) 
            else: # get last act for single timestep only 
                last_act = th.zeros_like(batch["actions_onehot"][:, ts]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t)]
                inputs.append(last_act)

        if self.args.obs_agent_id:
            agent_id_onehot = th.eye(self.n_agents, device=batch.device)
            agent_id_onehot = agent_id_onehot.expand(bs, max_t, self.n_agents, -1) if t is None \
                else agent_id_onehot.expand(bs, 1, self.n_agents, -1)
            inputs.append(agent_id_onehot)
        inputs = th.cat(inputs, dim=-1) # shape (bs, ts, n_agents, input_shape)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape