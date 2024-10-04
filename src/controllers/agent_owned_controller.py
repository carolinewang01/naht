from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class AgentOwnedMAC:
    '''Same as basic MAC except that each agent builds its own inputs. 
    Introduced for POAM.'''
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
                                                    t_ep, 
                                                    test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs, 
                                                            avail_actions, t_env, test_mode=test_mode)
        return chosen_actions, hidden_states

    def forward(self, ep_batch, t=None, test_mode=False):
        ts = slice(None) if t is None else slice(t, t+1)
        agent_outs, hidden_states = self.agent(ep_batch, t)
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

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape