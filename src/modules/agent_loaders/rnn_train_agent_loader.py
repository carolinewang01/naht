import torch as th
import torch.nn.functional as F

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agent_loaders.base_agent_loader import BaseAgentLoader


class RNNTrainAgentLoader(BaseAgentLoader):
    '''
    The purpose of this class is to wrap an RNNAgent for open training. 
    All agent-specific information has been abstracted to this class. 
    It should be possible to perform forward passes and select actions from within this class. 
    '''
    def __init__(self, args, scheme, model_path):
        super().__init__(args, scheme,
                         obs_last_action=args.obs_last_action,
                         obs_agent_id=args.obs_agent_id,
                         obs_team_composition=args.obs_team_composition,
                         n_agents=args.n_agents)
        
        input_shape = self._get_input_shape(scheme, args)
        
        self._build_agent(input_shape=input_shape)
        if model_path != "":
            raise NotImplementedError

        self.agent_output_type = self.args.agent_output_type
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)

        assert self.args.agent !="rnn_poam"
    
    def predict(self, ep_batch, agent_idx_list, 
                t_ep, t_env, bs, test_mode
                ):
        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        ret_t = ep_batch.max_seq_length if t_ep is None else 1
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts][:, :, agent_idx_list]
    
        # fwd pass
        inputs = self._build_inputs(ep_batch_sliced, t_ep, agent_idx=None) # build for all agents
        inputs = inputs[:, :, agent_idx_list]
        hidden_state = ep_batch_sliced["actor_hidden_states"][:, ts][:, :, agent_idx_list]
        agent_outs, hidden_state = self.policy(inputs, hidden_state)
        
        ret_shape = (ep_batch_sliced.batch_size, ret_t, len(agent_idx_list), -1)
        agent_outs = agent_outs.reshape(*ret_shape)
        
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                
                agent_outs[avail_actions == 0] = -1e10

        chosen_actions = self.action_selector.select_action(agent_outs, 
                                                            avail_actions,
                                                            t_env, 
                                                            test_mode=test_mode)
        chosen_actions = chosen_actions.reshape(*ret_shape)
        hidden_state = hidden_state.reshape(*ret_shape)
        return agent_outs, chosen_actions, hidden_state

    def _build_agent(self, input_shape):
        '''Initialize agent'''
        self.policy = agent_REGISTRY[self.args.agent](input_shape, self.args)        