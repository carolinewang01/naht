import json
from types import SimpleNamespace as SN
import torch as th
import torch.nn.functional as F

from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import *
from utils.load_utils import find_model_path


class RNNEvalAgentLoader(BaseAgentLoader):
    '''
    The purpose of this class is to wrap an RNNAgent for evaluation only. 
    All agent-specific information has been abstracted to this class. 
    It should be possible to load an RNNAgent from logs only,
    perform forward passes, and predict actions from within this class. 
    '''
    def __init__(self, 
                 args, scheme, 
                 model_path, 
                 load_step, load_agent_idx,
                 test_mode=True
                ):
        # load args from saved     
        config_path = f"{model_path.replace('models', 'sacred')}/1/config.json"
        self.saved_args = SN(**json.load(open(config_path, "r")))
        super().__init__(args, scheme, 
                         n_agents=args.n_agents,
                         obs_last_action=self.saved_args.obs_last_action, 
                         obs_agent_id=self.saved_args.obs_agent_id, 
                         )
        # overwrite below values of self.saved_args
        self.saved_args.n_agents = args.n_agents
        self.n_actions = self.saved_args.n_actions = args.n_actions
        self.agent_output_type = self.saved_args.agent_output_type
        self.action_selector = action_REGISTRY[self.saved_args.action_selector](self.saved_args)
        # use scheme of current runtime. if this doesn't match the training scheme, then 
        # agents won't be able to run anyways
        input_shape = self._get_input_shape(scheme, self.saved_args)
        assert load_step in ['best', 'last'] or load_step.isdigit(), "load_step must be 'best', 'last', or a digit"
        self.load(input_shape, model_path, 
                  load_step=load_step, load_agent_idx=load_agent_idx)

        self.batch_size_run = self.args.batch_size_run
        self.device = self.args.device
        try:
            self.policy.cuda(self.device)
        except Exception as e:
            print(e)
            print("Device is: ", self.device)
        self.test_mode = test_mode

    def predict(self, ep_batch, agent_idx, 
                t_ep, t_env, bs, 
                test_mode=None # can override self.test_mode through this arg
                ):
        if test_mode is None:
            test_mode = self.test_mode # default value

        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        ret_t = ep_batch.max_seq_length if t_ep is None else 1
        agent_slice = slice(agent_idx, agent_idx+1)
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts, agent_slice]
        
        # fwd pass
        inputs = self._build_inputs(ep_batch_sliced, t_ep, agent_idx=agent_idx)
        hidden_state = ep_batch_sliced["actor_hidden_states"][:, ts, agent_slice]
        agent_outs, hidden_state = self.policy(inputs, hidden_state)
        
        ret_shape = (ep_batch_sliced.batch_size, ret_t, 1, -1)
        agent_outs = agent_outs.reshape(*ret_shape)
        
        if self.agent_output_type == "pi_logits":
            if getattr(self.saved_args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs[avail_actions == 0] = -1e10

        chosen_actions = self.action_selector.select_action(agent_outs, 
                                                            avail_actions,
                                                            t_env, 
                                                            test_mode=test_mode)
        chosen_actions = chosen_actions.reshape(*ret_shape)
        hidden_state = hidden_state.reshape(*ret_shape)
        return agent_outs, chosen_actions, hidden_state

    def load(self, input_shape, path, load_step, load_agent_idx):
        # if load_step is 0, load max possible
        # if load_step is best, load best chkpt
        model_path, _ = find_model_path(path, load_step=load_step, logger=None)
        
        if self.saved_args.agent == "rnn":
            self.use_param_sharing = True
            self.policy = RNNAgent(input_shape, self.saved_args) 
            self.policy.load_state_dict(th.load(f"{model_path}/agent.th", 
                                                map_location=lambda storage, loc: storage))
        elif self.saved_args.agent == "rnn_ns": 
            self.use_param_sharing = False
            joint_policy = RNNNSAgent(input_shape, self.saved_args)
            joint_policy.load_state_dict(th.load(f"{model_path}/agent.th", 
                                                map_location=lambda storage, loc: storage))
            self.policy = joint_policy.agents[load_agent_idx]
        elif self.saved_args.agent == "rnn_norm":
            self.use_param_sharing = True
            self.policy = RNNNormAgent(input_shape, self.saved_args)
            self.policy.load_state_dict(th.load(f"{model_path}/agent.th", 
                                                map_location=lambda storage, loc: storage))
        elif self.saved_args.agent == "rnn_norm_ns":
            self.use_param_sharing = False
            joint_policy = RNNNormNSAgent(input_shape, self.saved_args)
            joint_policy.load_state_dict(th.load(f"{model_path}/agent.th", 
                                                map_location=lambda storage, loc: storage))
            self.policy = joint_policy.agents[load_agent_idx]
        else: 
            raise NotImplementedError
