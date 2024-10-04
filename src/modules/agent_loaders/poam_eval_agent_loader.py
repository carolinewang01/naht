import json
from types import SimpleNamespace as SN
import torch as th
import torch.nn.functional as F

from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import *
from utils.load_utils import find_model_path


class POAMEvalAgentLoader(BaseAgentLoader):
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
        # overwrite below values with self.saved_args
        self.saved_args.n_agents = args.n_agents
        self.n_actions = self.saved_args.n_actions = args.n_actions
        self.agent_output_type = self.saved_args.agent_output_type
        self.action_selector = action_REGISTRY[self.saved_args.action_selector](self.saved_args)

        assert self.saved_args.agent in ["rnn_poam", "rnn_liam"], "Only LIAM or POAM agents are supported for evaluation"

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
        # Note that bs is addressed in poam internals
        if t_ep == 0: # manual reset
            h_e, h = self.policy.init_hidden(self.batch_size_run)
            h_e.to(self.device)
            h.to(self.device)
            self.hidden_state = h_e, h
        # pass hidden internally to agent loader
        agent_outs, h_e, h = self.policy(ep_batch, t_ep, hidden_state=self.hidden_state)
        self.hidden_state = h_e, h
        agent_outs = agent_outs[bs, :, agent_slice]
        # define dummy hidden state for return purposes only 
        hidden_state = th.zeros(ep_batch_sliced.batch_size, ret_t, 1, self.args.hidden_dim).to(self.args.device)
        
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
        
        assert self.saved_args.agent == "rnn_poam"
        self.use_param_sharing = False
        from utils.encoder_decoder import Encoder, get_encoder_input_shape
        self.policy = RNNPOAMAgent(input_shape, self.saved_args)
        encoder_input_shape = get_encoder_input_shape(self.scheme)

        if hasattr(self.saved_args, "ed_hidden_dim"):
            saved_ed_hidden_dim = self.saved_args.ed_hidden_dim
        else: 
            saved_ed_hidden_dim = self.saved_args.hidden_dim  

        self.policy.encoder = Encoder(args=self.saved_args, 
                            input_dim=encoder_input_shape,
                            hidden_dim=saved_ed_hidden_dim, 
                            output_dim=self.saved_args.embed_dim)

        self.policy.load(path=f"{model_path}/agent.th")
        self.policy.encoder.load_state_dict(th.load(f"{model_path}/encoder.th", 
                                            map_location=lambda storage, loc: storage))