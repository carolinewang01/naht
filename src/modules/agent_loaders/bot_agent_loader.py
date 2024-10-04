from modules.agent_loaders.base_agent_loader import BaseAgentLoader
from modules.bots import REGISTRY as bot_REGISTRY

class BotAgentLoader(BaseAgentLoader):
    '''
    The purpose of this class is to wrap a hand-coded BotAgent. 
    All agent-specific information has been abstracted to this class. 
    It should be possible to load an BotAgent,
    perform forward passes, and predict actions from within this class. 
    '''
    def __init__(self, 
                 args, scheme,
                 bot_name
                 ):
        super().__init__(args, scheme, 
                         n_agents=args.n_agents,
                         obs_last_action=False, 
                         obs_agent_id=False, 
                         )
        self.bot_name = bot_name
        # note that this is the index used to load the agent from saved, 
        # not the agent's actual index in the episode

        self.batch_size_run = self.args.batch_size_run
        self.device = self.args.device
        self.load(bot_name=bot_name)

    def predict(self, ep_batch, 
                agent_idx, # within episode-team index
                t_ep, t_env, bs, 
                test_mode=False # not relevant for bots currently
                ):
        if t_ep == 0:
            self.policy.reset()

        ts = slice(None) if t_ep is None else slice(t_ep, t_ep+1)
        # ret_t = ep_batch.max_seq_length if t_ep is None else 1
        agent_slice = slice(agent_idx, agent_idx+1)
        ep_batch_sliced = ep_batch[bs]
        avail_actions = ep_batch["avail_actions"][bs, ts, agent_slice]
        dummy_hidden_state = ep_batch_sliced["actor_hidden_states"][:, ts, agent_slice]
        
        # fwd pass
        inputs = self._build_inputs(ep_batch_sliced, t_ep, agent_idx=agent_idx)
        actions = self.policy.select_action(obs=inputs, avail_actions=avail_actions, 
                                            ep_agent_idx=agent_idx)
        return None, actions, dummy_hidden_state

    
    def load(self, bot_name):
        # Load the agent
        if bot_name == "bit-matrix-game:static":
            self.policy = bot_REGISTRY[bot_name](device=self.device)
        elif bot_name == "bit-matrix-game:random":
            self.policy = bot_REGISTRY[bot_name](p=1./3., 
                                                 device=self.device)
        elif bot_name == "bit-matrix-game:explore":
            self.policy = bot_REGISTRY[bot_name](p1=1./6., p2=1./2., # 0.05, 
                                                 anneal_steps=10, device=self.device)
        elif bot_name == "bit-matrix-game:timestep":
            self.policy = bot_REGISTRY[bot_name](n_agents=self.n_agents, device=self.device)
        else:
            raise NotImplementedError(f"Bot not implemented: {bot_name}")
            