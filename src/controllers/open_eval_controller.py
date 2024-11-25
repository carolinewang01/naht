from modules.agent_loaders import REGISTRY as agent_loader_REGISTRY
import os
import random
import numpy as np
import torch as th


class OpenEvalMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.n_uncontrolled = args.n_uncontrolled
        self._build_agent_pool(scheme)
        self.sample_agent_team()

    def select_actions(self, ep_batch, 
                       t_ep, t_env, bs=slice(None), test_mode=True):
        # Get the agent outputs for each agent in the pool
        joint_act = []
        joint_hidden = []
        for agent_idx, subteam_idx, team_name in self._active_team:
            if team_name == "uncontrolled_agent_subteam":
                agent = self.uncontrolled_agent_pool[subteam_idx]
            else:
                assert team_name == "trained_agent_subteam"
                agent = self.trained_agent_pool[subteam_idx]

            _, act, hidden_state = agent.predict(ep_batch, agent_idx=agent_idx, 
                                                 t_ep=t_ep, t_env=t_env, bs=bs, test_mode=test_mode)
            joint_act.append(act)
            joint_hidden.append(hidden_state)
        joint_act = th.cat(joint_act, dim=2)
        joint_hidden = th.cat(joint_hidden, dim=2)
        return joint_act, joint_hidden

    def init_hidden(self, batch_size):
        '''A dummy function for open evaluation only.'''
        # return self.trained_agent_pool[0].init_hidden(batch_size)
        return th.zeros(batch_size, 1, self.n_agents, self.args.hidden_dim)

    def parameters(self):
        parameters = []
        for agent in [*self.trained_agent_pool, *self.uncontrolled_agent_pool]:
            parameters += list(agent.parameters())
        return parameters

    def cuda(self):
        for agent in [*self.trained_agent_pool, *self.uncontrolled_agent_pool]:
            agent.cuda()
        
    def sample_agent_team(self): 
        '''
        This function controls the openness of the evaluation.
        Randomly samples n_uncontrolled agents from the uncontrolled agent team.
        ''' 
        uncontrolled_agent_idxs = list(np.random.choice(len(self.uncontrolled_agent_pool), 
                                                     self.n_uncontrolled, 
                                                     replace=False))
        trained_agent_idxs = list(np.random.choice(len(self.trained_agent_pool), 
                                                      self.n_agents - self.n_uncontrolled, 
                                                      replace=False))
        # order agents from uncontrolled and trained teams randomly
        agent_order = list(range(self.n_agents))
        random.shuffle(agent_order)
        self._active_team = [(agent_order.pop(0), i, "uncontrolled_agent_subteam") for i in uncontrolled_agent_idxs] + \
                            [(agent_order.pop(0), i, "trained_agent_subteam") for i in trained_agent_idxs]
        
        # original agent order
        # self._active_team = [(i, i, "trained_agent_subteam") for i in range(self.n_agents)]
        # shuffled agent order
        # self._active_team = [(agent_order.pop(0), i, "trained_agent_subteam") for i in range(self.n_agents)]

        self._active_team = sorted(self._active_team, key=lambda x: x[0])
        # indices of the trained agents
        trained_agent_idxs = [agent_idx for agent_idx, _, team_name in self._active_team if team_name == "trained_agent_subteam"]
        return trained_agent_idxs

    def _build_agent_pool(self, scheme):
        '''
        Example subset of self.args: 
        base_checkpoint_path: ""
        trained_agents: 
            agent_0: 
                agent_loader: "rnn_agent_loader"
                agent_path: ""
                n_agents_to_populate: 2
                load_step: 0
        uncontrolled_agents:
            agent_0:
                agent_loader: "rnn_agent_loader"
                agent_path: ""
                n_agents_to_populate: 2
                load_step: 0
        '''
        base_uncntrl_path = self.args.base_uncntrl_path
        trained_agents_dict = self.args.trained_agents
        uncontrolled_agents_dict = self.args.uncntrl_agents

        self.trained_agent_pool = []
        for _, agent_cfg in trained_agents_dict.items():
            # each agent_cfg can be used to load multiple agents                        
            for i in range(agent_cfg["n_agents_to_populate"]):
                model_path = os.path.join(base_uncntrl_path, agent_cfg["agent_path"])
                agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](args=self.args, 
                                                                         scheme=scheme, 
                                                                         model_path=model_path, 
                                                                         load_step=agent_cfg["load_step"], 
                                                                         load_agent_idx=i
                                                                        )
                self.trained_agent_pool.append(agent)

        self.uncontrolled_agent_pool = []
        for _, agent_cfg in uncontrolled_agents_dict.items():
            for i in range(agent_cfg["n_agents_to_populate"]):
                if agent_cfg["agent_loader"] == "bot_agent_loader":
                        agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](
                            args=self.args, scheme=scheme,
                            bot_name=agent_cfg["bot_name"],
                        )
                else:                
                    model_path = os.path.join(base_uncntrl_path, agent_cfg["agent_path"])
                    agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](args=self.args, 
                                                                            scheme=scheme, 
                                                                            model_path=model_path, 
                                                                            load_step=agent_cfg["load_step"],
                                                                            load_agent_idx=i,
                                                                            )
                self.uncontrolled_agent_pool.append(agent)

        assert len(self.trained_agent_pool) + len(self.uncontrolled_agent_pool) >= self.n_agents
