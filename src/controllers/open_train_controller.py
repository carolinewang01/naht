from modules.agent_loaders import REGISTRY as agent_loader_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import os
import random
import numpy as np
import torch as th


class OpenTrainMAC:
    def __init__(self, scheme, groups, args):
        '''This class was based off the OpenEvalMAC'''
        self.n_agents = args.n_agents
        self.args = args
        self.n_unseen = args.n_unseen
        self._build_agent_pool(scheme)
        self.sample_agent_team()

        # hacky way to provide compatibility with learners
        self.action_selector = self.trained_agent.action_selector

    def select_actions(self, ep_batch,
                       t_ep, t_env, bs=slice(None), 
                       test_mode=False):
        '''Select joint action using the active team'''
        trained_agent_idxs = [agent_idx for agent_idx, _, team_name in self._active_team if team_name == "trained_agent_subteam"]
        _, trained_agent_act, trained_agent_hidden = self.trained_agent.predict(ep_batch, agent_idx_list=trained_agent_idxs,
                                                                                t_ep=t_ep, t_env=t_env, bs=bs,
                                                                                test_mode=test_mode)
        
        # compile outputs
        curr_agent_idx = 0
        joint_act = []
        joint_hidden = []
        for agent_idx, subteam_idx, team_name in self._active_team:
            if team_name == "unseen_agent_subteam":
                agent = self.unseen_agent_pool[subteam_idx]
                # unseen agents should be evaluated in test mode
                _, act, hidden_state = agent.predict(ep_batch, agent_idx=agent_idx, 
                                                     t_ep=t_ep, t_env=t_env, bs=bs
                                                     )
            else:
                assert team_name == "trained_agent_subteam"
                act = trained_agent_act[:, :, slice(curr_agent_idx, curr_agent_idx+1)]
                hidden_state = trained_agent_hidden[:, :, slice(curr_agent_idx, curr_agent_idx+1)]
                curr_agent_idx += 1
            
            joint_act.append(act)
            joint_hidden.append(hidden_state)
            
        joint_act = th.cat(joint_act, dim=2)
        joint_hidden = th.cat(joint_hidden, dim=2)
        return joint_act, joint_hidden
    
    def forward(self, ep_batch, t=None, test_mode=False):
        '''This function is used by learners only. Thus, we only execute the forward pass 
        using the trained agent.'''
        trained_agent_idxs = list(range(self.n_agents))
         # t_env used to select actions but we only need the logits here, so the value doesn't matter
        agent_out, _, hidden = self.trained_agent.predict(ep_batch, agent_idx_list=trained_agent_idxs,
                                                          t_ep=t, t_env=0,
                                                          bs=slice(None),
                                                          test_mode=test_mode)
        
        return agent_out, hidden

    def set_encoder(self, encoder): 
        self.trained_agent.policy.encoder = encoder
        
    def init_hidden(self, batch_size):
        '''A dummy function for open evaluation only.'''
        return th.zeros(batch_size, 1, self.n_agents, self.args.hidden_dim)

    def parameters(self):
        '''Return learnable parameters'''
        return self.trained_agent.parameters()
    
    def load_state(self, other_mac):
        '''Used by the Q-learning, QMIX, QTRAN and MADDPG learners'''
        self.trained_agent.load_state_dict(other_mac.trained_agent.state_dict())

    def cuda(self):
        for agent in [self.trained_agent, *self.unseen_agent_pool]:
            agent.cuda()
    
    def save_models(self, path):
        self.trained_agent.save_models(path)
    
    def load_models(self, path):
        # TODO check if this is correct
        self.trained_agent.load_state_dict(th.load("{}/agent.th".format(path)))

    def sample_agent_team(self): 
        '''
        This function controls the openness of the evaluation.
        Randomly samples n_unseen agents from the unseen agent team.
        ''' 
        # sample number of unseen agents
        if self.n_unseen is None: # sample n_unseen uniformly from 1 to n_agents-1
            n_unseen = np.random.randint(1, self.n_agents)
        else:
            n_unseen = self.n_unseen
        # sample unseen agent team
        active_unseen_team = np.random.choice(list(self.unseen_agent_teams.keys()))            
        self.unseen_agent_pool = self.unseen_agent_teams[active_unseen_team]
        unseen_agent_idxs = list(np.random.choice(len(self.unseen_agent_pool), 
                                                     n_unseen, 
                                                     replace=False))
        trained_agent_idxs = list(np.random.choice(range(self.n_agents), 
                                                   self.n_agents - n_unseen, 
                                                   replace=False))
        # order agents from unseen and trained teams randomly
        agent_order = list(range(self.n_agents))
        random.shuffle(agent_order)
        self._active_team = [(agent_order.pop(0), i, "unseen_agent_subteam") for i in unseen_agent_idxs] + \
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
        Example yaml to be loaded into args: 
        base_checkpoint_path: ""
        trained_agents:
            agent_0:
                agent_loader: "rnn_train_agent_loader"
                agent_path: "" # leave empty for training from scratch
        unseen_agents:
            agent_0:
                agent_loader: "rnn_eval_agent_loader"
                agent_path: ""
                n_agents_to_populate: 5
                load_step: best
        '''
        # initialize training agents
        agent_loader = self.args.trained_agents['agent_0']['agent_loader']
        agent_path = self.args.trained_agents['agent_0']['agent_path']
        self.trained_agent = agent_loader_REGISTRY[agent_loader](args=self.args,
                                                                 scheme=scheme,
                                                                 model_path=agent_path)
        
        # initialize+load unseen agents
        base_path = self.args.base_results_path
        unseen_agents_dict = self.args.unseen_agents
        self.unseen_agent_teams = {}

        for agent_nm, agent_cfg in unseen_agents_dict.items():
            self.unseen_agent_teams[agent_nm] = []
            use_param_sharing = False

            assert agent_cfg["n_agents_to_populate"] >= self.n_agents - 1
            for i in range(agent_cfg["n_agents_to_populate"]):
                # load in new agent only if param sharing is not used
                # else, python will place a reference to the single agent in all team slots
                if not use_param_sharing: 
                    if agent_cfg["agent_loader"] == "bot_agent_loader":
                        agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](
                            args=self.args, scheme=scheme,
                            bot_name=agent_cfg["bot_name"],
                        )
                    else:
                        model_path = os.path.join(base_path, agent_cfg["agent_path"])
                        agent = agent_loader_REGISTRY[agent_cfg['agent_loader']](args=self.args, 
                                                                                scheme=scheme, 
                                                                                model_path=model_path, 
                                                                                load_step=agent_cfg["load_step"],
                                                                                load_agent_idx=i, # only matters for ns methods
                                                                                test_mode=agent_cfg["test_mode"]
                                                                                )
                        use_param_sharing = agent.use_param_sharing
                self.unseen_agent_teams[agent_nm].append(agent)
