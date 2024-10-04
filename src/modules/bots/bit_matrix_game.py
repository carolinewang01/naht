import torch as th


class StaticBot:
    '''a bot for the bit matrix game that will select 1 if the index is 0, and 0 otherwise'''
    def __init__(self, device):
        self.device = device
    
    def reset(self):
        pass
    
    def select_action(self, obs, avail_actions, ep_agent_idx):
        '''ep_agent_idx is the agent index within the episode'''
        bs = obs.shape[:-1]
        if ep_agent_idx == 0:
            act = th.ones(bs + (1,), dtype=th.int32, device=self.device)
        else:
            act = th.zeros(bs + (1,), dtype=th.int32, device=self.device)
        return act
    
class RandomBot:
    '''a bot for the bit matrix game that will select 1 with probability p and 0 otherwise'''
    def __init__(self, p, device):
        self.p = p
        self.device = device
    
    def reset(self):
        pass
    
    def select_action(self, obs, avail_actions, ep_agent_idx):
        bs = obs.shape[:-1]
        act = (th.rand(bs + (1,), device=self.device) < self.p).int()
        return act

class ExploreBot:
    '''a bot that will select 1 with probability p1, 
    annealing to probability p2 over the course of the episode.
    '''
    def __init__(self, p1, p2, anneal_steps, device):
        self.p1 = p1
        self.p2 = p2
        self.anneal_steps = anneal_steps
        self.device = device
        self.ts = 0
    
    def reset(self): 
        self.ts = 0

    def select_action(self, obs, avail_actions, ep_agent_idx):
        bs = obs.shape[:-1]
        p = self.p1 + (self.p2 - self.p1) * min(1, self.ts / self.anneal_steps)
        act = (th.rand(bs + (1,), device=self.device) < p).int()
        self.ts += 1
        return act
    
class TimestepBot:
    '''a bot that will select 1 if the timestep mod the number of agents is equal to its own agent index
    within the team and 0 otherwise'''
    def __init__(self, n_agents, device):
        self.device = device
        # self.agent_idx = agent_idx
        self.n_agents = n_agents
        self.ts = 0
    
    def reset(self):
        self.ts = 0
    
    def select_action(self, obs, avail_actions, ep_agent_idx):
        bs = obs.shape[:-1]
        if (self.ts % self.n_agents) == ep_agent_idx:
            act = th.ones(bs + (1,), dtype=th.int32, device=self.device)
        else:
            act = th.zeros(bs + (1,), dtype=th.int32, device=self.device)
        self.ts += 1
        return act
    
