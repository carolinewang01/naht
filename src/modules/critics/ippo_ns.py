import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.critic_base import CriticBase
from modules.critics.ippo import IPPOCritic

class ValueNormalizerNS(): 
    def __init__(self, value_normalizers):
        self.value_normalizers = value_normalizers
    
    def normalize(self, returns):
        # input has shape (batch_size, episode_length, n_agents)
        returns = returns.unsqueeze(-1)
        normalized = []
        for agent_idx, normalizer in enumerate(self.value_normalizers):
            normalized.append(normalizer.normalize(returns[:, :, [agent_idx]]))
        return th.cat(normalized, dim=2).squeeze(-1)
    
    def update(self, returns):
        # input has shape (batch_size, episode_length, n_agents)
        returns = returns.unsqueeze(-1)
        for agent_idx, normalizer in enumerate(self.value_normalizers):
            normalizer.update(returns[:, :, [agent_idx]])
    
    def denormalize(self, values):
        # values has shape (batch_size, episode_length, n_agents)
        values = values.unsqueeze(-1)
        denormalized = []
        for agent_idx, normalizer in enumerate(self.value_normalizers):
            denormalized.append(normalizer.denormalize(values[:, :, [agent_idx]]))
        return th.cat(denormalized, dim=2).squeeze(-1) # shape (batch_size, episode_length, n_agents)


class IPPOCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(IPPOCriticNS, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.output_type = "v"

        # Set up network layers
        self.critics = th.nn.ModuleList([IPPOCritic(scheme, args) for _ in range(self.n_agents)])
        if self.critics[0].value_normalizer is None:
            self.value_normalizer = None
        else:
            self.value_normalizer = ValueNormalizerNS(value_normalizers=[c.value_normalizer for c in self.critics])

    def init_hidden(self): 
        # shape: bs, n_agents, hidden_dim
        return self.critics[0].init_hidden() 
    
    def forward(self, batch, hidden_state, t=None):
        inputs, bs, max_t = self.critics[0]._build_inputs(batch, t=t)
        qs = []
        hiddens = []
        for i in range(self.n_agents):
            # iteration over timestep dim occurs over OUTER loop
            q, h = self.critics[i](inputs[:, :, i], 
                                   hidden_state[:, :, i],
                                   build_inputs=False, 
                                   )
            qs.append(q)
            hiddens.append(h)
        q = th.stack(qs, dim=-2)
        hiddens = th.stack(hiddens, dim=-2)
        return q, hiddens

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self, device="cuda:0"):
        for c in self.critics:
            c.cuda(device=device)