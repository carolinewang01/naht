import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th

class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        return self.agents[0].init_hidden(batch_size)

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        for i in range(self.n_agents):
            q, h = self.agents[i](inputs[:, :, i], hidden_state[:, :, i])
            hiddens.append(h)
            qs.append(q)
        return th.stack(qs, dim=-2), th.stack(hiddens, dim=-2)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
