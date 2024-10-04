# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F
from utils.mlp import MLPBase
from utils.mappo_util import init_rnn, init_module


class RNNNormAgent(nn.Module):
    '''Identical to RNNAgent except applies layer normalization to inputs.
    This policy class was introduced for PPO.
    '''
    def __init__(self, input_shape, args):
        super(RNNNormAgent, self).__init__()
        self.args = args
        self.input_size = input_shape
        self.n_agents = args.n_agents
        self.base = MLPBase(input_shape, args.hidden_dim, 
                            n_hidden_layers=1,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )
        if self.args.use_rnn:
            self.rnn = init_rnn(nn.GRUCell(args.hidden_dim, args.hidden_dim), args.use_orthogonal_init)
            self.rnn_norm = nn.LayerNorm(args.hidden_dim)
        else:
            self.rnn = init_module(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.fc2 = init_module(nn.Linear(args.hidden_dim, args.n_actions))

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        return self.base.mlp.fc1[0].weight.new(batch_size, 1, self.n_agents, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        orig_batch_dims = inputs.shape[:-1]
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        inputs = inputs.reshape(-1, self.input_size)

        x = self.base(inputs)

        if self.args.use_rnn:
            h_out = self.rnn(x, h_in)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = F.relu(self.rnn(x))

        q = self.fc2(h_norm)
        return q.view(*orig_batch_dims, -1), h_out.view(*orig_batch_dims, -1)
