import torch.nn as nn


class RNNFeatureAgent(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

    def init_hidden(self, batch_size):
        return self.fc1.weight.new(batch_size, 1, self.n_agents, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = nn.functional.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.args.hidden_dim))
        return None, h