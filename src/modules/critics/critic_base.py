import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.popart import PopArt
from utils.mappo_util import init_module


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
    

class CriticBase(nn.Module):
    def __init__(self, args, input_shape, hidden_dim):
        super(CriticBase, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.args.use_popart:
            self.v_out = init_module(PopArt(hidden_dim, 1, device=args.device))
        else:
            self.v_out = init_module(nn.Linear(hidden_dim, 1))

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.v_out(x)
        return q