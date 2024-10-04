import torch.nn as nn
from .mappo_util import init_, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init(m):
            return init_(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(
            init(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_hidden_layers,
                 use_feature_norm=True, use_orthogonal=True, 
                 gain=0.01):
        super(MLPBase, self).__init__()
        self._use_feature_norm = use_feature_norm
        self._use_orthogonal = use_orthogonal
        self._layer_N = n_hidden_layers
        self.hidden_size = hidden_dim

        if self._use_feature_norm:
            self.feature_norm = nn.LayerNorm(input_shape)

        self.mlp = MLPLayer(input_shape, self.hidden_size,
                            self._layer_N, self._use_orthogonal, 
                            use_ReLU=True)

    def forward(self, x):
        if self._use_feature_norm:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x