import torch.nn as nn
import torch.nn.functional as F
import torch as th
from utils.mappo_util import init_rnn, init_module, get_clones
from utils.mlp import MLPBase


class Encoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.args = args
        if hasattr(self.args, "ed_hidden_dim"):
            self.hidden_dim = self.args.ed_hidden_dim
        else: 
            self.hidden_dim = self.args.hidden_dim  
        if hasattr(self.args, "n_ed_hidden_layers"):
            self._layer_N = self.args.n_ed_hidden_layers
        else: 
            self._layer_N = 1

        # create network layers
        self.rnn = init_rnn(nn.GRUCell(input_dim, hidden_dim), args.use_orthogonal_init)
        self.fc_h = init_module(nn.Linear(hidden_dim, hidden_dim))
        self.fc_list = get_clones(self.fc_h, self._layer_N)
        self.embedding = init_module(nn.Linear(hidden_dim, output_dim))

    def init_hidden(self, batch_size): 
        return self.embedding.weight.new(batch_size, 1, self.args.n_agents, self.hidden_dim).zero_()
    
    def forward(self, x, hidden):
        h_out = self.rnn(x, hidden)
        for i in range(self._layer_N):
            h_out = F.relu(self.fc_list[i](h_out))
        embedding = self.embedding(h_out)
        return embedding, h_out

    def forward_all(self, x):
        '''forward for all timesteps
        assumption: x has shape (bs, max_t, n_agents, input_dim)
        '''
        bs, max_t, n_agents, input_dim = x.shape
        
        hidden = self.init_hidden(bs)
        embeddings = []
        for t in range(max_t):
            x_t = x[:, t].reshape(-1, input_dim)
            hidden = hidden.view(-1, self.hidden_dim)
            embedding, hidden = self.forward(x_t, hidden)
            embeddings.append(embedding.view(bs, n_agents, -1))
        embeddings = th.stack(embeddings, dim=1)
        return embeddings

class Decoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, 
                 output_dim1, output_dim2):
        super(Decoder, self).__init__()
        self.args = args
        if hasattr(self.args, "n_ed_hidden_layers"):
            self._layer_N = self.args.n_ed_hidden_layers
        else: 
            self._layer_N = 1
        self.fc_h = init_module(nn.Linear(hidden_dim, hidden_dim))
        
        # define network structure
        self.fc1 = init_module(nn.Linear(input_dim, hidden_dim))
        self.fc1_list = get_clones(self.fc_h, self._layer_N)
        self.out1 = init_module(nn.Linear(hidden_dim, output_dim1))

        self.fc2 = init_module(nn.Linear(input_dim, hidden_dim))
        self.fc2_list = get_clones(self.fc_h, self._layer_N)
        self.out2 = init_module(nn.Linear(hidden_dim, output_dim2))

        self.ed_bce_loss = args.ed_bce_loss

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        for i in range(self._layer_N):
            h1 = F.relu(self.fc1_list[i](h1))
        pred1 = self.out1(h1)
        if self.ed_bce_loss:
            pred1 = th.sigmoid(pred1)

        h2 = F.relu(self.fc2(x))
        for i in range(self._layer_N):
            h2 = F.relu(self.fc2_list[i](h2))
        pred2 = self.out2(h2)

        return pred1, pred2
    
def build_encoder_inputs(n_agents, batch, t, 
                         concat_obs_act=True
                         ):
    '''build inputs for encoder'''
    max_t = batch.max_seq_length if t is None else 1
    ts = slice(None) if t is None else slice(t, t+1)
    
    # all keys in batch follow shape (bs, ts, n_agents, input_shape)
    obs_inputs_all = batch["obs"][:, ts]
    if t is None:
        last_act_inputs_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                             batch["actions_onehot"][:, slice(0, max_t-1)]],
                                             dim=1
                                            )
    else:
        last_act_inputs_all = th.zeros_like(batch["actions_onehot"][:, ts]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t)]

    # inputs_all have shape (bs, ts, n_agents, feat_shape)
    if concat_obs_act:
        inputs_all = th.cat([obs_inputs_all, last_act_inputs_all], dim=-1)
    else:
        inputs_all = (obs_inputs_all, last_act_inputs_all)
    return inputs_all

def build_decoder_targets(n_agents, 
                          batch, mask, t, 
                          concat_agents=False,
                          concat_obs_act=True
                         ):
    '''build targets for decoder.
    implement parameter sharing by reshaping the target to 
    (bs, ts, n_agents, n_agents - 1, feat_shape)
    '''
    bs = batch.batch_size
    max_t = batch.max_seq_length if t is None else 1
    ts = slice(None) if t is None else slice(t, t+1)
    
    # all keys in batch follow shape (bs, ts, n_agents, input_shape)
    obs_inputs_all, act_inputs_all, mask_inputs_all = [], [], []

    for agent_idx in range(n_agents):
        # get obs of OTHER agents 
        obs_input = th.cat([batch["obs"][:, ts, slice(0, agent_idx)],
                            batch["obs"][:, ts, slice(agent_idx+1, n_agents)]
                            ], dim=2)
        obs_inputs_all.append(obs_input)
        
        # get action of OTHER agents 
        act_input = th.cat([batch["actions_onehot"][:, ts, slice(0, agent_idx)],
                            batch["actions_onehot"][:, ts, slice(agent_idx+1, n_agents)]
                            ], dim=2)
        act_inputs_all.append(act_input)

        # reshape mask similarly to obs and act
        mask_input = th.cat([mask[:, ts, slice(0, agent_idx)],
                       mask[:, ts, slice(agent_idx+1, n_agents)]
                       ], dim=2)
        mask_inputs_all.append(mask_input)

    # inputs_all have shape (bs, ts, n_agents, n_agents - 1, feat_shape)
    obs_inputs_all = th.stack(obs_inputs_all, dim=2) 
    act_inputs_all = th.stack(act_inputs_all, dim=2)
    mask_inputs_all = th.stack(mask_inputs_all, dim=2).unsqueeze(-1)

    # combine last two dimensions
    if concat_agents:
        batch_dims = act_inputs_all.shape[:-2]
        # only needed if using obs of OTHER agents
        obs_inputs_all = obs_inputs_all.view(*batch_dims, -1)
        act_inputs_all = act_inputs_all.view(*batch_dims, -1)
        mask_inputs_all = mask_inputs_all.view(*batch_dims, -1)

    if concat_obs_act:
        out = th.cat([obs_inputs_all, act_inputs_all], dim=-1), mask_inputs_all
    else:
        out = (obs_inputs_all, act_inputs_all, mask_inputs_all)
    return out

def build_decoder_inputs(embeddings):
    '''add agent ids to encoer embeddings and repeat n-1 times
    embeddings: (bs, ts, n_agents, feat_shape)
    desired output shape: (bs, ts, n_agents, n_agents - 1, feat_shape + n_agents)
    '''
    bs, max_t, n_agents, _ = embeddings.shape
    embeddings = embeddings.unsqueeze(3).repeat(1, 1, 1, n_agents - 1, 1)
    
    # create onehot agent id for all agents except current
    agent_id_onehot = th.eye(n_agents, device=embeddings.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, n_agents, n_agents)
    other_agent_ids = []
    for agent_idx in range(n_agents):
        all_but_agent_idx = th.cat([agent_id_onehot[:, :, slice(0, agent_idx)],
                                    agent_id_onehot[:, :, slice(agent_idx+1, n_agents)]
                                    ], dim=2)    
        other_agent_ids.append(all_but_agent_idx)        

    # other_agent_ids has shape (bs, ts, n_agents, n_agents - 1, 1)
    other_agent_ids = th.stack(other_agent_ids, dim=2) 
    inputs_all = th.cat([embeddings, other_agent_ids], dim=-1)
    return inputs_all

def get_encoder_input_shape(scheme):
    # observations
    obs_size = scheme["obs"]["vshape"]
    act_size = scheme["actions_onehot"]["vshape"][0]
    return obs_size + act_size

def get_decoder_input_shape(n_agents, embed_dim):
    return embed_dim + n_agents

def get_decoder_target_shape(n_agents, scheme):
    # observations
    obs_size = scheme["obs"]["vshape"] # * (n_agents - 1)
    act_size = scheme["actions_onehot"]["vshape"][0] # * (n_agents - 1)
    return obs_size, act_size