# code adapted from https://github.com/wendelinboehmer/dc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.mlp import MLPBase
from utils.mappo_util import init_rnn, init_module
from utils.encoder_decoder import build_encoder_inputs


class RNNPOAMAgent(nn.Module):
    '''Identical to RNNNormAgent except possesses an encoder.
    '''
    def __init__(self, input_shape, args):
        super(RNNPOAMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_size = input_shape
        self.encoder = None
        # a hack to decouple POAM hidden size from loaded eval policy hidden sizes
        # checks for ed_hidden_dim; if none, defaults to hidden_dim
        if hasattr(self.args, "ed_hidden_dim"):
            self.hidden_dim = self.args.hidden_dim
            self.ed_hidden_dim = self.args.ed_hidden_dim
        else: 
            self.hidden_dim = self.ed_hidden_dim = self.args.hidden_dim  

        self.base = MLPBase(input_shape + args.embed_dim, 
                            self.hidden_dim, 
                            n_hidden_layers=1,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )
                           
        if self.args.use_rnn:
            self.rnn = init_rnn(nn.GRUCell(self.hidden_dim, self.hidden_dim), args.use_orthogonal_init)
            self.rnn_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.rnn = init_module(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc2 = init_module(nn.Linear(self.hidden_dim, args.n_actions))


    def init_hidden(self, batch_size):
        encoder_hidden = self.base.mlp.fc1[0].weight.new(batch_size, 1, self.n_agents, self.ed_hidden_dim).zero_()
        hidden_base = self.base.mlp.fc1[0].weight.new(batch_size, 1, self.n_agents, self.hidden_dim).zero_()
        return encoder_hidden, hidden_base 
        # return th.stack([encoder_hidden, hidden_base], dim=-1)

    def get_out_shapes(self, curr_t, batch_size, max_t ):
        if curr_t is None:
            ts = slice(None)
            out_batch_dim = (batch_size, max_t, self.n_agents)
            hidden_batch_dim = (batch_size, max_t, self.n_agents, self.hidden_dim)
            ed_hidden_batch_dim = (batch_size, max_t, self.n_agents, self.ed_hidden_dim)
        else: 
            ts = slice(curr_t, curr_t + 1)
            out_batch_dim = (batch_size, 1, self.n_agents)
            hidden_batch_dim = (batch_size, 1, self.n_agents, self.hidden_dim)
            ed_hidden_batch_dim = (batch_size, 1, self.n_agents, self.ed_hidden_dim)
        return ts, out_batch_dim, hidden_batch_dim, ed_hidden_batch_dim

    def forward(self, ep_batch, t, hidden_state=None):
        '''hidden_state arg used for open eval only'''
        # define out shapes
        ts, out_batch_dim, hidden_batch_dim, ed_hidden_batch_dim = self.get_out_shapes(curr_t=t, 
                                            batch_size=ep_batch.batch_size, 
                                            max_t=ep_batch.max_seq_length)
        # build inputs
        inputs = self._build_inputs(ep_batch, t)
        inputs = inputs.reshape(-1, self.input_size)

        # get hidden states from buffer
        if hidden_state is None:
            hidden_state = ep_batch["actor_hidden_states"][:, ts]
            h_e, h_in = hidden_state[:, :, :, :, 0], hidden_state[:, :, :, :, 1]
        else: # hack that applies only to the open eval runner
            h_e, h_in = hidden_state
        h_in = h_in.reshape(-1, self.hidden_dim)
        h_e = h_e.reshape(-1, self.ed_hidden_dim)
        # build embeddings
        enc_inputs = build_encoder_inputs(self.n_agents, ep_batch, t)
        enc_input_size = enc_inputs.shape[-1] # enc inputs has shape (bs, ts, n_agents, feat_size)
        enc_inputs = enc_inputs.reshape(-1, enc_input_size)

        embeddings, h_e = self.encoder(enc_inputs, h_e)

        # start forward pass
        inputs = th.cat([embeddings.detach(), inputs], dim=1)    
        x = F.relu(self.base(inputs))

        if self.args.use_rnn:
            h_out = self.rnn(x, h_in)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = F.relu(self.rnn(x))
        
        q = self.fc2(h_norm)
        h_e = h_e.view(*ed_hidden_batch_dim)
        h_out = h_out.view(*hidden_batch_dim)
        return q.view(*out_batch_dim, -1), h_e, h_out

    def _build_inputs(self, batch, t=None):
        '''
        Assumes homogenous agents with flat observations.
        Other MACs might want to e.g. delegate building inputs to each agent
        If t=None, then returns inputs for all timesteps.
        '''
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts])  # b1av
        if self.args.obs_last_action:
            if t is None: # build last act for all timesteps
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                       batch["actions_onehot"][:, slice(0, max_t-1)]],
                                       dim=1
                                       )
                inputs.append(last_act_all) 
            else: # get last act for single timestep only 
                last_act = th.zeros_like(batch["actions_onehot"][:, ts]) if t == 0 else batch["actions_onehot"][:, slice(t-1, t)]
                inputs.append(last_act)

        if self.args.obs_agent_id:
            agent_id_onehot = th.eye(self.n_agents, device=batch.device)
            agent_id_onehot = agent_id_onehot.expand(bs, max_t, self.n_agents, -1) if t is None \
                else agent_id_onehot.expand(bs, 1, self.n_agents, -1)
            inputs.append(agent_id_onehot)
        inputs = th.cat(inputs, dim=-1) # shape (bs, ts, n_agents, input_shape)
        return inputs

    def load(self, path):
        state_dict_all = th.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict_all)
