import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.popart import PopArt
from utils.mappo_util import init_module, init_rnn
from utils.mlp import MLPBase
from utils.encoder_decoder import build_encoder_inputs


class POAMCritic(nn.Module):
    def __init__(self, scheme, args):
        super(POAMCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_size = self._get_input_shape(scheme)
        if hasattr(self.args, "ed_hidden_dim"):
            self.hidden_dim = self.args.hidden_dim
            self.ed_hidden_dim = self.args.ed_hidden_dim
        else: 
            self.hidden_dim = self.ed_hidden_dim = self.args.hidden_dim  

        self.output_type = "v"

        self.encoder = None
        # Set up network layers
        self.base = MLPBase(self.input_size + args.embed_dim, 
                            self.hidden_dim, 
                            n_hidden_layers=1,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )
        if self.args.use_rnn: 
            self.rnn = init_rnn(nn.GRUCell(self.hidden_dim, self.hidden_dim), args.use_orthogonal_init)
            self.rnn_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.rnn = MLPBase(self.hidden_dim, self.hidden_dim, 
                            n_hidden_layers=0,
                            use_feature_norm=args.use_obs_norm, 
                            use_orthogonal=args.use_orthogonal_init
                            )

        if self.args.use_popart:
            self.v_out = init_module(PopArt(self.hidden_dim, 1, 
                                      norm_axes=3, device=args.device),
                                      gain=1.0)
        else: 
            self.v_out = init_module(nn.Linear(self.hidden_dim, 1, device=args.device),
                                     gain=1.0)
        
        self.value_normalizer = self.v_out if self.args.use_popart else None

    def init_hidden(self):
        # make hidden states on same device as model
        encoder_hidden = self.base.mlp.fc1[0].weight.new(self.args.batch_size, 1, self.n_agents, self.ed_hidden_dim).zero_()
        hidden_base = self.base.mlp.fc1[0].weight.new(self.args.batch_size, 1, self.n_agents, self.hidden_dim).zero_()
        return encoder_hidden, hidden_base 
        # return th.stack([encoder_hidden, hidden_base], dim=-1)

    def forward(self, batch, hidden_state, t=None, 
                build_inputs=True):
        # build inputs
        if build_inputs:
            inputs, bs, max_t = self._build_inputs(batch, t=t)
        else: 
            inputs = batch

        orig_batch_dims = inputs.shape[:-1]
        inputs = inputs.reshape(-1, self.input_size)
        
        # get hidden states
        h_e, h_in = hidden_state
        h_e = h_e.reshape(-1, self.ed_hidden_dim)
        h_in = h_in.reshape(-1, self.hidden_dim)
        
        # build embeddings
        enc_inputs = build_encoder_inputs(self.n_agents, batch, t)
        enc_input_size = enc_inputs.shape[-1]
        enc_inputs = enc_inputs.reshape(-1, enc_input_size)
        embeddings, h_e = self.encoder(enc_inputs, h_e)    

        # create 0s tensor that is same shape as embeddings
        # start forward pass   
        inputs = th.cat([embeddings.detach(), inputs], dim=1)
        x = self.base(inputs)

        if self.args.use_rnn:
            h_out = self.rnn(x, h_in)
            h_norm = self.rnn_norm(h_out)
        else:
            h_norm = h_out = self.rnn(x) # MlpBase applies relu automatically

        q = self.v_out(h_norm)
        # hidden_state = th.stack([h_e.view(*orig_batch_dims, -1),
        #                         h_out.view(*orig_batch_dims, -1)], 
        #                         dim=-1)
        h_e = h_e.view(*orig_batch_dims, -1)
        h_out = h_out.view(*orig_batch_dims, -1)
        return q.view(*orig_batch_dims, -1), h_e, h_out

    def _build_inputs(self, batch, t=None):
        '''if t=None, then returns inputs for all timesteps
        Otherwise, returns inputs for single timestep t
        '''
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        inputs.append(batch["obs"][:, ts])
        if self.args.obs_state:
            inputs.append(batch["state"][:, ts].unsqueeze(-2).expand(-1, -1, self.n_agents, -1))
        if self.args.obs_last_action:
            if t is None: # build last act for all timesteps
                last_act_all = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), 
                                       batch["actions_onehot"][:, slice(0, max_t-1)]],
                                       dim=1
                                       )
                inputs.append(last_act_all) 
            elif t == 0: # use existing slice
                inputs.append(th.zeros_like(batch["actions_onehot"][:, ts]))
            else: # use previous timestep slice
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_state:
            input_shape += scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents        
        return input_shape
