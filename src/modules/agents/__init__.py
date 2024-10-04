REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_norm_agent import RNNNormAgent
from .rnn_norm_ns_agent import RNNNormNSAgent
from .rnn_poam_agent import RNNPOAMAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["rnn_norm"] = RNNNormAgent
REGISTRY["rnn_norm_ns"] = RNNNormNSAgent
REGISTRY["rnn_poam"] = RNNPOAMAgent
