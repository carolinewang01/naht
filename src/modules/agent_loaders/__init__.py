REGISTRY = {}

from .rnn_eval_agent_loader import RNNEvalAgentLoader
from .rnn_train_agent_loader import RNNTrainAgentLoader
from .poam_eval_agent_loader import POAMEvalAgentLoader
from .poam_train_agent_loader import POAMTrainAgentLoader


REGISTRY["rnn_eval_agent_loader"] = RNNEvalAgentLoader
REGISTRY["rnn_train_agent_loader"] = RNNTrainAgentLoader
REGISTRY["poam_eval_agent_loader"] = POAMEvalAgentLoader
REGISTRY["poam_train_agent_loader"] = POAMTrainAgentLoader
