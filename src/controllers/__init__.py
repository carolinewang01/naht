REGISTRY = {}

from .basic_controller_old import BasicMACOld
from .basic_controller import BasicMAC
from .maddpg_controller import MADDPGMAC
from .agent_owned_controller import AgentOwnedMAC
from .open_eval_controller import OpenEvalMAC
from .open_train_controller import OpenTrainMAC

REGISTRY["basic_mac_old"] = BasicMACOld
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["agent_owned_mac"] = AgentOwnedMAC
REGISTRY["open_eval_mac"] = OpenEvalMAC
REGISTRY["open_train_mac"] = OpenTrainMAC