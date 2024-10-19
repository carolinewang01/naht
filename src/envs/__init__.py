from functools import partial
import sys
import os
import numpy as np

from smac.env import MultiAgentEnv, StarCraft2Env
from .gymma import GymmaWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )
REGISTRY["gymma"] = partial(env_fn, env=GymmaWrapper)
