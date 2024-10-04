#!/scratch/cluster/clw4542/conda_envs/oil/bin/python
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src/"))

import numpy as np
import yaml
from envs import REGISTRY as env_REGISTRY

def main():
    # read in env args 
    env_args = yaml.safe_load(open("src/config/envs/sc2.yaml", "r"))['env_args']
    env_args = {
        'map_name': '3s_vs_5z',
        **env_args

    }

    # init env
    env = env_REGISTRY["sc2"](**env_args)
    env_info = env.get_env_info()
    print("env info: ", env_info)

    # Reset the environment
    obs = env.reset()

    # Run an episode with random actions
    done = False
    while not done:
        avail_actions = env.get_avail_actions()
        # sample action from avail actions
        action = [np.random.choice(np.nonzero(avail)[0]) for avail in avail_actions]
        reward, done, _ = env.step(action)
        obs = env.get_obs()
        print("Obs min: ", np.min(obs), "Obs max: ", np.max(obs))

    print("EPISODE FINISHED SUCCESSFULLY")
    # Close the environment
    env.close()

if __name__ == "__main__":
    
    main()