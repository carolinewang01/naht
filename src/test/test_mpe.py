
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src/"))
# print(os.getcwd())

# from collections import OrderedDict
from modules.agent_loaders.rnn_eval_agent_loader import RNNEvalAgentLoader
from envs import GymmaWrapper


# # create team wrapper 
# class TeamWrapper(RecordVideo):
#     def __init__(self, model_paths: list, 
#                  load_steps: list, load_agent_idxs: list, 
#                  args, scheme):
#         # team is an OrderedDict
#         team = OrderedDict()
#         for i, model_path in enumerate(model_paths):
#             team[i] = RNNEvalAgentLoader(args, scheme, 
#                                          model_path=model_path, 
#                                          load_step=load_steps[i], 
#                                          load_agent_idx=load_agent_idxs[i], 
#                                          test_mode=True)
    
#     def step(self, joint_state, t_ep
#              ):
#         # TODO: create ep_batch from joint_state
#         ep_batch = None
#         joint_act = []
#         for agent_idx, agent in self.team.items():
#             # TODO: figure out how to handle hidden states
#             _, act, hidden_state = agent.predict(
#                 ep_batch, agent_idx, 
#                 t_ep=t_ep, t_env=None, bs=slice(None), 
#                 )
#             joint_act.append(act)
#         return th.stack(joint_act, dim=1)
    
if __name__ == '__main__':
    
    # Create the environment using the gymma wrapper
    env = GymmaWrapper(
        key='mpe:SimpleTag-v0',
        # key='mpe:SimpleSpread-v0',
        # key='mpe:PredatorPrey-v0',
        time_limit=150,
        seed=1111, 
        pretrained_wrapper='PretrainedTag',
        record_video=False
    )

    # Reset the environment
    obs = env.reset()
    done = False
    ts = 0
    while not done:
        # Choose a random action
        actions = env._env.action_space.sample()

        # Take a step in the environment
        reward, done, info = env.step(actions)
        env.render()
        ts +=1
        print("Time step:", ts)
        # print('Observation:', env._obs)
        # print('Reward:', reward)
        # print('Done:', done)
        # print('Info:', info)
        # print('\n')

    # Close the environment
    env.close()