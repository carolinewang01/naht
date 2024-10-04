from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size_run = self.args.batch_size_run
        assert self.batch_size_run == 1
        self.open_train_or_eval = self.args.open_train_or_eval

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.log_discounted_return = self.args.log_discounted_return

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size_run, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        if self.open_train_or_eval:
            self.trained_agent_idxs = self.mac.sample_agent_team()
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    @th.no_grad()
    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        actor_hidden_states = self.mac.init_hidden(batch_size=self.batch_size_run)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "actor_hidden_states": actor_hidden_states
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, actor_hidden_states = self.mac.select_actions(self.batch, 
                                                                   t_ep=self.t, t_env=self.t_env, 
                                                                   test_mode=test_mode)
            
            reward, terminated, env_info = self.env.step(actions.squeeze().cpu().numpy())
            if test_mode and self.args.render:
                self.env.render()

            if self.log_discounted_return:
                episode_return += reward * (self.args.gamma ** self.t)
            else:
                episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "actor_hidden_states": actor_hidden_states
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, actor_hidden_states = self.mac.select_actions(self.batch, 
                                                            #    hidden_states=actor_hidden_states, 
                                                               t_ep=self.t, t_env=self.t_env, 
                                                               test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else f"train_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if test_mode and self.args.test_verbose:
            print("Number of episodes collected: ", cur_stats["n_episodes"])
            print(f"Return so far:  {round(np.mean(cur_returns), 3)} +/- {round(np.std(cur_returns), 3)}" )

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        
        mean_test_return = None
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            mean_test_return = np.mean(self.test_returns)
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # filter batch data for data corresponding to trained agents only
        if self.open_train_or_eval:
            trainable_mask = self.compute_open_agent_mask(self.batch, self.trained_agent_idxs)
            self.batch.update({"trainable_agents": trainable_mask})

        return self.batch, mean_test_return

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def compute_open_agent_mask(self, batch, agent_idx_list):
        '''compute mask corresponding to trained agents only'''
        agent_mask = np.zeros((batch.batch_size, batch.max_seq_length, self.args.n_agents))
        # set entires of agent_mask coresponding to trained agents to 1
        agent_mask[:, :, agent_idx_list] = 1.0
        return agent_mask
