import os
import json
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.load_utils import find_model_path
from os.path import dirname, abspath
from os import makedirs

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Create the local results directory
    if args.local_results_path == "":
        args.local_results_path = dirname(dirname(abspath(__file__)))
    makedirs(args.local_results_path, exist_ok=True)

    # setup loggers
    logger = Logger(_log)

    if args.eval_mode == "open":
        # force sacred to dump to log every 5 seconds
        _run.beat_interval = 5

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.local_results_path, "tb_logs", args.expt_logname)
        logger.setup_tb(tb_log_dir)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    start_time = time.time()
    while True:
        runner.run(test_mode=True)
        # when num_test_episodes have run, the test_stats buffer will be cleared
        if len(runner.test_stats) == 0:
            break

    if args.eval_mode == "open":
        sacred_log_path = os.path.join(runner.logger._run_obj.observers[0].dir, "info.json")
        while not os.path.exists(sacred_log_path):
            time.sleep(1)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()
    end_time = time.time()
    print("Evaluation took {} seconds".format(end_time - start_time))

def run_sequential(args, logger):
    print("ENV IS : ", args.env)
    args.open_train_or_eval = True if "open" in args.mac else False
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "actor_hidden_states": {"vshape": (args.hidden_dim,), 
                                "group": "agents"},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8}
    }
    if args.open_train_or_eval: # track a mask relating to whether agents are trainable or not
        scheme["trainable_agents"] = {"vshape": (1,), "group": "agents", "dtype": th.bool}
    
    if "liam" in args.name or "poam" in args.name and not args.open_train_or_eval:
        scheme['actor_hidden_states']['vshape'] = (args.hidden_dim, 2) 
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    mac.cuda()

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    assert args.eval_mode in ["default", "open", None]
    if args.eval_mode != "open":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

        if args.use_cuda:
            learner.cuda()

    if args.checkpoint_path != "":
        model_path, timestep_to_load = find_model_path(args.checkpoint_path, args.load_step, logger=logger)
        logger.console_logger.info(f"Loading model from ts {timestep_to_load}, {model_path}")
        learner.load_models(model_path)
        # runner.t_env = timestep_to_load

    if args.eval_mode in ["default", "open"] or args.save_replay:
        runner.log_train_stats_t = runner.t_env
        evaluate_sequential(args, runner)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    best_test_return = -1000000

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch, _ = runner.run(test_mode=False) # batch_size_run eps collected
        buffer.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size): # when batch_size eps collected
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode) 

            if args.on_policy:
                buffer.clear()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size_run)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                _, mean_test_return = runner.run(test_mode=True)

            # save best checkpoint
            assert mean_test_return is not None
            if mean_test_return > best_test_return:
                best_test_return = mean_test_return
                save_path = os.path.join(args.local_results_path, "models", args.expt_logname, "best")
                os.makedirs(save_path, exist_ok=True)
                # make json file with best_test_return
                with open(os.path.join(save_path, "best_info.json"), 'w') as f:
                    json.dump({"best_test_return": best_test_return, "best_ts": str(runner.t_env)}, f)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
        
        # save at regular intervals 
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.expt_logname, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
