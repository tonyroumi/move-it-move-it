""" Script to play a checkpoint of an RL agent. """

import argparse
import contextlib
import os
import random
import sys
import time

import gymnasium as gym
import torch

import moveitmoveit

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO"],
    help="The RL algorithm used for training the agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Play with skrl agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "agent_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


        args_cli.seed = agent_cfg["seed"]

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", agent_cfg["experiment"]["directory"], args_cli.algorithm)
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            raise ValueError("Please specify a checkpoint path to play.")

        log_dir = os.path.dirname(os.path.dirname(resume_path))

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # get environment (step) dt for real-time evaluation
        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt

        from moveitmoveit.runners import OnPolicyRunner
        from moveitmoveit.utils.logger import Logger

        logger = Logger(backend="none", log_dir=log_dir)
        runner = OnPolicyRunner(
            cfg=agent_cfg,
            env=env,
            logger=logger,
        )

        configure_seed(env_cfg.seed, True)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load_checkpoint(resume_path, device=env.unwrapped.device)

        # reset environment
        obs, _ = env.reset()
        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()

                with torch.inference_mode():
                    actions = runner.agent.act(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(actions)

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
