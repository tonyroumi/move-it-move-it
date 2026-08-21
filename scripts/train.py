""" Script to train an RL agent. """

import argparse
import contextlib
import logging
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch

import moveitmoveit

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

# import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an agent to MOVE with IsaacLab")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO"],
    help="The RL algorithm used for training the agent to MOVE.",
)
# append AppLauncher cli args
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Train an agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "agent_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env_cfg.seed = agent_cfg["seed"]

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", agent_cfg["agent"]["experiment"]["directory"], args_cli.algorithm)
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {log_dir}")

        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # get checkpoint path (to resume training)
        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        start_time = time.time()

        from moveitmoveit.runners import OnPolicyRunner
        from moveitmoveit.utils.logger import Logger
        logger = Logger(log_dir)
        runner = OnPolicyRunner(
            cfg=agent_cfg["agent"],
            env=env,
            logger=logger,
        )

        # // can configure runner here.
        configure_seed(env_cfg.seed, True)

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        # run training
        try:
            runner.learn()
            
            # runner.run()
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")

            # save a final checkpoint
            total_timesteps = agent_cfg["trainer"]["timesteps"]
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
            # runner.agent.write_checkpoint(timestep=total_timesteps, timesteps=total_timesteps)
            print(f"[INFO] Saved final agent checkpoint to: {log_dir}/checkpoints")
            # close the simulator
            env.close()
        except KeyboardInterrupt:
            env.close()
            pass


if __name__ == "__main__":
    main()
