""" Script to train an RL agent. """

import argparse
import contextlib
import logging
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import yaml

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

import moveitmoveit
from moveitmoveit.utils.logger import Logger
from moveitmoveit.utils.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Train an agent to MOVE in IsaacLab")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--algo", type=str, required=True, choices=["PPO", "AMP"], help="RL algorithm.")
parser.add_argument("--manifest", type=str, required=True, help="Manifest file",)

add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Train an agent."""
    # env_cfg is still resolved through the hydra/task machinery so `env.scene.*`
    # style CLI overrides keep working, even though there's only one task now
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "agent_cfg_entry_point")

    # seed/experiment/logger/description are shared across algorithms, so they live in their
    # own config rather than being duplicated in each agents/<algo>.yaml
    with open(os.path.join(CONFIGS_DIR, "experiment.yaml"), "r") as f:
        experiment_cfg = yaml.safe_load(f)
    agent_cfg["seed"] = experiment_cfg["seed"]
    agent_cfg["experiment"] = experiment_cfg["experiment"]
    agent_cfg["logger"] = experiment_cfg["logger"]

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.motion_manifest = args_cli.manifest

        # specify directory for logging experiments: directory/experiment_name/algorithm/run
        log_root_path = os.path.join(
            "logs",
            agent_cfg["experiment"]["directory"],
            agent_cfg["experiment"]["experiment_name"],
            args_cli.algo,
        )
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name: {run_name}")

        agent_cfg["experiment"]["directory"] = log_root_path
        log_dir = os.path.join(log_root_path, run_name)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(args_cli.manifest, "r") as f:
            manifest_cfg = yaml.safe_load(f)
        dump_yaml(os.path.join(log_dir, "params", "manifest.yaml"), manifest_cfg)
        with open(os.path.join(log_dir, "description.txt"), "w") as f:
            f.write(experiment_cfg["description"])

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
        env_cfg.log_dir = log_dir

        env = gym.make(TASK_ID, cfg=env_cfg)

        start_time = time.time()

        logger = Logger(
            log_dir=log_dir,
            **agent_cfg["logger"],
            total_timesteps=agent_cfg["runner"]["timesteps"],
        )

        from moveitmoveit.runners import OnPolicyRunner

        runner = OnPolicyRunner(cfg=agent_cfg, env=env, logger=logger)

        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load_checkpoint(resume_path, env.unwrapped.device)

        try:
            runner.learn()

            print(f"Training time: {round(time.time() - start_time, 2)} seconds")

            timesteps = agent_cfg["runner"]["timesteps"]
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
            runner.agent.write_checkpoint(timesteps)
            print(f"[INFO] Saved final agent checkpoint to: {log_dir}/checkpoints")
            env.close()
        except KeyboardInterrupt:
            env.close()
            pass


if __name__ == "__main__":
    main()
