""" Script to train an RL agent. """

import argparse
import contextlib
import logging
import os
import sys
import time
from datetime import datetime

import gymnasium as gym

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

logger = logging.getLogger(__name__)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Train an agent to MOVE with IsaacLab")
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

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", agent_cfg["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # algo/reward folded into the run name since --task no longer encodes it
        log_dir += f"_{args_cli.algo}_{os.path.basename(args_cli.manifest).split('.')[0]}"
        print(f"Exact experiment name: {log_dir}")

        if agent_cfg["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['experiment']['experiment_name']}"
        agent_cfg["experiment"]["directory"] = log_root_path
        agent_cfg["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_yaml(os.path.join(log_dir, "params", "manifest.yaml"),args_cli.manifest)

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
        env_cfg.log_dir = log_dir

        env = gym.make(TASK_ID, cfg=env_cfg)

        start_time = time.time()

        run_logger = Logger(
            log_dir=log_dir,
            **agent_cfg["logger"],
            total_timesteps=agent_cfg["runner"]["timesteps"],
        )

        from moveitmoveit.runners import OnPolicyRunner

        runner = OnPolicyRunner(cfg=agent_cfg, env=env, logger=run_logger)

        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load_checkpoint(resume_path)

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
