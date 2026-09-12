""" Script to play a checkpoint of an RL agent. """

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import torch

import moveitmoveit
from moveitmoveit.utils.paths import resolve_checkpoint
from moveitmoveit.utils.logger import Logger

from isaaclab.utils.seed import configure_seed

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--step",
    type=int,
    default=None,
    help=(
        "Timestep of the checkpoint to play (checkpoints/{step}.pt), taken from the most recently "
        "written run directory. Ignored if --checkpoint is provided."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--joystick",
    action="store_true",
    default=False,
    help=(
        "Drive the policy's velocity command live from the keyboard instead of the env's sampled commands."
        " Requires a task configured with env.reward_type=joystick and a non-headless renderer."
    ),
)
parser.add_argument(
    "--joystick-overlay",
    action="store_true",
    default=False,
    help="With --joystick, overlay a faint keyboard on the renderer highlighting the currently held keys.",
)
parser.add_argument("--record", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument(
    "--record-length", type=int, default=200, help="Length of the recorded video (in steps)."
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.record:
    args_cli.enable_cameras = True


def main():
    """Play with skrl agent."""
    env_cfg, agent_cfg = resolve_task_config(TASK_ID, "agent_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        args_cli.seed = agent_cfg["seed"]

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", agent_cfg["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        resume_path = resolve_checkpoint(log_root_path, checkpoint=args_cli.checkpoint, step=args_cli.step)

        log_dir = os.path.dirname(os.path.dirname(resume_path))

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)

        keyboard = None
        keyboard_overlay = None
        if args_cli.joystick_overlay and not args_cli.joystick:
            raise ValueError("--joystick-overlay requires --joystick.")
        if args_cli.joystick:
            if args_cli.headless:
                raise ValueError("--joystick requires a renderer; do not combine it with --headless.")
            if not hasattr(env.unwrapped, "commands"):
                raise ValueError(
                    f"--joystick requires a command-conditioned task (got '{args_cli.task}', which has no"
                    " `commands` buffer)."
                )

            from moveitmoveit.utils.keyboard import Keyboard
            from isaaclab.devices import Se2KeyboardCfg

            keyboard = Keyboard(
                Se2KeyboardCfg(
                    sim_device=env.unwrapped.device,
                    v_x_sensitivity=env.unwrapped.cfg.command_lin_vel_range[1],
                    v_y_sensitivity=env.unwrapped.cfg.command_lat_vel_range[1],
                    omega_z_sensitivity=env.unwrapped.cfg.command_ang_vel_range[1],
                )
            )
            print(keyboard)

            if args_cli.joystick_overlay:
                from moveitmoveit.utils.keyboard import KeyboardOverlay

                keyboard_overlay = KeyboardOverlay()

        # get environment (step) dt for real-time evaluation
        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt

        # wrap for video recording
        if args_cli.record:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.record_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        from moveitmoveit.runners import OnPolicyRunner

        logger = Logger(backend="none", log_dir=log_dir)
        runner = OnPolicyRunner(
            cfg=agent_cfg,
            env=env,
            logger=logger,
        )

        configure_seed(env_cfg.seed, True)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load_checkpoint(resume_path, device=env.unwrapped.device)

        obs, _ = env.reset()
        timestep = 0
        try:
            while True:
                start_time = time.time()

                if keyboard is not None:
                    command = keyboard.advance().to(env.unwrapped.device)
                    env.unwrapped.commands[:] = command.unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                if keyboard_overlay is not None:
                    keyboard_overlay.update()

                with torch.inference_mode():
                    actions = runner.agent.act(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(actions)

                if args_cli.record:
                    timestep += 1
                    if timestep >= args_cli.record_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            if keyboard_overlay is not None:
                keyboard_overlay.close()


if __name__ == "__main__":
    main()
