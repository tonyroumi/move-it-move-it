"""Script to play a checkpoint of an RL agent."""

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import torch

from moveitmoveit import agent_cfg_entry_point
from moveitmoveit.motion.manifest import CommandIndex
from moveitmoveit.utils.logger import Logger

from isaaclab.utils.dict import update_class_from_dict
from isaaclab.utils.io import load_yaml
from isaaclab.utils.seed import configure_seed

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Play a checkpoint of an agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--joystick",
    type=str,
    default="none",
    choices=["none", "velocity", "point_goal"],
    help=(
        "Drive the policy's command live from the keyboard instead of the env's sampled commands"
    ),
)
parser.add_argument(
    "--camera",
    type=str,
    default=None,
    choices=["facing", "third-person", "torso-follow", "none"],
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
    # the agent config is the one saved with the checkpoint, read below; this one is discarded
    env_cfg, _ = resolve_task_config(TASK_ID, agent_cfg_entry_point("PPO"))

    with launch_simulation(env_cfg, args_cli):
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

        params_dir = os.path.join(log_dir, "params")

        print(f"[INFO] Loading configs from run directory: {log_dir}")

        update_class_from_dict(env_cfg, load_yaml(os.path.join(params_dir, "env.yaml")))
        agent_cfg = load_yaml(os.path.join(params_dir, "agent.yaml"))

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.camera_type = args_cli.camera if args_cli.camera is not None else env_cfg.camera_type
        env_cfg.random_reset = False

        env_cfg.resample_commands = args_cli.joystick == "none"
        args_cli.seed = agent_cfg["seed"]

        # create isaac environment
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)

        keyboard = None
        goal_marker = None
        if args_cli.joystick != "none":
            if args_cli.joystick == "point_goal" and not env.unwrapped.motions.has_goal.any():
                raise ValueError(
                    "--joystick-mode point_goal needs a checkpoint trained on a manifest motion that"
                    " declares goal_x/goal_y commands; this one has none, so its policy never sees a goal."
                )

            from moveitmoveit.utils.keyboard import Keyboard

            keyboard = Keyboard(args_cli.joystick, dt=env.unwrapped.step_dt, device=env.unwrapped.device)
            print(keyboard)

            if args_cli.joystick == "point_goal":
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

                goal_marker = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/PointGoal",
                        markers={
                            "goal": sim_utils.SphereCfg(
                                radius=0.1,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                            ),
                        },
                    )
                )

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
                    root_index = env.unwrapped.ref_body_index
                    command = keyboard.advance(
                        env.unwrapped.robot.data.body_pos_w.torch[:, root_index, :2],
                        env.unwrapped.robot.data.body_quat_w.torch[:, root_index],
                    )
                    env.unwrapped.commands[:] = command
                    if goal_marker is not None:
                        goal_xy = command[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]
                        goal_marker.visualize(translations=torch.nn.functional.pad(goal_xy, (0, 1), value=0.1))

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
            pass


if __name__ == "__main__":
    main()
