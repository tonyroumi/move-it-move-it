"""Script to play a checkpoint of an RL agent."""

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import torch

import moveitmoveit
from moveitmoveit.manifest import CommandIndex
from moveitmoveit.utils import transforms
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

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Play a checkpoint of an agent.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help=(
        "Path to a model checkpoint written by train.py (<run>/checkpoints/<name>.pt). The env and agent"
        " configs are loaded from that run's params/{env,agent}.yaml."
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
parser.add_argument(
    "--joystick-mode",
    type=str,
    default="velocity",
    choices=["velocity", "point_goal"],
    help=(
        "With --joystick, which command the keyboard drives: 'velocity' for the SE(2) lin/yaw"
        " command (original behavior; already in the robot's body frame, so up is always forward),"
        " or 'point_goal' for a world-space goal point that starts 1 m ahead of each env's start"
        " position along its start heading, stays put while the robot walks, and is moved with the"
        " same up/down/left/right keys relative to the robot's current heading (up pushes it the way"
        " the robot faces). Needs a checkpoint whose manifest declares goal_x/goal_y."
    ),
)
parser.add_argument(
    "--camera_type",
    type=str,
    default=None,
    choices=["facing", "third-person", "none"],
    help=(
        "Scripted camera mode: 'facing' follows in front of the robot looking back at it,"
        " 'third-person' follows behind and above it, 'none' disables scripted camera updates."
        " Defaults to the task's configured camera_type."
    ),
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


class HeadingRelativeGoal:
    """The world-space point goal of `--joystick-mode point_goal`, moved in the robot's heading frame.

    `PointGoalKeyboard` integrates the arrow keys into an XY offset; each step's change in that
    offset is rotated by the robot's current yaw before it moves the goal, so up always pushes the
    goal ahead of where the robot faces. The goal stays fixed in world space while no key is held.
    """

    def __init__(self, anchor_xy: torch.Tensor, anchor_rotations: torch.Tensor):
        self._anchor_xy = anchor_xy.clone()
        self._anchor_rotations = anchor_rotations.clone()
        self._goal_xy: torch.Tensor | None = None
        self._offset_xy: torch.Tensor | None = None

    def reset(self):
        """Back to the keyboard's offset from the start position, in the start heading (L key)."""
        self._goal_xy = None

    def update(self, offset_xy: torch.Tensor, root_rotations: torch.Tensor) -> torch.Tensor:
        """(N, 2) world-space goal for the keyboard's (N, 2) offset and the (N, 4) root rotations."""
        if self._goal_xy is None:
            self._goal_xy = self._anchor_xy + self._heading_to_world_xy(self._anchor_rotations, offset_xy)
        else:
            step_xy = offset_xy - self._offset_xy
            self._goal_xy = self._goal_xy + self._heading_to_world_xy(root_rotations, step_xy)
        self._offset_xy = offset_xy.clone()
        return self._goal_xy

    @staticmethod
    def _heading_to_world_xy(rotations: torch.Tensor, vectors_xy: torch.Tensor) -> torch.Tensor:
        """Rotate (N, 2) heading-frame XY vectors into world space by each rotation's yaw only."""
        vectors = torch.cat((vectors_xy, torch.zeros_like(vectors_xy[:, :1])), dim=-1)
        return transforms.quat_apply_yaw(rotations, vectors)[:, :2]


def main():
    """Play a trained agent checkpoint."""
    env_cfg, _ = resolve_task_config(TASK_ID, "agent_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        # train.py writes checkpoints to <run>/checkpoints/, next to the configs it ran with in <run>/params/
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

        # load the env/agent configs as they were dumped by train.py
        params_dir = os.path.join(log_dir, "params")

        print(f"[INFO] Loading configs from run directory: {log_dir}")

        update_class_from_dict(env_cfg, load_yaml(os.path.join(params_dir, "env.yaml")))
        agent_cfg = load_yaml(os.path.join(params_dir, "agent.yaml"))

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.camera_type = args_cli.camera_type if args_cli.camera_type is not None else env_cfg.camera_type
        env_cfg.random_reset = False
        # the keyboard owns the commands; the env's own resampling would overwrite them
        env_cfg.resample_commands = not args_cli.joystick
        args_cli.seed = agent_cfg["seed"]

        # create isaac environment
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)

        keyboard = None
        keyboard_overlay = None
        goal_marker = None
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

            from isaaclab.devices import Se2KeyboardCfg

            se2_keyboard_cfg = Se2KeyboardCfg(
                sim_device=env.unwrapped.device,
                v_x_sensitivity=1.5,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=1.0,
            )

            if args_cli.joystick_mode == "point_goal":
                if not env.unwrapped.motions.has_goal.any():
                    raise ValueError(
                        "--joystick-mode point_goal needs a checkpoint trained on a manifest motion that"
                        " declares goal_x/goal_y commands; this one has none, so its policy never sees a goal."
                    )

                from moveitmoveit.utils.keyboard import PointGoalKeyboard

                keyboard = PointGoalKeyboard(se2_keyboard_cfg, dt=env.unwrapped.step_dt)

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
            else:
                from moveitmoveit.utils.keyboard import Keyboard

                keyboard = Keyboard(se2_keyboard_cfg)
            print(keyboard)

            if args_cli.joystick_overlay:
                from moveitmoveit.utils.keyboard_overlay import KeyboardOverlay

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
        # the keyboard's goal offset is taken from each env's start position and heading, once, so the
        # goal is a fixed point in world space rather than one that moves along with the robot
        point_goal = None
        if args_cli.joystick and args_cli.joystick_mode == "point_goal":
            point_goal = HeadingRelativeGoal(
                env.unwrapped.robot.data.body_pos_w.torch[:, env.unwrapped.ref_body_index, :2],
                env.unwrapped.robot.data.body_quat_w.torch[:, env.unwrapped.ref_body_index],
            )
            keyboard.add_callback("L", point_goal.reset)
        timestep = 0
        try:
            while True:
                start_time = time.time()

                if keyboard is not None:
                    command = keyboard.advance().to(env.unwrapped.device)
                    command = command.unsqueeze(0).expand(env.unwrapped.num_envs, -1).clone()
                    if point_goal is not None:
                        goal_positions_xy = point_goal.update(
                            command[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1],
                            env.unwrapped.robot.data.body_quat_w.torch[:, env.unwrapped.ref_body_index],
                        )
                        command[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1] = goal_positions_xy

                        goal_height = torch.full(
                            (goal_positions_xy.shape[0], 1), 0.1, device=goal_positions_xy.device
                        )
                        goal_marker.visualize(translations=torch.cat((goal_positions_xy, goal_height), dim=-1))
                    env.unwrapped.commands[:] = command
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
