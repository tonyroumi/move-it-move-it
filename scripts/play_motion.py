""" Script to kinematically play back a single motion clip on the humanoid in Isaac Sim. """

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import torch

import moveitmoveit
from moveitmoveit.utils import transforms
from moveitmoveit.utils.paths import CONFIGS_DIR

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

parser = argparse.ArgumentParser(description="Play back a motion clip on the humanoid, kinematically.")
parser.add_argument(
    "--manifest",
    type=str,
    default=str(CONFIGS_DIR / "manifest.yaml"),
    help="Path to the motion manifest YAML listing the clips to choose from.",
)
parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Name of the motion (as declared in the manifest) to play. Omit to choose interactively.",
)
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument(
    "--random-reset",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply a fixed random yaw rotation about the z-axis to the clip, held for the whole"
    " playback, mirroring the rotation applied at reset during training.",
)
parser.add_argument(
    "--num-loops",
    type=int,
    default=0,
    help="Number of times to loop the clip before exiting. Defaults to 0 (loop forever).",
)
parser.add_argument(
    "--camera_type",
    type=str,
    default=None,
    choices=["facing", "third-person", "none"],
    help="Scripted camera mode. Defaults to the task's configured camera_type.",
)
parser.add_argument("--record", action="store_true", default=False, help="Record a video of the playback.")
parser.add_argument(
    "--record-length", type=int, default=200, help="Length of the recorded video (in steps)."
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.record:
    args_cli.enable_cameras = True


def _choose_motion(motion_names: list[str]) -> str:
    print("\n[INFO] Available motions:")
    for i, name in enumerate(motion_names):
        print(f"  [{i}] {name}")
    selection = input(f"Choose a motion [0-{len(motion_names) - 1}]: ").strip()
    if selection in motion_names:
        return selection
    if selection.isdigit() and 0 <= int(selection) < len(motion_names):
        return motion_names[int(selection)]
    raise ValueError(f"Invalid selection: '{selection}'")


def main():
    env_cfg, _ = resolve_task_config(TASK_ID, "agent_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        # a single env is enough to preview a clip
        env_cfg.scene.num_envs = 1
        env_cfg.motion_manifest = args_cli.manifest
        env_cfg.random_reset = args_cli.random_reset
        env_cfg.camera_type = args_cli.camera_type if args_cli.camera_type is not None else env_cfg.camera_type

        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
        unwrapped = env.unwrapped

        motions = unwrapped.motions
        motion_library = motions.library

        motion_name = args_cli.motion if args_cli.motion is not None else _choose_motion(motions.names)
        if motion_name not in motions.name_to_id:
            raise ValueError(f"Unknown motion '{motion_name}'. Available: {motions.names}")
        motion_id = motions.name_to_id[motion_name]
        duration = motion_library.duration[motion_id].item()
        print(f"[INFO] Playing motion '{motion_name}' (id={motion_id}, duration={duration:.2f}s)")

        env.reset()

        env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
        motion_ids = torch.full((unwrapped.num_envs,), motion_id, dtype=torch.long, device=unwrapped.device)
        physics_dt = unwrapped.sim.get_physics_dt()

        # sample a single random yaw up front and hold it for the whole playback (mirroring the
        # per-episode, not per-step, yaw randomization applied during training); `_compute_robot_state`
        # would otherwise draw a fresh yaw on every call, since it's normally only called once per
        # reset, so its own random-yaw handling is disabled here and applied manually below instead
        random_yaw = transforms.random_yaw_orientation(env_ids.shape[0], unwrapped.device) if args_cli.random_reset else None
        unwrapped.cfg.random_reset = False
        if random_yaw is not None:
            print(f"[INFO] Applying a fixed random yaw rotation about the z-axis: {random_yaw}")
            # `_compute_robot_state`/training only rotate orientation, velocity, and other bodies'
            # offsets *relative to the root* -- never the root's own absolute translation, since reward
            # tracking is translation-invariant. For a visible kinematic replay we additionally need to
            # rotate the root's translational path itself, pivoting about the clip's own start position,
            # so the character actually walks off in the direction it's now facing.
            zero_times = torch.zeros((unwrapped.num_envs,), device=unwrapped.device)
            clip_origin_positions = motion_library.frames(motion_ids, zero_times).body_positions
            clip_origin_position = clip_origin_positions[:, unwrapped.motion_ref_body_index]

        frames = []
        if args_cli.record:
            video_folder = os.path.join("logs", "videos", "play_motion")
            os.makedirs(video_folder, exist_ok=True)
            print(f"[INFO] Recording up to {args_cli.record_length} steps to: {video_folder}")

        clip_time = 0.0
        num_loops = 0
        try:
            while True:
                start_time = time.time()

                times = torch.full((unwrapped.num_envs,), clip_time, device=unwrapped.device)
                (
                    dof_positions,
                    dof_velocities,
                    body_positions,
                    body_rotations,
                    body_linear_velocities,
                    body_angular_velocities,
                ) = motion_library.frames(motion_ids, times)

                root_state, dof_pos, dof_vel = unwrapped._compute_robot_state(
                    env_ids,
                    dof_positions,
                    dof_velocities,
                    body_positions,
                    body_rotations,
                    body_linear_velocities,
                    body_angular_velocities,
                )
                if random_yaw is not None:
                    pivot = clip_origin_position + unwrapped.scene.env_origins[env_ids]
                    root_state[:, 0:3] = pivot + transforms.quat_apply(random_yaw, root_state[:, 0:3] - pivot)
                    root_state[:, 3:7] = transforms.quat_mul(random_yaw, root_state[:, 3:7])
                    root_state[:, 7:10] = transforms.quat_apply(random_yaw, root_state[:, 7:10])
                    root_state[:, 10:13] = transforms.quat_apply(random_yaw, root_state[:, 10:13])
                unwrapped._write_robot_state(env_ids, root_state, dof_pos, dof_vel)

                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=True)
                unwrapped.scene.update(dt=physics_dt)

                if unwrapped.render_enabled and env_cfg.camera_type != "none":
                    unwrapped._update_camera()

                if args_cli.record:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    # if len(frames) >= args_cli.record_length:
                        # break

                clip_time += physics_dt * args_cli.speed
                if clip_time >= duration:
                    clip_time = 0.0
                    num_loops += 1
                    if args_cli.num_loops and num_loops >= args_cli.num_loops:
                        break

                sleep_time = physics_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            env.close()

        if args_cli.record and frames:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            video_path = os.path.join(video_folder, f"{motion_name}.mp4")
            ImageSequenceClip(frames, fps=round(1.0 / physics_dt)).write_videofile(video_path)
            print(f"[INFO] Saved video to: {video_path}")


if __name__ == "__main__":
    main()
