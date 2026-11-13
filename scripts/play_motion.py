"""Script to play back a single motion clip in IsaacLab."""

import argparse
import contextlib
import os
import sys
import tempfile
import time

import gymnasium as gym
import torch
import yaml

from moveitmoveit import agent_cfg_entry_point

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

TASK_ID = "MoveIt-Humanoid-v0"

parser = argparse.ArgumentParser(description="Play back a motion clip on the humanoid.")
parser.add_argument("--motion", type=str, required=True, help="Path to the motion clip (.npz) to play.")
parser.add_argument(
    "--camera",
    type=str,
    default=None,
    choices=["facing", "third-person", "none"],
    help="camera mode. Defaults to the task's configured camera_type.",
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


def main():
    # only the env config is used; no agent runs here
    env_cfg, _ = resolve_task_config(TASK_ID, agent_cfg_entry_point("PPO"))

    motion_file = os.path.abspath(args_cli.motion)
    motion_name = os.path.splitext(os.path.basename(motion_file))[0]

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.random_reset = False
        env_cfg.camera_type = args_cli.camera if args_cli.camera is not None else env_cfg.camera_type

        # the env loads its clips from a manifest file, so describe the one clip in a throwaway manifest
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as manifest:
            yaml.safe_dump({"motions": {motion_name: {"file": motion_file}}}, manifest)
            manifest.flush()
            env_cfg.motion_manifest = manifest.name
            env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
        unwrapped = env.unwrapped

        reference = unwrapped.reference
        duration = unwrapped.motions.library.duration[0].item()
        print(f"[INFO] Playing motion '{motion_name}' (duration={duration:.2f}s)")

        env.reset()

        env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
        motion_ids = torch.zeros((unwrapped.num_envs,), dtype=torch.long, device=unwrapped.device)
        physics_dt = unwrapped.sim.get_physics_dt()

        video_frames = []
        if args_cli.record:
            video_folder = os.path.join("logs", "videos", "play_motion")
            os.makedirs(video_folder, exist_ok=True)
            print(f"[INFO] Recording up to {args_cli.record_length} steps to: {video_folder}")

        clip_time = 0.0
        try:
            while True:
                start_time = time.time()

                times = torch.full((unwrapped.num_envs,), clip_time, device=unwrapped.device)
                frames = reference.frames_for_reset(motion_ids, times)

                root_state, dof_pos, dof_vel = unwrapped._compute_robot_state(env_ids, frames)
                unwrapped._write_robot_state(env_ids, root_state, dof_pos, dof_vel)

                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=True)
                unwrapped.scene.update(dt=physics_dt)

                if unwrapped.render_enabled and env_cfg.camera_type != "none":
                    unwrapped._update_camera()

                if args_cli.record:
                    frame = env.render()
                    if frame is not None:
                        video_frames.append(frame)
                    if len(video_frames) >= args_cli.record_length:
                        break

                clip_time += physics_dt
                if clip_time >= duration:
                    clip_time = 0.0

                sleep_time = physics_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            env.close()

        if args_cli.record and video_frames:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            video_path = os.path.join(video_folder, f"{motion_name}.mp4")
            ImageSequenceClip(video_frames, fps=round(1.0 / physics_dt)).write_videofile(video_path)
            print(f"[INFO] Saved video to: {video_path}")


if __name__ == "__main__":
    main()
