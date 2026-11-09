""" Script to show a motion's `standing.pose` target on the humanoid in Isaac Sim, frozen. """

import argparse
import contextlib
import sys
import time

import torch

# lowest face of humanoid_28.usd's foot box below the foot body's origin (box center z -0.0225, half
# height 0.0275), so a flat foot rests on the ground with its origin this high above it
FOOT_SOLE_DEPTH = 0.05
FOOT_BODIES = ("right_foot", "left_foot")


def standing_pose(specs, targets: torch.Tensor, motion: str | None = None) -> tuple[int, torch.Tensor]:
    """The id and `standing.pose` row (robot joint order) of `motion`, or of the first motion
    declaring a pose. `targets` is ComposedMDP.standing_pose_targets, the table the reward reads."""
    with_pose = [spec.name for spec in specs if spec.standing is not None and spec.standing.pose]
    if motion is None:
        if not with_pose:
            raise ValueError("No motion in the manifest declares a standing.pose.")
        motion = with_pose[0]
    if motion not in with_pose:
        raise ValueError(f"Motion '{motion}' declares no standing.pose. Motions that do: {with_pose}")
    motion_id = [spec.name for spec in specs].index(motion)
    return motion_id, targets[motion_id]


def grounded_root_height(root_z: float, foot_z: list[float], ground_z: float = 0.0) -> float:
    """The root height that rests the lower foot's sole on the ground, given the feet's body heights
    with the root at `root_z`. Assumes the feet are flat."""
    return root_z - (min(foot_z) - FOOT_SOLE_DEPTH - ground_z)


def main(args_cli):
    import gymnasium as gym

    import moveitmoveit  # noqa: F401  (registers the task)

    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    task_id = "MoveIt-Humanoid-v0"
    env_cfg, _ = resolve_task_config(task_id, "agent_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        # a single env is enough to look at a pose
        env_cfg.scene.num_envs = 1
        env_cfg.random_reset = False
        env_cfg.camera_type = "none"
        env_cfg.motion_manifest = args_cli.manifest
        env = gym.make(task_id, cfg=env_cfg)
        unwrapped = env.unwrapped
        env.reset()

        robot = unwrapped.robot
        sim = unwrapped.sim
        physics_dt = sim.get_physics_dt()
        env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
        foot_ids = [robot.data.body_names.index(name) for name in FOOT_BODIES]

        motion_id, target = standing_pose(unwrapped.motions.specs, unwrapped.mdp.standing_pose_targets, args_cli.motion)
        print(f"[INFO] Standing pose of motion '{unwrapped.motions.specs[motion_id].name}' from {args_cli.manifest}:")
        for joint, value in zip(robot.data.joint_names, target.tolist()):
            print(f"  |-- {joint}: {value:+.3f}")

        dof_positions = target.unsqueeze(0).repeat(unwrapped.num_envs, 1)
        root_state = unwrapped.default_root_state[env_ids].clone()
        root_state[:, 0:3] += unwrapped.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0

        def place(root_state: torch.Tensor):
            unwrapped._write_robot_state(env_ids, root_state, dof_positions, torch.zeros_like(dof_positions))
            # forward kinematics only: moves the links (and what the viewer draws) without stepping physics
            sim.forward()
            unwrapped.scene.update(dt=physics_dt)

        # place once at the default height, then drop the root so the lower sole touches the ground
        place(root_state)
        foot_z = robot.data.body_pos_w.torch[0, foot_ids, 2].tolist()
        root_state[:, 2] = grounded_root_height(root_state[0, 2].item(), foot_z)
        place(root_state)

        root_pos = robot.data.root_pos_w.torch[0].tolist()
        foot_z = robot.data.body_pos_w.torch[0, foot_ids, 2].tolist()
        print(f"[INFO] Root at {[round(v, 3) for v in root_pos]}, foot origins at z={[round(v, 3) for v in foot_z]}")
        print("[INFO] Physics is frozen: the viewer stays live, but the simulation never steps. Ctrl+C to quit.")

        sim.set_camera_view(eye=(root_pos[0] + 3.0, root_pos[1] - 2.0, root_pos[2] + 0.5), target=tuple(root_pos))

        try:
            # render without stepping: no gravity, contacts or PD drives act, so the pose holds exactly
            while True:
                start_time = time.time()
                sim.render()
                sleep_time = physics_dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        env.close()


if __name__ == "__main__":
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli

    from moveitmoveit.utils.paths import CONFIGS_DIR

    parser = argparse.ArgumentParser(description="Show a motion's standing.pose target on the humanoid, frozen.")
    parser.add_argument(
        "--manifest", type=str, default=str(CONFIGS_DIR / "manifest.yaml"), help="Manifest declaring the standing.pose."
    )
    parser.add_argument(
        "--motion", type=str, default=None, help="Motion whose standing.pose to show. Defaults to the first that declares one."
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    main(args_cli)
