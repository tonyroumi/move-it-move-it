"""
Rollout motion clips through MuJoCo to extract kinematic state.

For each .npz clip in the input directory, sub-steps between consecutive
keyframes using mj_integratePos and saves the resulting kinematic quantities
at each sub-step.

Between keyframes i and i+1 (separated by frame_dt = 1/fps):
  - velocity is computed once via mj_differentiatePos
  - mj_integratePos propagates qpos forward by sub_dt each step
  - each intermediate state is recorded

Output length: (N - 1) * num_substeps + 1  where
  num_substeps = round(frame_dt / sub_dt)

Saved per-clip (float32 npz):
  root_pos      (M, 3)          world-frame root position
  root_rot      (M, 4)          root quaternion (w, x, y, z)
  root_vel      (M, 3)          root linear velocity
  root_ang_vel  (M, 3)          root angular velocity
  dof_pos       (M, nq)         full qpos
  body_pos      (M, nbody, 3)   world-frame body positions (xpos)
  joint_pos     (M, njoint, 3)  world-frame joint anchor positions (xanchor)
  dof_vel       (M, nv)         full qvel

Usage:
    # Keep original fps (one output frame per input frame)
    python rollout_motions.py data/humanoid --output data/humanoid_rollout

    # 4x upsampling for 30fps source → 120fps output
    python rollout_motions.py data/humanoid --output data/humanoid_rollout --timestep 0.00833
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


def find_xml(directory: Path) -> Path:
    xmls = sorted(directory.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No .xml file found in {directory}")
    if len(xmls) > 1:
        print(f"[warn] multiple XML files found, using {xmls[0].name}")
    return xmls[0]


def _record(
    data: mujoco.MjData,
    out: dict,
    idx: int,
) -> None:
    out["root_pos"][idx] = data.qpos[:3]
    out["root_rot"][idx] = data.qpos[3:7]
    out["root_vel"][idx] = data.qvel[:3]
    out["root_ang_vel"][idx] = data.qvel[3:6]
    out["dof_pos"][idx] = data.qpos[7:]
    out["body_pos"][idx] = data.xpos
    out["joint_pos"][idx] = data.xanchor
    out["dof_vel"][idx] = data.qvel[6:]


def rollout_clip(
    frames: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_dt: float,
    sub_dt: float,
) -> dict:
    """
    Kinematically replay frames with sub-step interpolation.

    Between each pair of keyframes the inter-frame velocity is held constant
    and mj_integratePos steps qpos forward by sub_dt, giving smooth
    intermediate states that respect the quaternion manifold.
    """
    N = frames.shape[0]
    num_substeps = max(1, round(frame_dt / sub_dt))
    actual_sub_dt = frame_dt / num_substeps
    total = (N - 1) * num_substeps + 1

    nv = model.nv
    nbody = model.nbody
    njoint = model.njnt

    out = dict(
        root_pos=np.zeros((total, 3), dtype=np.float32),
        root_rot=np.zeros((total, 4), dtype=np.float32),
        root_vel=np.zeros((total, 3), dtype=np.float32),
        root_ang_vel=np.zeros((total, 3), dtype=np.float32),
        dof_pos=np.zeros((total, model.nq-7), dtype=np.float32),
        body_pos=np.zeros((total, nbody, 3), dtype=np.float32),
        joint_pos=np.zeros((total, njoint, 3), dtype=np.float32),
        dof_vel=np.zeros((total, nv-6), dtype=np.float32),
    )

    qvel = np.zeros(nv)
    q_sub = np.zeros(model.nq)
    out_idx = 0

    for i in range(N - 1):
        # Constant velocity for this inter-frame interval
        mujoco.mj_differentiatePos(model, qvel, frame_dt, frames[i], frames[i + 1])

        for k in range(num_substeps):
            # Integrate from keyframe i by k * sub_dt
            q_sub[:] = frames[i]
            if k > 0:
                mujoco.mj_integratePos(model, q_sub, qvel, k * actual_sub_dt)

            data.qpos[:] = q_sub
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            _record(data, out, out_idx)
            out_idx += 1

    # Final keyframe — hold last velocity
    data.qpos[:] = frames[-1]
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    _record(data, out, out_idx)

    return out, num_substeps, actual_sub_dt


def process_directory(
    src_dir: Path,
    out_dir: Path,
    timestep: float | None,
) -> None:
    xml_path = find_xml(src_dir)
    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    clips = sorted(src_dir.glob("*.npz"))
    if not clips:
        print(f"No .npz files found in {src_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for clip_path in clips:
        d = np.load(str(clip_path), allow_pickle=False)
        name = str(d["name"])
        fps = float(d["fps"])
        frames = d["frames"].astype(np.float64)

        frame_dt = 1.0 / fps
        sub_dt = timestep if timestep is not None else frame_dt

        if frames.shape[1] != model.nq:
            print(f"[skip] {clip_path.name}: frame width {frames.shape[1]} != nq {model.nq}")
            continue

        data = mujoco.MjData(model)
        state, num_substeps, actual_sub_dt = rollout_clip(frames, model, data, frame_dt, sub_dt)
        out_frames = state["root_pos"].shape[0]
        out_fps = 1.0 / actual_sub_dt

        out_path = out_dir / clip_path.name
        np.savez_compressed(
            str(out_path),
            name=np.array(name),
            fps=np.float32(out_fps),
            dt=np.float32(actual_sub_dt),
            **state,
        )
        print(
            f"  {clip_path.name}: {frames.shape[0]} keyframes × {num_substeps} substeps"
            f" = {out_frames} frames @ {out_fps:.1f}fps → {out_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "src_dir",
        type=Path,
        help="Directory containing .npz clips and one .xml model",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: <src_dir>_rollout)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Sub-step dt in seconds (default: 1/fps — one output frame per keyframe)",
    )
    args = parser.parse_args()

    src_dir: Path = args.src_dir.resolve()
    if not src_dir.is_dir():
        print(f"Error: {src_dir} is not a directory")
        sys.exit(1)

    out_dir: Path = (
        args.output.resolve() if args.output
        else src_dir.parent / (src_dir.name + "_rollout")
    )

    process_directory(src_dir, out_dir, args.timestep)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
