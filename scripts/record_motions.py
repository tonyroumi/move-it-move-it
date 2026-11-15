"""Record one facing-camera video of every motion clip (.npz) under the data directory.

Each clip runs through ``scripts/play_motion.py --camera facing --record`` in its own process, for
just enough steps to play the clip once to its end without wrapping back to its start. Videos are
named by the clip's path under the data directory, so ``boxing/jab.npz`` and ``dance/jab.npz`` do
not overwrite each other. Run it with the same Python as play_motion (``./isaaclab.sh -p``).
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAY_MOTION = REPO_ROOT / "scripts" / "play_motion.py"

SIM_DT = 1 / 60
"""play_motion's step, the env's ``sim.dt`` in ``src/moveitmoveit/env_cfg.py``: one recorded frame
per step, while the clip clock advances by this much."""

PLAY_MOTION_VIDEO_DIR = Path("logs") / "videos" / "play_motion"
"""Where play_motion writes ``<clip stem>.mp4``, relative to its working directory."""


def find_clips(data_dir: Path) -> list[Path]:
    """Every .npz file under `data_dir`, recursively, in sorted order."""
    return sorted(path for path in Path(data_dir).rglob("*.npz") if path.is_file())


def clip_duration(clip: Path) -> float:
    """The clip's duration in seconds, as MotionLibrary computes it: (frames - 1) / fps."""
    with np.load(clip) as data:
        num_frames = data["dof_positions"].shape[0]
        fps = float(data["fps"])
    return (num_frames - 1) / fps


def record_length(duration: float, sim_dt: float = SIM_DT) -> int:
    """Steps that play a clip of `duration` once without wrapping.

    play_motion records a frame at clip times 0, dt, 2*dt, ... and wraps the clock to 0 once it
    reaches `duration`, so one playthrough is every step strictly before `duration`. The tolerance
    keeps a duration that is a whole number of steps from gaining a step to float error.
    """
    return max(1, math.ceil(duration / sim_dt - 1e-6))


def video_path(clip: Path, data_dir: Path, output_dir: Path) -> Path:
    """The clip's video: its path under `data_dir`, with .mp4 in place of .npz, under `output_dir`."""
    return Path(output_dir) / Path(clip).relative_to(data_dir).with_suffix(".mp4")


def record_clip(clip: Path, length: int, video: Path, play_motion_args: list[str]) -> None:
    """Record `clip` for `length` steps with play_motion and move its video to `video`."""
    staged = REPO_ROOT / PLAY_MOTION_VIDEO_DIR / f"{clip.stem}.mp4"
    # a video left by an earlier run must not pass for this one
    staged.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(PLAY_MOTION),
        "--motion",
        str(clip),
        "--camera",
        "facing",
        "--record",
        "--record-length",
        str(length),
        *play_motion_args,
    ]
    returncode = subprocess.run(command, cwd=REPO_ROOT).returncode
    if returncode != 0 or not staged.is_file():
        found = "found" if staged.is_file() else "missing"
        raise RuntimeError(f"play_motion exited with {returncode}; video {found} at {staged}")
    video.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, video)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a facing-camera video of every motion clip under the data directory.",
        epilog="Any other arguments are passed through to play_motion, e.g. --headless.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Directory searched for .npz clips. Defaults to the data directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "videos" / "record_motions",
        help="Directory the videos are written to, mirroring the clips' paths under the data directory.",
    )
    parser.add_argument("--force", action="store_true", help="Rerecord clips that already have a video.")
    args, play_motion_args = parser.parse_known_args()

    if args.data_dir is None:
        from moveitmoveit.utils.paths import MOVEITMOVEIT_DATA_DIR

        args.data_dir = MOVEITMOVEIT_DATA_DIR
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    clips = find_clips(data_dir)
    if not clips:
        sys.exit(f"No .npz clips found under {data_dir}")
    print(f"[INFO] Found {len(clips)} clips under {data_dir}")

    failed = []
    for index, clip in enumerate(clips, start=1):
        name = clip.relative_to(data_dir)
        video = video_path(clip, data_dir, output_dir)
        if video.is_file() and not args.force:
            print(f"[INFO] [{index}/{len(clips)}] Skipping {name}: video exists at {video}")
            continue
        duration = clip_duration(clip)
        length = record_length(duration)
        print(f"[INFO] [{index}/{len(clips)}] Recording {name} ({duration:.2f}s, {length} steps)")
        try:
            record_clip(clip, length, video, play_motion_args)
        except RuntimeError as error:
            print(f"[ERROR] [{index}/{len(clips)}] Failed {name}: {error}")
            failed.append(name)
            continue
        print(f"[INFO] [{index}/{len(clips)}] Wrote {name} to {video}")

    if failed:
        sys.exit(f"Failed to record {len(failed)} of {len(clips)} clips: {', '.join(map(str, failed))}")


if __name__ == "__main__":
    main()
