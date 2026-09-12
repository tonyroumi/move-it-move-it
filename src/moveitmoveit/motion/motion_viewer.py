# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import torch

try:
    from .motion_loader import MotionLoader
except ImportError:
    from motion_loader import MotionLoader


class MotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    """

    def __init__(
        self,
        motion_file: str,
        device: torch.device | str = "cpu",
        render_scene: bool = False,
        root_body: str | None = None,
    ) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.
            render_scene: Whether the scene (space occupied by the skeleton during movement)
                is rendered instead of a reduced view of the skeleton.
            root_body: Body used to compute the nominal locomotion velocity.
                If None, the first body in the motion data is used.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene

        # Load motions
        self._motion_loader = MotionLoader(
            motion_files=[motion_file],
            device=device,
        )

        self._num_frames = int(self._motion_loader.num_frames[0])
        self._current_frame = 0
        self._body_positions = (
            self._motion_loader.body_positions[0, : self._num_frames]
            .cpu()
            .numpy()
        )

        # Determine which body to use as the root.
        if root_body is None:
            self._root_body_index = 0
            self._root_body_name = self._motion_loader.body_names[0]
        else:
            if root_body not in self._motion_loader.body_names:
                raise ValueError(
                    f"Root body '{root_body}' not found. "
                    f"Available bodies: {self._motion_loader.body_names}"
                )

            self._root_body_index = self._motion_loader.body_names.index(root_body)
            self._root_body_name = root_body

        print("\nBody")
        for i, name in enumerate(self._motion_loader.body_names):
            minimum = np.min(
                self._body_positions[:, i],
                axis=0,
            ).round(decimals=2)

            maximum = np.max(
                self._body_positions[:, i],
                axis=0,
            ).round(decimals=2)

            print(
                f"  |-- [{name}] minimum position: {minimum}, "
                f"maximum position: {maximum}"
            )

        # Compute motion statistics.
        self._print_velocity_statistics()

    def _compute_nominal_velocity(self) -> tuple[np.ndarray, float]:
        """Compute the nominal linear velocity of the motion clip.

        The nominal velocity is defined as the net displacement of the root
        body divided by the duration of the clip.

        Returns:
            nominal_velocity: Average XYZ velocity in m/s.
            nominal_horizontal_speed: Magnitude of the XY velocity in m/s.
        """
        root_positions = self._body_positions[:, self._root_body_index]

        # N frames contain N - 1 time intervals.
        duration = (self._num_frames - 1) * self._motion_loader.dt

        if duration <= 0.0:
            raise ValueError("Motion clip must contain at least two frames.")

        displacement = root_positions[-1] - root_positions[0]

        nominal_velocity = displacement / duration

        nominal_horizontal_speed = np.linalg.norm(
            nominal_velocity[:2]
        )

        return nominal_velocity, nominal_horizontal_speed

    def _compute_frame_velocities(self) -> np.ndarray:
        """Compute root linear velocity between consecutive frames."""
        root_positions = self._body_positions[:, self._root_body_index]

        frame_velocities = np.diff(
            root_positions,
            axis=0,
        ) / self._motion_loader.dt

        return frame_velocities

    def _print_velocity_statistics(self) -> None:
        """Print nominal and frame-wise velocity statistics."""
        nominal_velocity, nominal_horizontal_speed = (
            self._compute_nominal_velocity()
        )

        frame_velocities = self._compute_frame_velocities()

        horizontal_speeds = np.linalg.norm(
            frame_velocities[:, :2],
            axis=1,
        )

        duration = (self._num_frames - 1) * self._motion_loader.dt

        print("\nMotion")
        print(f"  |-- root body: {self._root_body_name}")
        print(f"  |-- frames: {self._num_frames}")
        print(f"  |-- dt: {self._motion_loader.dt:.6f} s")
        print(f"  |-- duration: {duration:.3f} s")

        print("\nNominal velocity")
        print(
            "  |-- linear velocity: "
            f"[{nominal_velocity[0]:.3f}, "
            f"{nominal_velocity[1]:.3f}, "
            f"{nominal_velocity[2]:.3f}] m/s"
        )
        print(
            f"  |-- horizontal speed: "
            f"{nominal_horizontal_speed:.3f} m/s"
        )

        print("\nFrame-wise velocity")
        print(
            "  |-- mean linear velocity: "
            f"{np.mean(frame_velocities, axis=0).round(3)} m/s"
        )
        print(
            "  |-- std linear velocity: "
            f"{np.std(frame_velocities, axis=0).round(3)} m/s"
        )
        print(
            f"  |-- mean horizontal speed: "
            f"{np.mean(horizontal_speeds):.3f} m/s"
        )
        print(
            f"  |-- std horizontal speed: "
            f"{np.std(horizontal_speeds):.3f} m/s"
        )

    def _drawing_callback(self, frame: int) -> None:
        """Drawing callback called each frame."""
        vertices = self._body_positions[self._current_frame]

        # Draw skeleton state
        self._figure_axes.clear()
        self._figure_axes.scatter(
            *vertices.T,
            color="black",
            depthshade=False,
        )

        # Adjust axes according to motion view
        if self._render_scene:
            minimum = np.min(
                self._body_positions.reshape(-1, 3),
                axis=0,
            )
            maximum = np.max(
                self._body_positions.reshape(-1, 3),
                axis=0,
            )
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)

        else:
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array(
                [0.75 * np.max(maximum - minimum).item()] * 3
            )

        # Scale view
        self._figure_axes.set_xlim(
            (center[0] - diff[0], center[0] + diff[0])
        )
        self._figure_axes.set_ylim(
            (center[1] - diff[1], center[1] + diff[1])
        )
        self._figure_axes.set_zlim(
            (center[2] - diff[2], center[2] + diff[2])
        )
        self._figure_axes.set_box_aspect(
            aspect=diff / diff[0]
        )

        # Plot ground plane
        x, y = np.meshgrid(
            [center[0] - diff[0], center[0] + diff[0]],
            [center[1] - diff[1], center[1] + diff[1]],
        )
        self._figure_axes.plot_surface(
            x,
            y,
            np.zeros_like(x),
            color="green",
            alpha=0.2,
        )

        # Print metadata
        self._figure_axes.set_xlabel("X")
        self._figure_axes.set_ylabel("Y")
        self._figure_axes.set_zlabel("Z")
        self._figure_axes.set_title(
            f"frame: {self._current_frame}/{self._num_frames}"
        )

        # Increase frame counter
        self._current_frame += 1

        if self._current_frame >= self._num_frames:
            self._current_frame = 0

    def show(self) -> None:
        """Show motion."""
        self._figure = plt.figure()
        self._figure_axes = self._figure.add_subplot(
            projection="3d"
        )

        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._motion_loader.dt,
        )

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Motion file",
    )

    parser.add_argument(
        "--root-body",
        type=str,
        default=None,
        help=(
            "Body used to compute nominal locomotion velocity. "
            "Defaults to the first body."
        ),
    )

    parser.add_argument(
        "--render-scene",
        action="store_true",
        default=False,
        help=(
            "Whether the scene (space occupied by the skeleton during movement) "
            "is rendered instead of a reduced view of the skeleton."
        ),
    )

    parser.add_argument(
        "--matplotlib-backend",
        type=str,
        default="TkAgg",
        help="Matplotlib interactive backend",
    )

    args, _ = parser.parse_known_args()

    matplotlib.use(args.matplotlib_backend)

    viewer = MotionViewer(
        args.file,
        render_scene=args.render_scene,
        root_body=args.root_body,
    )

    viewer.show()