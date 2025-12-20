from src.skeletal_models import SkeletonTopology
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import math

class SkeletonVisualizer:
    """ Skeleton and motion data visualization utilities """
    @staticmethod
    def _set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max(x_range, y_range, z_range)

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def visualize_skeleton(global_position: ArrayLike, save_path: str):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        global_position = SkeletonUtils.prune_joints(global_position, cutoff=22)

        xs = global_position[:, 0]
        ys = global_position[:, 1]
        zs = global_position[:, 2]

        ax.scatter(xs, ys, zs, s=40, c="black")

        # Label each joint by its index
        for idx, (x, y, z) in enumerate(global_position):
            ax.text(x, y, z, str(idx), color="red", fontsize=8)

        # Make axes equal
        SkeletonVisualizer._set_axes_equal(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    
    @staticmethod
    def visualize_motion(global_positions: ArrayLike, save_path: str, fps: int = 30):
        T = global_positions.shape[0]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        frame_iter = tqdm(range(T), desc=f"Rendering frames", leave=False, position=1)
        for t in frame_iter:
            ax.cla()

            frame_pos = global_positions[t]  # [J, 3]
            xs = frame_pos[:, 0]
            ys = frame_pos[:, 1]
            zs = frame_pos[:, 2]

            ax.scatter(xs, ys, zs, s=40, c="black")

            for idx, (x, y, z) in enumerate(frame_pos):
                ax.text(x, y, z, str(idx), color="red", fontsize=8)

            SkeletonVisualizer._set_axes_equal(ax)

            # Render the frame on Agg canvas
            fig.canvas.draw()

            # Extract raw RGBA buffer
            buf = np.asarray(fig.canvas.buffer_rgba())

            # Convert RGBA → RGB
            rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)

            # Convert RGB → BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            out.write(bgr)

        out.release()
        plt.close(fig)
