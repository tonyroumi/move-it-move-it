from .kinematics import ForwardKinematics

from numpy.typing import ArrayLike
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

class SkeletonVisualizer:
    """Skeleton and motion data visualization utilities"""
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
    def visualize_skeleton(
      global_position: ArrayLike, 
      height: float,
      foot_idx: int, 
      head_idx: int, 
      save_path: str
    ):
        """ Visualize a single skeleton frame with height measurement. """        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        xs = global_position[:, 0]
        ys = global_position[:, 1]
        zs = global_position[:, 2]

        ax.scatter(xs, ys, zs, s=40, c="black")

        # Label each joint by its index
        for idx, (x, y, z) in enumerate(global_position):
            ax.text(x, y, z, str(idx), color="red", fontsize=8)

        # Get foot and head positions
        foot_pos = global_position[foot_idx]
        head_pos = global_position[head_idx]
        
        # Calculate the direction from foot to head
        direction = head_pos - foot_pos
        direction_normalized = direction / np.linalg.norm(direction)
        
        expected_head_pos = foot_pos + direction_normalized * height
        
        ax.plot([foot_pos[0], expected_head_pos[0]], 
                [foot_pos[1], expected_head_pos[1]], 
                [foot_pos[2], expected_head_pos[2]], 
                'b-', linewidth=2, label=f'Height: {height:.3f}')
        
        ax.legend()

        # Make axes equal
        SkeletonVisualizer._set_axes_equal(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def visualize_motion(global_positions: ArrayLike, save_path: str, fps: int = 30):
        """ Visualize motion sequence as a video. """     
        if global_positions.ndim == 4:
            num_windows, window_size, num_joints, _ = global_positions.shape
            # Stitch all windows together along the time dimension
            global_positions = global_positions.reshape(num_windows * window_size, num_joints, 3)
        
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

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            rgb = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            out.write(bgr)

        out.release()
        plt.close(fig)

    @staticmethod
    def visualize_dataset(cross_motion_dataloader: DataLoader, save_path: str):
        (Path(save_path) / "domain_a").mkdir(parents=True, exist_ok=True)
        (Path(save_path) / "domain_b").mkdir(parents=True, exist_ok=True)

        motion_datasets = cross_motion_dataloader.dataset.domains
        for i, batch in enumerate(cross_motion_dataloader):
            B, _, T = batch.motions[0].shape

            positions_A = ForwardKinematics.forward_batched(
                batch.rotations[0][:, :-3]
                  .reshape(B, -1, 4, T)
                  .permute(0, 3, 1, 2), 
                batch.offsets[0].reshape(B, -1, 3),
                batch.rotations[0][:, -3:],
                motion_datasets[0].topology
            )
            positions_B = ForwardKinematics.forward_batched(
                batch.rotations[1][:, :-3]
                  .reshape(B, -1, 4, T)
                  .permute(0, 3, 1, 2), 
                batch.offsets[1].reshape(B, -1, 3),
                batch.rotations[1][:, -3:],
                motion_datasets[1].topology
            )
            
            print("Visualizing gt positions...")
            SkeletonVisualizer.visualize_motion(batch.gt_positions[0], f"{save_path}/domain_a/gt_position_batch_{i}.mp4")
            SkeletonVisualizer.visualize_motion(batch.gt_positions[1], f"{save_path}/domain_b/gt_position_batch_{i}.mp4")

            print("Visualizing positions from fk...")
            SkeletonVisualizer.visualize_motion(positions_A, f"{save_path}/domain_a/rotations_batch_{i}.mp4")
            SkeletonVisualizer.visualize_motion(positions_B, f"{save_path}/domain_b/rotations_batch_{i}.mp4")