from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.array import ArrayLike

class SkeletonVisualizer:
    """ Skeleton and motion data visualization utilities """
    def __init__(self, save_path: str):
        self.save_path = save_path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def _set_axes_equal(cls, ax):
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

    @classmethod
    def visualize_skeleton(cls, global_position: ArrayLike, height: float, foot_idx: int, head_idx: int, save_path: str):
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
        
        # Calculate the expected head position based on height
        expected_head_pos = foot_pos + direction_normalized * height
        
        # Draw a line from foot to expected head position based on height
        ax.plot([foot_pos[0], expected_head_pos[0]], 
                [foot_pos[1], expected_head_pos[1]], 
                [foot_pos[2], expected_head_pos[2]], 
                'b-', linewidth=2, label=f'Height: {height:.3f}')
        
        ax.legend()

        # Make axes equal
        cls._set_axes_equal(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    
   
    @classmethod
    def visualize_motion(cls, global_positions: ArrayLike, save_path: str):
        # Reshape from [num_windows, 64, num_joints, 3] to [T, num_joints, 3]
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
            30,
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

            cls._set_axes_equal(ax)

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
    
