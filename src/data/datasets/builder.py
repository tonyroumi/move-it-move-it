"""
MotionDataset builder to construct MotionDataset for a single motion domain.
"""

from typing import Any, Dict, Tuple
import numpy as np
import torch

from src.data.adapters.base import DataSourceAdapter
from src.data.metadata import SkeletonMetadata, MotionSequence
from src.skeletal.utils import SkeletonUtils
from src.utils import ArrayUtils, Logger

class MotionDatasetBuilder:
    def __init__(
        self, 
        adapter: DataSourceAdapter, 
        data_config: Dict[str, Any],
        logger: Logger = Logger()
    ):
        self.adapter = adapter
        self.data_config = data_config
        self.logger = logger

        self.window_size = data_config['window_size']
        self.downsample_fps = data_config['downsample_fps']

        self.device = self.adapter.device

    def get_or_process(self, character: str):
        """ 
        Load or generate all skeleton metadata and motion data for a given character. 
        
        Raises:
            FileNotFoundError: If no raw files exist for provided character.
        """ 
        cache_path = (
            self.adapter.cache_dir / character
        )
        processed_files = list(cache_path.glob("*"))

        raw_path = (
            self.adapter.raw_dir / character
        )
        raw_files = list(raw_path.glob("*"))

        if not raw_files:
            raise FileNotFoundError(f"No raw files for {character}")

        skel_path = (
            self.adapter.skeleton_dir / f"{character}.npz"
        )
        # If skeleton doesn't exist, extract it. 
        if not skel_path.exists():
            skeleton = self.adapter.extract_skeleton(character)
        else:
            skeleton = SkeletonMetadata.load(skel_path)
        
        offsets = skeleton.offsets
        edge_topology = skeleton.edge_topology
        ee_ids = skeleton.ee_ids
        height = skeleton.height

        # If there are no processed files, proccess from raw dir. 
        if not processed_files:
            motions = self.adapter.extract_motion(character)
        else:
            motions = []
            for processed_motion in processed_files:
                motions.append(MotionSequence.load(processed_motion))

        windowed_motion, windowed_pos = zip(
            *[self._process_motion_sequence(m) for m in motions]
        )
        all_windowed_motion, all_windowed_pos = np.vstack(windowed_motion), np.vstack(windowed_pos)
        ee_vels = SkeletonUtils.get_ee_velocity(all_windowed_pos, skeleton) 
        #we might want to reshape after this. right now the shape is [num_w, 63, 5, 3]

        num_frames = all_windowed_motion.shape[0] * all_windowed_motion.shape[1]
        self.logger.info(f"{character} contains {num_frames} frames")

        return (ArrayUtils.to_torch(all_windowed_motion, self.device),
                ArrayUtils.to_torch(all_windowed_pos, self.device),
                ArrayUtils.to_torch(offsets, self.device).reshape(-1),  # (3*edges)
                ArrayUtils.to_torch(edge_topology, self.device, torch.int),
                ArrayUtils.to_torch(ee_ids, self.device, torch.int),
                ArrayUtils.to_torch(ee_vels, self.device),
                ArrayUtils.to_torch(height, self.device))
    
    def _process_motion_sequence(self, motion: MotionSequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Downsample and break up motion data into fixed size windows """
        rotations = motion.rotations.reshape(motion.rotations.shape[0], -1)
        root_pos = motion.positions[:,0]

        full_motion = np.hstack([root_pos, rotations])
        full_motion = self._downsample(data=full_motion, fps_in=motion.fps)
        full_position = self._downsample(data=motion.positions, fps_in=motion.fps)

        # window into [num_windows, window_size, feature_dim]
        windowed_motion = self._get_windows(full_motion)
        windowed_position = self._get_windows(full_position)

        windowed_motion = np.transpose(windowed_motion, axes=(0, 2, 1))
        return windowed_motion, windowed_position

    def _downsample(self, data: np.ndarray, fps_in: float):
        """ Downsample by uniform subsampling. """
        stride = int(round(fps_in / self.downsample_fps))
        return data[::stride]

    def _get_windows(self, data: np.ndarray):
        """ 
        Slice a motion array of shape [T, J*4+3] into windows of shape [window_size, J*4+3].

        Returns:
          [num_windows, window_size, J*4+7].
        """
        T = data.shape[0]
        step = self.window_size // 2

        n_window = T // step - 1
        windows = []

        for i in range(n_window):
            start = i * step
            end = start + self.window_size
            if end > T:
                break
            windows.append(data[start:end])

        return np.stack(windows)
