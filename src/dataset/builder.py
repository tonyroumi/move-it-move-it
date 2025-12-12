"""
MotionDataset builder to construct MotionDataset for a single motion domain.
"""

from src.data_processing import DataSourceAdapter
from src.data_processing import SkeletonMetadata, MotionSequence
from src.utils.data import ArrayUtils

from typing import Any, Dict
import numpy as np
import torch

class MotionDatasetBuilder:
    def __init__(
        self, 
        adapter: DataSourceAdapter, 
        data_config: Dict[str, Any],
        device: torch.device = 'cpu',
    ):
        self.adapter = adapter
        self.data_config = data_config

        self.window_size = data_config['window_size']
        self.downsample_fps = data_config['downsample_fps']
        self.include_root_quat = data_config['include_root_quat']

        self.device = device

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
        topology = skeleton.topology
        ee_ids = skeleton.ee_ids
        height = skeleton.height

        # If there are no processed files, proccess from raw dir. 
        if not processed_files:
            motions = self.adapter.extract_motion(character)
        else:
            motions = []
            for processed_motion in processed_files:
                motions.append(MotionSequence.load(processed_motion))

        windowed_motion = [self._process_motion_sequence(m) for m in motions]
        all_windowed_motion = np.vstack(windowed_motion)

        return (ArrayUtils.to_torch(all_windowed_motion, self.device),
                ArrayUtils.to_torch(offsets, self.device),
                ArrayUtils.to_torch(topology, self.device),
                ArrayUtils.to_torch(ee_ids, self.device),
                ArrayUtils.to_torch(height, self.device))
    
    def _process_motion_sequence(self, motion: MotionSequence):
        """ Downsample and break up motion data into fixed size windows """
        rotations = motion.rotations.reshape(motion.rotations.shape[0], -1)
        if not self.include_root_quat:
            rotations = rotations[:, 4:]
                        
        root_pos = motion.positions[:,0]

        full_motion = np.hstack([root_pos, rotations])
        full_motion = self._downsample(motion=full_motion, fps_in=motion.fps)

        # window into [num_windows, window_size, feature_dim]
        windowed = self._get_windows(full_motion)
        return windowed

    def _downsample(self, motion: np.ndarray, fps_in: float):
        """ Downsample by uniform subsampling. """
        stride = int(round(fps_in / self.downsample_fps))
        return motion[::stride]

    def _get_windows(self, motion: np.ndarray):
        """ 
        Slice a motion array of shape [T, J*4+3] into windows of shape [window_size, J*4+3].

        Returns:
          [num_windows, window_size, J*4+7].
        """
        T = motion.shape[0]
        step = self.window_size // 2

        n_window = T // step - 1
        windows = []

        for i in range(n_window):
            start = i * step
            end = start + self.window_size
            if end > T:
                break
            windows.append(motion[start:end])

        return np.stack(windows)
