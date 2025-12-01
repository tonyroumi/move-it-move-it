from .normalization import NormalizationStats

from src.data_processing import DataSourceAdapter
from src.utils.data import _to_torch
from src.data_processing import SkeletonMetadata, MotionSequence

from torch.utils.data import Dataset
from typing import List, Any, Dict
import numpy as np
import torch


class MotionDatasetBuilder:
    """ MotionDataset builder to construct model Dataset object. """
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

        self.device = device

    def get_or_process(self, character: str):
        """ Load or generate all motion data and skeleton metadata for a given character. """ 
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
            all_motion = []
            for proccesed_motion in processed_files:
                motion = MotionSequence.load(proccesed_motion)

                rotations = motion.rotations
                rotations = rotations.reshape(rotations.shape[0], -1)
                root_orient = motion.root_orient

                full_motion = np.hstack([root_orient, rotations])
                full_motion = self._downsample(motion=full_motion, 
                                               fps_in=motion.fps)

                windowed_motion = self._get_windows(full_motion)

                all_motion.append(windowed_motion)

        all_motion = np.vstack(all_motion)

        return (_to_torch(all_motion, self.device),
                _to_torch(offsets, self.device),
                _to_torch(topology, self.device),
                _to_torch(ee_ids, self.device),
                _to_torch(height, self.device))

    def _downsample(self, motion: np.ndarray, fps_in: float):
        """
        Downsample by uniform subsampling.
        fps_out must be <= fps_in.
        """
        if self.downsample_fps > fps_in:
            raise ValueError("fps_out must be <= fps_in for simple subsampling.")

        stride = int(round(fps_in / self.downsample_fps))
        return motion[::stride]

    def _get_windows(self, motion: np.ndarray):
        """ 
        Slice a motion array of shape [T, J*4+7] into windows of shape [window_size, J*4+7].

        Returns:
          [num_windows, window_size, J*4+7].
        """
        T = motion.shape[0]
        ws = self.window_size

        num_windows = T // ws
        out = []

        for i in range(num_windows):
            start = i * ws
            end = start + ws
            out.append(motion[start:end])

        return np.stack(out)

class MotionDataset(Dataset):
    """
    Represents all motion windows for a motion domain where all characters share the same topology
    and end-effectors.

    Each sample is:
       rotations: [window, n_joints*4]
       offsets: ?
    """

    def __init__(
        self,
        builder: MotionDatasetBuilder,
        characters: List[str],
    ):
        self.motion_seqs = []
        self.offsets = []
        self.char_index = []         # maps each sample → char_id
        self.char_meta = {}          # char_name → {offsets, height}

        shared_topology = None
        shared_ee_ids = None

        for char_id, char in enumerate(characters):
            (
                motion_windows,      # [num_windows, (T) window, 7+(J-1)*4]
                offsets,
                topology,
                ee_ids,
                height,
            ) = builder.get_or_process(char)

            if shared_topology is None:
                shared_topology = topology
                shared_ee_ids = ee_ids
            else:
                if topology != shared_topology:
                    raise ValueError(f"Topology mismatch for character {char}")
                if ee_ids != shared_ee_ids:
                    raise ValueError(f"EE ids mismatch for character {char}")

            self.char_meta[char] = {
                "height": height,
                "id": char_id,
            }
            self.motion_seqs.append(motion_windows)
            self.offsets.append(offsets)
            self.char_index.extend([char_id] * motion_windows.shape[0])

        self.motion_seqs = torch.cat(self.motion_seqs, dim=0)
        self.offsets = torch.cat(self.offsets, dim=0)
        self.char_index = torch.tensor(self.char_index, dtype=torch.long)

        self.topology = shared_topology
        self.ee_ids = shared_ee_ids

        self.norm_stats = self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        # mean/var over batch and time
        mean = torch.mean(self.samples, dim=(0, 2), keepdim=True)
        var  = torch.var(self.samples, dim=(0, 2), keepdim=True)
        return NormalizationStats(mean, var)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
          rotations: Tensor [window, features]
          metadata:  dict with per-character info (height, offsets, char_id)
        """
        char_id = self.char_index[idx]

        # find which character this corresponds to
        char_name = list(self.char_meta.keys())[char_id]
        meta = self.char_meta[char_name]

        return {
            "motion": self.samples[idx],
            "char_id": char_id,
            "offsets": meta["offsets"],
            "height": meta["height"],
        }
