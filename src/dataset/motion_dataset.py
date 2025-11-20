from torch.utils.data import Dataset
import torch
from .normalization import NormalizationStats
from typing import List
from src.data_processing import DataSourceAdapter

class MotionDatasetBuilder:
    def __init__(
        self, 
        adapter: DataSourceAdapter, 
        window_size: int = 64
    ):
        self.adapter = adapter
        self.window_size = window_size

    def get_or_process(self, character: str):
        """
         [N, T, J, 3]
        """ 

        cache_path = (
            self.adapter.cache_dir / character
        )
        cache_path.mkdir(parents=True, exist_ok=True)


        raw_path = (
            self.adapter.raw_dir / character
        )
        files = raw_path.glob("*")

        if not files:
            raise FileNotFoundError(f"No raw files for {character}")

        skel_path = (
            self.adapter.skeleton_dir / f"{character}.npy"
        )
        if not skel_path.exists():
            
            skel = self.adapter.extract_skeleton(raw_path)
            skel_path.parent.mkdir(parents=True)
            skel_path.write_text(skel.model_dump_json(indent=2))

        # 3. process motions
        motions = []
        for f in files:
            motion = self.adapter.extract_motion(str(f))         # MotionSequence
            canon = self.canonical_mapper(motion)                # → canonical joints
            motions.append(canon)

        # 4. concatenate all sequences for the character
        all_motion = torch.cat(motions, dim=0)

        # 5. sliding windows
        windows = self._get_windows(all_motion)

        torch.save(windows, cache_path)
        return windows

    def _get_windows(self, motion):
        T = motion.shape[0]
        ws = self.window_size
        out = []
        for i in range(T - ws + 1):
            out.append(motion[i:i+ws])
        return torch.stack(out)

class MotionDataset(Dataset):
    def __init__(
        self, 
        builder: MotionDatasetBuilder, 
        characters: List[str]
    ):
        self.samples = []

        for char in characters:
            processed = builder.get_or_process(char)
            self.samples.append(processed)

        self.samples = torch.cat(self.samples, dim=0)
    
    def _compute_normalization_stats(self):
        mean = torch.mean(self.samples, (0,2), keepdim=True)   
        var =  torch.var(self.samples, (0,2), keepdim=True)
        return NormalizationStats(mean, var)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]