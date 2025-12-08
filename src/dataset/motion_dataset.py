from .builder import MotionDatasetBuilder
from .normalization import NormalizationStats

from torch.utils.data import Dataset
from typing import List
import torch

class MotionDataset(Dataset):
    """
    Represents all motion for a motion domain where all characters share the same topology
    and end-effectors.

    Each sample contains:

        motion: Windowed motion from all characters in the dataset.
        offsets: Offsets of only the characters in the motion dataset stacked for batch size.
        height: Heights of only the characters in the motion dataset stacked for batch size. 
    """

    def __init__(
        self,
        name: str,
        characters: List[str],
        builder: MotionDatasetBuilder,
    ):
        self.name = name

        self.char_index = []         # maps each sample → char_id
        self.char_meta = {}

        shared_topology = None
        shared_ee_ids = None

        motion_seqs = []
        for char_id, char in enumerate(characters):
            (
                motion_windows,
                offsets,
                topology,
                ee_ids,
                height,
            ) = builder.get_or_process(char)

            if shared_topology is None:
                shared_topology = topology
                shared_ee_ids = ee_ids
            else:
                if not torch.equal(topology, shared_topology):
                    raise ValueError(f"Topology mismatch for character {char}")
                if not torch.equal(ee_ids, shared_ee_ids):
                    raise ValueError(f"EE ids mismatch for character {char}")

            self.char_meta[char] = {
                "offsets": offsets,
                "height": height,
                "id": char_id,
            }
            motion_seqs.append(motion_windows)
            self.char_index.extend([char_id] * motion_windows.shape[0])

        self.samples = torch.cat(motion_seqs, dim=0)
        self.char_index = torch.tensor(self.char_index, dtype=torch.long)

        self.topology = shared_topology
        self.ee_ids = shared_ee_ids

        self.norm_stats = self._compute_normalization_stats()
        self.samples = self.norm_stats.norm(self.samples)

    def denorm(self, motion: torch.Tensor):
        return self.norm_stats.denorm(motion)
    
    def _compute_normalization_stats(self):
        # mean/var over batch and time
        mean = torch.mean(self.samples, dim=(0, 1), keepdim=True)
        var  = torch.var(self.samples, dim=(0, 1), keepdim=True)
        return NormalizationStats(mean, var)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        char_id = self.char_index[idx]

        # find which character this corresponds to
        char_name = list(self.char_meta.keys())[char_id]
        meta = self.char_meta[char_name]

        return self.samples[idx], meta["offsets"], meta["height"]

class CrossDomainMotionDataset(Dataset):
    """ Combines two MotionDataset instances for two different motion domains. """
    def __init__(
        self,
        domain_A: MotionDataset,
        domain_B: MotionDataset
    ):
        self.domain_A = domain_A
        self.domain_B = domain_B
        self.domains = {
            domain_A.name: domain_A,
            domain_B.name: domain_B
        }

    def denorm(self, domain_name: str, motion: torch.Tensor) -> torch.Tensor:
        return self.domains[domain_name].denorm(motion)
    
    def __len__(self):
        return max(len(self.domain_A, len(self.domain_B)))
    
    def __getitem__(self, idx):
        idx_A = idx % len(self.domain_A)
        idx_B = idx % len(self.domain_B)
        
        sample_A = self.domain_A[idx_A]
        sample_B = self.domain_B[idx_B]

        return sample_A, sample_B