"""
Normalization and torch dataset classes for single and cross motion domains.
"""

from .builder import MotionDatasetBuilder
from .normalization import NormalizationStats
from src.skeletal_models import SkeletonTopology
from src.utils import SkeletonUtils

from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import List, Tuple
import torch

@dataclass
class PairedSample:
    motions: Tuple[torch.Tensor, torch.Tensor]
    offsets: Tuple[torch.Tensor, torch.Tensor]
    heights: Tuple[torch.Tensor, torch.Tensor]

    def to(self, device: torch.device):
        return PairedSample(
            motions=tuple(m.to(device) for m in self.motions),
            offsets=tuple(o.to(device) for o in self.offsets),
            heights=tuple(h.to(device) for h in self.heights),
        )

def paired_collate(batch):
    motions = tuple(
        torch.stack([
            pad_root_flat(b.motions[i])
            for b in batch
        ])
        for i in range(2)
    )

    offsets = tuple(
        torch.stack([b.offsets[i] for b in batch])
        for i in range(2)
    )

    heights = tuple(
        torch.stack([b.heights[i] for b in batch])
        for i in range(2)
    )

    return PairedSample(motions, offsets, heights)

class MotionDataset(Dataset):
    """
    Represents all motion for a motion domain where all characters share the same topology
    and end-effectors.

    __getitem__ returns:
        motion, offsets, height
    """
    def __init__(
        self,
        characters: List[str],
        builder: MotionDatasetBuilder,
    ):
        self.char_index = []         # maps each sample → char_id
        self.char_meta = {}

        shared_topology = None
        shared_ee_ids = None

        motion_seqs = []
        for char_id, char in enumerate(characters):
            (
                motion_windows,
                offsets,
                edge_topology,
                ee_ids,
                height,
            ) = builder.get_or_process(char)

            if shared_topology is None:
                shared_topology = edge_topology
                shared_ee_ids = ee_ids
            else:
                if not torch.equal(edge_topology, shared_topology):
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

        shared_adjacency = SkeletonUtils.construct_adj(shared_topology)

        self.topology = SkeletonTopology(edge_topology=shared_topology, edge_adjacency=shared_adjacency, ee_ids=shared_ee_ids) #Mby here compute the edge list and adjacency

        self.norm_stats = self._compute_normalization_stats()
        self.samples = self.norm_stats.norm(self.samples)

    def denorm(self, motion: torch.Tensor):
        return self.norm_stats.denorm(motion[:,1:,:])
    
    def _compute_normalization_stats(self):
        # mean/var over batch and time
        mean = torch.mean(self.samples, dim=(0, 2), keepdim=True)
        var  = torch.var(self.samples, dim=(0, 2), keepdim=True)
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
    """ 
    Combines two MotionDataset instances for two different motion domains.
    
    __getitem__ returns:
        PairedSample(motions, offsets, heights)
    """
    def __init__(
        self,
        domain_A: MotionDataset,
        domain_B: MotionDataset
    ):
        self.domain_A = domain_A
        self.domain_B = domain_B
        self.topologies = [self.domain_A.topology, self.domain_B.topology]

    def denorm(
        self, 
        motion: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denormed_A = self.domain_A.denorm(motion[0])
        denormed_B = self.domain_B.denorm(motion[1])
        return denormed_A, denormed_B
    
    def __len__(self):
        return max(len(self.domain_A), len(self.domain_B))

    def __getitem__(self, idx):
        idx_A = idx % len(self.domain_A)
        idx_B = idx % len(self.domain_B)

        return PairedSample(
            motions=(self.domain_A[idx_A][0], self.domain_B[idx_B][0]),
            offsets=(self.domain_A[idx_A][1], self.domain_B[idx_B][1]),
            heights=(self.domain_A[idx_A][2], self.domain_B[idx_B][2])
        )

def pad_root_flat(motion: torch.Tensor) -> torch.Tensor:
    """
    motion: (T, 3 + 4*J)
    returns: (T, 4 + 4*J)
    """
    D, T = motion.shape

    padded = torch.zeros(D + 1, T, device=motion.device, dtype=motion.dtype)

    # root translation -> xyz0
    padded[1:4, :] = motion[:3, :]

    # shift rotations by 1
    padded[4:, :] = motion[3:, :]

    return padded