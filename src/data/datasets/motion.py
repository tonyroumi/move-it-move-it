"""
Normalization and torch dataset classes for single and cross motion domains.
"""

from .builder import MotionDatasetBuilder

from torch.utils.data import Dataset
from typing import List, Tuple
import torch

from src.core.normalization import NormalizationStats
from src.core.types import PairedSample, SkeletonTopology
from src.skeletal.utils import SkeletonUtils

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

        motion_seqs, pos_seqs, ee_vels = [], [], []
        for char_id, char in enumerate(characters):
            (
                motion_windows,
                position_windows,
                offsets,
                edge_topology,
                ee_ids,
                ee_vel,
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
            pos_seqs.append(position_windows)
            ee_vels.append(ee_vel)
            self.char_index.extend([char_id] * motion_windows.shape[0])

        self.rotations = torch.cat(motion_seqs, dim=0)
        self.gt_positions = torch.cat(pos_seqs, dim=0)
        self.gt_ee_vels = torch.cat(ee_vels, dim=0)
        self.char_index = torch.tensor(self.char_index, dtype=torch.long)

        shared_adjacency = SkeletonUtils.construct_adj(shared_topology)

        self.topology = SkeletonTopology(edge_topology=shared_topology, edge_adjacency=shared_adjacency, ee_ids=shared_ee_ids) #Mby here compute the edge list and adjacency

        self.norm_stats = self._compute_normalization_stats()
        self.motion_samples = self.norm_stats.norm(self.rotations)
    
    def _compute_normalization_stats(self):
        # mean/var over batch and time
        mean = torch.mean(self.rotations, dim=(0, 2), keepdim=True)
        var  = torch.var(self.rotations, dim=(0, 2), keepdim=True)
        return NormalizationStats(mean, var)

    def __len__(self):
        return self.motion_samples.shape[0]

    def __getitem__(self, idx):
        char_id = self.char_index[idx]

        # find which character this corresponds to
        char_name = list(self.char_meta.keys())[char_id]
        meta = self.char_meta[char_name]

        return self.rotations[idx], \
               self.motion_samples[idx], \
               meta["offsets"], \
               meta["height"], \
               self.gt_positions[idx], \
               self.gt_ee_vels[idx], \

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
        self.domains = [self.domain_A, self.domain_B]
        self.topologies = [self.domain_A.topology, self.domain_B.topology]

    def __len__(self):
        return max(len(self.domain_A), len(self.domain_B))

    def __getitem__(self, idx):
        idx_A = idx % len(self.domain_A)
        idx_B = idx % len(self.domain_B)

        return PairedSample(
            rotations=(self.domain_A[idx_A][0], self.domain_B[idx_B][0]),
            motions=(self.domain_A[idx_A][1], self.domain_B[idx_B][1]),
            offsets=(self.domain_A[idx_A][2], self.domain_B[idx_B][2]),
            heights=(self.domain_A[idx_A][3], self.domain_B[idx_B][3]),
            gt_positions=(self.domain_A[idx_A][4], self.domain_B[idx_B][4]),
            gt_ee_vels=(self.domain_A[idx_A][5], self.domain_B[idx_B][5])
        )
