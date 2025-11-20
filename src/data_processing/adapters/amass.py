from ..base import DataSourceAdapter
from ..metadata import SkeletonMetadata, MotionSequence
from src.utils import _to_torch

from pathlib import Path
from typing import List, Tuple

from scipy.spatial.transform import Rotation as R
import numpy as np

from human_body_prior.body_model.body_model import BodyModel

class AMASSTAdapter(DataSourceAdapter):
    dataset_name = "amass"
    support_name = "body_models"
    joint_cutoff = 22 # Joints > 22 are hands, fingers, toes
    """
    Adapter for AMASS dataset using the SMPL BodyModel.
    
    Extracts skeleton topology, offsets, and other metadata.
    """
    
    def __init__(self): super().__init__(self.dataset_name)

    def download(self, **kwargs) -> None:
        print("=" * 70)
        print("AMASS Dataset Setup Instructions")
        print("=" * 70)
        
        print("\n1. Download AMASS Motion Data:")
        print("   Visit: https://amass.is.tue.mpg.de/")
        print("   - Create account (free for research)")
        print("   - Download desired subsets (CMU, BMLrub, ACCAD, etc.)")
        print(f"   - Extract NPZ files to: {self.data_dir}")
        
        print("\n2. Download SMPL Body Models:")
        print("   Visit: https://smpl.is.tue.mpg.de/")
        print("   - Register and download SMPL+H model")
        print(f"   - Extract to: .../smplh/")
        print("   Expected structure:")
        print(f"     {self.data_dir}/")
        print("         male/model.npz")
        print("         female/model.npz")
        print("         neutral/model.npz")
        
        print("\n3. Install Dependencies:")
        print("   pip install torch")
        print("   pip install git+https://github.com/nghorbani/human_body_prior")
        print("   pip install git+https://github.com/nghorbani/body_visualizer")
        
        print("=" * 70) 
    
    def extract_skeleton(self, file_path: str = None) -> SkeletonMetadata:
        """
        Extract skeleton metadata from body model.
        
        Args:
            file_path: Path to the raw character directory
        
        Returns:
            SkeletonMetadata with topology, offsets, end effectors, and height
        """
        self._post_init(file_path)
        
        body = self.body_model(betas=self.betas.unsqueeze(0)) # BodyModel expects shape [1, num_betas]
        parent_kintree = self.full_kintree[0]
        
        J0 = body.Jtr[0] # T pose
        offsets = J0 - J0[parent_kintree]

        n_joints = len(parent_kintree)
        indices = np.arange(1, n_joints)

        # No hands, fingers, or toes
        indices = indices[indices < 22]
        offsets = offsets[indices]

        topology = self._build_topology()
        end_effectors = self._find_ee()
        height = self._compute_height() #TODO
        
        return SkeletonMetadata(topology=topology,
                                offsets=offsets,
                                end_effectors=end_effectors,
                                height=0)

    def extract_motion(self, file_path: str) -> MotionSequence:
        return MotionSequence #TODO

    def _post_init(self, file_path: Path):
        character_skeleton = np.load(file_path / "shape.npz")
        character_gender = character_skeleton["gender"]
        character_betas = character_skeleton["betas"]

        num_betas = len(character_betas)
        self.betas = _to_torch(character_betas, self.device)

        bm_path = str(self.dataset_dir / self.support_name / character_gender / "model.npz")
        self.body_model = BodyModel(bm_path, num_betas=num_betas).to(self.device)  

        self.full_kintree = self.body_model.kintree_table
        self.pruned_kintree = self._prune_kintree()
    
    def _prune_kintree(self):
        parent = self.full_kintree[0]
        children = self.full_kintree[1]

        pruned_parent   = [p if p <= self.joint_cutoff else -1 for p in parent]
        pruned_children = [c if c <= self.joint_cutoff else -1 for c in children]

        # Remove (-1,-1) pairs
        out_parent = []
        out_child  = []
        for j, (p, c) in enumerate(zip(pruned_parent, pruned_children)):
            if not (p == -1 and c == -1) and j < 22:
                out_parent.append(p)
                out_child.append(c)
        kintree = np.vstack([out_parent, out_child])

        return kintree
    
    def _build_topology(self) -> List[Tuple[int]]:
        parent_kintree = self.pruned_kintree[0]
        
        # Build topology as (parent, child) tuples
        topology = []
        for child_idx in range(1, len(parent_kintree)):  # skip root
            parent_idx = parent_kintree[child_idx]
            topology.append((int(parent_idx), int(child_idx)))

        return topology

    def _find_ee(self):
        parent_kintree = self.pruned_kintree[0]
        children = {i: [] for i in range(len(parent_kintree))}
        for j, p in enumerate(parent_kintree):
            if p >= 0:
                children[p].append(j)

        leaves = [j for j, c in children.items() if len(c) == 0]
        return leaves

    def _compute_height(self):
        return 0 #TODO
    