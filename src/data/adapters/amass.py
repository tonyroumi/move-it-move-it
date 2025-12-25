"""
Adapter for AMASS dataset using the SMPL BodyModel.
    
Extracts skeleton topology, offsets, motion sequences, and other metadata.
"""

from .base import DataSourceAdapter
from ..metadata import SkeletonMetadata, MotionSequence

from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import torch

from src.skeletal.utils import SkeletonUtils
from src.utils import ArrayUtils, RotationUtils

class AMASSTAdapter(DataSourceAdapter):
    DATASET_NAME = "amass"
    SUPPORT_NAME = "body_models"
    JOINT_CUTOFF = 22 # Joints > 21 are hands, fingers, toes. (zero based)
    HEAD_IDX = 15
    FOOT_IDX = 10

    def __init__(self, device: torch.device = 'cpu'): super().__init__(self.DATASET_NAME, device)

    def _post_init(self, character: str):
        """ Initialize character specific BodyModel """
        from human_body_prior.body_model.body_model import BodyModel

        character_dir = self.raw_dir / character
        character_skeleton = np.load(character_dir / "shape.npz")
        character_gender = character_skeleton["gender"]
        character_betas = character_skeleton["betas"]

        cache_dir = self.cache_dir / character
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.motion_seqs = [str(f) for f in character_dir.glob("*.npz") if f.name != "shape.npz"]

        num_betas = len(character_betas)
        self.betas = ArrayUtils.to_torch(character_betas, self.device)

        bm_path = str(self.dataset_dir / self.SUPPORT_NAME / character_gender / "model.npz")
        self.body_model = BodyModel(bm_path, num_betas=num_betas).to(self.device)  

        self.full_kintree = self.body_model.kintree_table
        self.parent_kintree = self.full_kintree[0]
        self.pruned_kintree = self._prune_kintree()

        self.num_joints = len(self.parent_kintree)

    def download(self, **kwargs) -> None:
        print("=" * 70)
        print("AMASS Dataset Setup Instructions")
        print("=" * 70)

        print("\n1. Download AMASS Motion Data:")
        print("   Visit: https://amass.is.tue.mpg.de/")
        print("   - Create an account (free for research)")
        print("   - Download the desired subsets (HUMAN4D, ACCAD, CMU, etc.)")
        print("   - Extract the downloaded archives")

        print("\n   Required directory structure:")
        print(f"   {self.raw_dir}/...")
        print("           <character_001>/")
        print("               motion_0001.npz")
        print("               motion_0002.npz")
        print("               ...")
        print("               shape.npz")
        print("           <character_002>/")
        print("               ...")

        print(f"\n   Place all extracted character folders into: {self.raw_dir}")

        print("\n2. Download SMPL Body Models:")
        print("   Visit: https://smpl.is.tue.mpg.de/")
        print("   - Register and download the SMPL+H model package")
        print("   - Extract the model files")

        print("\n   Required directory structure:")
        print("   data/amass/body_models/")
        print("       female/model.npz")
        print("       male/model.npz")
        print("       neutral/model.npz")

        print("=" * 70)
    
    def extract_skeleton(self, character: str) -> SkeletonMetadata:
        """ Extract and save skeleton metadata from a particular character. """

        self._post_init(character)
        
        body = self.body_model(betas=self.betas.unsqueeze(0)) # BodyModel expects shape [1, num_betas]
        
        J0 = body.Jtr[0] # T pose
        offsets = J0 - J0[self.parent_kintree]

        offsets = SkeletonUtils.prune_joints(offsets, self.JOINT_CUTOFF)
        offsets[0] = torch.zeros(3) # No root offset
        offsets *= 100 # convert to cm

        edge_topology = self._build_edge_topology()
        ee_ids = self._find_ee()
        height = self._compute_height(offsets)

        skeleton = SkeletonMetadata(edge_topology=ArrayUtils.to_numpy(edge_topology),
                                    offsets=ArrayUtils.to_numpy(offsets),
                                    ee_ids=ArrayUtils.to_numpy(ee_ids),
                                    height=ArrayUtils.to_numpy(height),
                                    kintree=ArrayUtils.to_numpy(self.pruned_kintree))
        skeleton.save(self.skeleton_dir / (str(character) + ".npz") )

        return skeleton

    def extract_motion(self, character: str) -> List[MotionSequence]:
        """ Extract and save all skeleton motion data for a particular character. """
        self._post_init(character)

        sequences : List[MotionSequence] = []

        for npz_file in tqdm(self.motion_seqs, desc="Extracting motion sequences"):
            fname = npz_file.split('/')[-1]
            tqdm.write(f"Processing: {fname}")

            data = np.load(npz_file)

            pose_body = ArrayUtils.to_torch(data["poses"], self.device)
            trans = ArrayUtils.to_torch(data["trans"], self.device)
            
            out = self.body_model(
                root_orient=pose_body[:,0:3],
                pose_body=pose_body[:,3:66],
                trans=trans,
                betas=self.betas.unsqueeze(0)
            )

            aa = out.full_pose.reshape(-1, self.num_joints, 3)
            quat_rotations = RotationUtils.aa_to_quat(aa.reshape(-1, 3))
            quat_rotations = quat_rotations.reshape(-1, self.num_joints, 4)
            quat_rotations = SkeletonUtils.prune_joints(quat_rotations, cutoff=self.JOINT_CUTOFF, discard_root=True)

            positions = SkeletonUtils.prune_joints(out.Jtr , cutoff=self.JOINT_CUTOFF) * 100 # convert to cm

            motion_sequence = MotionSequence(name=fname,
                                             positions=ArrayUtils.to_numpy(positions),
                                             rotations=ArrayUtils.to_numpy(quat_rotations),
                                             fps=ArrayUtils.to_numpy(data['mocap_framerate']),)
            motion_sequence.save(self.cache_dir / character / fname)

            sequences.append(motion_sequence)

        return sequences
    
    def _prune_kintree(self):
        """ Removes joints from the kintree outside of the joint_cutoff """
        parent = self.full_kintree[0]
        children = self.full_kintree[1]

        pruned_parent = [
            p if p <= self.JOINT_CUTOFF and joint <= self.JOINT_CUTOFF else -1
            for joint, p in enumerate(parent)
        ]
        pruned_children = [
            c if c <= self.JOINT_CUTOFF and joint <= self.JOINT_CUTOFF else -1
            for joint, c in enumerate(children)
        ]

        # Remove (-1,-1) pairs
        out_parent = []
        out_child  = []
        for j, (p, c) in enumerate(zip(pruned_parent, pruned_children)):
            if (j == 0 or (p != -1 and c != -1)) and j < 22:
                out_parent.append(p)
                out_child.append(c)
        kintree = np.vstack([out_parent, out_child])

        return kintree
    
    def _build_edge_topology(self) -> List[Tuple[int]]:
        """ Constructs the edge topology for a skeleton as (parent, child) joint tuples. """
        parent_kintree = self.pruned_kintree[0]
        
        # Build edge topology as (parent, child) joint tuples
        topology = []
        for child_idx, parent_idx in enumerate(parent_kintree[1:], start=1):  # skip root
            topology.append((int(parent_idx), int(child_idx)))

        return topology

    def _find_ee(self):
        """ Finds the skeleton's end effectors by traversing the tree for leaf joints. """
        parent_kintree = self.pruned_kintree[0]
        children = {i: [] for i in range(len(parent_kintree))}
        for j, p in enumerate(parent_kintree):
            if p >= 0:
                children[p].append(j)

        leaves = [j for j, c in children.items() if len(c) == 0]
        return leaves

    def _compute_height(self, offset: np.ndarray):
        """ Computes the height by summing the size of each offset vector from the head to the feet. """
        # foot to pelvis
        h1 = 0.0
        p = self.FOOT_IDX
        while p != 0:
            h1 += np.linalg.norm(offset[p])
            p = self.parent_kintree[p]

        # pelvis to head
        h2 = 0.0
        p = self.HEAD_IDX
        while p != 0:
            h2 += np.linalg.norm(offset[p])
            p = self.parent_kintree[p]

        return h1 + h2
