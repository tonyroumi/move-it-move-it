from ..base import DataSourceAdapter
from ..metadata import SkeletonMetadata, MotionSequence

from src.utils.data import _to_torch, _to_numpy
from src.utils.rotation import axis_angle_to_quat
from src.utils.skeleton import prune_joints
from src.utils.transforms import global_aa_to_local_quat_batched

from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import torch

class AMASSTAdapter(DataSourceAdapter):
    dataset_name = "amass"
    support_name = "body_models"
    joint_cutoff = 22 # Joints > 22 are hands, fingers, toes
    head_idx = 15
    foot_idx = 10
    """
    Adapter for AMASS dataset using the SMPL BodyModel.
    
    Extracts skeleton topology, offsets, motion sequences, and other metadata.
    """
    
    def __init__(self, device: torch.device = 'cpu'): super().__init__(self.dataset_name, device)

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
        self.betas = _to_torch(character_betas, self.device)

        bm_path = str(self.dataset_dir / self.support_name / character_gender / "model.npz")
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
        """
        Extract and save skeleton metadata from a particular character.
        
        Args:
            character: character name as it exists in amass/raw.
        
        Returns:
            SkeletonMetadata: topology, offsets, ee_idsor indices, and height.
        """
        self._post_init(character)
        
        body = self.body_model(betas=self.betas.unsqueeze(0)) # BodyModel expects shape [1, num_betas]
        
        J0 = body.Jtr[0] # T pose
        offsets = J0 - J0[self.parent_kintree]

        offsets = prune_joints(offsets, self.joint_cutoff, exclude_root=True)

        topology = self._build_topology()
        ee_ids = self._find_ee()
        height = self._compute_height(offsets)

        skeleton = SkeletonMetadata(topology=_to_numpy(topology),
                                    offsets=offsets,
                                    ee_ids=_to_numpy(ee_ids),
                                    height=_to_numpy(height))
        skeleton.save(self.skeleton_dir / (str(character) + ".npz") )

        return skeleton

    def extract_motion(self, character: str) -> List[MotionSequence]:
        """
        Extract and save all skeleton motion data for a particular character.
        
        Args:
            character: character name as it exists in amass/raw.
        
        Returns:
            List[MotionSequence]: root orientations, quat rotations, fps. 
        """
        self._post_init(character)

        sequences : List[MotionSequence] = []

        for npz_file in tqdm(self.motion_seqs, desc="Extracting motion sequences"):
            fname = npz_file.split('/')[-1]
            tqdm.write(f"Processing: {fname}")

            data = np.load(npz_file)

            pose_body = _to_torch(data["poses"], self.device)
            trans = _to_torch(data["trans"], self.device)
            
            out = self.body_model(
                root_orient = pose_body[:,0:3],
                pose_body=pose_body[:,3:66],
                trans=trans,
                betas=self.betas.unsqueeze(0)
            )

            aa = out.full_pose.reshape(-1, self.num_joints, 3)
            aa = prune_joints(aa, cutoff=self.joint_cutoff, exclude_root=True)
            quat_rotations = global_aa_to_local_quat_batched(aa, self.parent_kintree, include_root=False)

            root_pos = out.Jtr[:, 0]                              # [T, 3]
            root_quat = axis_angle_to_quat(out.full_pose[:, :3])  # [T, 4]
            root_global = np.concatenate([root_pos, root_quat], axis=-1)

            motion_sequence = MotionSequence(root_orient=_to_numpy(root_global),
                                             rotations=_to_numpy(quat_rotations),
                                             fps=_to_numpy(data['mocap_framerate']))
            motion_sequence.save(self.cache_dir / character / fname)

            sequences.append(motion_sequence)

        return sequences
    
    def _prune_kintree(self):
        """ Removes joints from the kintree based outside of the joint_cutoff """
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
        """ Constructs the topology for a skeleton as (parent, child) joint tuples. """
        parent_kintree = self.pruned_kintree[0]
        
        # Build topology as (parent, child) joint tuples
        topology = []
        for child_idx in range(1, len(parent_kintree)):  # skip root
            parent_idx = parent_kintree[child_idx]
            topology.append((int(parent_idx), int(child_idx)))

        return topology

    def _find_ee(self):
        """ Finds the skeleton's end effectors by traversiing the tree for leaf joints. """
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
        p = self.foot_idx
        while p != 0:
            h1 += np.linalg.norm(offset[p])
            p = self.parent_kintree[p]

        # pelvis to head
        h2 = 0.0
        p = self.head_idx
        while p != 0:
            h2 += np.linalg.norm(offset[p])
            p = self.parent_kintree[p]

        return h1 + h2
