from typing import List

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.utils.visualization import SkeletonVisualizer
from src.utils import ForwardKinematics, SkeletonUtils, ArrayUtils

from ..metadata import MotionSequence, SkeletonMetadata
from .base import BaseAdapter
from ..mj_generator import SyntheticMotionGenerator

class MuJoCoAdapter(BaseAdapter):
    DATASET_NAME = "mujoco"

    def __init__(self, device="cpu"):
        super().__init__(self.DATASET_NAME, device)

    def _post_init(self, character: str):
        """
        Initialize from a MuJoCo XML model.

        Expected directory layout:
            raw/<character>/<character>.xml   (the MuJoCo model)
        """
        self.character = character
        self.character_dir = self.raw_dir / character

        cache_dir = self.cache_dir / character
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = str(self.character_dir / f"{character}.xml")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Build kinematic tree from MuJoCo model
        # We only include bodies that have joints (actuated bodies)
        # plus the world body as root, matching the kintree convention
        # where kintree[i] = parent of joint/body i.
        self._build_kintree()
        self._compute_canonical_frame()

    def _compute_canonical_frame(self):
        """
        Compute R_world_to_canonical: the rotation that maps world-frame
        coordinates to a canonical frame where the model's forward direction
        (+x local axis of the root body in T-pose) aligns with +z.

        Stored as self.R_world_to_canonical (3x3 ndarray).
        """
        tmp = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, tmp)
        tmp.qpos[:] = 0
        mujoco.mj_forward(self.model, tmp)

        root_quat = tmp.xquat[1]  # wxyz, MuJoCo body 1 = root body
        R = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, root_quat)
        R_root = R.reshape(3, 3)

        forward_world = R_root[:, 0].copy()  # +x local = forward in world
        forward_world /= np.linalg.norm(forward_world)

        world_up = np.array([0.0, 0.0, 1.0])
        helper = np.array([1.0, 0.0, 0.0]) if abs(np.dot(forward_world, world_up)) > 0.999 else world_up

        new_x = np.cross(helper, forward_world)
        new_x /= np.linalg.norm(new_x)
        new_y = np.cross(forward_world, new_x)
        new_y /= np.linalg.norm(new_y)

        # R_canonical columns = canonical axes expressed in world coordinates.
        # new_z = forward_world => model forward maps to +z in canonical frame.
        R_canonical = np.stack([new_x, new_y, forward_world], axis=1)
        self.R_world_to_canonical = R_canonical.T  # world -> canonical

    def _build_kintree(self):
        """
        Build parent array and joint-to-body mapping from the MuJoCo model.

        Convention: body 0 = world (excluded from motion data).
        kintree[i] = parent body index of body i, with body 0's parent = -1.
        """
        model = self.model

        self.body_names = []
        self.kintree_list = []

        # Map: body_id -> index in our flattened array
        # We include all bodies (body 0 = root)
        self.body_id_to_idx = {}

        for bid in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
            self.body_names.append(name)
            self.body_id_to_idx[bid] = bid

            parent_bid = model.body_parentid[bid]
            if bid == 1: #root 
                self.kintree_list.append(-1)
            else:
                self.kintree_list.append(parent_bid - 1)

        self.kintree = np.array(self.kintree_list, dtype=np.int32)
        self.num_bodies = model.nbody-1

    def extract_skeleton(self, character: str) -> SkeletonMetadata:
        """
        Extract skeleton metadata from MuJoCo model.

        Computes T-pose offsets (in cm), edge topology, end-effector IDs,
        and skeleton height.
        """
        self._post_init(character)

        model = self.model
        data = self.data

        mujoco.mj_resetData(model, data)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)

        offsets = np.zeros((self.num_bodies, 3), dtype=np.float64)

        root_pos = data.xpos[1]  # MuJoCo body 1 = root body

        canon_pos = np.zeros((self.num_bodies, 3), dtype=np.float64)
        for bid in range(self.num_bodies):
            p_world = data.xpos[bid + 1]
            canon_pos[bid] = self.R_world_to_canonical @ (p_world - root_pos)

        for bid in range(self.num_bodies):
            parent_bid = self.kintree[bid]
            if parent_bid == -1:
                offsets[bid] = np.zeros(3)
            else:
                offsets[bid] = canon_pos[bid] - canon_pos[parent_bid]
        
        offsets *= 100.0 # convert to cm

        edge_topology = SkeletonUtils.construct_edge_topology(self.kintree)
        ee_ids = SkeletonUtils.find_ee(self.kintree)
        height = SkeletonUtils.compute_height(self.kintree, offsets, ee_ids)

        skeleton = SkeletonMetadata(
            edge_topology=ArrayUtils.to_numpy(edge_topology),
            offsets=ArrayUtils.to_numpy(offsets),
            ee_ids=ArrayUtils.to_numpy(ee_ids),
            height=ArrayUtils.to_numpy(height),
            kintree=ArrayUtils.to_numpy(self.kintree),
        )

        skeleton.save(self.skeleton_dir / f"{character}.npz")
        return skeleton

    def extract_motion(
        self,
        character: str,
        num_episodes: int = 50,
        episode_length: float = 20.0,
        tasks: list[str] | None = None,
        kp_scale: float = 0.2,
        seed: int = 42,
        target_fps: int = 60,
    ) -> List[MotionSequence]:
        """
        Generate synthetic motion sequences for the character.

        Args:
            target_fps: Output framerate. Simulation data is Gaussian-filtered
                        and downsampled from the sim rate to this fps.
        """
        self._post_init(character)

        skeleton = SkeletonMetadata.load(
            self.data_dir / "skeletons" / f"{character}.npz"
        )

        # Generate raw simulation data
        generator = SyntheticMotionGenerator(
            self.model_path,
            kp_scale=kp_scale,
            seed=seed,
        )
        raw_motions = generator.generate(
            num_episodes=num_episodes,
            episode_length=episode_length,
            tasks=tasks,
        )

        sim_fps = int(round(1.0 / self.model.opt.timestep))
        stride = max(1, sim_fps // target_fps)
        # Gaussian sigma sized to act as anti-aliasing low-pass before striding
        smooth_sigma = stride / 2.0

        sequences: List[MotionSequence] = []

        for i, motion in enumerate(tqdm(raw_motions, desc="Processing synthetic motions")):
            task_name = motion["task"]
            fname = f"{task_name}_{i:04d}"

            T = motion["xquat"].shape[0]

            world_quats = motion["xquat"]  # (T, nbody, 4) wxyz

            body_world_quats = world_quats[:, 1:, :]

            parent_rel_quats_wxyz = np.zeros((T, self.num_bodies, 4), dtype=np.float64)

            parent_rel_quats_wxyz[:, 0] = body_world_quats[:, 0]

            for bid in range(1, self.num_bodies):
                parent_bid = self.kintree[bid]  # local index of parent
                parent_rel_quats_wxyz[:, bid] = _quat_mul_wxyz_batch(
                    _quat_conj_wxyz_batch(body_world_quats[:, parent_bid]),
                    body_world_quats[:, bid],
                )

            # Convert wxyz -> xyzw: roll last axis left by 1
            # [w, x, y, z] -> [x, y, z, w]
            parent_rel_quats = np.concatenate(
                [parent_rel_quats_wxyz[:, :, 1:], parent_rel_quats_wxyz[:, :, :1]],
                axis=-1,
            )

            # Discard root body (index 0) from rotations, matching AMASS/Bandai
            rotations = parent_rel_quats[:, 1:]  # (T, num_bodies-1, 4) xyzw

            # ─── Root position in canonical frame (z = model forward) ───
            # MuJoCo xpos index 1 = root body (index 0 = world body at origin).
            root_pos_world = motion["xpos"][:, 1, :]  # (T, 3) metres
            root_pos = (self.R_world_to_canonical @ root_pos_world.T).T * 100.0  # cm, canonical

            # ─── Smooth then downsample ───
            # Gaussian filter anti-aliases before striding; renormalise quats.
            rotations_s = gaussian_filter1d(rotations, sigma=smooth_sigma, axis=0)
            norms = np.linalg.norm(rotations_s, axis=-1, keepdims=True)
            rotations_s /= np.where(norms > 0, norms, 1.0)

            root_pos_s = gaussian_filter1d(root_pos, sigma=smooth_sigma, axis=0)

            rotations_ds = rotations_s[::stride]
            root_pos_ds = root_pos_s[::stride]

            fk_positions = ForwardKinematics.forward(
                quaternions=ArrayUtils.to_torch(rotations_ds),
                offsets=skeleton.offsets,
                root_pos=ArrayUtils.to_torch(root_pos_ds),
                topology=skeleton,
            )

            motion_seq = MotionSequence(
                name=fname,
                positions=ArrayUtils.to_numpy(fk_positions),
                rotations=ArrayUtils.to_numpy(rotations_ds),
                fps=target_fps,
            )

            out_path = self.cache_dir / character / f"{fname}.npz"
            motion_seq.save(out_path)
            sequences.append(motion_seq)

        return sequences

def _quat_conj_wxyz_batch(q: np.ndarray) -> np.ndarray:
    """Conjugate of a batch of unit quaternions. q: (..., 4) wxyz."""
    out = q.copy()
    out[..., 1:] *= -1
    return out


def _quat_mul_wxyz_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batch quaternion multiplication. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)