from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from moveitmoveit.src.sim import Skeleton
import moveitmoveit.src.transforms as transforms
from utils import Logger

from .motion_clip import MotionClip
#TODO would it be faster if clips were already on gpu then when i sample for the start state make then np for mujoco?
class MotionLibrary:
    """ Collection of motion clips with weighted random sampling. """

    def __init__(
        self,
        srcs: List[str],
        skeleton: Skeleton,
        logger: Logger,
    ) -> None:
        self._clips: List[MotionClip] = [MotionClip.load(src) for src in srcs]

        # Weight proportional to duration. why? TODO
        weights = [c.duration for c in self._clips]
        w = np.array(weights, dtype=np.float64)
        self._weights = w / w.sum()

        self._skeleton = skeleton
        self.logger = logger

        self._load()

    def _load(self):
        """
        This function depends on the notion that joint posiitons and rotations are on a per joint basis. 
        It assumes that joints are quats and are in the form w,x,y,z.  These are likely local rotations and local positions. 
        NOT GLOBAL 
        """
        motion_lengths = []
        motion_root_pos_delta = []
        motion_num_frames = []

        frame_root_pos = []
        frame_root_rot = []
        frame_root_vel = []
        frame_root_ang_vel = []
        frame_joint_rot = []
        frame_joint_pos = []
        frame_dof_vel = []

        total_length = 0 
        for clip in self._clips:
            fps = clip.fps
            dt = 1 / fps
            num_frames = clip.positions.shape[0]
            curr_len = 1.0 / fps * (num_frames - 1)
            total_length += curr_len
            # these should be tensors. no reason to not be. 
            root_pos = np.array(clip.positions[:, 0, :])
            root_pos_delta = root_pos[-1] - root_pos[0]
            root_pos_delta[-1] = 0.0
            root_rot = np.array(clip.rotations[:, 0, :])
            rotations = np.array(clip.rotations[:, 1:, :])
            rotations = transforms.quat_pos(rotations)
            positions = clip.positions[:, 1:, :]
            
            root_vel = np.zeros_like(root_pos)
            root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
            root_vel[-1, :] = root_vel[-2, :]

            root_ang_vel = np.zeros_like(root_pos)
            root_drot = transforms.quat_diff(root_rot[:-1, :], root_rot[1:, :])
            root_ang_vel[:-1, :] = fps * transforms.quat_to_exp_map(root_drot)
            root_ang_vel[-1, :] = root_ang_vel[-2, :]

            dof_vel = self._skeleton.compute_dof_vel(rotations, dt)

            motion_lengths.append([curr_len])
            motion_root_pos_delta.append(root_pos_delta)
            motion_num_frames.append(num_frames)
            
            frame_root_pos.append(root_pos)
            frame_root_rot.append(root_rot)
            frame_root_vel.append(root_vel)
            frame_root_ang_vel.append(root_ang_vel)
            frame_joint_rot.append(rotations)
            frame_joint_pos.append(positions)
            frame_dof_vel.append(dof_vel)

        self._motion_lengths = np.concatenate(motion_lengths, axis=0)
        self._motion_root_pos_delta = np.concatenate(motion_root_pos_delta, axis=0)
        self._motion_num_frames = np.concatenate(motion_num_frames, axis=0)

        self._frame_root_pos = np.concatenate(frame_root_pos, axis=0)
        self._frame_root_rot = np.concatenate(frame_root_rot, axis=0)
        self._frame_root_vel = np.concatenate(frame_root_vel, axis=0)
        self._frame_root_ang_vel = np.concatenate(frame_root_ang_vel, axis=0)
        self._frame_joint_rot = np.concatenate(frame_joint_rot, axis=0)
        self._frame_joint_pos = np.concatenate(frame_joint_pos, axis=0)
        self._frame_dof_vel = np.concatenate(frame_dof_vel, axis=0)

        self._motion_ids = np.arange(self.num_clips)

        self.logger.info(f"Loaded {len(self._clips)} motions with a total length of {total_length}")

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    @property
    def total_length(self) -> int:
        return np.sum(self._motion_lengths)

    def get_clip(self, idx: int) -> MotionClip:
        return self._clips[idx]
    
    def sample_motions(self, n: int) -> List[int]:
        # They use a multinomial distribution here. I don't know if we need that..
        rng = np.random.default_rng()
        motion_ids = rng.choice(len(self._clips), size=n, p=self._weights)
        return motion_ids

    def sample_times(self, motion_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniformly sample *n* (clip_index, time) pairs,
        weighted by clip duration.
        """
        phase = np.random(len(motion_ids))
        lengths = self._motion_lengths[motion_ids]

        start_times = phase * lengths
        return start_times

    def sample_frame_data(self, n: int) -> List[dict]:
        """
        Sample *n* random frames and return their data dicts.
        """
        clip_ids, frame_ids = self.sample_frames(n)
        return [
            self._clips[cid].get_frame(fid)
            for cid, fid in zip(clip_ids, frame_ids)
        ]

    def sample_start_state(self) -> Tuple[int, int, dict]:
        """ Sample a single random starting state for episode initialisation.

        Returns
            clip_id
            frame_id
            frame 
        """
        cids, fids = self.sample_frames(1)
        return int(cids[0]), int(fids[0]), self._clips[cids[0]].get_frame(fids[0])
    
    def calculate_motion_frame(self, motion_ids, motion_times):
        """
        This will be easier with actual data seen.
        """
        # First calculate the phase of a particular motion:
        number_frames = self._motion_num_frames[motion_ids]
        motion_lengths = self._motion_lengths[motion_ids]

        # we want to compute the times here relative to the actual clip frames. This will depend on us can't really copy.
        # alls we want is the motion frame from a time sampled during a window of data. 
        phase = motion_times / motion_lengths
        phase = phase - np.floor(phase)
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (number_frames - 1)).long()
        frame_idx1 = np.min(frame_idx0 + 1, number_frames - 1)
        blend = phase * (number_frames - 1) - frame_idx0

        root_pos0 = self._frame_root_pos[frame_idx0]
        root_pos1 = self._frame_root_pos[frame_idx1]
        
        root_rot0 = self._frame_root_rot[frame_idx0]
        root_rot1 = self._frame_root_rot[frame_idx1]

        root_vel = self._frame_root_vel[frame_idx0]
        root_ang_vel = self._frame_root_ang_vel[frame_idx0]

        joint_rot0 = self._frame_joint_rot[frame_idx0]
        joint_rot1 = self._frame_joint_rot[frame_idx1]

        joint_pos0 = self._frame_joint_pos[frame_idx0]
        joint_pos1 = self._frame_joint_pos[frame_idx1]

        dof_vel = self._frame_dof_vel[frame_idx0]

        blend_unsq = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsq) * root_pos0 + blend_unsq * root_pos1
        root_rot = transforms.slerp(root_rot0, root_rot1, blend)
        
        joint_rot = transforms.slerp(joint_rot0, joint_rot1, blend_unsq)
        joint_pos =  transforms.slerp(joint_pos0, joint_pos1, blend_unsq)

        root_pos_deltas = self._motion_root_pos_delta[motion_ids]

        phase = motion_times / motion_lengths
        phase = np.floor(phase)
        phase = phase.unsqueeze(-1)
        
        root_pos_offset = np.zeros((motion_ids.shape[0], 3), device=self._device)
        root_pos_offset = phase * root_pos_deltas

        root_pos += root_pos_offset

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, joint_pos, dof_vel

    