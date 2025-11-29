from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class SkeletonMetadata:
    topology: List[Tuple[int, int]]
    offsets: np.ndarray
    end_effectors: List[int]
    height: float
    root_joint: int = 0

    def save(self, path: str):
        """
        Save skeleton metadata as a .npz file.
        """
        np.savez_compressed(
            path,
            topology=np.asarray(self.topology, dtype=np.int32),
            offsets=self.offsets.astype(np.float32),
            end_effectors=np.asarray(self.end_effectors, dtype=np.int32),
            height=np.float32(self.height),
            root_joint=np.int32(self.root_joint),
        )

    @classmethod
    def load(cls, path: str) -> "SkeletonMetadata":
        """
        Load metadata from a .npz file and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        topology = [tuple(x) for x in d["topology"]]
        offsets = d["offsets"]
        end_effectors = d["end_effectors"].tolist()
        height = float(d["height"])
        root_joint = int(d["root_joint"])

        return cls(
            topology=topology,
            offsets=offsets,
            end_effectors=end_effectors,
            height=height,
            root_joint=root_joint,
        )

@dataclass
class MotionSequence:
    rotations: np.ndarray
    positions: Optional[np.ndarray] = None
    fps: float = 30.0
    sequence_name: str = ""

    def save(self, path: str):
        """
        Save motion sequence as compressed .npz.
        """
        np.savez_compressed(
            path,
            rotations=self.rotations.astype(np.float32),
            positions=self.positions.astype(np.float32) if self.positions is not None else None,
            fps=np.float32(self.fps),
            sequence_name=np.string_(self.sequence_name),
        )

    @classmethod
    def load(cls, path: str) -> "MotionSequence":
        """
        Load a motion sequence from disk and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        rotations = d["rotations"]
        positions = d["positions"] if d["positions"] is not None else None

        fps = float(d["fps"])
        sequence_name = str(d["sequence_name"].astype(str))

        return cls(
            rotations=rotations,
            positions=positions,
            fps=fps,
            sequence_name=sequence_name,
        )