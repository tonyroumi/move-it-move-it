from dataclasses import dataclass
import numpy as np

@dataclass
class SkeletonMetadata:
    topology: np.ndarray
    offsets: np.ndarray
    end_effectors: np.ndarray
    height: np.ndarray

    def save(self, path: str):
        """
        Save skeleton metadata as a .npz file.
        """
        np.savez_compressed(
            path,
            topology=self.topology, #float32
            offsets=self.offsets, #float32
            end_effectors=self.end_effectors, # int32
            height=self.height, #float32
        )
        print(f"SAVED SKELETON TO: {path}")

    @classmethod
    def load(cls, path: str) -> "SkeletonMetadata":
        """
        Load metadata from a .npz file and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        topology = d["topology"]
        offsets = d["offsets"]
        end_effectors = d["end_effectors"]
        height = d["height"]
        root_joint = d["root_joint"]

        return cls(
            topology=topology,
            offsets=offsets,
            end_effectors=end_effectors,
            height=height,
            root_joint=root_joint,
        )

@dataclass
class MotionSequence:
    root_orient: np.ndarray
    rotations: np.ndarray
    fps: float

    def save(self, path: str):
        """
        Save motion sequence as compressed .npz.
        """
        np.savez_compressed(
            path,
            root_orient=self.root_orient, #float32
            rotations=self.rotations, #float32
            fps=self.fps,  #float32
        )
        print(f"SAVED MOTION TO: {path}")

    @classmethod
    def load(cls, path: str) -> "MotionSequence":
        """
        Load a motion sequence from disk and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        root_orient = d["root_orient"]
        rotations = d["rotations"]
        fps = d["fps"]

        return cls(
            root_orient=root_orient,
            rotations=rotations,
            fps=fps,
        )