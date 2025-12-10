from dataclasses import dataclass
import numpy as np

@dataclass
class SkeletonMetadata:
    topology: np.ndarray
    offsets: np.ndarray
    ee_ids: np.ndarray
    height: np.ndarray
    kintree: np.ndarray

    def save(self, path: str):
        """
        Save skeleton metadata as a .npz file.
        """
        np.savez_compressed(
            path,
            topology=self.topology,
            offsets=self.offsets,
            ee_ids=self.ee_ids,
            height=self.height,
            kintree=self.kintree,
        )
        print(f"Saving motion sequence: {path}")

    @classmethod
    def load(cls, path: str) -> "SkeletonMetadata":
        """
        Load metadata from a .npz file and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        topology = d["topology"]
        offsets = d["offsets"]
        ee_ids = d["ee_ids"]
        height = d["height"]
        kintree = d["kintree"]
        return cls(
            topology=topology,
            offsets=offsets,
            ee_ids=ee_ids,
            height=height,
            kintree=kintree,
        )

@dataclass
class MotionSequence:
    positions: np.ndarray
    rotations: np.ndarray
    fps: float = 60

    def save(self, path: str):
        """
        Save motion sequence as compressed .npz.
        """
        np.savez_compressed(
            path,
            positions=self.positions,
            rotations=self.rotations,
            fps=self.fps,
        )
        print(f"Saved motion sequence: {path}")

    @classmethod
    def load(cls, path: str) -> "MotionSequence":
        """
        Load a motion sequence from disk and reconstruct the dataclass.
        """
        d = np.load(path, allow_pickle=False)

        positions = d["positions"]
        rotations = d["rotations"]
        fps = d["fps"]

        print(f"Loaded motion sequence file: {path}. Total number of frames: {rotations.shape[0]}. FPS: {fps}")

        return cls(
            positions=positions,
            rotations=rotations,
            fps=fps,
        )