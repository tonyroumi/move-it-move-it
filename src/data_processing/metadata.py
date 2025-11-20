from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class SkeletonMetadata:
    topology: List[Tuple[int, int]]
    offsets: np.ndarray
    end_effectors: List[int]
    height: float
    joint_names: List[str]
    root_joint: int = 0

@dataclass
class MotionSequence:
    rotations: np.ndarray
    positions: Optional[np.ndarray] = None
    fps: float = 30.0
    sequence_name: str = ""