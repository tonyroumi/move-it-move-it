from .metadata import SkeletonMetadata, MotionSequence

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import torch

class DataSourceAdapter(ABC):
    """Abstract base class for processing different motion capture data sources"""
    def __init__(self, dataset_name: str, device: torch.device):
        root = Path(__file__).resolve().parent.parent.parent

        data_dir = root / "data"

        self.skeleton_dir = data_dir / "skeletons"
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories inside data
        self.dataset_dir = data_dir / dataset_name
        self.raw_dir = self.dataset_dir / "raw"
        self.cache_dir = self.dataset_dir / "processed"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = device

    @abstractmethod
    def download(self, **kwargs) -> None:
        """ Abstract method to download mocap data. """
        pass
    
    @abstractmethod
    def extract_skeleton(self, file_path: str) -> SkeletonMetadata:
        """ Abstract method to extract skeleton metadata. """
        pass
    
    @abstractmethod
    def extract_motion(self, file_path: str) -> List[MotionSequence]:
        """ Abstract method to extract a sequence of motions """
        pass