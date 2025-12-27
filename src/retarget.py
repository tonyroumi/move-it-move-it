from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

from src.data import SkeletonMetadata, MotionSequence
from src.data import AMASSTAdapter
from src.data.datasets import CrossDomainMotionDataset, MotionDataset, MotionDatasetBuilder, paired_collate
from src.models.networks import SkeletalGAN
from src.training import SkeletalGANTrainer
from src.utils import set_seed, Logger, SkeletonVisualizer, ForwardKinematics, ArrayUtils, DataUtils
import numpy as np

def get_windows(data):
        """ 
        Slice a motion array of shape [T, J*4+3] into windows of shape [window_size, J*4+3].

        Returns:
          [num_windows, window_size, J*4+7].
        """
        T = data.shape[0]
        step = 64 // 2

        n_window = T // step - 1
        windows = []

        for i in range(n_window):
            start = i * step
            end = start + 64
            if end > T:
                break
            windows.append(data[start:end])

        return np.stack(windows)

def main() -> None:
    
    model = SkeletalGAN.load_for_inference("outputs/2025-12-26/20-09-25/skeletal_gan_epoch000.pt")

    Aude = SkeletonMetadata.load("data/skeletons/Aude.npz")
    Karim = SkeletonMetadata.load("data/skeletons/Karim.npz")

    Aude_motion = MotionSequence.load("data/amass/processed/Aude/INF_Basketball_S2_01_poses.npz")

    rots = ArrayUtils.to_torch(Aude_motion.rotations).flatten(-2)
    root_pos = ArrayUtils.to_torch(Aude_motion.positions[:, 0])
    full_motion = torch.hstack([rots, root_pos])
    full_motion = ArrayUtils.to_torch(get_windows(full_motion)).transpose(1,2)
    full_motion = model.domains[0].norm(full_motion)

    aude_off = ArrayUtils.to_torch( Aude.offsets.reshape(-1)).unsqueeze(0)
    karim_off = ArrayUtils.to_torch( Karim.offsets.reshape(-1)).unsqueeze(0)

    rotations = model.translate(full_motion, aude_off, karim_off)
    B, _, T = rotations.shape
    pos = ForwardKinematics.forward_batched(
         rotations[:, :-3]
          .reshape(B, -1, 4, T)
          .permute(0, 3, 1, 2),
         ArrayUtils.to_torch(Karim.offsets.reshape(-1, 3)).unsqueeze(0),
         rotations[:, -3:],
         Karim)
    SkeletonVisualizer.visualize_motion(pos, "KARIM/ret.mp4")

if __name__ == "__main__":
    import debugpy
    print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    print("[DEBUG] Debugger attached.")
    
    main()