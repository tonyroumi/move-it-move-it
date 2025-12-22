import torch

from src.core.types import PairedSample

def paired_collate(batch):
    rotations = tuple(
        torch.stack([b.rotations[i] for b in batch])
        for i in range(2)
    )

    motions = tuple(
        torch.stack([
            pad_root_flat(b.motions[i])
            for b in batch
        ])
        for i in range(2)
    )

    offsets = tuple(
        torch.stack([b.offsets[i] for b in batch])
        for i in range(2)
    )

    heights = tuple(
        torch.stack([b.heights[i] for b in batch])
        for i in range(2)
    )

    gt_positions = tuple(
        torch.stack([b.gt_positions[i] for b in batch])
        for i in range(2)
    )

    gt_ee_vels = tuple(
        torch.stack([b.gt_ee_vels[i] for b in batch])
        for i in range(2)
    )

    return PairedSample(
        rotations=rotations,
        motions=motions,
        offsets=offsets,
        heights=heights,
        gt_positions=gt_positions,
        gt_ee_vels=gt_ee_vels)


def pad_root_flat(motion: torch.Tensor) -> torch.Tensor:
    """
    motion: (T, 3 + 4*J)
    returns: (T, 4 + 4*J)
    """
    D, T = motion.shape

    padded = torch.zeros(D + 1, T, device=motion.device, dtype=motion.dtype)

    # root translation -> xyz0
    padded[1:4, :] = motion[:3, :]

    # shift rotations by 1
    padded[4:, :] = motion[3:, :]

    return padded
