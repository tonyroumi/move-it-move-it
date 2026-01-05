"""
Motion Retargeting Script

Retargets motion from a source skeleton to a target skeleton using a trained SkeletalGAN model.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from src.data import SkeletonMetadata, MotionSequence
from src.models.networks import SkeletalGAN
from src.utils import SkeletonVisualizer, ForwardKinematics, ArrayUtils

class MotionRetargeter:
    """Handles motion retargeting between different skeletal structures."""
    
    def __init__(self, model_path: str, window_size: int = 64):
        """
        Initialize the retargeter.
        
        Args:
            model_path: Path to the trained SkeletalGAN model
            window_size: Size of motion windows for processing
        """
        self.model = SkeletalGAN.load_for_inference(model_path)
        self.window_size = window_size
        self.window_step = window_size // 2
    
    def create_motion_windows(self, motion_data: np.ndarray) -> np.ndarray:
        """
        Slice motion array into overlapping windows.
        
        Args:
            motion_data: Motion array of shape [T, features]
            
        Returns:
            Windowed motion of shape [num_windows, window_size, features]
        """
        T = motion_data.shape[0]
        num_windows = T // self.window_step - 1
        windows = []
        
        for i in range(num_windows):
            start = i * self.window_step
            end = start + self.window_size
            if end > T:
                break
            windows.append(motion_data[start:end])
        
        return np.stack(windows) if windows else np.array([])
    
    def prepare_motion(self, motion_seq: MotionSequence) -> torch.Tensor:
        """
        Prepare motion sequence for model input.
        
        Args:
            motion_seq: Source motion sequence
            
        Returns:
            Normalized windowed motion tensor
        """
        # Combine rotations and root position
        rots = ArrayUtils.to_torch(motion_seq.rotations).flatten(-2)
        root_pos = ArrayUtils.to_torch(motion_seq.positions[:, 0])
        full_motion = torch.hstack([rots, root_pos])
        
        # Create windows and normalize
        windowed = ArrayUtils.to_torch(self.create_motion_windows(full_motion))
        windowed = windowed.transpose(1, 2)
        return self.model.domains[0].norm(windowed)
    
    def retarget(
        self,
        motion_seq: MotionSequence,
        source_skeleton: SkeletonMetadata,
        target_skeleton: SkeletonMetadata
    ) -> np.ndarray:
        """
        Retarget motion from source to target skeleton.
        
        Args:
            motion_seq: Source motion sequence
            source_skeleton: Source skeleton metadata
            target_skeleton: Target skeleton metadata
            
        Returns:
            Retargeted global positions [B, T, J, 3]
        """
        # Prepare inputs
        motion_windows = self.prepare_motion(motion_seq)
        source_offsets = ArrayUtils.to_torch(
            source_skeleton.offsets.reshape(-1)
        ).unsqueeze(0)
        target_offsets = ArrayUtils.to_torch(
            target_skeleton.offsets.reshape(-1)
        ).unsqueeze(0)
        
        # Translate motion
        translated_motion = self.model.translate(
            motion_windows, source_offsets, target_offsets
        )
        
        # Compute forward kinematics
        B, _, T = translated_motion.shape
        rotations = translated_motion[:, :-3].reshape(B, -1, 4, T).permute(0, 3, 1, 2)
        root_positions = translated_motion[:, -3:]
        
        global_positions = ForwardKinematics.forward_batched(
            rotations,
            ArrayUtils.to_torch(target_skeleton.offsets.reshape(-1, 3)).unsqueeze(0),
            root_positions,
            target_skeleton
        )
        
        return global_positions


def setup_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        'root': output_dir,
        'videos': output_dir / 'videos',
        'skeletons': output_dir / 'skeletons'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Retarget motion between different skeletal structures'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained SkeletalGAN model'
    )
    parser.add_argument(
        '--source-skeleton',
        type=str,
        required=True,
        help='Path to source skeleton metadata (.npz)'
    )
    parser.add_argument(
        '--target-skeleton',
        type=str,
        required=True,
        help='Path to target skeleton metadata (.npz)'
    )
    parser.add_argument(
        '--motion',
        type=str,
        required=True,
        help='Path to source motion sequence (.npz)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/retargeting',
        help='Output directory for visualizations (default: outputs/retargeting)'
    )
    parser.add_argument(
        '--source-foot-idx',
        type=int,
        default=20,
        help='Foot joint index for source skeleton (default: 20)'
    )
    parser.add_argument(
        '--source-head-idx',
        type=int,
        default=4,
        help='Head joint index for source skeleton (default: 4)'
    )
    parser.add_argument(
        '--target-foot-idx',
        type=int,
        default=15,
        help='Foot joint index for target skeleton (default: 15)'
    )
    parser.add_argument(
        '--target-head-idx',
        type=int,
        default=10,
        help='Head joint index for target skeleton (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main retargeting pipeline."""
    args = parse_args()
    
    # Setup output directories
    output_dirs = setup_output_dirs(Path(args.output_dir))
    
    # Extract names for file naming
    source_name = Path(args.source_skeleton).stem
    target_name = Path(args.target_skeleton).stem
    motion_name = Path(args.motion).stem
    
    print(f"Loading model from: {args.model}")
    print(f"Source skeleton: {source_name}")
    print(f"Target skeleton: {target_name}")
    print(f"Motion: {motion_name}")
    print(f"Output directory: {output_dirs['root']}")
    print("-" * 60)
    
    # Load data
    print("Loading skeletons and motion...")
    source_skeleton = SkeletonMetadata.load(args.source_skeleton)
    target_skeleton = SkeletonMetadata.load(args.target_skeleton)
    source_motion = MotionSequence.load(args.motion)
    
    # Initialize retargeter
    print("Initializing retargeter...")
    retargeter = MotionRetargeter(args.model)
    
    # Perform retargeting
    print("Retargeting motion...")
    retargeted_positions = retargeter.retarget(
        source_motion,
        source_skeleton,
        target_skeleton
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Source skeleton T-pose
    source_skeleton_path = output_dirs['skeletons'] / f"{source_name}_tpose.png"
    SkeletonVisualizer.visualize_skeleton(
        global_position=source_motion.positions[0],
        height=source_skeleton.height,
        foot_idx=args.source_foot_idx,
        head_idx=args.source_head_idx,
        save_path=str(source_skeleton_path)
    )
    print(f"  ✓ Saved source skeleton: {source_skeleton_path.relative_to(output_dirs['root'])}")
    
    # Target skeleton T-pose (if motion available)
    # Note: This would require target motion data, skipping for now
    
    # Source motion video
    source_motion_path = output_dirs['videos'] / f"{source_name}_{motion_name}_original.mp4"
    SkeletonVisualizer.visualize_motion(
        source_motion.positions,
        str(source_motion_path)
    )
    print(f"  ✓ Saved source motion: {source_motion_path.relative_to(output_dirs['root'])}")
    
    # Retargeted motion video
    retargeted_path = output_dirs['videos'] / f"{source_name}_to_{target_name}_{motion_name}.mp4"
    SkeletonVisualizer.visualize_motion(
        retargeted_positions,
        str(retargeted_path)
    )
    print(f"  ✓ Saved retargeted motion: {retargeted_path.relative_to(output_dirs['root'])}")
    
    print("-" * 60)
    print("Retargeting complete!")
    print(f"All outputs saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()