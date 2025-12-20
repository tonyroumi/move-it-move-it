"""
Trainer for an unpaired skeletal motion GAN.
"""

from src.data_processing import SkeletonMetadata
from src.skeletal_models import SkeletalGAN, SkeletonTopology
from src.utils import Logger, ForwardKinematics, SkeletonUtils
from .losses import LossBundle

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    num_epochs: int = 5
    checkpoint_interval: int = 1
    checkpoint_dir: str = "./checkpoints"

class SkeletalGANTrainer:
    def __init__(
        self,
        model: SkeletalGAN,
        losses: LossBundle,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: torch.optim.lr_scheduler,
        scheduler_D: torch.optim.lr_scheduler,
        train_loader: DataLoader,
        config: TrainConfig,
        logger: Logger = Logger(),
        device: torch.device = 'cpu',
    ):
        self.model : SkeletalGAN = model.to(device)
        self.topologies : List[SkeletonTopology] = self.model.topologies
        self.num_topologies = len(self.topologies)

        self.losses = losses
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D

        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.logger = logger

        self.denorm = train_loader.dataset.denorm

        self.logger.info(f"Trainer Initialized on: {device}")
    
    def train(self) -> None:
        total_loss = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss += self._train_one_epoch(epoch)

            self.logger.log(f"Epoch: {epoch}, Total Loss: {total_loss/{epoch+1}}")

            self._step_schedulers()
            self.logger.epoch()

            if (epoch % self.config.checkpoint_interval == 0):
                self._save_checkpoint(epoch)
    
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        for step, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            offset_features = self.model.forward_offsets(offsets=batch.offsets)
            
            (latents, 
             reconstructed,
             original_rot,
             reconstructed_rot,
             original_quats,
             reconstructed_quats,
             original_world_pos,
             reconstructed_world_pos,
             ee_vels) = self._reconstruct(motions=batch.motions, offsets=batch.offsets, offset_features=offset_features, heights=batch.heights)

            (retargeted_reconstruction,
             retargeted_latents,
             retargeted_world_pos,
             retargeted_ee_vels) = self._retarget(latents=latents, offsets=batch.offsets, offset_features=offset_features, heights=batch.heights)

            # Generator
            self.model.discriminators_requires_grad_(False)
            self.optimizer_G.zero_grad()
            self._backward_G(batch.motions, reconstructed,
                             original_rot, reconstructed_rot,
                             original_world_pos, reconstructed_world_pos,
                             latents, retargeted_latents,
                             ee_vels, retargeted_ee_vels,
                             batch.heights)
            self.optimizer_G.step()

            # Discriminator
            self.model.discriminators_requires_grad_(True)
            self.optimizer_D.zero_grad()
            self._backward_D(original_world_pos, reconstructed_world_pos)
            self.optimizer_D.step()

            self.logger.step()
            
    def _reconstruct(
        self,
        motions: Tuple[torch.Tensor, torch.Tensor], 
        offsets: Tuple[torch.Tensor, torch.Tensor], 
        offset_features: Tuple[torch.Tensor, torch.Tensor],
        heights: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encodes paired motion sequences and reconstructs them, computing 
        reconstruction losses on rotations, root translation, and joint world-space positions
        via forward kinematics.
        """
        #TODO(anthony) loop once at the top. modify functions appropraitely. 
        B, C, T = motions[0].shape
        num_joints = C // 4

        latents, reconstructed = self.model.encode_decode_motion(motions=motions, 
                                                                 offsets=offset_features)

        original_rot = self.denorm(motions)
        reconstructed_rot = self.denorm(reconstructed)

        #Prepare inputs for fk
        original_root_pos = [r[:, :3] for r in original_rot]
        reconstructed_root_pos = [r[:, :3] for r in reconstructed_rot]

        original_quats = [
            r[:, 3:].reshape(B, T, num_joints-1, 4)
            for r in original_rot
        ]
        reconstructed_quats = [
            r[:, 3:].reshape(B, T, num_joints-1, 4)
            for r in reconstructed_rot
        ]
        offsets = [o.reshape(B, -1, 3) for o in offsets]

        original_world_pos = ForwardKinematics.forward_batched(
            quaternions=original_quats,
            offsets=offsets,
            root_pos=original_root_pos,
            topologies=self.topologies,
            world=True
        ) #False

        reconstructed_world_pos = ForwardKinematics.forward_batched(
            quaternions=reconstructed_quats, 
            offsets=offsets, 
            root_pos=reconstructed_root_pos,
            topologies=self.topologies, 
            world=True
        )

        ee_vels = SkeletonUtils.get_ee_velocity(original_world_pos, topologies=self.topologies)
        ee_vels = tuple(
            v / h[:, None, None, None]
            for v, h in zip(ee_vels, heights)
        )
        
        return (latents, 
                reconstructed,
                original_rot,
                reconstructed_rot,
                original_quats,
                reconstructed_quats,
                original_world_pos,
                reconstructed_world_pos,
                ee_vels)

    def _retarget(
        self,
        latents: Tuple[torch.Tensor, torch.Tensor],
        offsets: Tuple[torch.Tensor, torch.Tensor],
        offset_features: Tuple[torch.Tensor, torch.Tensor],
        heights: Tuple[torch.Tensor]
    ):
        """
        # Same domain mapping and cross domain mapping:
        # False: Same domain model inputs
        # True (reversed): Cross domain model inputs
        """
        B, T, C = 256, 64, 91
        num_joints = 22
        #This is incorrect need to store the latents and outputs appropriately. 
        for rev in (False, True):
            latent_domains = latents[::-1] if rev else latents
            offset_domains = offsets[::-1] if rev else offsets
            offset_feature_domains = offset_features[::-1] if rev else offset_features
            height_domains = heights[::-1] if rev else heights

            retargeted_reconstruction = self.model.decode_latent_motion(latent_domains, offset_feature_domains)
            retargeted_latents = self.model.encode_motion(retargeted_reconstruction, offsets=offset_feature_domains)

            #Prepare inputs for fk
            retargeted_rotations = self.denorm(retargeted_reconstruction)

            retargeted_root_pos = [r[:, :3] for r in retargeted_rotations]

            retargeted_quats = [
                r[:, 3:].reshape(B, T, num_joints-1, 4)
                for r in retargeted_rotations
            ]
            offsets = [o.reshape(B, -1, 3) for o in offset_domains]

            retargeted_world_pos = ForwardKinematics.forward_batched(
                quaternions=retargeted_quats,
                offsets=offsets,
                root_pos=retargeted_root_pos,
                topologies=self.topologies,
                world=True
            ) #False

            retargeted_ee_vels = SkeletonUtils.get_ee_velocity(retargeted_world_pos, topologies=self.topologies)
            retargeted_ee_vels = tuple(
                v / h[:, None, None, None]
                for v, h in zip(retargeted_ee_vels, height_domains)
            )

            return (retargeted_reconstruction,
                    retargeted_latents,
                    retargeted_world_pos,
                    retargeted_ee_vels)

    def _backward_D(
        self,
        original_world_pos, reconstructed_world_pos
    ):
        loss_D = 0
        for skel in range(self.num_topologies):
            fake = self.model.query(reconstructed_world_pos[2 - skel])
            pred_real = self.model.discriminate(original_world_pos)
            real_D_loss = self.losses.lsgan(pred_real, True)

            pred_fake = self.model.discriminate(fake.detach())
            fake_D_loss = self.losses.lsgan(pred_fake, False)

            loss_D += (real_D_loss + fake_D_loss) * 0.5

        loss_D.backward()
         

    def _backward_G(
        self, 
        motions, reconstructed,
        original_rot, reconstructed_rot,
        original_world_pos, reconstructed_world_pos,
        latents, retargeted_latents,
        ee_vels, retargeted_ee_vels,
        heights
    ) -> None:
        #Reconstruction losses
        rec_motion_loss = self.losses.mse(preds=reconstructed, gts=motions)
        self.logger.log_metric("rec_motion_loss", rec_motion_loss)

        original_root_pos = [r[:, :3] / heights[i] for i, r in enumerate(original_rot)]
        reconstructed_root_pos = [r[:, :3] / heights[i] for i, r in enumerate(reconstructed_rot)]
        rec_root_pos_loss = self.losses.mse(preds=reconstructed_root_pos,
                                            gts=original_root_pos) 
        self.logger.log_metric("rec_root_pos_loss", value=rec_root_pos_loss)

        original_world_pos = [r / heights[i] for i, r in enumerate(original_world_pos)]
        reconstructed_world_pos = [r / heights[i] for i, r in enumerate(reconstructed_world_pos)]
        rec_joint_pos_loss = self.losses.mse(preds=reconstructed_world_pos, gts=original_world_pos)
        self.logger.log_metric("rec_joint_pos_loss", value=rec_joint_pos_loss)

        rec_loss = rec_motion_loss + (rec_root_pos_loss * 2.5 + rec_joint_pos_loss) * 100

        #Retargeting losses
        for src in range(self.num_topologies):
            for dst in range(self.num_topologies):
                cycle_loss = self.losses.mae(latents, retargeted_latents) #This is incorrect
                self.logger.log_metric("cycle_loss", value=cycle_loss)

                ee_loss = self.losses.ee(ee_vels, retargeted_ee_vels)
                self.logger.log_metric("ee_loss", value=ee_loss)

                if src != dst:
                    G_loss = self.losses.lsgan(g_args=self.model.discriminate(reconstructed_world_pos))
                    self.logger.log_metric("g_loss", value=G_loss)
        
        total_G_loss = rec_loss * 5 + \
                       cycle_loss * 2.5 + \
                       ee_loss + 50 * \
                       G_loss

        total_G_loss.backward()


    
    def verbose(self):
        res = {'rec_loss_0': self.rec_losses[0].item(),
               'rec_loss_1': self.rec_losses[1].item(),
               'cycle_loss': self.cycle_loss.item(),
               'ee_loss': self.ee_loss.item(),
               'D_loss_gan': self.loss_D.item(),
               'G_loss_gan': self.loss_G.item()}
        return sorted(res.items(), key=lambda x: x[0])

    def _save_checkpoint(self, epoch: int) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "scheduler_G": self.scheduler_G.state_dict(),
            "scheduler_D": self.scheduler_D.state_dict()
        }

        path = f"{self.config.checkpoint_dir}/skeletal_gan_epoch{epoch:03d}.pt"
        torch.save(state, path)

        self.logger.log(f"Checkpoint saved: {path}")

    def _step_schedulers(self) -> None:
        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()