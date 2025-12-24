"""
Trainer for an unpaired skeletal motion GAN.
"""
from .losses import LossBundle

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple, List
import torch

from src.core.types import MotionOutput, PairedSample
from src.skeletal.models.gan import SkeletalGAN
from src.utils import Logger, ImagePool

@dataclass
class TrainConfig:
    num_epochs: int = 5
    checkpoint_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    buffer_size: int = 50

class SkeletalGANTrainer:
    def __init__(
        self,
        model: SkeletalGAN,
        losses: LossBundle,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        # scheduler_G: torch.optim.lr_scheduler,
        # scheduler_D: torch.optim.lr_scheduler,
        train_loader: DataLoader,
        config: TrainConfig,
        logger: Logger = Logger(),
        device: torch.device = 'cpu',
    ):
        self.model : SkeletalGAN = model.to(device)

        self.losses = losses
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        # self.scheduler_G = scheduler_G
        # self.scheduler_D = scheduler_D

        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.logger = logger

        self.image_pools : List[ImagePool] = [
            ImagePool(config.buffer_size) 
            for _ in range(len(self.model.topologies))
        ]

        self.logger.info(f"Trainer Initialized on: {device}")
    
    def train(self) -> None:
        total_loss = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            epoch_loss += self._train_one_epoch()
            
            total_loss += epoch_loss / (epoch+1)

            self.logger.info(f"Epoch: {epoch}, Total Loss: {total_loss}")
            self.logger.log_metric("total_loss", total_loss)

            self._step_schedulers()
            self.logger.epoch()

            if (epoch % self.config.checkpoint_interval == 0):
                self._save_checkpoint(epoch)
    
    def _train_one_epoch(self) -> Any:
        total_epoch_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)

            rec_outputs, ret_outputs = self.model(batch)

            # Generator
            self.model.discriminators_requires_grad_(False)
            self.optimizer_G.zero_grad()
            generator_loss = self._backward_G(rec_outputs, ret_outputs, batch=batch)
            self.optimizer_G.step()

            # Discriminator
            self.model.discriminators_requires_grad_(True)
            self.optimizer_D.zero_grad()
            discriminator_loss = self._backward_D(ret_outputs, original_world_pos=batch.gt_positions)
            self.optimizer_D.step()

            total_epoch_loss += (generator_loss + discriminator_loss)
    
            self.logger.step()

        return total_epoch_loss

    def _backward_D(
        self,
        ret_outputs: Dict[Tuple[int, int], MotionOutput],
        original_world_pos: Tuple[torch.Tensor, torch.Tensor] 
    ) -> torch.Tensor:
        """
        A->A, A->B, B->A, B->B
        """
        tot_D_loss = 0.0
        for (src, dst), out in ret_outputs.items(): #TODO(anthony) unsure if we want to do this for each cross section. only want A->A, A->B. sum like that 
            pred_fake = self.model.forward_discriminator(
                self.image_pools[dst].query(out.positions.flatten(start_dim=-2)).detach(),
                idx=src
            )

            pred_real = self.model.forward_discriminator(
                original_world_pos[dst].flatten(start_dim=-2),
                idx=src
            )

            loss_D = self.losses.lsgan(d_args=pred_real, g_args=pred_fake)
            loss_D.backward()

            tot_D_loss += loss_D

        return tot_D_loss       

    def _backward_G(
        self, 
        rec_outputs: Dict[int, MotionOutput],
        ret_outputs: Dict[Tuple[int, int], MotionOutput],
        batch: PairedSample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -----------------------
        # Reconstruction loss
        # -----------------------
        rec_loss = 0
        for i, out in rec_outputs.items():
            rec_motion_loss = self.losses.mse(pred=out.motion, gt=batch.motions[i])
            self.logger.log_metric(f"rec_motion_loss_{i}", rec_motion_loss)

            original_root_pos = batch.rotations[i][:, -3:] / batch.heights[i][:, None, None]
            rec_root_pos = out.rotations[:, -3:] / batch.heights[i][:, None, None]
            rec_root_pos_loss = self.losses.mse(
                pred=rec_root_pos,
                gt=original_root_pos
            ) 
            self.logger.log_metric(f"rec_root_pos_loss_{i}", value=rec_root_pos_loss)

            original_world_pos = batch.gt_positions[i] / batch.heights[i][:, None, None, None]
            rec_world_pos = out.positions / batch.heights[i][:, None, None, None]
            rec_joint_pos_loss = self.losses.mse(pred=rec_world_pos, gt=original_world_pos)
            self.logger.log_metric(f"rec_joint_pos_loss_{i}", value=rec_joint_pos_loss)

            rec_loss += rec_motion_loss + (rec_root_pos_loss * 2.5 + rec_joint_pos_loss) * 100

        # -----------------------
        # Retargetting loss
        # -----------------------
        tot_cycle_loss, tot_ee_loss, tot_G_loss = 0, 0, 0
        for (src, dst), out in ret_outputs.items():
            cycle_loss = self.losses.mae(pred=rec_outputs[src].latents, gt=out.latents)
            self.logger.log_metric(f"cycle_loss_{src}->{dst}", value=cycle_loss)
            tot_cycle_loss += cycle_loss

            ee_loss = self.losses.ee(pred=out.ee_vels, gt=batch.gt_ee_vels[src])
            self.logger.log_metric(f"ee_loss_{src}->{dst}", value=ee_loss)
            tot_ee_loss += ee_loss

            if src != dst:
                G_loss = self.losses.lsgan(
                    g_args=self.model.forward_discriminator(out.positions.flatten(start_dim=-2), dst)
                )
                self.logger.log_metric(f"g_loss_{src}->{dst}", value=G_loss)
                tot_G_loss += G_loss
        
        total_G_loss = rec_loss * 5 + \
                       tot_cycle_loss * 2.5 + \
                       tot_ee_loss + 50 * \
                       tot_G_loss
        total_G_loss.backward()

        return total_G_loss

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            # "scheduler_G": self.scheduler_G.state_dict(),
            # "scheduler_D": self.scheduler_D.state_dict(),
        }

        path = ckpt_dir / f"skeletal_gan_epoch{epoch:03d}.pt"
        torch.save(state, path)

        self.logger.info(f"Checkpoint saved to: {path}")

    # def _step_schedulers(self) -> None:
    #     self.scheduler_G.step()
    #     self.scheduler_D.step()
