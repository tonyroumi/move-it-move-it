"""
Trainer for an unpaired skeletal motion GAN.
"""

from src.skeletal_models import SkeletalGAN
from src.utils import Logger

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    num_epochs: int
    checkpoint_interval: int = 1
    checkpoint_dir: str = "./checkpoints"

class SkeletalGANTrainer:
    def __init__(
        self,
        model: SkeletalGAN,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: torch.optim.lr_scheduler,
        scheduler_D: torch.optim.lr_scheduler,
        train_loader: DataLoader,
        config: TrainConfig,
        logger: Logger = Logger(),
        device: torch.device = 'cpu',
    ):
        self.model = model.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D

        self.denom = train_loader.dataset.denorm

        self.logger.log(f"Trainer Initialized on: {device}")
    
    def train(self) -> None:
        for epoch in range(self.config.num_epochs):
            self.logger.log(f"EPOCH: {epoch}")
            self.model.train()
            self._train_one_epoch(epoch)
            self._step_schedulers()

            if (epoch % self.config.checkpoint_interval == 0):
                self._save_checkpoint(epoch)
    
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            self.denom(motion_A)
            motion_A, offsets_A, height_A = batch[0]
            motion_B, offsets_B, height_B = batch[1]
            # PICK BACK UP HERE
            # Im going to need to implement imagePool, forward/inverse kinematic SPEND TIME HERE, valididate static_encoder config. 
            # Looooking good.



            offset_features = self.model.encode_offsets(batch.offsets)

            motion_denorm = batch.motion.denorm

            latent, reconstructed = self.model.encode_decode_motion(batch["motion"], offset_features)

            reconstructed.denorm()

            # we need forward kinematics here.
            # self.fk.forward(reconstructed, batch.offsets)




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