import torch
import torch.nn as nn

class LSGANLoss(nn.Module):
    """
    Least Squares GAN loss.

    Discriminator targets:
        real -> 1
        fake -> 0

    Generator target:
        fake -> 1
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def d_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        real_targets = torch.ones_like(d_real)
        fake_targets = torch.zeros_like(d_fake)

        loss_real = self.mse(d_real, real_targets)
        loss_fake = self.mse(d_fake, fake_targets)

        return 0.5 * (loss_real + loss_fake)

    def g_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        real_targets = torch.ones_like(d_fake)
        return self.mse(d_fake, real_targets)

class EELoss(nn.Module):
    """ End-Effector velocity loss. """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, norm_eps=0.008) -> torch.Tensor:
        ee_loss = self.mse(pred, gt)
 
        gt_norm = torch.norm(gt, dim=-1)
        contact_idx = gt_norm < self.norm_eps
        extra_ee_loss = self.mse(pred[contact_idx], gt[contact_idx])
 
        return ee_loss + extra_ee_loss * 100