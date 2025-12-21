"""
Skeletal GAN for unpaired motion translation from one motion domain to another. 
"""

from dataclasses import dataclass
from .autoencoder import SkeletalAutoEncoder
from .discriminator import SkeletalDiscriminator
from .encoder import SkeletalEncoder

from src.dataset import PairedSample
from src.utils import ImagePool
from src.utils import Logger, ForwardKinematics, SkeletonUtils

from typing import Dict, Any, Tuple, Literal, List

import torch
import torch.nn as nn

@dataclass
class SkeletonTopology:
    edge_topology: torch.Tensor = None
    joint_topology: torch.Tensor = None
    edge_adjacency: torch.Tensor = None
    ee_ids: torch.Tensor = None

@dataclass
class MotionOutput:
    motion: torch.Tensor          # normalized motion
    latents: torch.Tensor
    rotations: torch.Tensor       # denormalized
    positions: torch.Tensor       # FK world positions
    ee_vels: torch.Tensor

@dataclass
class RetargetOutput:
    src_domain: int
    dst_domain: int
    output: MotionOutput

class SkeletalDomainModule(nn.Module):
    def __init__(
        self,
        topology: SkeletonTopology,
        static_encoder_params: Dict[str, Any],
        auto_encoder_params: Dict[str, Any],
        discriminator_params: Dict[str, Any],
        buffer_size: int = 50
    ):
        super().__init__()

        self.topology = topology
        self.num_joints = len(topology.joint_topology)
        self.num_characters = 1

        self.static_encoder = SkeletalEncoder(adj_init=topology.edge_adjacency,
                                              edge_init=topology.edge_topology,
                                              params=static_encoder_params)
        self.auto_encoder = SkeletalAutoEncoder(adj_init=topology.edge_adjacency,
                                                edge_init=topology.edge_topology,
                                                params=auto_encoder_params)
        
        pooled_info = self.static_encoder.pooling_hierarchy
        self.discriminator = SkeletalDiscriminator(pooled_info=pooled_info,
                                                   discriminator_params=discriminator_params)    
        
        self.fake_pool = ImagePool(buffer_size)
    
    def forward(self, motions: torch.Tensor, offsets: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.auto_encoder(motions, offsets)

    def forward_offset(self, offset: torch.Tensor):
        return self.static_encoder(offset.unsqueeze(-1))

    def encode(self, reconstructed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.encoder(reconstructed, offset=offset)
    
    def decode(self, latent_representation: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.decoder(latent_representation, offset=offset)

    def denorm(self, motions) -> torch.Tensor:
        pass
        
    def generator_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters())

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def discriminators_requires_grad_(self, requires_grad: bool = True):
        for para in self.discriminator_parameters():
            para.requires_grad = requires_grad
    
    def discriminate_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.discriminator(motion)
    
    def query(self, motion: torch.Tensor) -> torch.Tensor:
        return self.fake_pool.query(motion)

class SkeletalGAN(nn.Module):   
    def __init__(
        self,
        topologies: Tuple[SkeletonTopology, SkeletonTopology],
        gan_params: Dict[str, Any],
    ):
        super().__init__()     

        self.topologies = topologies
        
        self.domain_A = SkeletalDomainModule(self.topologies[0], **gan_params)
        self.domain_B = SkeletalDomainModule(self.topologies[1], **gan_params)
        self.domains = nn.ModuleList([self.domain_A, self.domain_B])
    
    def forward(self, batch: PairedSample, phase: Literal["reconstruction", "retarget"]):
        B, _, T = batch.motions[0].shape

        outputs = []
        offset_features = self._forward_offsets(offsets=batch.offsets)

        for i, domain in enumerate(self.domains):
            latents, reconstructed = domain(batch.motions, offset_features[i], i)
            
            original_rot = domain.denorm(batch.motions[i])
            reconstructed_rot = domain.denorm(reconstructed)

            orig_pos = ForwardKinematics.forward_batched(
                quaternions=original_rot.reshape(B, T, domain.num_joints-1, 4),
                offsets=batch.offsets[i].reshape(B, -1, 3),
                root_pos=original_rot[:, :3],
                topology=domain.topology,
                world=True
            ) #False

            reconstructed_pos = ForwardKinematics.forward_batched(
                quaternions=reconstructed_rot.reshape(B, T, domain.num_joints-1, 4), 
                offsets=batch.offsets[i].reshape(B, -1, 3), 
                root_pos=reconstructed_rot[:, :3],
                topology=domain.topology,
                world=True
            )

            ee_vels = SkeletonUtils.get_ee_velocity(orig_pos, topology=domain.topology)
            ee_vels /= batch.heights[:, None, None, None]

            outputs.append({
                "latents": latents,
                "reconstructed": reconstructed,
                "orig_pos": orig_pos,
                "recon_pos": reconstructed_pos,
                "ee_vels": ee_vels,
            })

        if phase == "reconstruction":
            return {"reconstruction": outputs}

        elif phase == "retarget":
            return self._retarget(outputs, offset_features)

    def _retarget(self, outputs: Dict[str, torch.Tensor], offset_features: Tuple[torch.Tensor], heights: Tuple[torch.Tensor]) -> Dict[Tuple[int], RetargetOutput]:
        retargets = {}
        for i, src in enumerate(self.domains):
            for j, dst in enumerate(self.domains):
                retargetted_motion = dst.decode(outputs[src]["latents"], offset_features[j])
                retargetted_latents = dst.encode(retargetted_motion, offset_features[j])

                retargetted_rots = dst.denorm(retargetted_motion)
                retargetted_pos = ForwardKinematics.forward_batched(
                    quaternions=retargetted_rots.reshape(B, T, dst.num_joints-1, 4),
                    offsets=offset_features[j][0].reshape(B, -1, 3),
                    root_pos=retargetted_rots[:, :3],
                    topologies=dst.topology,
                    world=True
                ) 

                retargetted_ee_vels = SkeletonUtils.get_ee_velocity(retargetted_pos, topology=dst.topology)
                retargetted_ee_vels /= heights[j][:, None, None, None]

                retargets[(i, j)] = RetargetOutput(
                    src_domain=i,
                    dst_domain=j,
                    output=MotionOutput(
                        motion=retargetted_motion,
                        latents=retargetted_latents,
                        rotations=retargetted_rots,
                        positions=retargetted_pos,
                        ee_vels=retargetted_ee_vels,
                    ),
                )
        return retargets
    
    def generator_parameters(self):
        return self.domain_A.generator_parameters() + self.domain_B.generator_parameters() 

    def discriminator_parameters(self):
        return list(self.domain_A.discriminator_parameters()) + list(self.domain_B.discriminator_parameters())
      
    def discriminators_requires_grad_(self, requires_grad: bool = True) -> None:
        self.domain_A.discriminators_requires_grad_(requires_grad)
        self.domain_B.discriminators_requires_grad_(requires_grad)
    
    def _forward_offsets(
        self, 
        offsets: Tuple[torch.Tensor, torch.Tensor], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        offsets_A = self.domain_A.forward_offset(offsets[0])
        offsets_B = self.domain_B.forward_offset(offsets[1])
        return offsets_A, offsets_B