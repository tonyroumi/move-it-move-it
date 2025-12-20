"""
Skeletal GAN for unpaired motion translation from one motion domain to another. 
"""

from dataclasses import dataclass
from .autoencoder import SkeletalAutoEncoder
from .discriminator import SkeletalDiscriminator
from .encoder import SkeletalEncoder

from src.utils import ImagePool

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

@dataclass
class SkeletonTopology:
    edge_topology: torch.Tensor = None
    joint_topology: torch.Tensor = None
    edge_adjacency: torch.Tensor = None
    ee_ids: torch.Tensor = None

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

        self.static_encoder = SkeletalEncoder(adj_init=topology.edge_adjacency,
                                              edge_init=topology.edge_topology,
                                              params=static_encoder_params)
        self.auto_encoder = SkeletalAutoEncoder(adj_init=topology.edge_adjacency,
                                                edge_init=topology.edge_topology,
                                                params=auto_encoder_params)
        
        pooled_info = self.static_encoder.pooling_hierarchy
        self.discriminator = SkeletalDiscriminator(pooled_info=pooled_info,
                                                   discriminator_params=discriminator_params)    
        
        self.buffer = ImagePool(buffer_size)
        
    def generator_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters())

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def discriminators_requires_grad_(self, requires_grad: bool = True):
        for para in self.discriminator_parameters():
            para.requires_grad = requires_grad
    
    def forward_offset(self, offset: torch.Tensor):
        return self.static_encoder(offset.unsqueeze(-1))
    
    def encode_decode_motion(self, motion: torch.Tensor, offset: torch.Tensor):
        return self.auto_encoder(motion, offset)
    
    def encode_motion(self, reconstructed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.encoder(reconstructed, offset=offset)
    
    def decode_latent_motion(self, latent_representation: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.decoder(latent_representation, offset=offset)
    
    def discriminate_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.discriminator(motion)
    
    def query(self, motion: torch.Tensor) -> torch.Tensor:
        return self.buffer.query(motion)

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
    
    def generator_parameters(self):
        return self.domain_A.generator_parameters() + self.domain_B.generator_parameters() 

    def discriminator_parameters(self):
        return list(self.domain_A.discriminator_parameters()) + list(self.domain_B.discriminator_parameters())
      
    def discriminators_requires_grad_(self, requires_grad: bool = True) -> None:
        self.domain_A.discriminators_requires_grad_(requires_grad)
        self.domain_B.discriminators_requires_grad_(requires_grad)
    
    def forward_offsets(
        self, 
        offsets: Tuple[torch.Tensor, torch.Tensor], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        offsets_A = self.domain_A.forward_offset(offsets[0])
        offsets_B = self.domain_B.forward_offset(offsets[1])
        return offsets_A, offsets_B

    def encode_decode_motion(
        self,
        motions: Tuple[torch.Tensor, torch.Tensor],
        offsets: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Auto-encode motion for a given domain. """
        latent_A, reconstructed_motion_A = self.domain_A.encode_decode_motion(motions[0], offsets[0])
        latent_B, reconstructed_motion_B = self.domain_B.encode_decode_motion(motions[1], offsets[1])

        latents = (latent_A, latent_B)
        reconstructed = (reconstructed_motion_A, reconstructed_motion_B)
        return latents, reconstructed

    def decode_latent_motion(
        self,
        latents: torch.Tensor,
        offsets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Decode latent motion for a given domain. """
        reconstructed_motion_A = self.domain_A.decode_latent_motion(latents[0], offsets[0])
        reconstructed_motion_B = self.domain_B.decode_latent_motion(latents[1], offsets[1])

        return reconstructed_motion_A, reconstructed_motion_B

    def encode_motion(
        self,
        motions: torch.Tensor,
        offsets: torch.Tensor
    ):
        """Encode motion into latent representation for a given domain. """
        latent_A = self.domain_A.encode_motion(motions[0], offsets[0])
        latent_B = self.domain_B.encode_motion(motions[1], offsets[1])
        return latent_A, latent_B
    
    def discriminate(self, motions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Evaluate the discriminator for a given domain. """
        discriminator_guess_A = self.domain_A.discriminate_motion(motions[0])
        discriminator_guess_B = self.domain_B.discriminate_motion(motions[1])
        return discriminator_guess_A, discriminator_guess_B
    
    def query(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        motions_A  = self.domain_A.query(motion[0])
        motions_B  = self.domain_B.query(motion[1])
        return motions_A, motions_B
