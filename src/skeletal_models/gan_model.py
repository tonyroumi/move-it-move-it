from .autoencoder import SkeletalAutoEncoder
from .discriminator import SkeletalDiscriminator
from .encoder import SkeletalEncoder

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

class SkeletalDomainModule(nn.Module):
    def __init__(
        self,
        topology: Dict[str, Any],
        static_encoder_params: Dict[str, Any],
        auto_encoder_params: Dict[str, Any],
        discriminator_params: Dict[str, Any],
    ):
        super().__init__()
        self.static_encoder = SkeletalEncoder(adj_init=topology['adj'],
                                              edge_init=topology['edges'],
                                              encoder_params=static_encoder_params)
        self.auto_encoder = SkeletalAutoEncoder(adj_init=topology['adj'],
                                                edge_init=topology['edges'],
                                                auto_encoder_params=auto_encoder_params)
        
        pooled_info = self.static_encoder.pooling_hierarchy
        self.discriminator = SkeletalDiscriminator(pooled_info=pooled_info,
                                                   discriminator_params=discriminator_params)
        
    def generator_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters())

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def encode_offsets(self, offset: torch.Tensor):
        return self.static_encoder(offset)
    
    def encode_decode_motion(self, motion: torch.Tensor, offset: torch.Tensor):
        return self.auto_encoder(motion, offset)
    
    def encode_motion(self, reconstructed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.encoder(reconstructed, offset)
    
    def decode_latent_motion(self, latent_representation: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return self.auto_encoder.decoder(latent_representation, offset)
    
    def discriminate_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.discriminator(motion)

class SkeletalGAN(nn.Module):   
    def __init__(
        self,
        topologies: Tuple[Dict[str, Any], Dict[str, Any]],
        gan_params: Dict[str, Any],
    ):
        super().__init__()        
        self.topology_A, self.topology_B = topologies
        
        self.domain_A = SkeletalDomainModule(self.topology_A, **gan_params)
        self.domain_B = SkeletalDomainModule(self.topology_B, **gan_params)
        self.domains = {"A": self.domain_A, "B": self.domain_B}
      
    def discriminators_requires_grad_(self, requires_grad: bool = True):
        for domain in self.domains.values():
            for para in domain.discriminator_parameters():
                para.requires_grad = requires_grad
    
    def encode_offsets(self, offset: torch.Tensor, domain: str):
        """ Encode offsets for a given domain. """
        return self.domains[domain].encode_offsets(offset)

    def encode_decode_motion(
        self,
        motion: Dict[str, torch.Tensor],
        offset: Dict[str, torch.Tensor],
        domain: str
    ):
        """ Auto-encode motion for a given domain. """
        return self.domains[domain].encode_decode_motion(motion, offset)

    def decode_latent_motion(
        self,
        domain: str,
        latent: torch.Tensor,
        offsets: torch.Tensor
    ): 
        """ Decode latent motion for a given domain. """
        return self.domains[domain].decode_latent_motion(latent, offsets)

    def encode_motion(
        self,
        domain: str,
        motion: torch.Tensor,
        offsets: torch.Tensor
    ):
        """Encode motion into latent representation for a given domain. """
        return self.domains[domain].encode_motion(motion, offsets)
    
    def discriminate(
        self,
        domain: str,
        motion: torch.Tensor,
        offsets: torch.Tensor
    ):
        """ Evaluate the discriminator for a given domain. """
        return self.domains[domain].discriminate_motion(motion, offsets)