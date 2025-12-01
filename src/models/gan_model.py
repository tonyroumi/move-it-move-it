from .encoder import SkeletalEncoder
from .autoencoder import SkeletalAutoEncoder
from .discriminator import SkeletalDiscriminator

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalGAN(nn.Module):   
    def __init__(
        self,
        topologies: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
        logger: Optional[Any] = None
    ):
        super().__init__()
        
        self.config = config
        self.logger = logger
        self.domain_names = list(topologies.keys())
        
        self.topologies = topologies
        
        self.static_encoders = nn.ModuleDict()
        self.auto_encoders = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()
        
        for name, topology in self.topologies.items():
            self.static_encoders[name] = self._create_static_encoder(topology)
            self.auto_encoders[name] = self._create_autoencoder(topology)
            self.discriminators[name] = self._create_discriminator(
                topology,
                self.static_encoders[name].pooling_hierarchy
            )
    
    def _create_static_encoder(self, topology: Dict[str, Any]) -> nn.Module:
        return SkeletalEncoder(
            adj_init=topology['adj'],
            edge_init=topology['edges'],
            encoder_params=self.config['static_encoder']
        )
    
    def _create_autoencoder(self, topology: Dict[str, Any]) -> nn.Module:
        return SkeletalAutoEncoder(
            adj_init=topology['adj'],
            edge_init=topology['edges'],
            auto_encoder_params=self.config['auto_encoder']
        )
    
    def _create_discriminator(
        self,
        pooled_info: Any
    ) -> nn.Module:
        return SkeletalDiscriminator(
            pooled_info=pooled_info,
            discriminator_params=self.config['discriminator']
        )