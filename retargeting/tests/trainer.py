from src.data import AMASSTAdapter
from src.training import SkeletalGANTrainer
from src.data.datasets import CrossDomainMotionDataset, MotionDataset, MotionDatasetBuilder
from src.skeletal.models import SkeletalGAN
from src.utils import DataUtils
from typing import List, Callable
from src.training import LossBundle
from src.training.trainer import TrainConfig
import torch

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5679))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")
     
    dataset_config = DataUtils.load_yaml("/workspace/configs/default/motion_dataset.yaml")
    auto_encoder_params = DataUtils.load_yaml("/workspace/configs/default/autoencoder.yaml")
    discriminator_params = DataUtils.load_yaml("/workspace/configs/default/discriminator.yaml")
    offset_encoder_params = DataUtils.load_yaml("/workspace/configs/default/offset_encoder.yaml")
    gan_params = {"auto_encoder_params" : auto_encoder_params,
                  "discriminator_params": discriminator_params,
                  "offset_encoder_params": offset_encoder_params}

    amass_adapter = AMASSTAdapter()
    dataset_builder = MotionDatasetBuilder(adapter=amass_adapter, data_config=dataset_config)
    dataset_A = MotionDataset(characters=["Aude", "Carine"], builder=dataset_builder)
    dataset_A_norm_stats = dataset_A.norm_stats
    dataset_B = MotionDataset(characters=["Karim"], builder=dataset_builder)
    dataset_B_norm_stats = dataset_B.norm_stats

    #The issue lies in where we are creating our edge list. 
    # Remember for edges specifically, edge 0 corresponds to the edge from 0->1.
    # Normal joint indexing is accounting for the root which we do consider. 
    # Keep this in mind for when constructing the adjacency list.

    cross_domain_dataset = CrossDomainMotionDataset(dataset_A, dataset_B)
    train_loader = torch.utils.data.DataLoader(cross_domain_dataset, batch_size=256)
    
    model = SkeletalGAN(topologies=cross_domain_dataset.topologies, gan_params=gan_params, normalization_stats = (dataset_A_norm_stats, dataset_B_norm_stats))

    optimizer_G = torch.optim.Adam(
      model.generator_parameters(),
      lr=0.0002,
      betas=(0.9, 0.999),
   )

    optimizer_D = torch.optim.Adam(
      model.discriminator_parameters(),
      lr=0.0002,
      betas=(0.9, 0.999),
   )

  #   scheduler_G = torch.optim.lr_scheduler.StepLR(
  #     optimizer_G,
  #     step_size=50,
  #     gamma=0.5,
  #     )

  #   scheduler_D = torch.optim.lr_scheduler.StepLR(
  #     optimizer_D,
  #     step_size=50,
  #     gamma=0.5,
  #  )

    trainer = SkeletalGANTrainer(model=model, 
                                 losses=LossBundle(), 
                                 optimizer_G=optimizer_G, 
                                 optimizer_D=optimizer_D, 
                                #  scheduler_G=scheduler_G, 
                                #  scheduler_D=scheduler_D, 
                                 train_loader=train_loader, 
                                 config = TrainConfig)
    trainer.train()


# Final todos tony:

#TODO(anthony) fix height. their heights are like 152 cm? 175 cm? HEIGHT
#TODO(anthony) verify that motion is in radians. DATA
#TODO(anthony) unsure if we want to do this for each cross section. only want A->A, A->B. sum like that . DISCRIMINATOR BACKWARD
# #TODO(anthony) They randomly select characters for this. Not sure if we want to do that. hold off for now. RETARGETTING FORWARD
#  #TODO(anthony) pick back up here. we need to pad the last row and discard it somehow.  ENCODER,DECODER
#TODO(anthony) we can take their mixamo data and make sure our pooling operations are the same. 