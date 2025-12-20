from src.data_processing import AMASSTAdapter
from src.training import SkeletalGANTrainer
from src.dataset import CrossDomainMotionDataset, MotionDataset, MotionDatasetBuilder
from src.dataset.motion_dataset import paired_collate
from src.data_processing import AMASSTAdapter
from src.skeletal_models import SkeletalGAN
from src.utils import DataUtils
from typing import List
from src.data_processing import SkeletonMetadata
from src.training import LossBundle
from src.training.trainer import TrainConfig
import torch
from src.utils.skeleton import SkeletonUtils

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")
     
    dataset_config = DataUtils.load_yaml("/workspace/configs/default/motion_dataset.yaml")
    auto_encoder_params = DataUtils.load_yaml("/workspace/configs/default/autoencoder.yaml")
    discriminator_params = DataUtils.load_yaml("/workspace/configs/default/discriminator.yaml")
    static_encoder_params = DataUtils.load_yaml("/workspace/configs/default/static_encoder.yaml")
    gan_params = {"auto_encoder_params" : auto_encoder_params,
                  "discriminator_params": discriminator_params,
                  "static_encoder_params": static_encoder_params}

    amass_adapter = AMASSTAdapter()
    dataset_builder = MotionDatasetBuilder(adapter=amass_adapter, data_config=dataset_config)
    dataset_A = MotionDataset(characters=["Aude", "Carine"], builder=dataset_builder)
    dataset_B = MotionDataset(characters=["Karim", "Medhi"], builder=dataset_builder)

    #The issue lies in where we are creating our edge list. 
    # Remember for edges specifically, edge 0 corresponds to the edge from 0->1.
    # Normal joint indexing is accounting for the root which we do consider. 
    # Keep this in mind for when constructing the adjacency list.

    cross_domain_dataset = CrossDomainMotionDataset(dataset_A, dataset_B)
    train_loader = torch.utils.data.DataLoader(cross_domain_dataset, batch_size=256, collate_fn=paired_collate)
    
    model = SkeletalGAN(topologies=cross_domain_dataset.topologies, gan_params=gan_params)

    optimizer_G = torch.optim.Adam(
      model.generator_parameters(),
      lr=0.003,
      betas=(0.5, 0.999),
   )

    optimizer_D = torch.optim.Adam(
      model.discriminator_parameters(),
      lr=0.003,
      betas=(0.5, 0.999),
   )

    scheduler_G = torch.optim.lr_scheduler.StepLR(
      optimizer_G,
      step_size=50,
      gamma=0.5,
      )

    scheduler_D = torch.optim.lr_scheduler.StepLR(
      optimizer_D,
      step_size=50,
      gamma=0.5,
   )

    trainer = SkeletalGANTrainer(model=model, 
                                 losses=LossBundle(), 
                                 optimizer_G=optimizer_G, 
                                 optimizer_D=optimizer_D, 
                                 scheduler_G=scheduler_G, 
                                 scheduler_D=scheduler_D, 
                                 train_loader=train_loader, 
                                 config = TrainConfig)
    trainer.train()