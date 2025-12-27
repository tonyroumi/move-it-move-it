from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

from src.data import AMASSTAdapter
from src.data.datasets import CrossDomainMotionDataset, MotionDataset, MotionDatasetBuilder, paired_collate
from src.models.networks import SkeletalGAN
from src.training import SkeletalGANTrainer
from src.utils import set_seed, Logger, SkeletonVisualizer

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
   
    logger = Logger(log_dir=output_dir, verbose=cfg.verbose)
    
    set_seed(cfg.seed)
    logger.info(f"Seed set to: {cfg.seed}")

    # Datasets
    amass_dataset_builder = MotionDatasetBuilder(
        adapter=AMASSTAdapter(cfg.device), 
        data_config=cfg.dataset, 
        logger=logger,
    )
    
    dataset_A = MotionDataset(
        characters=cfg.data.characters_a, 
        builder=amass_dataset_builder
    )
    dataset_A_norm_stats = dataset_A.norm_stats

    dataset_B = MotionDataset(
        characters=cfg.data.characters_b, 
        builder=amass_dataset_builder
    )
    dataset_B_norm_stats = dataset_B.norm_stats
    
    cross_domain_dataset = CrossDomainMotionDataset(dataset_A, dataset_B)
    
    train_loader = torch.utils.data.DataLoader(
        cross_domain_dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=paired_collate,
        num_workers=cfg.get("num_workers", 4),
        shuffle=True,
        pin_memory=(cfg.device == 'cuda')
    ) 

    if cfg.get("visualize_data_first", False):
        SkeletonVisualizer.visualize_dataset(train_loader, save_path=output_dir)
    
    # Model
    model_config = cfg.model 
    model = SkeletalGAN(
        topologies=cross_domain_dataset.topologies,
        normalization_stats=(dataset_A_norm_stats, dataset_B_norm_stats),
        offset_encoder_params=model_config.offset_encoder,
        auto_encoder_params=model_config.autoencoder,
        discriminator_params=model_config.discriminator
    )

    # Trainer
    trainer = SkeletalGANTrainer(
        model=model,
        optimizer_G=torch.optim.Adam(
            model.generator_parameters(),
            lr=cfg.train.learning_rate,
            betas=cfg.train.betas,
        ),
        optimizer_D=torch.optim.Adam(
            model.discriminator_parameters(),
            lr=cfg.train.learning_rate,
            betas=cfg.train.betas,
        ),
        train_loader=train_loader,
        checkpoint_dir=output_dir,
        config=cfg.train,
        logger=logger,
        device=cfg.device
    )
    

    if cfg.get("resume_from") is not None:
        trainer.load_checkpoint(cfg.resume_from)
    
    # ============================================================================
    # Start Training
    # ============================================================================
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()