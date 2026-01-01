"""
This script is the main entry point for training the CLSS model.

It handles setting up the distributed environment, parsing arguments, preparing the
datasets and dataloaders, initializing the model, and running the training loop
using PyTorch Lightning.
"""

import os
import warnings
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
import torch
from infra import (
    setup_wandb,
    setup_trainer,
)
from args import setup_args
from clss import CLSSModel
from datamodule import CLSSDataModule


@rank_zero_only
def save_checkpoint(
    trainer: pl.Trainer, wandb_logger: WandbLogger, checkpoint_path: str
) -> None:
    trainer.save_checkpoint(
        os.path.join(checkpoint_path, "models", f"{wandb_logger.experiment.name}.lckpt")
    )


def main():
    """Main training function that can be called as a console script."""
    warnings.filterwarnings("ignore", category=UserWarning)

    args = setup_args()

    print(args)

    pl.seed_everything(args.seed, workers=True)

    # Set up distributed environment if running with SLURM
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPUs.")

    # Set up Wandb logging only on rank 0
    wandb_logger = setup_wandb(args)

    # Initialize the PyTorch Lightning trainer
    trainer = setup_trainer(args.epochs, wandb_logger, args.checkpoint_path, num_nodes=args.num_nodes)
    
    # Validate checkpoint if resuming
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from_checkpoint}")
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")

    # Initialize DataModule
    datamodule = CLSSDataModule(
        dataset_path=args.dataset_path,
        dataset_size_limit=args.dataset_size_limit,
        validation_dataset_frac=args.validation_dataset_frac,
        esm_checkpoint=args.esm_checkpoint,
        structures_dir=args.structures_dir,
        train_pickle_file=args.train_pickle_file,
        validation_pickle_file=args.validation_pickle_file,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Define the model
    model = CLSSModel(
        esm2_checkpoint=args.esm_checkpoint,  # Specify the ESM2 model variant you want to use
        hidden_dim=args.hidden_projection_dim,  # Dimension of the projection head
        learning_rate=args.learning_rate,  # Learning rate
        random_sequence_stretches=args.random_sequence_stretches,
        random_stretch_min_size=args.random_stretch_min_size,
        should_learn_temperature=args.learn_temperature,
        init_temperature=args.init_temperature,
        should_load_esm3=False,
    )

    # Train the model (with optional checkpoint resumption)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)

    save_checkpoint(trainer, wandb_logger, args.checkpoint_path)


if __name__ == "__main__":
    main()
