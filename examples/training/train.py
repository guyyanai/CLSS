"""
This script is the main entry point for training the CLSS model.

It handles setting up the distributed environment, parsing arguments, preparing the
datasets and dataloaders, initializing the model, and running the training loop
using PyTorch Lightning.
"""

import os
import warnings
import atexit
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.distributed
from infra import (
    setup_wandb,
    setup_process_group,
    destroy_process_group,
    setup_dataset,
    setup_dataloaders,
    setup_trainer,
)
from args import setup_args
from clss import CLSSModel


@rank_zero_only
def save_checkpoint(
    trainer: pl.Trainer, wandb_logger: WandbLogger, checkpoint_path: str
) -> None:
    trainer.save_checkpoint(
        os.path.join(checkpoint_path, "models", f"{wandb_logger.experiment.name}.lckpt")
    )


def main():
    """Main training function that can be called as a console script."""
    setup_process_group()
    atexit.register(destroy_process_group)

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
    trainer = setup_trainer(args.epochs, wandb_logger)

    # Load dataset
    train_dataset, val_dataset = setup_dataset(
        dataset_path=args.dataset_path,
        dataset_size_limit=args.dataset_size_limit,
        validation_dataset_frac=args.validation_dataset_frac,
        esm_checkpoint=args.esm_checkpoint,
        structures_dir=args.structures_dir,
        train_pickle_file=args.train_pickle_file,
        validation_pickle_file=args.validation_pickle_file,
        seed=args.seed,
    )
    torch.distributed.barrier()

    # Create DataLoaders
    train_dataloader, val_dataloader = setup_dataloaders(
        train_dataset, val_dataset, args.batch_size
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

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    torch.distributed.destroy_process_group()

    save_checkpoint(trainer, wandb_logger, args.checkpoint_path)


if __name__ == "__main__":
    main()
