"""
This module provides infrastructure setup functions for the training script.

It includes helper functions for setting up Weights & Biases logging, initializing
the distributed process group, creating dataloaders, preparing datasets, and
configuring the PyTorch Lightning trainer.
"""

import os
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch

@rank_zero_only
def setup_wandb(args: object) -> WandbLogger:
    """Initializes and configures Weights & Biases (Wandb) logging.

    This function should only be called on the main process (rank zero) to avoid
    multiple initializations. It sets up a Wandb logger with the specified project
    name, run name, and save directory. It also logs the script's arguments and
    the world size.

    Args:
        args (object): An object containing the script's arguments (e.g., from argparse).
                       It should have attributes like `run_name` and `checkpoint_path`.

    Returns:
        WandbLogger: The configured Wandb logger instance.
    """
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        name=args.run_name,
        project=args.wandb_project,
        save_dir=os.path.join(args.checkpoint_path, "logs"),
    )

    # Update experiment config with parameters
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config["world_size"] = world_size

    return wandb_logger


def setup_trainer(epochs: int, wandb_logger: WandbLogger, checkpoint_dir: str, num_nodes: int = 1) -> pl.Trainer:
    """Configures and returns a PyTorch Lightning Trainer.

    This function sets up a PyTorch Lightning `Trainer` with configurations
    suitable for multi-GPU, multi-node training on a SLURM cluster. It uses
    the DDPStrategy and is configured to log with Wandb.

    Args:
        epochs (int): The total number of epochs for training.
        wandb_logger (WandbLogger): The Wandb logger instance to use for logging.
        checkpoint_dir (str): Directory to save checkpoints to.
        num_nodes (int): Number of nodes for distributed training.

    Returns:
        pl.Trainer: The configured PyTorch Lightning Trainer.
    """
    # Configure periodic checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, "checkpoints"),
        filename="clss-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,  # Always save last checkpoint for easy resumption
        every_n_epochs=1,
    )
    
    return pl.Trainer(
        logger=wandb_logger,  # Use the WandB logger
        max_epochs=epochs,  # Number of epochs to train
        accelerator="cuda",
        devices=torch.cuda.device_count(),  # type: ignore # Number of GPUs to use, set to 0 if you don't have a GPU
        log_every_n_steps=1,
        num_nodes=num_nodes,
        check_val_every_n_epoch=1,  # Frequency of validation
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(
            find_unused_parameters=True, cluster_environment=SLURMEnvironment()
        ),
    )


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
