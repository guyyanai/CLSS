"""
This module provides infrastructure setup functions for the training script.

It includes helper functions for setting up Weights & Biases logging, initializing
the distributed process group, creating dataloaders, preparing datasets, and
configuring the PyTorch Lightning trainer.
"""

import os
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
import random
import numpy as np
from dataset import CLSSDataset
from typing import Tuple

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


def setup_process_group():
    """Sets up the distributed process group for multi-GPU training.

    This function initializes the process group for distributed training using
    PyTorch's `init_process_group`. It uses the 'nccl' backend and expects
    the 'env://' initialization method. It also sets the appropriate CUDA
    device for the current process based on its local rank.
    """
    # Assign the correct GPU to the process based on its LOCAL_RANK
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"world size: {world_size}, local rank: {local_rank}")

    print(f"Setting up process group")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=local_rank,
        world_size=world_size,
        timeout=timedelta(seconds=3600),
    )
    print(
        f"Finished setting up process group, current rank: {torch.distributed.get_rank()}"
    )

    torch.cuda.set_device(local_rank)

def destroy_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def setup_dataloaders(
    train_dataset: CLSSDataset, val_dataset: CLSSDataset, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Creates DataLoader objects for training and validation sets.

    This function prepares DataLoaders for the training and validation datasets.
    It uses a `DistributedSampler` for the training loader to ensure that data is
    correctly distributed across multiple GPUs.

    Args:
        train_dataset (CLSSDataset): The dataset for training.
        val_dataset (CLSSDataset): The dataset for validation.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
    """
    sampler: DistributedSampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=5
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=5
    )

    return train_dataloader, val_dataloader


def setup_dataset(
    dataset_path: str,
    dataset_size_limit: int,
    validation_dataset_frac: float,
    esm_checkpoint: str,
    structures_dir: str,
    train_pickle_file: str,
    validation_pickle_file: str,
    seed: int,
) -> Tuple[CLSSDataset, CLSSDataset]:
    """Prepares the training and validation datasets.

    This function reads a dataset from a CSV file, splits it into training and
    validation sets, and creates `CLSSDataset` objects for both.

    Args:
        dataset_path (str): Path to the CSV file containing dataset information (e.g., ECOD UIDs).
        dataset_size_limit (int): The maximum number of samples to use from the dataset.
        validation_dataset_frac (float): The fraction of the dataset to be used for validation.
        esm_checkpoint (str): The Hugging Face Hub checkpoint for the EsmTokenizer.
        structures_dir (str): The root directory containing the PDB structure files.
        train_pickle_file (str): Path to the pickle file for caching the processed training dataset.
        validation_pickle_file (str): Path to the pickle file for caching the processed validation dataset.
        seed (int): The random seed for sampling and train/validation splitting.

    Returns:
        tuple[CLSSDataset, CLSSDataset]: A tuple containing the training and validation datasets.
    """
    dataframe = pd.read_csv(dataset_path, dtype={"ecod_uid": str})
    dataframe = dataframe.sample(dataset_size_limit, random_state=seed)
    train_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=validation_dataset_frac, random_state=seed
    )

    # New dataset:
    train_dataset = CLSSDataset(
        esm_checkpoint,
        structures_dir,
        train_dataframe["ecod_uid"].tolist(),
        pickle_file=train_pickle_file,
    )

    validation_dataset = CLSSDataset(
        esm_checkpoint,
        structures_dir,
        validation_dataframe["ecod_uid"].tolist(),
        pickle_file=validation_pickle_file,
    )

    return train_dataset, validation_dataset


def setup_trainer(epochs: int, wandb_logger: WandbLogger) -> pl.Trainer:
    """Configures and returns a PyTorch Lightning Trainer.

    This function sets up a PyTorch Lightning `Trainer` with configurations
    suitable for multi-GPU, multi-node training on a SLURM cluster. It uses
    the DDPStrategy and is configured to log with Wandb.

    Args:
        epochs (int): The total number of epochs for training.
        wandb_logger (WandbLogger): The Wandb logger instance to use for logging.

    Returns:
        pl.Trainer: The configured PyTorch Lightning Trainer.
    """
    return pl.Trainer(
        logger=wandb_logger,  # Use the WandB logger
        max_epochs=epochs,  # Number of epochs to train
        accelerator="cuda",
        devices=torch.cuda.device_count(),  # type: ignore # Number of GPUs to use, set to 0 if you don't have a GPU
        log_every_n_steps=1,
        num_nodes=1,
        check_val_every_n_epoch=1,  # Frequency of validation
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
