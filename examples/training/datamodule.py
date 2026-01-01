import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional
from dataset import CLSSDataset

class CLSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        dataset_size_limit: int,
        validation_dataset_frac: float,
        esm_checkpoint: str,
        structures_dir: str,
        train_pickle_file: str,
        validation_pickle_file: str,
        seed: int,
        batch_size: int,
        num_workers: int = 5,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_size_limit = dataset_size_limit
        self.validation_dataset_frac = validation_dataset_frac
        self.esm_checkpoint = esm_checkpoint
        self.structures_dir = structures_dir
        self.train_pickle_file = train_pickle_file
        self.validation_pickle_file = validation_pickle_file
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[CLSSDataset] = None
        self.val_dataset: Optional[CLSSDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Read and split dataframe
        dataframe = pd.read_csv(self.dataset_path, dtype={"ecod_uid": str})
        
        # Sample if limit is set
        if self.dataset_size_limit < len(dataframe):
            dataframe = dataframe.sample(self.dataset_size_limit, random_state=self.seed)
            
        train_dataframe, validation_dataframe = train_test_split(
            dataframe, test_size=self.validation_dataset_frac, random_state=self.seed
        )

        # Create datasets
        self.train_dataset = CLSSDataset(
            self.esm_checkpoint,
            self.structures_dir,
            train_dataframe["ecod_uid"].tolist(),
            pickle_file=self.train_pickle_file,
        )

        self.val_dataset = CLSSDataset(
            self.esm_checkpoint,
            self.structures_dir,
            validation_dataframe["ecod_uid"].tolist(),
            pickle_file=self.validation_pickle_file,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
