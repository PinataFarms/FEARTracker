from typing import Dict, Tuple, Any, List, Union, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import DummyLogger
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate

from model_training.utils.logger import create_logger

logger = create_logger(__name__)


def get_collate_for_dataset(dataset: Union[Dataset, ConcatDataset]) -> Callable:
    """
    Returns collate_fn function for dataset. By default, default_collate returned.
    If the dataset has method get_collate_fn() we will use it's return value instead.
    If the dataset is ConcatDataset, we will check whether all get_collate_fn() returns
    the same function.

    Args:
        dataset: Input dataset

    Returns:
        Collate function to put into DataLoader
    """
    collate_fn = default_collate

    if hasattr(dataset, "get_collate_fn"):
        collate_fn = dataset.get_collate_fn()

    if isinstance(dataset, ConcatDataset):
        collates = [get_collate_for_dataset(ds) for ds in dataset.datasets]
        if len(set(collates)) != 1:
            raise ValueError("Datasets have different collate functions")
        collate_fn = collates[0]
    return collate_fn


class BaseLightningModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], train: Dataset, val: Dataset) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.use_ddp = self.config.get("accelerator", None) == "ddp"
        self.epoch_num = 0
        self.tensorboard_logger = None

    @property
    def is_master(self) -> bool:
        """
        Returns True if the caller is the master node (Either code is running on 1 GPU or current rank is 0)
        """
        return (self.use_ddp is False) or (torch.distributed.get_rank() == 0)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=self.config.get("metric_mode", "min"),
                                                               factor=0.5,
                                                               patience=5,
                                                               min_lr=1e-6)
        scheduler_config = {"scheduler": scheduler, "monitor": self.config.get("metric_to_monitor", "valid/loss")}
        return [optimizer], [scheduler_config]

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, self.config, "train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, self.config, "val")

    def on_epoch_end(self) -> None:
        self.epoch_num += 1

    def on_pretrain_routine_start(self) -> None:
        if not isinstance(self.logger, DummyLogger):
            for logger in self.logger:
                if isinstance(logger, TensorBoardLogger):
                    self.tensorboard_logger = logger

    def _get_dataloader(self, dataset: Dataset, config: Dict[str, Any], loader_name: str) -> DataLoader:
        """
        Instantiate DataLoader for given dataset w.r.t to config and mode.
        It supports creating a custom sampler.
        Note: For DDP mode, we support custom samplers, but trainer must be called with:
            >>> replace_sampler_ddp=False

        Args:
            dataset: Dataset instance
            config: Dataset config
            loader_name: Loader name (train or val)

        Returns:

        """
        collate_fn = get_collate_for_dataset(dataset)

        dataset_config = config[loader_name]
        if "sampler" not in dataset_config or dataset_config["sampler"] == "none":
            sampler = None
        else:
            sampler = self._build_sampler(dataset_config, dataset)

        drop_last = loader_name == "train"

        if self.use_ddp:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            sampler = DistributedSampler(dataset, world_size, local_rank)

        should_shuffle = (sampler is None) and (loader_name == "train")
        batch_size = self._get_batch_size(loader_name)
        # Number of workers must not exceed batch size
        num_workers = min(batch_size, self.config["num_workers"])
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return loader

    def _get_batch_size(self, mode: str = "train") -> int:
        if isinstance(self.config["batch_size"], dict):
            return self.config["batch_size"][mode]
        return self.config["batch_size"]
