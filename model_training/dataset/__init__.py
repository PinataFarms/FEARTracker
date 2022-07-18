from typing import Dict, Any

from got10k.datasets import VOT, GOT10k, NfS
from torch.utils.data import Dataset, ConcatDataset

from model_training.utils import create_logger
from .siam_dataset import SiameseTrackingDataset
from .tracking_dataset import TrackingDataset

logger = create_logger(__name__)


def dummy_collate(batch: Any) -> Any:
    return batch


class SequenceDatasetWrapper(Dataset):
    _datasets = {
        "nfs": NfS,
        "got10k": GOT10k,
        "vot": VOT,
    }

    def __init__(self, dataset_name: str, dataset: Dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return self.dataset_name

    def __getitem__(self, index: int):
        image_files, annotations = self.dataset[index]
        return image_files, annotations, self.dataset_name

    def get_collate_fn(self) -> Any:
        return dummy_collate

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        dataset_name = config.pop("name")
        dataset = cls._datasets[dataset_name](**config)
        return cls(dataset_name=dataset_name, dataset=dataset)


def get_tracking_dataset(config: Dict) -> TrackingDataset:
    datasets = {
        "siam": SiameseTrackingDataset,
    }
    cls = datasets[config["dataset"]["dataset_type"]]
    return cls.from_config(config)


def get_tracking_datasets(config) -> [ConcatDataset, ConcatDataset]:
    train_datasets = []
    for dataset_config in config["train"]["datasets"]:
        ds = get_tracking_dataset(dict(dataset=dataset_config, tracker=config["tracker"]))
        logger.info("Train dataset %s %d", str(ds), len(ds))
        train_datasets.append(ds)

    val_datasets = []
    for dataset_config in config["val"]["datasets"]:
        ds = SequenceDatasetWrapper.from_config(dataset_config)
        logger.info("Valid dataset %s %d", str(ds), len(ds))
        val_datasets.append(ds)
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


__all__ = [
    "SiameseTrackingDataset",
    "TrackingDataset",
    "get_tracking_dataset",
    "get_tracking_datasets",
]
