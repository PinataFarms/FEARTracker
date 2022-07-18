import random
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ItemSampler(ABC):
    def __init__(
        self,
        data_path: str,
        negative_ratio: float,
        frame_offset: int,
        num_samples: int,
        clip_range: bool = False,
    ):
        self.data_path = data_path
        self.negative_ratio = negative_ratio
        self.frame_offset = frame_offset
        self.num_samples = num_samples
        self.data = None
        self.mapping = None
        self.clip_range = clip_range

    def __len__(self):
        pass

    def resample(self) -> None:
        pass

    @abstractmethod
    def _read_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def parse_samples(self) -> None:
        pass

    @abstractmethod
    def extract_sample(self, idx: int) -> Dict[str, Any]:
        pass


class TrackSampler(ItemSampler):
    def __init__(
        self,
        data_path: str,
        negative_ratio: float,
        frame_offset: int,
        num_samples: int,
        clip_range: bool = False,
    ):
        self.epoch_data = None
        self.template_data = None
        self.num_tracks = None
        super().__init__(
            data_path=data_path,
            negative_ratio=negative_ratio,
            frame_offset=frame_offset,
            num_samples=num_samples,
            clip_range=clip_range,
        )

    def __len__(self) -> int:
        return len(self.epoch_data)

    def _read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        negative = data[data["presence"] == 0]
        negative_ratio = len(negative) / len(data)
        num_neg_samples_to_drop = max(0, int((negative_ratio - self.negative_ratio) * len(data)))
        dropped_negatives = np.random.choice(negative.index, num_neg_samples_to_drop, replace=False)
        data = data.drop(dropped_negatives)
        data = data.reset_index(drop=True)
        return data

    def resample(self) -> None:
        if self.num_tracks == len(self.template_data):
            self.epoch_data = self.template_data.sample(self.num_samples).reset_index(drop=True)
        else:
            self.epoch_data = (
                self.template_data.groupby("track_id")
                .sample(int(ceil((self.num_samples / self.num_tracks))), replace=True)
                .sample(self.num_samples)
                .reset_index(drop=True)
            )

    def parse_samples(self) -> None:
        self.data = self._read_data()
        self.template_data = self.data[(self.data["presence"] == 1) & (~self.data["near_corner"])]
        self.num_tracks = len(self.template_data["track_id"].unique())
        self.mapping = self.data.groupby(["track_id"]).groups
        self.resample()

    def extract_sample(self, idx: int) -> Dict[str, Any]:
        template_item = self.epoch_data.iloc[idx]
        track_indices = self.mapping[template_item["track_id"]]

        if self.clip_range:
            search_items = self.data.iloc[track_indices]
            search_item = (
                search_items[
                    (search_items["frame_index"] > template_item["frame_index"] - self.frame_offset)
                    & (search_items["frame_index"] < template_item["frame_index"] + self.frame_offset)
                ]
                .sample(1)
                .iloc[0]
            )
        else:
            search_index = random.choice(track_indices)
            search_item = self.data.iloc[search_index]
        return dict(template=template_item, search=search_item)


class FrameSampler(ItemSampler):
    def __init__(
        self,
        data_path: str,
        negative_ratio: float,
        frame_offset: int,
        num_samples: int,
        clip_range: bool = False,
    ):
        super().__init__(
            data_path=data_path,
            negative_ratio=negative_ratio,
            frame_offset=frame_offset,
            num_samples=num_samples,
            clip_range=clip_range,
        )
        self.indices = None

    def __len__(self):
        return min(self.num_samples, len(self.indices))

    def _read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        negative = data[data["presence"] == 0]
        negative_ratio = len(negative) / len(data)
        num_neg_samples_to_drop = max(0, int((negative_ratio - self.negative_ratio) * len(data)))
        dropped_negatives = np.random.choice(negative.index, num_neg_samples_to_drop)
        data = data.drop(dropped_negatives)
        return data.reset_index(drop=True)

    def parse_samples(self) -> None:
        self.data = self._read_data()
        self.mapping = self._get_mapping(self.data)
        self.indices = self._get_image_indices(self.data)
        if self.num_samples is None:
            self.num_samples = len(self.indices)

    def extract_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get search and template rows from csv
        """
        dataset_index = self.indices[idx]
        template_item = self.data.iloc[dataset_index]
        track_indices = self.mapping[template_item["track_id"]]
        if self.clip_range:
            search_items = self.data.iloc[track_indices]
            search_item = (
                search_items[
                    (search_items["frame_index"] > template_item["frame_index"] - self.frame_offset)
                    & (search_items["frame_index"] < template_item["frame_index"] + self.frame_offset)
                ]
                .sample(1)
                .iloc[0]
            )
        else:
            search_index = random.choice(track_indices)
            search_item = self.data.iloc[search_index]
        return dict(template=template_item, search=search_item)

    @staticmethod
    def _get_image_indices(data: pd.DataFrame) -> List:
        return list(data[(data["presence"] == 1) & (~data["near_corner"])].index)

    @staticmethod
    def _get_mapping(dataset: pd.DataFrame) -> Dict[Any, List[int]]:
        mapping = dict()
        for track_id, group in dataset.groupby("track_id"):
            mapping[track_id] = group.index
        return mapping
