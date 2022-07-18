from abc import abstractmethod, ABC
from collections import namedtuple
from typing import Dict, Any, Union, Optional

import numpy as np
import torch

from model_training.utils.utils import unravel_index, make_grid

TrackerEncodeResult = namedtuple("TrackerEncodeResult", ["regression_map", "classification_label"])
TrackerDecodeResult = namedtuple("TrackerDecodeResult", ["bbox", "pred_coords"])


class BoxCoder(ABC):
    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        super().__init__()
        self.tracker_config = tracker_config
        self.grid_x, self.grid_y = make_grid(
            tracker_config["score_size"], tracker_config["total_stride"], tracker_config["instance_size"]
        )

    def to_device(self, device: Union[str, int]) -> "BoxCoder":
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        return self

    @abstractmethod
    def encode(self, bboxes: np.array) -> TrackerEncodeResult:
        """

        :param bboxes: np.array - [x, y, w, h]
        :return: encoded_info: TrackerEncodeResult - regression_map: np.array(batch, 4, 25, 25),
                                                     classification_label: np.array(batch, 1, 25, 25),
        """
        pass

    @abstractmethod
    def decode(
        self,
        regression_map: torch.Tensor,
        classification_map: torch.Tensor,
        use_sigmoid: bool = True,
    ) -> TrackerDecodeResult:
        """
        :param regression_map: torch.Tensor(batch, 4, 25, 25) - Regression output from a tracking net
        :param classification_map: torch.Tensor(batch, 1, 25, 25) - Classification output from a tracking net
        :param use_sigmoid: torch.Tensor - Use sigmoid or not (for classification_labels we don`t need it)
        :return: decoded_info: TrackerDecodeResult - bbox, pred_coords
        """
        pass


class MobileTrackBoxCoder(BoxCoder):
    def __init__(self, tracker_config: Dict[str, Any]) -> None:
        super().__init__(tracker_config=tracker_config)

    @torch.no_grad()
    def encode(self, bboxes: torch.Tensor) -> TrackerEncodeResult:
        """
        :param bboxes: torch.Tensor(batch, 4) - Boxes in xywh format
        :return: encoded_info: TrackerEncodeResult - regression_map: torch.Tensor(batch, 4, 25, 25),
                                                     classification_label: torch.Tensor(batch, 1, 25, 25)
        """
        bboxes = bboxes.unsqueeze(-1).unsqueeze(-1)
        left = self.grid_x - bboxes[:, 0]
        top = self.grid_y - bboxes[:, 1]
        right = bboxes[:, 0] + bboxes[:, 2] - self.grid_x
        bottom = bboxes[:, 1] + bboxes[:, 3] - self.grid_y
        regression_map = torch.stack((left, top, right, bottom), dim=1).float()
        regression_map_min, _ = torch.min(regression_map, dim=1)
        classification_label = (regression_map_min.unsqueeze(1) > 0).float()
        return TrackerEncodeResult(regression_map=regression_map, classification_label=classification_label)

    @torch.no_grad()
    def decode(
        self,
        regression_map: torch.Tensor,
        classification_map: torch.Tensor,
        use_sigmoid: bool = True,
    ) -> TrackerDecodeResult:
        """
        :param regression_map: torch.Tensor - Regression output from a tracking net
        :param classification_map: torch.Tensor - Classification output from a tracking net
        :param use_sigmoid: torch.Tensor - Use sigmoid or not (for classification_labels we don`t need it)
        :return: decoded_info: TrackerDecodeResult - bbox (in xywh format), pred_coords
        """
        if use_sigmoid:
            classification_map = classification_map.float().sigmoid()
        classification_map = classification_map[:, 0, :, :]

        pred_location = torch.stack(
            [
                self.grid_x - regression_map[:, 0, ...],
                self.grid_y - regression_map[:, 1, ...],
                self.grid_x + regression_map[:, 2, ...],
                self.grid_y + regression_map[:, 3, ...],
            ],
            dim=1,
        )
        bboxes, pred_coords = list(), list()
        for one_classification_map, one_pred_location in zip(classification_map, pred_location):
            r_max, c_max = unravel_index(torch.argmax(one_classification_map), one_classification_map.shape)
            output = [x[r_max, c_max] for x in one_pred_location]
            bbox = torch.stack([output[0], output[1], output[2] - output[0], output[3] - output[1]])
            bboxes.append(bbox)
            pred_coords.append((r_max.item(), c_max.item()))
        return TrackerDecodeResult(bbox=torch.stack(bboxes), pred_coords=pred_coords)


def get_box_coder(tracker_config: Dict[str, Any], tracker_name: str = "ocean") -> Optional[BoxCoder]:
    """

    :param tracker_config: Dict[str, Any]
    :param tracker_name: str - name of the tracker
    :return: box_coder: BoxCoder - box coder instance
    """
    if tracker_name == "mobile_track":
        return MobileTrackBoxCoder(tracker_config=tracker_config)
    return None
