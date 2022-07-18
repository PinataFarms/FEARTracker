from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Callable, Union, Optional

import albumentations as albu
import numpy as np
import torch
import torch.nn as nn

from model_training.dataset.box_coder import TrackerDecodeResult
from model_training.utils.utils import to_device, limit, squared_size


class TrackingState:
    def __init__(self) -> None:
        super().__init__()
        self.frame_h = 0
        self.frame_w = 0
        self.bbox: Optional[np.array] = None
        self.mapping: Optional[np.array] = None
        self.prev_size = None
        self.mean_color = None

    def save_frame_shape(self, frame: np.ndarray) -> None:
        self.frame_h = frame.shape[0]
        self.frame_w = frame.shape[1]


class Tracker(ABC):
    def __init__(self, model: nn.Module, cuda_id: Union[int, str] = 0, **tracking_config: Any) -> None:
        super().__init__()
        self.cuda_id = cuda_id
        self.tracking_config = tracking_config
        self.tracking_state = TrackingState()
        self.net = model
        self.box_coder = self.get_box_coder(tracking_config, cuda_id)
        self._template_features = None
        self._template_transform = self._get_default_transform(img_size=tracking_config["template_size"])
        self._search_transform = self._get_default_transform(img_size=tracking_config["instance_size"])
        self.window = self._get_tracking_window(tracking_config["windowing"], tracking_config["score_size"])
        self.to_device(cuda_id)

    @staticmethod
    def _array_to_batch(x: np.ndarray) -> torch.Tensor:
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    @abstractmethod
    def get_box_coder(self, tracking_config, cuda_id: int = 0):
        pass

    def to_device(self, cuda_id):
        self.cuda_id = cuda_id
        self.window = to_device(self.window, cuda_id)
        self.box_coder = self.box_coder.to_device(self.cuda_id)

    @staticmethod
    def _get_tracking_window(windowing: str, score_size: int) -> torch.Tensor:
        """

        :param windowing: str - window creation type
        :param score_size: int - size of classification map
        :return: window: np.array
        """
        if windowing == "cosine":
            return torch.from_numpy(np.outer(np.hanning(score_size), np.hanning(score_size)))
        return torch.ones(int(score_size), int(score_size))

    @staticmethod
    def _get_default_transform(img_size):
        pipeline = albu.Compose(
            [
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def process(a):
            r = pipeline(image=a)
            return r["image"]

        return process

    def _rescale_bbox(self, bbox: np.array, padded_box) -> np.array:
        w_scale = padded_box[2] / self.tracking_config["instance_size"]
        h_scale = padded_box[3] / self.tracking_config["instance_size"]
        bbox[0] = round(bbox[0] * w_scale + padded_box[0])
        bbox[1] = round(bbox[1] * h_scale + padded_box[1])
        bbox[2] = max(3, round(bbox[2] * w_scale))
        bbox[3] = max(3, round(bbox[3] * h_scale))
        return list(map(int, bbox))

    def _get_scale(self, bbox: np.ndarray) -> int:
        wc_z = bbox[2] + self.tracking_config["search_context"] * sum(bbox[2:])
        hc_z = bbox[3] + self.tracking_config["search_context"] * sum(bbox[2:])
        return max(round(np.sqrt(wc_z * hc_z)), 1)

    def _preprocess_image(self, image: np.ndarray, transform: Callable) -> torch.Tensor:
        img = transform(image[:, :, :3])
        if image.shape[2] > 3:
            img = np.concatenate([img, image[:, :, 3:]], axis=2)
        img = self._array_to_batch(img).float()
        img = to_device(img, cuda_id=self.cuda_id)
        return img

    def reset(self) -> None:
        self._template_features = None

    def initialize(self, image: np.ndarray, rect: np.array, **kwargs) -> None:
        """
        args:
            img(np.ndarray): RGB image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        pass

    def update(self, image: np.ndarray, *kw) -> Dict[str, Any]:
        """
        args:
            img(np.ndarray): RGB image
        return:
            bbox(np.array):[x, y, width, height]
        """
        return {"bbox": self.tracking_state.bbox}

    def _smooth_size(self, size: np.array, prev_size: np.array, lr: float) -> Tuple[float, float]:
        """
        Tracking smoothing logic matches the code of Siamese Tracking
        https://www.robots.ox.ac.uk/~luca/siamese-fc.html
        :param size: np.array([w, h]) - predicted bbox size
        :param prev_size: np.array([w, h]) - bbox size on previous frame
        :param lr: float - smoothing learning rate
        :return: Tuple[float, float] - smoothed size
        """
        size = size * lr
        prev_size = prev_size * (1 - lr)
        w = prev_size[0] + lr * (size[0] + prev_size[0])
        h = prev_size[1] + lr * (size[1] + prev_size[1])
        return w, h

    def _get_point_offset(self, pred_bbox: np.array) -> Tuple[float, float]:
        pred_xs = pred_bbox[0] + (pred_bbox[2] / 2)
        pred_ys = pred_bbox[1] + (pred_bbox[3] / 2)

        diff_xs = pred_xs - self.tracking_config["instance_size"] // 2
        diff_ys = pred_ys - self.tracking_config["instance_size"] // 2
        return diff_xs, diff_ys

    def _postprocess_bbox(
            self, decoded_info: TrackerDecodeResult, cls_score: np.array, penalty: Any = None
    ) -> np.array:
        pred_bbox = np.squeeze(decoded_info.bbox.cpu().numpy())
        if not self.tracking_config.get("smooth", False):
            return pred_bbox

        prev_size = self.tracking_state.prev_size
        r_max, c_max = decoded_info.pred_coords[0]
        # size learning rate
        lr = (penalty[r_max, c_max] * cls_score[r_max, c_max] * self.tracking_config["lr"]).item()

        pred_size = np.array(pred_bbox[2:])
        pred_w, pred_h = self._smooth_size(pred_size, prev_size=prev_size, lr=lr)
        predicted_bbox = np.array([pred_bbox[0], pred_bbox[1], pred_w, pred_h])
        return predicted_bbox

    def _confidence_postprocess(
            self, cls_score: np.ndarray, regression_map: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """

        :param cls_score: torch.Tensor - classification score
        :param pred_location: torch.Tensor - predicted locations
        :param prev_size: np.array - size from previous frame
        :return: penalty_score: np.ndarray - updated cls_score
        """
        if not self.tracking_config.get("smooth", False):
            return cls_score, None
        prev_size = self.tracking_state.prev_size
        pred_location = torch.stack(
            [
                self.box_coder.grid_x - regression_map[:, 0, ...],
                self.box_coder.grid_y - regression_map[:, 1, ...],
                self.box_coder.grid_x + regression_map[:, 2, ...],
                self.box_coder.grid_y + regression_map[:, 3, ...],
            ],
            dim=1,
        )[0]
        s_c = limit(
            squared_size(pred_location[2] - pred_location[0], pred_location[3] - pred_location[1])
            / (squared_size(prev_size[0], prev_size[1]))
        )  # scale penalty
        r_c = limit(
            (prev_size[0] / prev_size[1])
            / ((pred_location[2] - pred_location[0]) / (pred_location[3] - pred_location[1]))
        )  # ratio penalty

        penalty = torch.exp(-(r_c * s_c - 1) * self.tracking_config["penalty_k"])
        pscore = penalty * cls_score

        # window penalty
        pscore = (
                pscore * (1 - self.tracking_config["window_influence"])
                + self.window * self.tracking_config["window_influence"]
        )
        return pscore, penalty.cpu().numpy()
