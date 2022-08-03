from collections import deque
from typing import Dict, Any, Tuple

import numpy as np
import torch

from model_training.dataset.box_coder import TrackerDecodeResult, FEARBoxCoder
from model_training.utils.utils import get_extended_crop, clamp_bbox
from model_training.tracker import Tracker
from model_training.utils.constants import TARGET_CLASSIFICATION_KEY, TARGET_REGRESSION_LABEL_KEY


class FEARTracker(Tracker):
    def get_box_coder(self, tracking_config, cuda_id: int = 0):
        return FEARBoxCoder(tracker_config=tracking_config)

    def initialize(self, image: np.ndarray, rect: np.array, **kwargs) -> None:
        """
        args:
            img(np.ndarray): RGB image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        rect = clamp_bbox(rect, image.shape)
        self.tracking_state.bbox = rect
        self.tracking_state.paths = deque([rect], maxlen=10)
        self.tracking_state.mean_color = np.mean(image, axis=(0, 1))
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            offset=self.tracking_config["template_bbox_offset"],
            crop_size=self.tracking_config["template_size"],
        )
        self._template_features = self.get_template_features(image, rect)

    def get_template_features(self, image, rect):
        template_crop, template_bbox, _ = get_extended_crop(
            image=image,
            bbox=rect,
            offset=self.tracking_config["template_bbox_offset"],
            crop_size=self.tracking_config["template_size"],
        )
        img = self._preprocess_image(template_crop, self._template_transform)
        return self.net.get_features(img)

    def update(self, image: np.ndarray, *kw) -> Dict[str, Any]:
        """
        args:
            img(np.ndarray): RGB image
        return:
            bbox(np.array):[x, y, width, height]
        """
        search_crop, search_bbox, padded_bbox = get_extended_crop(
            image=image,
            bbox=self.tracking_state.bbox,
            crop_size=self.tracking_config["instance_size"],
            offset=self.tracking_config["search_context"],
            padding_value=self.tracking_state.mean_color,
        )
        self.tracking_state.mapping = padded_bbox
        self.tracking_state.prev_size = search_bbox[2:]
        pred_bbox, _ = self.track(search_crop)
        pred_bbox = self._rescale_bbox(pred_bbox, self.tracking_state.mapping)
        pred_bbox = clamp_bbox(pred_bbox, image.shape)
        self.tracking_state.bbox = pred_bbox
        self.tracking_state.paths.append(pred_bbox)
        return dict(bbox=pred_bbox)

    def track(self, search_crop: np.ndarray):
        search_crop = self._preprocess_image(search_crop, self._search_transform)
        track_result = self.net.track(search_crop, self._template_features)
        return self._postprocess(track_result=track_result)

    def _postprocess(self, track_result: Dict[str, torch.Tensor]) -> Tuple[np.array, float]:
        cls_score = track_result[TARGET_CLASSIFICATION_KEY].detach().float().sigmoid()
        regression_map = track_result[TARGET_REGRESSION_LABEL_KEY].detach().float()
        classification_map, penalty = self._confidence_postprocess(cls_score=cls_score, regression_map=regression_map)
        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=classification_map,
            regression_map=track_result[TARGET_REGRESSION_LABEL_KEY],
            use_sigmoid=False,
        )
        cls_score = np.squeeze(cls_score)
        pred_bbox = self._postprocess_bbox(decoded_info=decoded_info, cls_score=cls_score, penalty=penalty)
        r_max, c_max = decoded_info.pred_coords[0]
        return pred_bbox, cls_score[r_max, c_max]
