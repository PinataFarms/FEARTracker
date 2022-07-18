from typing import Dict, Any, Tuple

import albumentations as A
import numpy as np
import torch
from pytorch_toolbelt.utils import image_to_tensor

from model_training.dataset.aug import TRACKING_AUGMENTATIONS
from model_training.dataset.box_coder import MobileTrackBoxCoder
from model_training.dataset.tracking_dataset import TrackingDataset
from model_training.dataset.utils import get_regression_weight_label
from model_training.utils.utils import ensure_bbox_boundaries
from model_training.utils.constants import (
    TARGET_CLASSIFICATION_KEY,
    TARGET_REGRESSION_WEIGHT_KEY,
    TARGET_REGRESSION_LABEL_KEY,
    TARGET_VISIBILITY_KEY,
    TRACKER_TARGET_SEARCH_IMAGE_KEY,
    TRACKER_TARGET_TEMPLATE_IMAGE_KEY,
    TRACKER_TARGET_BBOX_KEY,
    TRACKER_TEMPLATE_BBOX_KEY,
)
from model_training.utils.logger import create_logger

logger = create_logger(__name__)


class SiameseTrackingDataset(TrackingDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.box_coder = MobileTrackBoxCoder(config["tracker"])

    def _transform(self, item_data: Any) -> Any:
        search_presence = item_data["search_presence"]
        template_crop, template_bbox, search_crop, search_bbox = self._get_crops(item_data)
        template_crop, search_crop = self._add_color_augs(search_image=search_crop, template_image=template_crop)
        template_crop, template_bbox = self.transform(image=template_crop, bbox=template_bbox)
        search_crop, search_bbox = self.transform(image=search_crop, bbox=search_bbox)

        crop_size = self.sizes_config["search_image_size"]
        search_bbox = ensure_bbox_boundaries(np.array(search_bbox), img_shape=(crop_size, crop_size))

        grid_size = self.config["regression_weight_label_size"]
        if search_presence:
            regression_weight_label = get_regression_weight_label(search_bbox, crop_size, grid_size)
            encoded_result = self.box_coder.encode(torch.from_numpy(search_bbox).reshape(1, 4))
            regression_map = encoded_result.regression_map[0]
            classification_label = encoded_result.classification_label[0]
        else:
            regression_weight_label = torch.zeros(grid_size, grid_size)
            regression_map = torch.zeros(4, grid_size, grid_size)
            classification_label = torch.zeros(1, grid_size, grid_size)
        return {
            TARGET_REGRESSION_LABEL_KEY: regression_map,
            TARGET_CLASSIFICATION_KEY: classification_label,
            TARGET_REGRESSION_WEIGHT_KEY: regression_weight_label,
            TRACKER_TARGET_TEMPLATE_IMAGE_KEY: image_to_tensor(template_crop),
            TRACKER_TEMPLATE_BBOX_KEY: torch.tensor(template_bbox),
            TRACKER_TARGET_SEARCH_IMAGE_KEY: image_to_tensor(search_crop),
            TRACKER_TARGET_BBOX_KEY: torch.tensor(search_bbox),
            TARGET_VISIBILITY_KEY: np.expand_dims(search_presence, axis=0),
        }

    def _add_color_augs(self, search_image: np.ndarray, template_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        color_aug = A.Compose(TRACKING_AUGMENTATIONS, additional_targets={"search_image": "image"})
        aug_res = color_aug(image=template_image, search_image=search_image)
        return aug_res["image"], aug_res["search_image"]
