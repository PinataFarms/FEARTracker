import os
import random
from abc import ABC
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np
from hydra.utils import instantiate
from torch.utils.data.dataloader import default_collate

from model_training.dataset.aug import BBoxCropWithOffsets, PHOTOMETRIC_AUGMENTATIONS, get_normalize_fn
from model_training.dataset.utils import handle_empty_bbox, read_img
from model_training.utils.utils import ensure_bbox_boundaries, convert_center_to_bbox, get_extended_crop
from model_training.utils.constants import (
    TRACKER_TARGET_SEARCH_FILENAME_KEY,
    TRACKER_TARGET_TEMPLATE_FILENAME_KEY,
    TRACKER_TARGET_SEARCH_INDEX_KEY,
    TRACKER_TARGET_TEMPLATE_INDEX_KEY,
    IMAGE_FILENAME_KEY,
    DATASET_NAME_KEY,
    SAMPLE_INDEX_KEY
)


def collate_fn(batch: Any) -> Any:
    """
    Almost the same as default_collate, but does not collate indexes and filenames (they are kepts as lists)
    """
    skip_keys = [IMAGE_FILENAME_KEY, SAMPLE_INDEX_KEY, DATASET_NAME_KEY]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch_collated: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch_collated[k] = out

    return batch_collated


class TrackingDataset(ABC):
    def __init__(self, config):
        dataset_config = config["dataset"]
        self.sizes_config = dataset_config.pop("sizes")
        self.config = dataset_config
        self.item_sampler = instantiate(dataset_config["sampling"])
        self.item_sampler.parse_samples()
        self.max_deep_supervision_stride: Optional[int] = config.get("max_deep_supervision_stride", None)
        self.search_context = self.sizes_config["search_context"] * 2
        self.common_transforms = PHOTOMETRIC_AUGMENTATIONS

    def __str__(self):
        return self.config["sampling"]["data_path"]

    def __len__(self) -> int:
        return len(self.item_sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_anno = self._get_item_anno(idx=idx)
        item_data = self._parse_anno(item_anno)
        item_data = self._transform(item_data)
        item_dict = self._form_anno_dict(item_data)
        item_dict = self._add_index(idx, item_anno, item_dict)
        return item_dict

    def _get_item_anno(self, idx: int) -> Any:
        return self.item_sampler.extract_sample(idx=idx)

    def _parse_anno(self, item_anno: Any) -> Any:
        template_item, search_item = item_anno["template"], item_anno["search"]
        template_image = read_img(os.path.join(self.config["root"], template_item["img_path"]))
        search_image = read_img(os.path.join(self.config["root"], search_item["img_path"]))

        template_bbox = ensure_bbox_boundaries(eval(template_item["bbox"]), img_shape=template_image.shape[:2])
        search_bbox = ensure_bbox_boundaries(eval(search_item["bbox"]), img_shape=search_image.shape[:2])
        return dict(
            template_image=template_image,
            template_bbox=template_bbox,
            search_image=search_image,
            search_bbox=search_bbox,
            search_presence=search_item["presence"],
        )

    def _form_anno_dict(self, item_data: Any) -> Any:
        return item_data

    def _add_index(self, idx: int, annotation: Any, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        template_item, search_item = annotation["template"], annotation["search"]
        item_dict.update(
            {
                TRACKER_TARGET_SEARCH_FILENAME_KEY: search_item["img_path"],
                TRACKER_TARGET_TEMPLATE_FILENAME_KEY: template_item["img_path"],
                TRACKER_TARGET_SEARCH_INDEX_KEY: search_item.name,
                TRACKER_TARGET_TEMPLATE_INDEX_KEY: template_item.name,
                SAMPLE_INDEX_KEY: idx,
                DATASET_NAME_KEY: search_item["dataset"],
            }
        )
        return item_dict

    def _get_search_context(self):
        context_range = self.sizes_config.get("context_range", 0.5)
        min_context = self.search_context - context_range / 2
        return random.random() * context_range + min_context

    def get_search_transform(self, image: np.array, bbox: np.array) -> Tuple[np.array, np.array]:
        search_size = self.sizes_config["search_image_size"]
        crop, bbox, padded_bbox = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=search_size * 2,
            offset=self._get_search_context(),
        )
        bbox_crop = convert_center_to_bbox(
            [
                crop.shape[0] // 2,
                crop.shape[1] // 2,
                search_size,
                search_size,
            ],
        )
        crop_aug = BBoxCropWithOffsets(
            bbox_crop=bbox_crop,
            scale=self.sizes_config["search_image_scale"],
            shift=self.sizes_config["search_image_shift"],
            crop_size=search_size,
        )
        result = crop_aug(image=crop, bboxes=[bbox])
        crop, bbox = result["image"], result["bboxes"][0]
        bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(bbox),
                img_shape=(search_size, search_size),
            )
        )
        return crop, bbox

    def get_template_transform(self, image: np.array, bbox: np.array) -> Tuple[np.array, np.array]:
        template_size = self.sizes_config["template_image_size"]
        crop, bbox, _ = get_extended_crop(
            image=image,
            bbox=bbox,
            crop_size=template_size,
            offset=self.sizes_config["template_bbox_offset"],
        )
        bbox = handle_empty_bbox(
            ensure_bbox_boundaries(
                np.array(bbox),
                img_shape=(template_size, template_size),
            )
        )
        return crop, bbox

    def resample(self):
        self.item_sampler.resample()

    def transform(self, image, bbox):
        """
        image - crop image with centred bbox with additional context amount
        bbox - centred bounding box inside crop

        Here we add some shift, scale transformations, because we have crop with centred bounding box
        We do it in following steps:
        1) Apply photometric transformations to image
        2) get centred square (crop_size, crop_size) crop bounding box, which contains centred object bounding box
        3) get nearly centred centred crop from centred crop bbox using BBoxCropWithOffsets and apply changes to
        object bounding box and image
        """
        full_aug_list = [*self.common_transforms, get_normalize_fn(self.config.get("normalize", "imagenet"))]
        bbox_params = {"format": "coco", "min_visibility": 0, "label_fields": ["category_id"], "min_area": 0}
        transform = A.Compose(full_aug_list, bbox_params=bbox_params)
        result = transform(image=image, bboxes=[bbox], category_id=["bbox"])
        image, bbox = result["image"], np.array(result["bboxes"][0])
        return image, bbox

    def _get_crops(self, item_data):
        template_crop, template_bbox = self.get_template_transform(
            item_data["template_image"], item_data["template_bbox"]
        )
        search_crop, search_bbox = self.get_search_transform(item_data["search_image"], item_data["search_bbox"])
        return template_crop, template_bbox, search_crop, search_bbox

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrackingDataset":
        return cls(config)

    def get_collate_fn(self) -> Any:
        """
        Returns default collate method. If you need custom collate logic - override this
        method and return custom collate function.
        BaseModel will check whether dataset has 'collate_fn' attribute and use it whenever possible.
        """
        return collate_fn
