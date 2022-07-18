import random
from typing import Optional, List, Union, Tuple, Any, Dict

import cv2
import numpy as np
import albumentations as A

PHOTOMETRIC_AUGMENTATIONS = [
    A.OneOf(
        [
            A.Blur(),
            A.MotionBlur(),
            A.MedianBlur(),
            A.GaussianBlur(),
            A.GlassBlur(),
        ],
        p=0.2,
    ),
    A.OneOf(
        [A.GaussNoise(var_limit=(10, 35)), A.ImageCompression(quality_lower=50), A.ISONoise(), A.MultiplicativeNoise()],
        p=0.2,
    ),
    A.OneOf([A.RandomRain(), A.RandomShadow()], p=0.05),
    A.Downscale(scale_min=0.5, scale_max=0.5, p=0.2),
]

TRACKING_AUGMENTATIONS = [
    A.OneOf(
        [
            A.ToGray(),
            A.ToSepia(),
        ],
        p=0.05,
    ),
    A.OneOf(
        [
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
            A.Emboss(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.Equalize(),
            A.ColorJitter(),
            A.RandomToneCurve(),
        ],
        p=0.5,
    ),
]


class BBoxCropWithOffsets(A.DualTransform):
    """
    This augmentation get crop from image in following steps:
    1) Get initial bounding box `bbox_crop`
    2) Slightly transfrom it with scale and offset
    3) Get crop from modified bounding box
    4) Rescale bounding box to (crop_size, crop_size) square
    """

    def __init__(
        self,
        bbox_crop: List[int],
        scale: Union[float, Tuple[float, float]],
        shift: Union[float, Tuple[float, float]],
        crop_size: int,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        """

        Args:
            bbox_crop: initial bounding box, where to crop
            scale: scale, used to modify initial bounding box - from 0. to 1. in percentages relative to width and height
            shift: shift, used to modify initial bounding box - in pixels, max_offset from initial bounding box center
            crop_size: output image size after getting crop and resizing it to square
            always_apply: is always apply transformation
            p: probability of transformation use
        """
        super().__init__(always_apply, p)
        self.scale = A.to_tuple(scale)
        self.shift = A.to_tuple(shift)
        self.bbox_crop = bbox_crop
        self.crop_size = crop_size

    @property
    def targets_as_params(self) -> List:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transformations to initial bounding box crop and return new bounding box
        """
        x, y, w, h = self.bbox_crop
        img_h, img_w = params["image"].shape[:2]
        scale_x = random.uniform(min(self.scale), max(self.scale))
        scale_y = random.uniform(min(self.scale), max(self.scale))
        shift_x = random.uniform(min(self.shift), max(self.shift))
        shift_y = random.uniform(min(self.shift), max(self.shift))
        new_x = max(0, x - scale_x * w / 2 + shift_x)
        new_y = max(0, y - scale_y * h / 2 + shift_y)
        new_w = min(img_w, new_x + w + scale_x * w) - new_x
        new_h = min(img_h, new_y + h + scale_y * h) - new_y
        return {"modified_bbox_crop": [new_x, new_y, new_w, new_h]}

    def apply(self, image: np.array, **params: Any) -> np.array:
        return self.affine_crop(image, params["modified_bbox_crop"], self.crop_size)

    def apply_to_bbox(self, bbox: np.array, **params: Any) -> np.array:
        """
        Get new bounding box coordinates taking into account that we crop initial image and resize this crop to
        `self.crop_size`
        """
        crop_bbox = params["modified_bbox_crop"]
        new_x = (bbox[0] - crop_bbox[0]) * self.crop_size / crop_bbox[2]
        new_y = (bbox[1] - crop_bbox[1]) * self.crop_size / crop_bbox[3]
        new_w = bbox[2] * self.crop_size / crop_bbox[2]
        new_h = bbox[3] * self.crop_size / crop_bbox[3]
        if new_x < 0:
            new_x, new_w = 0, new_w + new_x
        if new_y < 0:
            new_y, new_h = 0, new_y + new_h
        new_w = min(self.crop_size, new_x + new_w) - new_x
        new_h = min(self.crop_size, new_y + new_h) - new_y
        return tuple([int(new_x), int(new_y), int(new_w), int(new_h)])

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return "bbox_crop", "scale", "shift", "crop_size"

    @staticmethod
    def affine_crop(image: np.array, bbox: List[int], out_size: int) -> np.array:
        """
        Get image specific image crop from `bbox` and resize it to (out_size, out_size) square
        """
        crop_bbox = [float(x) for x in bbox]
        a = (out_size - 1) / (crop_bbox[2])
        b = (out_size - 1) / (crop_bbox[3])
        c = -a * crop_bbox[0]
        d = -b * crop_bbox[1]
        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_size, out_size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return crop


def get_normalize_fn(mode: str = "imagenet") -> Optional[A.Normalize]:
    if mode == "imagenet":
        return A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif mode == "mean":
        return A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        return None
