import random
from typing import Tuple

import cv2
import numpy as np
import torch


def convert_center_to_bbox(center: np.array) -> np.array:
    """
    Args:
        center: np.array - bbox in xcycwh format
    Returns:
        bbox: np.array - bbox in xywh format
    """
    return np.array([center[0] - center[2] / 2, center[1] - center[3] / 2, center[2], center[3]]).astype("int")


def get_regression_weight_label(
    bbox, image_size: int = 255, map_size: int = 25, r_pos: int = 2, r_neg: int = 0
) -> torch.Tensor:
    bbox_c_x, bbox_c_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
    sz_x, sz_y = np.floor(float(bbox_c_x / image_size * map_size)), np.floor(float(bbox_c_y / image_size * map_size))
    x, y = np.meshgrid(np.arange(0, map_size) - sz_x, np.arange(0, map_size) - sz_y)

    dist_to_center = np.abs(x) + np.abs(y)
    label = np.where(
        dist_to_center <= r_pos,
        np.ones_like(y),
        np.where(dist_to_center < r_neg, 0.5 * np.ones_like(y), np.zeros_like(y)),
    )
    return torch.from_numpy(label)


def read_img(path: str) -> np.array:
    """
    Args:
        path: image path
    Returns: image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.copy()


def get_max_side_near_bbox(bbox: np.array, frame: np.array) -> Tuple[np.array, str]:
    """
    return on of sides relative to bbox with maximum area and its name
    Args:
        bbox: in xywh format
        frame: image
    Returns: np.array, side
    """
    sides = [frame[:, : bbox[0]], frame[:, bbox[0] + bbox[2] :], frame[: bbox[1], :], frame[bbox[1] + bbox[3] :]]
    side_names = ["left", "right", "top", "bottom"]

    max_side, max_side_name, max_area = None, None, None
    for side, side_name in zip(sides, side_names):
        side_area = side.shape[0] * side.shape[1]
        if max_area is None or max_area < side_area:
            max_side, max_side_name, max_area = side, side_name, side_area
    return max_side, max_side_name


def get_similar_random_crop(area: float, shape: Tuple[int, int]) -> np.array:
    """
    return crop with similar to selected area inside image with selected shape
    Args:
        area: estimated area of wanted crop
        shape: image shape where to get crop
    """
    crop_area = random.normalvariate(area, area / 12)
    crop_first_side = random.normalvariate(crop_area ** 0.5, (crop_area ** 0.5) / 8)
    crop_second_side = crop_area / crop_first_side
    if shape[0] > shape[1]:
        crop_h, crop_w = max(crop_first_side, crop_second_side), min(crop_first_side, crop_second_side)
    else:
        crop_h, crop_w = min(crop_first_side, crop_second_side), max(crop_first_side, crop_second_side)
    crop_w, crop_h = int(min(crop_w, shape[1])), int(min(crop_h, shape[0]))
    crop_x, crop_y = random.randint(0, shape[1] - crop_w), random.randint(0, shape[0] - crop_h)
    return np.array([crop_x, crop_y, crop_w, crop_h]).astype("int")


def get_negative_crop(bbox: np.array, image: np.array) -> np.array:
    """
    get crop outside bounding box on image
    Args:
        bbox: in xywh format
        image: np.array
    Returns: bbox with negative crop in xywh format
    """
    side, side_name = get_max_side_near_bbox(bbox, image)
    negative_bbox = get_similar_random_crop(bbox[2] * bbox[3], side.shape)
    if side_name == "right":
        negative_bbox[0] += bbox[0] + bbox[2]
    elif side_name == "bottom":
        negative_bbox[1] += bbox[1] + bbox[3]
    return negative_bbox


def convert_xywh_to_xyxy(bbox: np.array) -> np.array:
    """
    Args:
        bbox: in xywh format
    Returns:
        bbox: in xyxy format
    """
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox


def convert_bbox_to_center(bbox: np.array) -> np.array:
    """
    Args:
        bbox: np.array - bbox in xywh format
    Returns:
        center: np.array - bbox in xcycwh format
    """
    return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]).astype("int")


def augment_context(
    context: np.array, min_scale: float, max_scale: float, min_shift: float, max_shift: float
) -> np.array:
    """
    Augument context bbox for more robust training
    Args:
        context: bbox in xywh format to augment
        min_scale: minimum scale change for side to scale
        max_scale: maximum scale change for side to scale
        min_shift: minimum shift change for side to scale
        max_shift: maximum shift change for side to scale
    These values can be negative for shrinking and padding to left. Scale and shift applied at once for two sides
    Returns:
        bbox: new context bbox in xywh format
    """
    xc, yc, w, h = convert_bbox_to_center(context)
    context_side = (context[2] * context[3]) ** 0.5
    scale = random.uniform(min_scale, max_scale) * random.choice([-1.0, 1.0])
    shift = random.uniform(min_shift, max_shift) * random.choice([-1.0, 1.0])
    w_new = w + context_side * scale
    h_new = h + context_side * scale
    xc_new = xc + context_side * shift
    yc_new = yc + context_side * shift
    return convert_center_to_bbox([xc_new, yc_new, w_new, h_new])


def handle_empty_bbox(bbox: np.array, min_bbox: int = 3) -> np.array:
    bbox[2] = max(bbox[2], min_bbox)
    bbox[3] = max(bbox[3], min_bbox)
    return bbox
