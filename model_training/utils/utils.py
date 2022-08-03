from typing import Tuple, Dict, Any, List, Union, Optional

import torch
import cv2
import numpy as np
import albumentations as A

BBox = Union[List, np.array]


def to_device(x: Union[torch.Tensor, torch.nn.Module], cuda_id: int = 0) -> torch.Tensor:
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def get_iou(bb1: np.array, bb2: np.array) -> float:
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    x_a = np.max((x1, x2))
    y_a = np.max((y1, y2))
    x_b = np.min((x1 + w1, x2 + w2))
    y_b = np.min((y1 + h1, y2 + h2))
    inter_area = np.max(((x_b - x_a + 1), 0)) * np.max(((y_b - y_a + 1), 0))
    box_a_area = ((x1 + w1) - x1 + 1) * ((y1 + h1) - y1 + 1)
    box_b_area = ((x2 + w2) - x2 + 1) * ((y2 + h2) - y2 + 1)
    iou = inter_area / (box_a_area + box_b_area - inter_area)
    return iou


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def ensure_bbox_boundaries(bbox: np.array, img_shape: Tuple[int, int]) -> np.array:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def limit(radius: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    if isinstance(radius, torch.Tensor):
        return torch.maximum(radius, 1.0 / radius)
    return np.maximum(radius, 1.0 / radius)


def squared_size(w: int, h: int) -> Union[torch.Tensor, float]:
    pad = (w + h) * 0.5
    size = (w + pad) * (h + pad)
    if isinstance(size, torch.Tensor):
        return torch.sqrt(size)
    return np.sqrt(size)


def python2round(float_num: float) -> float:
    """
    Use python2 round function in python3
    """
    if round(float_num + 1) - round(float_num) != 1:
        return float_num + abs(float_num) / float_num * 0.5
    return round(float_num)


def bbox_from_cxy_wh(position: np.array, size: np.array) -> np.array:
    return np.array(
        [
            float(max(float(0), position[0] - size[0] / 2)),
            float(max(float(0), position[1] - size[1] / 2)),
            float(size[0]),
            float(size[1]),
        ]
    )


def position_from_bbox(bbox: np.array) -> np.array:
    lx, ly, w, h = bbox
    position = np.array([lx + w / 2, ly + h / 2])
    return position


def get_subwindow_tracking(
    frame: np.ndarray, bbox: np.array, template_size: int, original_sz: int, avg_chans: np.array
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    SiamFC type cropping
    """
    crop_info = dict()
    position = position_from_bbox(bbox=bbox)
    sz = original_sz
    im_sz = frame.shape
    c = (original_sz + 1) / 2
    context_xmin = round(position[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(position[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0.0, -context_xmin))
    top_pad = int(max(0.0, -context_ymin))
    right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = tuple(map(int, frame.shape))
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = frame
        if top_pad:
            te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad :, :] = avg_chans
        im_patch_original = te_im[
            int(context_ymin) : int(context_ymax + 1), int(context_xmin) : int(context_xmax + 1), :
        ]
    else:
        tete_im = np.zeros(frame.shape[0:2])
        im_patch_original = frame[
            int(context_ymin) : int(context_ymax + 1), int(context_xmin) : int(context_xmax + 1), :
        ]

    if not np.array_equal(template_size, original_sz):
        im_patch = cv2.resize(im_patch_original, (template_size, template_size))
    else:
        im_patch = im_patch_original

    crop_info["crop_cords"] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info["empty_mask"] = tete_im
    crop_info["pad_info"] = [top_pad, left_pad, r, c]

    return im_patch, crop_info


def unravel_index(index: Any, shape: Tuple[int, int]) -> Tuple[int, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


@torch.no_grad()
def make_grid(score_size: int, total_stride: int, instance_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Each element of feature map on input search image
    :return: H*W*2 (position for each element)
    """

    x, y = np.meshgrid(
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
        np.arange(0, score_size) - np.floor(float(score_size // 2)),
    )

    grid_x = x * total_stride + instance_size // 2
    grid_y = y * total_stride + instance_size // 2
    grid_x = torch.from_numpy(grid_x[np.newaxis, :, :])
    grid_y = torch.from_numpy(grid_y[np.newaxis, :, :])
    return grid_x, grid_y


def clamp_bbox(bbox: np.array, shape: Tuple[int, int], min_side: int = 3) -> np.array:
    bbox = ensure_bbox_boundaries(bbox, img_shape=shape)
    x, y, w, h = bbox
    img_h, img_w = shape[0], shape[1]
    if w < min_side:
        w = min_side
        x -= max(0, x + w - img_w)
    if h < min_side:
        h = min_side
        y -= max(0, y + h - img_h)
    return np.array([x, y, w, h])


def get_extended_crop(
    image: np.array, bbox: np.array, crop_size: int, offset: float, padding_value: np.array = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Extend bounding box by {offset} percentages, pad all sides, rescale to fit {crop_size} and pad all sides to make
    {side}x{side} crop
    Args:
        image: np.array
        bbox: np.array - in xywh format
        crop_size: int - output crop size
        offset: float - how much percentages bbox extend
        padding_value: np.array - value to pad

    Returns:
        crop_image: np.array
        crop_bbox: np.array
    """
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))
    bbox_params = {"format": "coco", "min_visibility": 0, "label_fields": ["category_id"], "min_area": 0}
    resize_aug = A.Compose([A.Resize(crop_size, crop_size)], bbox_params=bbox_params)
    context = extend_bbox(bbox, offset)
    pad_left, pad_top = max(-context[0], 0), max(-context[1], 0)
    pad_right, pad_bottom = max(context[0] + context[2] - image.shape[1], 0), max(
        context[1] + context[3] - image.shape[0], 0
    )
    crop = image[
        context[1] + pad_top : context[1] + context[3] - pad_bottom,
        context[0] + pad_left : context[0] + context[2] - pad_right,
    ]

    padded_crop = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padding_value
    )
    padded_bbox = np.array([bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]])
    padded_bbox = ensure_bbox_boundaries(padded_bbox, img_shape=padded_crop.shape[:2])
    result = resize_aug(image=padded_crop, bboxes=[padded_bbox], category_id=["bbox"])
    image, bbox = result["image"], np.array(result["bboxes"][0])
    return image, bbox, context


def rescale_crop(image: np.array, bbox: np.array, out_size: int, padding: Any = (0, 0, 0)) -> Tuple[np.array, np.array]:
    """
    Take crop from bbox position and rescale it to out_size
    Args:
        image: np.array
        bbox: bbox in xywh
        out_size: output image size
        padding: padding values
    Returns:
        crop: np.array - image crop rescaled to out_size
        mapping: np.array - transformation matrix to revert or reapply
    """
    a = (out_size - 1) / bbox[2]
    b = (out_size - 1) / bbox[3]
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_size, out_size), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop, mapping


def get_side_with_context(bbox: np.array, context_amount: float) -> float:
    """
    Args:
        bbox: bbox in xywh format
        context_amount: float, how much additional context to add

    Returns:
        scale - scaling value
    """
    w, h = bbox[2], bbox[3]
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    return max(round(np.sqrt(wc_z * hc_z)), 1)


def get_crop_context(
    image: np.array,
    bbox: BBox,
    context_amount: float = 0.5,
    bbox_side_ratio: float = 0.25,
    crop_size: int = 512,
    padding_value: Optional[np.array] = None,
) -> Tuple[np.array, BBox, Any]:
    """
    image: np.array - image
    bbox: in xywh format
    context_amount: float (0.-1.) how much additional context to add
    small_crop_size: bounding box size in rescaled image
    large_crop_size: rescaled image size
    Get image crop around bounding box with additional context amount and rescale it properly
    Note: Bounding box is centred in crop
    """
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))
    side_size = int(crop_size * bbox_side_ratio)
    cx, cy = [bbox[0] + (bbox[2] / 2.0), bbox[1] + (bbox[3] / 2.0)]
    s_z = get_side_with_context(bbox, context_amount)
    scale_z = side_size / s_z
    d_search = (crop_size - side_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    crop_image, mapping = rescale_crop(image, convert_center_to_bbox([cx, cy, s_x, s_x]), crop_size, padding_value)
    crop_bbox = transform_bbox(bbox, mapping)
    return crop_image, crop_bbox, mapping


def convert_center_to_bbox(center: np.array) -> np.array:
    """
    Args:
        center: np.array - bbox in xcycwh format
    Returns:
        bbox: np.array - bbox in xywh format
    """
    return np.array([center[0] - center[2] / 2, center[1] - center[3] / 2, center[2], center[3]]).astype("int")


def transform_bbox(bbox: np.array, mapping: np.array, inverse: bool = False) -> np.array:
    """
    Args:
        bbox: bbox in xywh format
        mapping: mapping matrix to get global coordinates
        inverse: apply mapping or inverse mapping to undo it

    Returns: np.array - bounding box in xywh format

    """
    if inverse:
        mapping = np.linalg.pinv(np.concatenate([mapping, np.array([0, 0, 1]).reshape(1, 3)], axis=0))[:2]
    transformed_points = cv2.transform(get_points(bbox), mapping)
    x, y = transformed_points[0, 0]
    w, h = transformed_points[2, 0] - transformed_points[0, 0]
    return np.array([x, y, w, h]).astype("int")


def get_points(bbox: np.array) -> np.array:
    """
    Args:
        bbox: np.array - bounding box in xywh format
    Returns: np.array - array of points
    """
    return (
        np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[1] + bbox[3]],
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                [bbox[0] + bbox[2], bbox[1]],
            ]
        )
        .reshape((-1, 1, 2))
        .astype("float")
    )
