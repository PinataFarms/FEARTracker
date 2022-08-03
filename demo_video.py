import os
import cv2
import imageio.v3 as iio
import numpy as np
from fire import Fire
from hydra.utils import instantiate
from typing import Optional, List

from model_training.tracker.fear_tracker import FEARTracker
from model_training.utils.torch import load_from_lighting
from model_training.utils.hydra import load_hydra_config_from_path


def get_tracker(config_path: str, config_name: str, weights_path: str) -> FEARTracker:
    config = load_hydra_config_from_path(config_path=config_path, config_name=config_name)
    model = instantiate(config["model"])
    model = load_from_lighting(model, weights_path).cuda().eval()
    tracker: FEARTracker = instantiate(config["tracker"], model=model)
    return tracker


def track(tracker: FEARTracker, frames: List[np.ndarray], initial_bbox: np.ndarray) -> List[np.ndarray]:
    tracked_bboxes = [initial_bbox]
    tracker.initialize(frames[0], initial_bbox)
    for frame in frames[1:]:
        tracked_bbox = tracker.update(frame)["bbox"]
        tracked_bboxes.append(tracked_bbox)
    return tracked_bboxes


def draw_bbox(image: np.ndarray, bbox: np.ndarray, width: int = 5) -> np.ndarray:
    image = image.copy()
    x, y, w, h = bbox
    return cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


def visualize(frames: List[np.ndarray], tracked_bboxes: List[np.ndarray]):
    visualized_frames = []
    for frame, bbox in zip(frames, tracked_bboxes):
        visualized_frames.append(draw_bbox(frame, bbox))
    return visualized_frames


def main(
    initial_bbox: List[int] = [163, 53, 45, 174],
    video_path: str = "assets/test.mp4",
    output_path: str = "outputs/test.mp4",
    config_path: str = "model_training/config",
    config_name: str = "fear_tracker",
    weights_path: str = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
):
    tracker = get_tracker(config_path=config_path, config_name=config_name, weights_path=weights_path)
    video, metadata = iio.imread(video_path), iio.immeta(video_path, exclude_applied=False)
    initial_bbox = np.array(initial_bbox)
    tracked_bboxes = track(tracker, video, initial_bbox)
    visualized_video = visualize(video, tracked_bboxes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iio.imwrite(output_path, visualized_video, fps=metadata["fps"])


if __name__ == '__main__':
    Fire(main)
