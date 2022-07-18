from typing import Dict, Tuple, Any, Iterable

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor
from pytorch_toolbelt.utils.visualization import hstack_autopad, vstack_autopad, vstack_header
from torch import Tensor
from torch.utils.data import ConcatDataset
from torchmetrics import MetricCollection
from torchvision.ops import box_convert, box_iou

from model_training.dataset.box_coder import TrackerDecodeResult, MobileTrackBoxCoder
from model_training.dataset.utils import read_img
from model_training.metrics import DatasetAwareMetric, BoxIoUMetric, TrackingFailureRateMetric, box_iou_metric
from model_training.train.base_lightning_model import BaseLightningModel
from model_training.train.loss import FEARLoss
from model_training.utils.utils import get_iou
from model_training.utils.constants import (
    TRACKER_TARGET_TEMPLATE_IMAGE_KEY,
    TRACKER_TARGET_SEARCH_IMAGE_KEY,
    TRACKER_TARGET_BBOX_KEY,
    TRACKER_TARGET_TEMPLATE_FILENAME_KEY,
    TRACKER_TARGET_SEARCH_FILENAME_KEY,
    TRACKER_TARGET_NEGATIVE_IMAGE_KEY,
    TARGET_CLASSIFICATION_KEY,
    TARGET_REGRESSION_WEIGHT_KEY,
    TARGET_REGRESSION_LABEL_KEY,
    TARGET_VISIBILITY_KEY,
    DATASET_NAME_KEY,
)
from model_training.utils.logger import create_logger

logger = create_logger(__name__)


class FEARLightningModel(BaseLightningModel):
    input_type = torch.float32
    target_type = torch.float32

    def __init__(self, model, config, train, val) -> None:
        super().__init__(model, config, train, val)
        self.tracker = instantiate(config["tracker"], model=self.model)
        self.box_coder = MobileTrackBoxCoder(tracker_config=config["tracker"])
        self.metrics = MetricCollection(
            {
                "box_iou": BoxIoUMetric(compute_on_step=True),
                "failure_rate": TrackingFailureRateMetric(compute_on_step=True),
            }
        )
        self.dataset_aware_metric = DatasetAwareMetric(metric_name="box_iou", metric_fn=box_iou_metric)
        #ToDo: Move loss creation to Hydra
        self.criterion = FEARLoss(coeffs=config["loss"]["coeffs"])

    def training_step(self, batch: Dict[str, Any], batch_nb: int):
        loss, outputs = self._training_step(batch=batch, batch_nb=batch_nb)
        return loss

    def _training_step(self, batch: Dict[str, Any], batch_nb: int):
        inputs, targets = self.get_input(batch)
        outputs = self.model.forward(inputs)
        loss = self.criterion(outputs, targets)
        total_loss, loss_dict = self.compute_loss(loss)

        decoded_info: TrackerDecodeResult = self.box_coder.decode(
            classification_map=outputs[TARGET_CLASSIFICATION_KEY],
            regression_map=outputs[TARGET_REGRESSION_LABEL_KEY],
        )
        pred_boxes = box_convert(decoded_info.bbox, "xywh", "xyxy")
        gt_boxes = box_convert(targets[TRACKER_TARGET_BBOX_KEY], "xywh", "xyxy")
        visibility_mask = (targets[TARGET_VISIBILITY_KEY][:, 0] == 1).tolist()
        datasets = list(np.array(batch[DATASET_NAME_KEY])[visibility_mask])
        pred_boxes = pred_boxes[visibility_mask]
        gt_boxes = gt_boxes[visibility_mask]
        ious = box_iou(pred_boxes, gt_boxes)

        metrics = self.metrics(ious)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )

        self.dataset_aware_metric.update(mode="train", datasets=datasets, outputs=ious)
        self.log(f"train/loss", total_loss, prog_bar=True, sync_dist=self.use_ddp)
        for key, loss in loss_dict.items():
            self.log(f"train/{key}_loss", loss, sync_dist=self.use_ddp)
        return {"loss": total_loss}, outputs

    def validation_step(self, batch: Tuple[Any, Any, str], batch_nb: int) -> Dict[str, Any]:
        _iou_threshold = 0.01
        max_samples = self.config.get("max_val_samples", 200)
        seq_ious = []
        for image_files, annotations, dataset_name in batch:
            self.tracker.initialize(read_img(image_files[0]), list(map(int, annotations[0])))
            num_samples = min(max_samples, len(annotations))
            ious = []
            failure_map = []
            for i in range(1, num_samples):
                bbox = self.tracker.update(read_img(image_files[i]))["bbox"]
                iou = get_iou(np.array(bbox), np.array(list(map(int, annotations[i]))))
                ious.append(iou)
                failure_map.append(int(iou < _iou_threshold))

            mean_iou = np.mean(ious)
            self.log(
                "valid/metrics/box_iou",
                mean_iou,
                on_epoch=True,
            )
            self.log(
                f"valid/metrics/{dataset_name}_box_iou",
                mean_iou,
                on_epoch=True,
            )
            self.log(
                f"valid/metrics/{dataset_name}_failure_rate",
                np.mean(failure_map),
                on_epoch=True,
            )
            seq_ious.append(mean_iou)
        return {"box_iou": np.mean(seq_ious)}

    def compute_loss(self, loss: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Return tuple of loss tensor and dictionary of named losses as second argument (if possible)
        """
        total_loss = 0
        for k, v in loss.items():
            total_loss = total_loss + v

        return total_loss, loss

    def get_input(self, data: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        input_keys = [
            TRACKER_TARGET_TEMPLATE_IMAGE_KEY,
            TRACKER_TARGET_SEARCH_IMAGE_KEY,
            TRACKER_TARGET_NEGATIVE_IMAGE_KEY,
        ]
        target_keys = [
            TARGET_CLASSIFICATION_KEY,
            TARGET_REGRESSION_LABEL_KEY,
            TARGET_REGRESSION_WEIGHT_KEY,
            TRACKER_TARGET_BBOX_KEY,
            TARGET_VISIBILITY_KEY,
        ]
        inputs_dict = self._convert_inputs(data, input_keys)
        targets_dict = self._convert_inputs(data, target_keys)
        inputs = [inputs_dict[TRACKER_TARGET_TEMPLATE_IMAGE_KEY], inputs_dict[TRACKER_TARGET_SEARCH_IMAGE_KEY]]
        if TRACKER_TARGET_NEGATIVE_IMAGE_KEY in inputs_dict:
            inputs.append(inputs_dict[TRACKER_TARGET_NEGATIVE_IMAGE_KEY])
        return tuple(inputs), targets_dict

    def _convert_inputs(self, data: Dict[str, Any], keys: Iterable) -> Dict[str, Any]:
        returned = dict()
        for key in keys:
            if key not in data:
                continue
            gt_map = data[key]
            if isinstance(gt_map, np.ndarray):
                gt_map = torch.from_numpy(gt_map)
            if isinstance(gt_map, list):
                gt_map = [item.to(dtype=self.input_type, device=self.device, non_blocking=True) for item in gt_map]
            else:
                gt_map = gt_map.to(dtype=self.input_type, device=self.device, non_blocking=True)
            returned[key] = gt_map
        return returned

    def on_train_epoch_start(self) -> None:
        self.dataset_aware_metric.reset("train")
        super().on_train_epoch_start()
        self.box_coder.to_device(self.device)

    def on_validation_epoch_start(self) -> None:
        self.dataset_aware_metric.reset("valid")
        super().on_validation_epoch_start()
        self.box_coder.to_device(self.device)

    def on_train_epoch_end(self, outputs: Any) -> None:
        metrics = self.dataset_aware_metric.compute("train")
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )
        super().on_train_epoch_end(outputs)

    def on_validation_epoch_end(self) -> None:
        metrics = self.dataset_aware_metric.compute("valid")
        for metric_name, metric_value in metrics.items():
            self.log(
                f"valid/metrics/{metric_name}",
                metric_value,
                on_epoch=True,
            )
        self.update_offset()
        self.resample_datasets()
        super().on_validation_epoch_end()

    def on_pretrain_routine_start(self) -> None:
        super().on_pretrain_routine_start()
        self.box_coder.to_device(self.device)
        self.tracker.to_device(self.device)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.box_coder.to_device(self.device)
        self.tracker.to_device(self.device)

    def _denormalize_img(self, input: np.ndarray, idx: int) -> np.ndarray:
        return rgb_image_from_tensor(input[idx])

    def get_visuals(
        self,
        inputs: Dict[str, Tensor],
        outputs: Dict[str, Any],
        score: float,
        max_images=None,
    ) -> np.ndarray:
        decoded_results = self.box_coder.decode(
            regression_map=outputs[TARGET_REGRESSION_LABEL_KEY],
            classification_map=outputs[TARGET_CLASSIFICATION_KEY],
        )
        template_imgs, search_imgs = inputs[TRACKER_TARGET_TEMPLATE_IMAGE_KEY], inputs[TRACKER_TARGET_SEARCH_IMAGE_KEY]
        template_filenames, search_filenames = (
            inputs[TRACKER_TARGET_TEMPLATE_FILENAME_KEY],
            inputs[TRACKER_TARGET_SEARCH_FILENAME_KEY],
        )
        num_images = len(template_imgs)
        if max_images is not None:
            num_images = min(num_images, max_images)

        batch_images = []
        for idx in range(num_images):
            template_img = rgb_image_from_tensor(template_imgs[idx][:3]).copy()
            search_img = rgb_image_from_tensor(search_imgs[idx][:3]).copy()
            pred_x, pred_y, pred_w, pred_h = map(int, decoded_results.bbox.cpu().tolist()[idx])
            gt_x, gt_y, gt_w, gt_h = map(int, inputs[TRACKER_TARGET_BBOX_KEY][idx].cpu().tolist())
            gt_color = (0, 0, 250) if inputs[TARGET_VISIBILITY_KEY][idx].item() == 0.0 else (250, 0, 0)
            search_img = cv2.rectangle(search_img, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 250, 0), 2)
            search_img = cv2.rectangle(search_img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), gt_color, 2)
            img = hstack_autopad(
                [
                    template_img,
                    search_img,
                ]
            )
            img = vstack_header(img, f"S: {inputs[DATASET_NAME_KEY][idx]}, {search_filenames[idx]}")
            img = vstack_header(img, f"T: {inputs[DATASET_NAME_KEY][idx]}, {template_filenames[idx]}")
            batch_images.append(img)

        res_img = vstack_autopad(batch_images)
        res_img = vstack_header(res_img, f"Batch Score {score:.4f}")
        return res_img

    def resample_datasets(self):
        train_dataset = self.train_dataloader().dataset
        datasets_to_update = train_dataset.datasets if type(train_dataset) is ConcatDataset else [train_dataset]
        for dataset in datasets_to_update:
            dataset.resample()

    def update_offset(self):
        if "dynamic_frame_offset" not in self.config:
            return
        params = self.config["dynamic_frame_offset"]
        start_epoch = params["start_epoch"]
        freq = params["freq"]
        step = params["step"]
        max_value = params["max_value"]

        train_dataset = self.train_dataloader().dataset
        datasets_to_update = train_dataset.datasets if type(train_dataset) is ConcatDataset else [train_dataset]
        if (self.current_epoch + 1) >= start_epoch and (self.current_epoch + 1) % freq == 0:
            for dataset in datasets_to_update:
                frame_offset = dataset.item_sampler.frame_offset
                updated_frame_offset = min(max_value, frame_offset + step)
                dataset.item_sampler.frame_offset = updated_frame_offset
                logger.info(
                    f"{dataset.config['root']} frame_offset updated from {frame_offset} to {updated_frame_offset}"
                )
