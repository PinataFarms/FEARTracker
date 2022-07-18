from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from model_training.utils.constants import (
    TARGET_REGRESSION_LABEL_KEY,
    TARGET_REGRESSION_WEIGHT_KEY,
    TARGET_CLASSIFICATION_KEY,
)


def calc_iou(reg_target: torch.Tensor, pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    target_area = (reg_target[..., 0] + reg_target[..., 2]) * (reg_target[..., 1] + reg_target[..., 3])
    pred_area = (pred[..., 0] + pred[..., 2]) * (pred[..., 1] + pred[..., 3])

    w_intersect = torch.min(pred[..., 0], reg_target[..., 0]) + torch.min(pred[..., 2], reg_target[..., 2])
    h_intersect = torch.min(pred[..., 3], reg_target[..., 3]) + torch.min(pred[..., 1], reg_target[..., 1])

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    return (area_intersect + smooth) / (area_union + smooth)


class BoxLoss(nn.Module):
    """
    BBOX Loss: optimizes IoU of bounding boxes
    Original implentation:
    losses = -torch.log(calc_iou(reg_target=target, pred=pred)) was computationally unstable
    those was replaced with: 1 - IoU
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        losses = 1 - calc_iou(reg_target=target, pred=pred)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()


class FEARLoss(nn.Module):
    def __init__(self, coeffs: Dict[str, float]):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = BoxLoss()
        self.coeffs = coeffs

    def _regression_loss(
        self, bbox_pred: torch.Tensor, reg_target: torch.Tensor, reg_weight: torch.Tensor
    ) -> torch.Tensor:
        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self.regression_loss(bbox_pred_flatten, reg_target_flatten)

        return loss

    def _weighted_cls_loss(self, pred: torch.Tensor, label: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
        if len(select.size()) == 0:
            return torch.Tensor([0])
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.classification_loss(pred, label)

    def _classification_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze()
        neg = label.data.eq(0).nonzero().squeeze()

        loss_pos = self._weighted_cls_loss(pred, label, pos)
        loss_neg = self._weighted_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def forward(self, outputs: Dict[str, torch.Tensor], gt: Dict[str, Any]) -> Dict[str, Any]:
        regression_loss = self._regression_loss(
            bbox_pred=outputs[TARGET_REGRESSION_LABEL_KEY],
            reg_target=gt[TARGET_REGRESSION_LABEL_KEY],
            reg_weight=gt[TARGET_REGRESSION_WEIGHT_KEY],
        )
        classification_loss = self._classification_loss(
            pred=outputs[TARGET_CLASSIFICATION_KEY], label=gt[TARGET_CLASSIFICATION_KEY]
        )
        return {
            TARGET_CLASSIFICATION_KEY: classification_loss * self.coeffs[TARGET_CLASSIFICATION_KEY],
            TARGET_REGRESSION_LABEL_KEY: regression_loss * self.coeffs[TARGET_REGRESSION_LABEL_KEY],
        }
