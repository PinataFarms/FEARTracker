from typing import Optional, Callable, Any

import torch
import torch.nn
from torch import Tensor
from torchmetrics import Metric

__all__ = ["BoxIoUMetric", "TrackingFailureRateMetric", "box_iou_metric"]


def box_iou_metric(iou_matrix: Tensor, gts: Optional[Any] = None) -> Tensor:
    return torch.diagonal(iou_matrix, 0)


class BoxIoUMetric(Metric):
    """Compute the IoU metric for bounding boxes with averaging across individual images."""

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("ious", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, iou_matrix: Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            iou_matrix: (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element
        """
        iou: Tensor = torch.mean(box_iou_metric(iou_matrix=iou_matrix))

        self.ious += iou
        self.total += 1

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy over state.
        """
        return torch.mean(self.ious / self.total)


class TrackingFailureRateMetric(Metric):
    """Compute the Normalized Failure Rate metric for tracking which equals to number of times IoU between pred and gt bboxes == 0"""

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("failure_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, iou_matrix: Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            iou_matrix: (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element
        """
        batch_size = iou_matrix.size()[0]
        failure_rate: Tensor = 1 - (torch.count_nonzero(torch.diagonal(iou_matrix)) / batch_size)

        self.failure_rate += failure_rate
        self.total += 1

    def compute(self) -> None:
        """
        Computes accuracy over state.
        """
        return torch.mean(self.failure_rate / self.total)
