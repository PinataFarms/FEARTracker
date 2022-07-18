from functools import partial
from collections import defaultdict
from typing import Optional, Callable, List, Dict, Any

import torch
import torch.nn
import pytorch_toolbelt.utils.distributed as tbt
from torch import Tensor

__all__ = ["DatasetAwareMetric"]


class DatasetAwareMetric:
    """
    Computes given metric alongs all training datasets, returns a dict of key - dataset_name and value - metric value.
    This class supports automatic handling of distributed training
    """

    def __init__(
        self,
        metric_name: str,
        metric_fn: Callable,
        modes: Optional[List[str]] = None,
    ):
        if modes is None:
            modes = ["train", "valid"]
        self.modes = modes
        self.metric_name = metric_name
        self.metric_fn = metric_fn
        self.metrics: Dict[str, Dict[str, float]] = {name: {} for name in modes}
        self.total: Dict[str, Dict[str, float]] = {name: {} for name in modes}

    @torch.no_grad()
    def update(self, mode: str, datasets: List[str], outputs: Any, gts: Optional[Any] = None) -> None:
        """

        Args:
            mode: str - train, val, test
            datasets: List[str] of a len() = batch_size of datasets
            outputs: Neural Network Outputs
            gts: Optional - Ground Truth Data

        """
        metrics: Tensor = self.metric_fn(outputs, gts)
        if len(datasets) != len(metrics):
            raise ValueError(
                f"Detected mismatch between number of dataset labels ({len(datasets)}) and number of elements in metrics ({len(metrics)})."
                f"For DatasetAwareMetric, metric_fn must return non-reduced tensor of shape [B] at least."
            )
        for dataset, metric_value in zip(datasets, metrics):
            if dataset not in self.metrics[mode]:
                self.metrics[mode][dataset] = metric_value.detach().clone().cpu()
                self.total[mode][dataset] = 1
            else:
                self.metrics[mode][dataset] += metric_value.detach().clone().cpu()
                self.total[mode][dataset] += 1

    @torch.no_grad()
    def compute(self, mode: str) -> Dict[str, Tensor]:
        """
        Reduces and returns dataset aware metrics
        Args:
            mode: str - train, val, test

        Returns: Dict[str, Tensor] - dataset aware metrics

        """

        # Gather metrics from all nodes. If we run in single-gpu, this will be a list of 1 element.
        # For N-gpu DDP, it will be a list of N.
        all_nodes_metrics: List[Dict[str, Tensor]] = tbt.all_gather(self.metrics[mode])
        all_nodes_totals: List[Dict[str, int]] = tbt.all_gather(self.total[mode])

        # Reduce metrics
        reduced_metrics: Dict[str, Tensor] = defaultdict(partial(torch.tensor, 0.0, device="cpu"))
        reduced_totals: Dict[str, int] = defaultdict(int)

        for node_metrics in all_nodes_metrics:
            for k, metric in node_metrics.items():
                reduced_metrics[k] += metric

        for node_totals in all_nodes_totals:
            for k, total in node_totals.items():
                reduced_totals[k] += total

        return {f"{k}_{self.metric_name}": torch.mean(v / reduced_totals[k]) for k, v in reduced_metrics.items()}

    def reset(self, mode: str) -> None:
        """

        Args:
            mode: str - train, val, test

        """
        self.metrics[mode] = {}
        self.total[mode] = {}
