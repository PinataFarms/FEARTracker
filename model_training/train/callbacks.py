import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from model_training.utils.torch import any2device, evaluating


class BaseCallback(Callback, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass


class ModelCheckpointCallback(ModelCheckpoint, BaseCallback):
    """
    Drop-in replacement for pl.ModelCheckpoint with the support of keys `metric/metric_name'
    """

    def format_checkpoint_name(self, metrics: Dict[str, Any], ver: Optional[int] = None) -> str:
        filename = self._format_checkpoint_name(
            self.filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name
        )
        filename = self.sanitize_metric_name(filename)

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    @staticmethod
    def sanitize_metric_name(metric_name: str) -> str:
        """
        Replace characters in string that are not path-friendly with underscore
        """
        for s in ["?", "/", "\\", ":", "<", ">", "|", "'", '"', "#", "="]:
            metric_name = metric_name.replace(s, "_")
        return metric_name

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[List["ModelCheckpointCallback"]]:
        if "experiment" not in config.keys():
            return None
        if "checkpoint_metrics" in config.keys():
            monitor_keys = config["checkpoint_metrics"]
        else:
            monitor_keys = [
                {"name": config.get("metric_to_monitor", "valid/loss"), "mode": config.get("metric_mode", "min")}
            ]
        return [
            cls(
                dirpath=os.path.join(config["experiment"]["folder"], config["experiment"]["name"], "checkpoints"),
                verbose=True,
                save_top_k=config.get("save_top_k", 1),
                monitor=metric_config["name"],
                mode=metric_config["mode"],
                save_last=config.get("save_last", True),
                save_weights_only=config.get("checkpoints_save_weights_only", True),
                filename=f"{{epoch:04d}}-{{{metric_config['name']}:.4f}}",
            )
            for metric_config in monitor_keys
        ]


class EarlyStoppingCallback(EarlyStopping, BaseCallback):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[BaseCallback]:
        if "early_stopping" not in config.keys():
            return None
        return cls(
            monitor=config.get("metric_to_monitor", "valid/loss"),
            min_delta=0.00,
            patience=config["early_stopping"],
            verbose=False,
            mode=config.get("metric_mode", "min"),
        )


class BestWorstMinerCallback(BaseCallback):
    """
    Tracks batches along the epoch and keeps the best & worst batch (input & outputs) to visualize
    them when epoch ends
    """

    best_input = None
    best_output = None
    best_score: Optional[float] = None

    worst_input = None
    worst_output = None
    worst_score: Optional[float] = None

    def __init__(
        self,
        target_metric_minimize: bool = True,
        min_delta: float = 1e-4,
        metric_to_monitor: str = "loss",
        max_images: int = 16,
    ) -> None:
        """
        Args:
            target_metric_minimize: True means smaller values are better than larger ones (e.g loss);
                False means bigger values are better than smaller ones (e.g IoU)
            min_delta: Min difference in score to update new best/worst batch
        """
        super().__init__()

        self.max_images = max_images
        self.metric_to_monitor = metric_to_monitor
        self.target_metric_minimize = target_metric_minimize
        if target_metric_minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
            self.is_worse = lambda score, worst: score >= (worst + min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)
            self.is_worse = lambda score, worst: score <= (worst - min_delta)

    def reset(self) -> None:
        self.best_score = None
        self.best_input = None
        self.best_output = None

        self.worst_score = None
        self.worst_input = None
        self.worst_output = None

    def _log_best_worst_batch(self, loader_name: str, pl_module: pl.LightningModule) -> None:
        if pl_module.is_master and self.best_score and self.worst_score:
            best_batch = pl_module.get_visuals(
                self.best_input, self.best_output, self.best_score, max_images=self.max_images
            )

            worst_batch = pl_module.get_visuals(
                self.worst_input, self.worst_output, self.worst_score, max_images=self.max_images
            )
            best_batch_name = f"{loader_name}/best_batch"
            worst_batch_name = f"{loader_name}/worst_batch"

            pl_module.tensorboard_logger.experiment.add_image(
                tag=best_batch_name, img_tensor=best_batch, global_step=pl_module.epoch_num, dataformats="HWC"
            )

            pl_module.tensorboard_logger.experiment.add_image(
                tag=worst_batch_name, img_tensor=worst_batch, global_step=pl_module.epoch_num, dataformats="HWC"
            )

        self.reset()

    def _check_score(self, score: float, pl_module: pl.LightningModule, batch: Any) -> None:
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.best_input = batch
            outputs = self._get_output(pl_module=pl_module, input=batch)
            self.best_output = outputs

        if self.worst_score is None or self.is_worse(score, self.worst_score):
            self.worst_score = score
            self.worst_input = batch
            outputs = self._get_output(pl_module=pl_module, input=batch)
            self.worst_output = outputs

    @torch.no_grad()
    def _get_output(self, pl_module: pl.LightningModule, input: Any) -> Any:
        with evaluating(pl_module):
            input = any2device(input, pl_module.device)
            inputs, _ = pl_module.get_input(input)
            outputs = pl_module.forward(inputs)
        return outputs

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.metric_to_monitor in outputs.keys():
            score = float(outputs[self.metric_to_monitor].cpu().item())
        else:
            metrics = trainer.logger_connector.logged_metrics
            score_key = f"valid/{self.metric_to_monitor}"
            if score_key not in metrics.keys():
                return
            score = float(metrics[score_key])
        self._check_score(score=score, pl_module=pl_module, batch=batch)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        metrics = trainer.logger_connector.logged_metrics
        score_key = f"train/{self.metric_to_monitor}"
        if score_key in metrics.keys():
            score = float(metrics[score_key])
            self._check_score(score=score, pl_module=pl_module, batch=batch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_best_worst_batch("valid", pl_module)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_best_worst_batch("train", pl_module)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.reset()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["BestWorstMinerCallback"]:
        if "best_worst_miner" not in config:
            return None
        return cls(
            target_metric_minimize=config["best_worst_miner"]["metric_mode"] == "min",
            metric_to_monitor=config["best_worst_miner"]["metric_to_monitor"],
            min_delta=config["best_worst_miner"].get("min_delta", 1e-4),
            max_images=config["best_worst_miner"].get("max_images", 16),
        )
