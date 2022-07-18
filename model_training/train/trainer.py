import os
from typing import Dict, Any, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

from model_training.train.callbacks import ModelCheckpointCallback, EarlyStoppingCallback, BestWorstMinerCallback


def _init_logger(config: Dict[str, Any]) -> List[LightningLoggerBase]:
    if "experiment" not in config.keys():
        return []
    experiment_dir = os.path.join(config["experiment"]["folder"], config["experiment"]["name"])
    version_tag = config["experiment"]["version"] if "version" in config["experiment"].keys() else 0
    tt_logger = TensorBoardLogger(
        save_dir=os.path.join(experiment_dir, "logs"),
        version=version_tag,
        name="logs",
    )
    return [tt_logger]


def to_iterable(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]


def get_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """
    Return callbacks, order matters
    """
    callback_classes = [
        ModelCheckpointCallback,
        EarlyStoppingCallback,
        BestWorstMinerCallback,
    ]
    callbacks: List[pl.Callback] = []
    for callback_class in callback_classes:
        callback = callback_class.from_config(config)
        if callback is not None:
            callbacks = callbacks + to_iterable(callback)
    return callbacks


#ToDo: Refactor Trainer creation -> Move to Hydra
def get_trainer(config: Dict[str, Any]) -> pl.Trainer:
    trainer = pl.Trainer(
        logger=_init_logger(config),
        gpus=config["gpus"],
        accelerator=config.get("accelerator", None),
        sync_batchnorm=config.get("sync_bn", False),
        auto_scale_batch_size=config.get("auto_bs", False),
        benchmark=config.get("cuda_benchmark", True),
        precision=config.get("precision", 32),
        callbacks=get_callbacks(config=config),
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        gradient_clip_val=config.get("gradient_clip_val", 0),
        val_check_interval=config.get("val_check_interval", 1.0),
        limit_train_batches=config.get("train_percent", 1.0),
        limit_val_batches=config.get("val_percent", 1.0),
        progress_bar_refresh_rate=config.get("progress_bar_refresh_rate", 10),
        num_sanity_val_steps=config.get("sanity_steps", 5),
        log_every_n_steps=1,
        auto_lr_find=config.get("auto_lr", False),
        replace_sampler_ddp=config.get("replace_sampler_ddp", True),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
    )
    return trainer
