import os
from typing import Dict, Any

import yaml
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig
from .logger import create_logger

logger = create_logger(__name__)


def load_hydra_config(hydra_config: DictConfig) -> Dict[str, Any]:
    """
    Load hydra config and returns ready-to-use dict.

    Notes:
        This function also restores current working directory (Hydra change it internally)

    Args:
        hydra_config:

    Returns:

    """
    os.chdir(get_original_cwd())
    return yaml.load(OmegaConf.to_yaml(hydra_config, resolve=True), Loader=yaml.FullLoader)


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config
