from typing import Optional

import coremltools
import numpy as np
import torch
from fire import Fire
from hydra.utils import instantiate

from evaluate.coreml_utils import coreml4_convert
from model_training.utils.hydra import load_yaml
from model_training.utils.torch import load_from_lighting

FEAR_DESCRIPTION = dict(
    model_name="Model",
    inputs=[
        dict(
            name="image",
            description="Search image",
            type="image",
            color_layout="RGB",
            shape=[1, 3, 256, 256],
        ),
        dict(
            name="template_features",
            description="Template image features",
            type="tensor",
            dtype=np.float32,
            shape=[1, 256, 8, 8],
        ),
    ],
    outputs=[
        dict(
            name="bbox",
            description="Bbox prediction",
        ),
        dict(
            name="cls",
            description="Classification predictions",
        ),
    ],
    metadata=dict(
        author="PiñataFarms",
        short_description="FEAR: Fast, Efficient, Accurate and Robust Visual Tracker",
        version="0.1",
    ),
    minimum_deployment_target=coremltools.target.iOS14,
)


class CoreMLTrackingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]]


def main(
    config_path: str = "model_training/config/model/fear.yaml",
    weights_path: Optional[str] = "evaluate/checkpoints/FEAR-XS-NoEmbs.ckpt",
):
    config = load_yaml(config_path)
    model = instantiate(config)
    if weights_path is not None:
        model = load_from_lighting(model, weights_path, map_location="cpu")
    model = CoreMLTrackingWrapper(model)
    model = model.eval()
    coreml4_convert(model, FEAR_DESCRIPTION)


if __name__ == '__main__':
    Fire(main)
