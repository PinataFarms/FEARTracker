import torch
from fire import Fire
from hydra.utils import instantiate
from thop import profile
from thop.utils import clever_format

from model_training.utils.hydra import load_yaml


class ProfileTrackingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]]


def main(config_path: str = "model_training/config/model/fear.yaml"):
    config = load_yaml(config_path)
    model = instantiate(config)
    model = ProfileTrackingWrapper(model)

    search_inp = torch.rand(1, 3, 256, 256)
    template_inp = torch.rand(1, 256, 8, 8)
    macs, params = profile(model, inputs=(search_inp, template_inp), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)


if __name__ == '__main__':
    Fire(main)
