from typing import Any, Union, Tuple, List

import torch
import torch.nn as nn
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.model = self._load_model()
        self.stages = self._get_stages()
        self.encoder_channels = {
           "layer0": 352,
           "layer1": 112,
           "layer2": 32,
           "layer3": 24,
           "layer4": 16,
        }

    def _load_model(self) -> Any:
        model_name = "fbnet_c"
        model = fbnet(model_name, pretrained=self.pretrained)
        return model

    def _get_stages(self) -> List[Any]:
        stages = [
            self.model.backbone.stages[:2],
            self.model.backbone.stages[2:5],
            self.model.backbone.stages[5:9],
            self.model.backbone.stages[9:18],
            self.model.backbone.stages[18:23],
        ]
        return stages

    def forward(self, x: Any) -> List[Any]:
        encoder_maps = []
        for stage in self.stages:
            x = stage(x)
            encoder_maps.append(x)
        return encoder_maps


class SepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdjustLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, crop_rate: int = 4):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.size_threshold = 20
        self.crop_rate = crop_rate

    def forward(self, x):
        x_ori = self.downsample(x)
        adjust = x_ori
        return adjust


class MatrixMobile(nn.Module):
    """
    Encode backbone feature
    """

    def __init__(self, in_channels, out_channels, conv_block: str = "regular"):
        super().__init__()
        self.matrix11_s = nn.Sequential(
            SepConv(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        return z.reshape(z.size(0), z.size(1), -1), self.matrix11_s(x)


class MobileCorrelation(nn.Module):
    """
    Mobile Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 64, conv_block: str = "regular"):
        super().__init__()
        self.enc = nn.Sequential(
            SepConv(num_channels + num_corr_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        b, c, w, h = x.size()
        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([x, s], dim=1)
        s = self.enc(s)
        return s


class BoxTower(nn.Module):
    """
    Box Tower for FCOS regression
    """

    def __init__(
        self,
        towernum: int = 4,
        conv_block: str = "regular",
        inchannels: int = 512,
        outchannels: int = 256,
        mobile: bool = False,
    ):
        super().__init__()
        tower = []
        cls_tower = []
        # encode backbone
        self.cls_encode = MatrixMobile(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        self.reg_encode = MatrixMobile(in_channels=inchannels, out_channels=outchannels, conv_block=conv_block)
        self.cls_dw = MobileCorrelation(num_channels=outchannels, conv_block=conv_block)
        self.reg_dw = MobileCorrelation(num_channels=outchannels, conv_block=conv_block)

        # box pred head
        for i in range(towernum):
            tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        # cls tower
        for i in range(towernum):
            cls_tower.append(SepConv(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module("bbox_tower", nn.Sequential(*tower))
        self.add_module("cls_tower", nn.Sequential(*cls_tower))

        # reg head
        self.bbox_pred = SepConv(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = SepConv(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)))

    def forward(self, search, kernel, update=None):
        # encode first
        if update is None:
            cls_z, cls_x = self.cls_encode(kernel, search)  # [z11, z12, z13]
        else:
            cls_z, cls_x = self.cls_encode(update, search)  # [z11, z12, z13]

        reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]

        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)
        x_reg = self.bbox_tower(reg_dw)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.exp(x)

        # cls tower
        c = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(c)

        return x, cls, cls_dw, x_reg
