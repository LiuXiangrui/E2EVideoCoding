import torch
import torch.nn as nn

from abc import ABCMeta

from Modules.Utils import EncUnitFVC


class PreTransformABC(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.pre_transform = None

    def forward(self, img: torch.Tensor, *args):
        return self.pre_transform(img, *args)


class IdentityPreTransform(PreTransformABC):
    def __init__(self):
        super().__init__()
        self.pre_transform = nn.Identity()


class FeaturePreTransformFVC(PreTransformABC):
    def __init__(self):
        super().__init__()
        self.pre_transform = EncUnitFVC(in_channels=3, out_channels=64)
