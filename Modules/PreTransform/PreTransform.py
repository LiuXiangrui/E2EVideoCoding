import torch
import torch.nn as nn

from abc import ABCMeta


class PreTransformABC(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor, *args):
        return self.pre_transform(img, *args)


class IdentityTransform(PreTransformABC):
    def __init__(self):
        super().__init__()
        self.pre_transform = nn.Sequential(nn.Identity(), )


if __name__ == "__main__":
    a = IdentityTransform()
    a(torch.randn(1,2,3))