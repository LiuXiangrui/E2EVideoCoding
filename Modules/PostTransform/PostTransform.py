import torch
import torch.nn as nn

from abc import ABCMeta


class PostTransformABC(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.post_transform = nn.Sequential()

    def forward(self, img: torch.Tensor, *args):
        return self.post_transform(img, *args)


class IdentityTransform(PostTransformABC):
    def __init__(self):
        super().__init__()
        self.post_transform = nn.Sequential(nn.Identity(), )


if __name__ == "__main__":
    a = IdentityTransform()
    a(torch.randn(1,2,3))