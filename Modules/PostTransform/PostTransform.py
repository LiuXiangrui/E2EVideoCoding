import torch
import torch.nn as nn

from abc import ABCMeta


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResBlocks(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(channels=channels),
            ResBlock(channels=channels),
            ResBlock(channels=channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PostTransformABC(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.post_transform = nn.Sequential()

    def forward(self, img: torch.Tensor, *args):
        return self.post_transform(img, *args)


class IdentityPostTransform(PostTransformABC):
    def __init__(self):
        super().__init__()
        self.post_transform = nn.Sequential(nn.Identity(), )


class FeaturePostTransformFVC(PostTransformABC):
    def __init__(self):
        super().__init__()
        self.pre_transform = nn.Sequential(
            ResBlocks(channels=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
