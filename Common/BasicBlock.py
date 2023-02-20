import torch
from torch import nn as nn


class SubPixelConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * upscale_factor ** 2, kernel_size=3, stride=1,
                      padding=1),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, activation: nn.Module, head_act: bool = False, tail_act: bool = False, **kwargs):
        super().__init__()
        kwargs["inplace"] = True
        self.net = [
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            activation(**kwargs),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        ]

        if head_act:
            self.net.insert(0, activation(**kwargs))
        if tail_act:
            self.net.append(activation(**kwargs))

        self.net = nn.Sequential(*self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.net(x)
        return x


class EncUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResBlock(channels=out_channels, activation=nn.ReLU),
            ResBlock(channels=out_channels, activation=nn.ReLU),
            ResBlock(channels=out_channels, activation=nn.ReLU)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        return x


class DecUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResBlock(channels=in_channels, activation=nn.ReLU),
            ResBlock(channels=in_channels, activation=nn.ReLU),
            ResBlock(channels=in_channels, activation=nn.ReLU)
        )
        self.tail = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return x
