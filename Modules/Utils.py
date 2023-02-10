import torch
from torch import nn as nn
from torch.nn import functional as F


def optical_flow_warp(x: torch.Tensor, optical_flow: torch.Tensor) -> torch.Tensor:
    B, C, H, W = optical_flow.shape
    axis_hor = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    axis_ver = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([axis_hor, axis_ver], dim=1).to(optical_flow.device)

    optical_flow = torch.cat([optical_flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                              optical_flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)

    warped_x = F.grid_sample(input=x, grid=(grid + optical_flow).permute(0, 2, 3, 1),
                             mode="bilinear", padding_mode="border", align_corners=False)
    return warped_x


class OpticalFlowWarpNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return optical_flow_warp(x, optical_flow=offset)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

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


