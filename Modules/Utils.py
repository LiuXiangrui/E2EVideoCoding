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


class ResBlockFVC(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EncUnitFVC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlockFVC(channels=out_channels),
            ResBlockFVC(channels=out_channels),
            ResBlockFVC(channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        return x


class DecUnitFVC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.tail = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=5, stride=2, padding=2, output_padding=1)

        self.res_blocks = nn.Sequential(
            ResBlockFVC(channels=in_channels),
            ResBlockFVC(channels=in_channels),
            ResBlockFVC(channels=in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return x

