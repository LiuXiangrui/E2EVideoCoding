import torch
import torch.nn as nn


class ResidualBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.downsample = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride)\
            if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.downsample(x)


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.upsample(x)


class DepthConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth_kernel: int = 3, stride: int = 1):
        super().__init__()
        dw_ch = in_ch * 1

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=dw_ch, kernel_size=1, stride=stride),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels=dw_ch, out_channels=dw_ch, kernel_size=depth_kernel, padding=depth_kernel // 2, groups=dw_ch),
            nn.Conv2d(in_channels=dw_ch, out_channels=out_ch, kernel_size=1)
        )

        self.downsample = nn.Identity()
        if stride == 2:
            self.downsample = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        elif in_ch != out_ch:
            self.downsample = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.downsample(x)


class ConvFFN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth_kernel: int = 3, stride: int = 1, slope_depth_conv=0.01, slope_ffn=0.1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride),
            ConvFFN(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SubPixelConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upscale_factor: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * upscale_factor ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
