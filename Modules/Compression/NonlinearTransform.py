from abc import ABCMeta

import torch
from compressai.layers import GDN
from torch import nn as nn

from Modules.Utils import ResBlock, EncUnit, DecUnit


class TransformABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class AnalysisTransform(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        )
    
    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)
    

class SynthesisTransform(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
        )

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)


class ContextualAnalysisTransform(TransformABC):
    def __init__(self, in_channels: int, ctx_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + ctx_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        )

    def forward(self, inputs: torch.Tensor, *args,  **kwargs) -> torch.Tensor:
        assert "ctx" in kwargs, "Context should be provided!"
        latents = self.transform(torch.cat([inputs, kwargs["ctx"]], dim=1))
        return latents


class ContextualSynthesisTransform(TransformABC):
    def __init__(self, in_channels: int, ctx_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.contextual_transform = nn.Sequential(
            nn.Conv2d(in_channels=ctx_channels + internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            ResBlock(channels=internal_channels, activation=nn.ReLU, head_act=True),
            ResBlock(channels=internal_channels, activation=nn.ReLU, head_act=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert "ctx" in kwargs, "Context should be provided!"
        inputs = self.transform(inputs)
        inputs = self.contextual_transform(torch.cat([inputs, kwargs["ctx"]], dim=1))
        return inputs


class HyperAnalysisTransform(TransformABC):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)


class HyperSynthesisTransform(TransformABC):
    def __init__(self, **kwargs):
        if "channels" in kwargs:
            channels = kwargs["channels"]
        else:
            channels = [kwargs["in_channels"], kwargs["internal_channels"], kwargs["internal_channels"], kwargs["out_channels"]]

        super().__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=channels[1], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[2], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)


class AnalysisTransformWithResBlocks(TransformABC):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            EncUnit(in_channels=in_channels, out_channels=internal_channels),
            EncUnit(in_channels=internal_channels, out_channels=internal_channels),
            EncUnit(in_channels=internal_channels, out_channels=internal_channels),
            EncUnit(in_channels=internal_channels, out_channels=out_channels)
        )

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)


class SynthesisTransformWithResBlocks(TransformABC):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            DecUnit(in_channels=in_channels, out_channels=internal_channels),
            DecUnit(in_channels=internal_channels, out_channels=internal_channels),
            DecUnit(in_channels=internal_channels, out_channels=internal_channels),
            DecUnit(in_channels=internal_channels, out_channels=out_channels)
        )

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        return self.transform(inputs)