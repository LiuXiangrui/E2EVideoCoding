import math

import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Model.Common.Compression import AnalysisTransform, SynthesisTransform, HyperSynthesisTransform, HyperAnalysisTransform
from Model.Common.Compression import FactorizedCompression, HyperpriorCompression


class AnalysisTransformMotionCompression(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        transform = [
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        ]

        # initialize
        for i, module in enumerate(transform):
            if not isinstance(module, nn.Conv2d):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SynthesisTransformMotionCompression(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        transform = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        ]

        # initialize
        for i, module in enumerate(transform):
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class MotionCompression(FactorizedCompression):
    def __init__(self, N: int, M: int):
        super().__init__()
        self.analysis_transform = AnalysisTransformMotionCompression(in_channels=2, internal_channels=N, out_channels=M)
        self.synthesis_transform = SynthesisTransformMotionCompression(in_channels=M, internal_channels=N, out_channels=2)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)


class ResiduesCompression(HyperpriorCompression):
    def __init__(self, N: int, M: int):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=3, internal_channels=N, out_channels=M, kernel_size=5)
        self.synthesis_transform = SynthesisTransform(in_channels=M, internal_channels=N, out_channels=3, kernel_size=5)
        self.hyper_analysis_transform = HyperAnalysisTransform(activation=nn.ReLU, in_channels=M, internal_channels=N, out_channels=N)
        self.hyper_synthesis_transform = HyperSynthesisTransform(activation=nn.ReLU, in_channels=N, internal_channels=N, out_channels=M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
