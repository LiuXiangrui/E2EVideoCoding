import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Model.Common.Compression import AnalysisTransform, SynthesisTransform
from Model.Common.Compression import FactorizedCompression, HyperpriorCompression


class AnalysisTransformMotionCompression(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SynthesisTransformMotionCompression(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class MotionCompression(FactorizedCompression):
    def __init__(self, N: int = 128, M: int = 128):
        super().__init__()
        self.analysis_transform = AnalysisTransformMotionCompression(in_channels=2, internal_channels=N, out_channels=M)
        self.synthesis_transform = SynthesisTransformMotionCompression(in_channels=M, internal_channels=N, out_channels=2)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)


class HyperAnalysisTransformResiduesCompression(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class HyperSynthesisTransformResiduesCompression(nn.Module):
    def __init__(self, **kwargs):
        if "channels" in kwargs:
            channels = kwargs["channels"]
        else:
            channels = [kwargs["in_channels"], kwargs["internal_channels"], kwargs["internal_channels"], kwargs["out_channels"]]

        super().__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=channels[1], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[2], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class ResiduesCompression(HyperpriorCompression):
    def __init__(self, N: int, M: int):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=3, internal_channels=N, out_channels=M, kernel_size=5)
        self.synthesis_transform = SynthesisTransform(in_channels=M, internal_channels=N, out_channels=3, kernel_size=5)
        self.hyper_analysis_transform = HyperAnalysisTransformResiduesCompression(in_channels=M, internal_channels=N, out_channels=N)
        self.hyper_synthesis_transform = HyperSynthesisTransformResiduesCompression(in_channels=N, internal_channels=N, out_channels=M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)
