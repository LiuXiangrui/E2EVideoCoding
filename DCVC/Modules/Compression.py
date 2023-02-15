import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import MaskedConv2d, GDN
from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Common.Compression import JointAutoregressiveCompression
from Common.Compression import AnalysisTransform, SynthesisTransform
from Common.Compression import ContextualAnalysisTransform, ContextualSynthesisTransform
from Common.Compression import HyperAnalysisTransform, HyperSynthesisTransform


class MotionCompression(JointAutoregressiveCompression):
    def __init__(self, latents_channels: int = 128, hyper_channels: int = 64):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=2, internal_channels=128, out_channels=latents_channels, kernel_size=3)
        self.synthesis_transform = SynthesisTransform(in_channels=latents_channels, internal_channels=128, out_channels=2, kernel_size=3)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=128, internal_channels=64, out_channels=64)
        self.hyper_synthesis_transform = HyperSynthesisTransform(channels=[64, 64, 96, 256])
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=426, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=426, out_channels=341, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=341, out_channels=256, kernel_size=1)
        )
        self.context_prediction = MaskedConv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=1)
        self.entropy_bottleneck = EntropyBottleneck(channels=hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)


class ContextualCompression(JointAutoregressiveCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = ContextualAnalysisTransform(in_channels=3, ctx_channels=64, internal_channels=64, out_channels=96, kernel_size=5)
        self.synthesis_transform = ContextualSynthesisTransform(in_channels=96, ctx_channels=64, internal_channels=64, out_channels=3, kernel_size=5)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=96, internal_channels=64, out_channels=64)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=64, internal_channels=64, out_channels=96)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=320, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=320, out_channels=256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=1)
        )
        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=2, padding=2),
        )
        self.context_prediction = MaskedConv2d(in_channels=96, out_channels=192, kernel_size=5, padding=2, stride=1)
        self.entropy_bottleneck = EntropyBottleneck(channels=64)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> dict:
        temporal_prior = self.temporal_prior_encoder(ctx)

        y = self.analysis_transform(x, ctx=ctx)
        z = self.hyper_analysis_transform(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis_transform(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat([temporal_prior, params, ctx_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.synthesis_transform(y_hat, ctx=ctx)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x: torch.Tensor, ctx: torch.Tensor = None) -> dict:
        temporal_prior = self.temporal_prior_encoder(ctx)

        y = self.analysis_transform(x, ctx=ctx)
        z = self.hyper_analysis_transform(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.hyper_synthesis_transform(z_hat)
        params = torch.cat([temporal_prior, params], dim=1)

        s = 4  # scaling factor between z and y
        kernel_size = self.context_prediction.weight.shape[-1]  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(y_hat[i: i + 1], params=params[i: i + 1],
                                       height=y_height, width=y_width, kernel_size=kernel_size, padding=padding)
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings: list, shape: list, ctx: torch.Tensor = None) -> dict:
        assert isinstance(strings, list) and len(strings) == 2
        temporal_prior = self.temporal_prior_encoder(ctx)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.hyper_synthesis_transform(z_hat)
        params = torch.cat([temporal_prior, params], dim=1)

        s = 4  # scaling factor between z and y
        kernel_size = self.context_prediction.weight.shape[-1]  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it, so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros((z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding), device=z_hat.device)

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(y_string, y_hat=y_hat[i: i + 1], params=params[i: i + 1], height=y_height, width=y_width, kernel_size=kernel_size, padding=padding)

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.synthesis_transform(y_hat, ctx=ctx)
        return {"x_hat": x_hat}



