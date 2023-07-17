import math
from typing import Union

import torch
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from torch import nn as nn

from Model.Common.BasicBlock import SubPixelConv, ResBlock, EncUnit, DecUnit


class FactorizedCompression(CompressionModel):
    def __init__(self):
        super().__init__()
        self.analysis_transform = None
        self.synthesis_transform = None
        self.entropy_bottleneck = None

    def forward(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.synthesis_transform(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            }
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    @torch.no_grad()
    def decompress(self, strings: list, shape: list) -> dict:
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.synthesis_transform(y_hat)
        return {"x_hat": x_hat}


class HyperpriorCompression(CompressionModel):
    def __init__(self):
        super().__init__()
        self.analysis_transform = None
        self.synthesis_transform = None
        self.hyper_analysis_transform = None
        self.hyper_synthesis_transform = None
        self.entropy_bottleneck = None
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.hyper_synthesis_transform(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.synthesis_transform(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.hyper_synthesis_transform(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    @torch.no_grad()
    def decompress(self, strings: list, shape: list) -> dict:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.hyper_synthesis_transform(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.synthesis_transform(y_hat)
        return {"x_hat": x_hat}


class JointAutoregressiveCompression(CompressionModel):
    def __init__(self):
        super().__init__()
        self.analysis_transform = None
        self.synthesis_transform = None
        self.hyper_analysis_transform = None
        self.hyper_synthesis_transform = None
        self.entropy_parameters = None
        self.context_prediction = None
        self.entropy_bottleneck = None
        self.gaussian_conditional = None

    def forward(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.hyper_synthesis_transform(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat([params, ctx_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.synthesis_transform(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.hyper_synthesis_transform(z_hat)

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

    @torch.no_grad()
    def _compress_ar(self, y_hat: torch.Tensor, params: torch.Tensor, height: int, width: int, kernel_size: int, padding: int) -> bytes:
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(y_crop, weight=masked_weight, bias=self.context_prediction.bias)

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()

        return string

    @torch.no_grad()
    def decompress(self, strings: list, shape: list) -> dict:
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.hyper_synthesis_transform(z_hat)

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
        x_hat = self.synthesis_transform(y_hat)
        return {"x_hat": x_hat}

    @torch.no_grad()
    def _decompress_ar(self, y_string: bytes, y_hat: torch.Tensor, params: torch.Tensor, height: int, width: int, kernel_size: int, padding: int):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the autoregressive nature of the
        # decoding... See more recent publication where they use an
        # autoregressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(y_crop, weight=self.context_prediction.weight, bias=self.context_prediction.bias)
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat([p, ctx_p], dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv


class AnalysisTransform(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int, stride: int = 2):
        super().__init__()
        transform = [
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            GDN(in_channels=internal_channels),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        ]

        for i, module in enumerate(transform):
            if not isinstance(module, nn.Conv2d):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SynthesisTransform(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int, stride: int = 2):
        super().__init__()
        transform = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
            GDN(in_channels=internal_channels, inverse=True),
            nn.ConvTranspose2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1),
        ]

        for i, module in enumerate(transform):
            if not isinstance(module, nn.Conv2d):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class ContextualAnalysisTransform(nn.Module):
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

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.transform(torch.cat([x, ctx], dim=1))


class ContextualSynthesisTransform(nn.Module):
    def __init__(self, in_channels: int, ctx_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.transform = nn.Sequential(
            SubPixelConv(in_channels=in_channels, out_channels=internal_channels, upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            SubPixelConv(in_channels=internal_channels, out_channels=internal_channels, upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            SubPixelConv(in_channels=internal_channels, out_channels=internal_channels, upscale_factor=2),
            GDN(in_channels=internal_channels, inverse=True),
            ResBlock(channels=internal_channels, activation=nn.LeakyReLU, tail_act=True, negative_slope=0.1),
            SubPixelConv(in_channels=internal_channels, out_channels=internal_channels, upscale_factor=2)
        )

        self.contextual_transform = nn.Sequential(
            nn.Conv2d(in_channels=ctx_channels + internal_channels, out_channels=internal_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            ResBlock(channels=internal_channels, activation=nn.ReLU, head_act=True),
            ResBlock(channels=internal_channels, activation=nn.ReLU, head_act=True),
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.contextual_transform(torch.cat([self.transform(x), ctx], dim=1))


class HyperAnalysisTransform(nn.Module):
    def __init__(self, activation: Union[nn.ReLU, nn.LeakyReLU], **kwargs):
        super().__init__()
        if "channels" in kwargs:
            channels = kwargs["channels"]
        else:
            channels = [kwargs["in_channels"], kwargs["internal_channels"], kwargs["internal_channels"], kwargs["out_channels"]]

        transform = [
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1, padding=1),
            activation(inplace=True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=5, stride=2, padding=2),
            activation(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=5, stride=2, padding=2)
        ]

        for i, module in enumerate(transform):
            if not isinstance(module, nn.Conv2d):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class HyperSynthesisTransform(nn.Module):
    def __init__(self, activation: Union[nn.ReLU, nn.LeakyReLU], **kwargs):
        if "channels" in kwargs:
            channels = kwargs["channels"]
        else:
            channels = [kwargs["in_channels"], kwargs["internal_channels"], kwargs["internal_channels"], kwargs["out_channels"]]

        super().__init__()
        transform = [
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=channels[1], kernel_size=5, stride=2, padding=2, output_padding=1),
            activation(inplace=True),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[2], kernel_size=5, stride=2, padding=2, output_padding=1),
            activation(inplace=True),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=1, padding=1),
        ]

        for i, module in enumerate(transform):
            if not isinstance(module, nn.ConvTranspose2d):
                continue
            out_channels, in_channels, _, _ = module.weight.data.shape
            torch.nn.init.xavier_normal_(module.weight.data, gain=math.sqrt((out_channels + in_channels) / in_channels))
            torch.nn.init.constant_(module.bias.data, val=0.01)

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class AnalysisTransformWithResBlocks(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            EncUnit(in_channels=in_channels, out_channels=internal_channels),
            EncUnit(in_channels=internal_channels, out_channels=internal_channels),
            EncUnit(in_channels=internal_channels, out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SynthesisTransformWithResBlocks(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            DecUnit(in_channels=in_channels, out_channels=internal_channels),
            DecUnit(in_channels=internal_channels, out_channels=internal_channels),
            DecUnit(in_channels=internal_channels, out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
