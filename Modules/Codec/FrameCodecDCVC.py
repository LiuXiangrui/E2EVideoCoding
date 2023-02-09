import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models import CompressionModel


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.net(x)
        return x


class ContextualSynthesisTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=64, inverse=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=64, inverse=True),
            ResBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            GDN(in_channels=64, inverse=True),
            ResBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.contextual_transform = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResBlock(channels=64),
            ResBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, rec_latents: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        latent_feats = self.transform(rec_latents)
        rec_frame = self.contextual_transform(torch.cat([latent_feats, ctx], dim=1))
        return rec_frame


class FrameCodecDCVC(CompressionModel):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=64)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=2, padding=2),
        )

        self.analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=67, out_channels=64, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=64),
            ResBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=64),
            ResBlock(channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
        )

        self.synthesis_transform = ContextualSynthesisTransform()

        self.hyper_analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        )

        self.hyper_synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
        )

        self.autoregressive_prediction = MaskedConv2d(in_channels=96, out_channels=192, kernel_size=5, padding=2, stride=1)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=320, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=320, out_channels=256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=1)
        )

        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, cur_frame: torch.Tensor, temporal_ctx: torch.Tensor) -> dict:
        temporal_prior = self.temporal_prior_encoder(temporal_ctx)

        latents = self.analysis_transform(torch.cat([cur_frame, temporal_ctx], dim=1))

        hyperprior = self.hyper_analysis_transform(latents)
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        hyperprior = self.hyper_synthesis_transform(rec_hyperprior)

        rec_latents = self.gaussian_conditional.quantize(latents, "noise" if self.training else "dequantize")
        auto_prior = self.autoregressive_prediction(rec_latents)

        gaussian_params = self.entropy_parameters(torch.cat([temporal_prior, hyperprior, auto_prior], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1)
        _, latents_likelihoods = self.gaussian_conditional(latents, scales_hat, means=means_hat)
        rec_frame = self.synthesis_transform(rec_latents, ctx=temporal_ctx)
        return {
            "rec_frame": rec_frame,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
