import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models import CompressionModel


class MotionCodecDCVC(CompressionModel):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=64)

        self.analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )

        self.synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.hyper_analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        )

        self.hyper_synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

        self.context_prediction = MaskedConv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=1)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=426, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=426, out_channels=341, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=341, out_channels=256, kernel_size=1)
        )

        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, offset: torch.Tensor) -> dict:
        latents = self.analysis_transform(offset)

        hyperprior = self.hyper_analysis_transform(latents)
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        hyper_params = self.hyper_synthesis_transform(rec_hyperprior)

        rec_latents = self.gaussian_conditional.quantize(latents, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(rec_latents)

        gaussian_params = self.entropy_parameters(torch.cat([hyper_params, ctx_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, latents_likelihoods = self.gaussian_conditional(latents, scales_hat, means=means_hat)
        rec_offset = self.synthesis_transform(rec_latents)
        return {
            "rec_offset": rec_offset,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
