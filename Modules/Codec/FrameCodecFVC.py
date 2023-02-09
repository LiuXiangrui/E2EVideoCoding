import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from Modules.Utils import EncUnitFVC, DecUnitFVC


class FrameCodecFVC(CompressionModel):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=128)

        self.analysis_transform = nn.Sequential(
            EncUnitFVC(in_channels=64, out_channels=128),
            EncUnitFVC(in_channels=128, out_channels=128),
            EncUnitFVC(in_channels=128, out_channels=128),
            EncUnitFVC(in_channels=128, out_channels=128)
        )

        self.synthesis_transform = nn.Sequential(
            DecUnitFVC(in_channels=128, out_channels=128),
            DecUnitFVC(in_channels=128, out_channels=128),
            DecUnitFVC(in_channels=128, out_channels=128),
            DecUnitFVC(in_channels=128, out_channels=64),
        )

        self.hyper_analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        )

        self.hyper_synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, cur: torch.Tensor, pred: torch.Tensor):
        residuals = cur - pred
        latents = self.analysis_transform(residuals)
        hyperprior = self.hyper_analysis_transform(torch.abs(latents))
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        scales_hat = self.hyper_synthesis_transform(rec_hyperprior)
        rec_latents, latents_likelihoods = self.gaussian_conditional(latents, scales_hat)
        rec_residuals = self.synthesis_transform(rec_latents)
        rec_cur = pred + rec_residuals
        return {
            "rec_cur": rec_cur,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
