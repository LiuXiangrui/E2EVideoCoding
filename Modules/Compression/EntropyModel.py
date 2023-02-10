from abc import ABCMeta, abstractmethod

import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d, GDN
from torch import nn as nn


class EntropyModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, latents: torch.Tensor, *args, **kwargs):
        raise NotImplementedError


class FactorizedEntropyModel(EntropyModel):
    def __init__(self, **kwargs):
        super().__init__()
        assert "channels" in kwargs
        self.entropy_bottleneck = EntropyBottleneck(channels=kwargs["channels"])

    def forward(self, latents: torch.Tensor, *args, **kwargs) -> tuple:
        latents_hat, likelihoods = self.entropy_bottleneck(latents)
        return latents_hat, likelihoods


class ScaleGaussianEntropyModel(EntropyModel):
    def __init__(self):
        super().__init__()
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, latents: torch.Tensor, *args, **kwargs) -> tuple:
        assert "scales_hat" in kwargs, "Scales should be provided!"
        latents_hat, likelihoods = self.gaussian_conditional(latents, kwargs["scales_hat"])
        return latents_hat, likelihoods


class AutoRegressiveEntropyModel(ScaleGaussianEntropyModel):
    def __init__(self, **kwargs):
        super().__init__()
        assert "latent_channels" in kwargs
        latent_channels = kwargs["latent_channels"]
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(latent_channels * 12 // 3, latent_channels * 10 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels * 10 // 3, latent_channels * 8 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_channels * 8 // 3, latent_channels * 6 // 3, kernel_size=1)
        )
        self.autoregressive_prediction = MaskedConv2d(in_channels=latent_channels, out_channels=latent_channels * 2,
                                                      kernel_size=5, padding=2, stride=1)

    def forward(self, latents: torch.Tensor, **kwargs) -> tuple:
        assert "hyperprior" in kwargs, "Hyper-prior should be provided!"
        latents_hat = self.gaussian_conditional.quantize(latents, "noise" if self.training else "dequantize")
        ar_prior = self.autoregressive_prediction(latents_hat)
        gaussian_params = self.entropy_parameters(torch.cat([kwargs["hyperprior"], ar_prior], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1)
        _, likelihoods = self.gaussian_conditional(latents, scales_hat, means=means_hat)
        return latents_hat, likelihoods


class AREntropyModelWithTemporalPrior(AutoRegressiveEntropyModel):
    def __init__(self, **kwargs):
        assert "latent_channels" in kwargs and "ctx_channels" in kwargs
        latent_channels = kwargs["latent_channels"]
        ctx_channels = kwargs["ctx_channels"]
        super().__init__(latent_channels=latent_channels)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(in_channels=ctx_channels, out_channels=ctx_channels, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=ctx_channels),
            nn.Conv2d(in_channels=ctx_channels, out_channels=ctx_channels, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=ctx_channels),
            nn.Conv2d(in_channels=ctx_channels, out_channels=ctx_channels, kernel_size=5, stride=2, padding=2),
            GDN(in_channels=ctx_channels),
            nn.Conv2d(in_channels=ctx_channels, out_channels=latent_channels, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, latents: torch.Tensor, *args, **kwargs) -> tuple:
        assert "hyperprior" in kwargs and "ctx" in kwargs, "Hyper-prior and context should be provided!"

        temporal_prior = self.temporal_prior_encoder(kwargs["ctx"])

        latents_hat = self.gaussian_conditional.quantize(latents, "noise" if self.training else "dequantize")
        ar_prior = self.autoregressive_prediction(latents_hat)
        gaussian_params = self.entropy_parameters(torch.cat([temporal_prior, kwargs["hyperprior"], ar_prior], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(chunks=2, dim=1)
        _, likelihoods = self.gaussian_conditional(latents, scales_hat, means=means_hat)
        return latents_hat, likelihoods
