from abc import ABCMeta

import torch
from compressai.models import CompressionModel

from Modules.Compression.EntropyModel import EntropyModel
from Modules.Compression.NonlinearTransform import TransformABC


class FactorizedCompression(CompressionModel, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.analysis_transform = TransformABC()
        self.synthesis_transform = TransformABC()
        self.entropy_bottleneck = EntropyModel()

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> dict:
        latents = self.analysis_transform(inputs, args, kwargs)
        latents_hat, latents_likelihoods = self.entropy_bottleneck(latents, args, kwargs)
        inputs_hat = self.synthesis_transform(latents_hat, args, kwargs)
        return {
            "inputs_hat": inputs_hat,
            "likelihoods": {"latents": latents_likelihoods},
        }


class HyperpriorCompression(FactorizedCompression, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.hyper_analysis_transform = TransformABC()
        self.hyper_synthesis_transform = TransformABC()
        self.entropy_model = EntropyModel()

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> dict:
        latents = self.analysis_transform(inputs, args, kwargs)

        hyperprior = self.hyper_analysis_transform(latents, args, kwargs)
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        hyperprior = self.hyper_synthesis_transform(rec_hyperprior)
        latents_hat, latents_likelihoods = self.entropy_model(latents, hyperprior, args, kwargs)

        inputs_hat = self.synthesis_transform(latents_hat, args, kwargs)
        return {
            "inputs_hat": inputs_hat,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }


class AutoregressiveCompression(HyperpriorCompression, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.analysis_transform = TransformABC()
        self.synthesis_transform = TransformABC()
        self.hyper_analysis_transform = TransformABC()
        self.hyper_synthesis_transform = TransformABC()
        self.entropy_model = EntropyModel()

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> dict:
        latents = self.analysis_transform(inputs, args, kwargs)

        hyperprior = self.hyper_analysis_transform(latents, args, kwargs)
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        hyperprior = self.hyper_synthesis_transform(rec_hyperprior)
        latents_hat, latents_likelihoods = self.entropy_model(latents, hyperprior=hyperprior)

        inputs_hat = self.synthesis_transform(latents_hat, args, kwargs)
        return {
            "inputs_hat": inputs_hat,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
