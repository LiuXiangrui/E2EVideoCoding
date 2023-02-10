import torch

from Modules.Compression.CompressionNetwork import HyperpriorCompression
from Modules.Compression.EntropyModel import FactorizedEntropyModel, ScaleGaussianEntropyModel
from Modules.Compression.NonlinearTransform import AnalysisTransform, SynthesisTransform, \
    HyperAnalysisTransform, HyperSynthesisTransform


class FrameCompression(HyperpriorCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=3, internal_channels=128, out_channels=192)
        self.synthesis_transform = SynthesisTransform(in_channels=192, internal_channels=128, out_channels=3)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=192, internal_channels=128, out_channels=128)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=128, internal_channels=128, out_channels=192)
        self.entropy_bottleneck = FactorizedEntropyModel(channels=128)
        self.entropy_model = ScaleGaussianEntropyModel()

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        assert "pred" in kwargs.keys()
        residuals = inputs - kwargs["pred"]

        latents = self.analysis_transform(residuals)
        hyperprior = self.hyper_analysis_transform(torch.abs(latents))
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        scales_hat = self.hyper_synthesis_transform(rec_hyperprior)
        rec_latents, latents_likelihoods = self.entropy_model(latents, scales_hat=scales_hat)
        rec_residuals = self.synthesis_transform(rec_latents)
        inputs_hat = kwargs["pred"] + rec_residuals

        return {
            "inputs_hat": inputs_hat,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
