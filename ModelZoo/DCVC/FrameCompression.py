import torch

from Modules.Compression.CompressionNetwork import AutoregressiveCompression
from Modules.Compression.EntropyModel import FactorizedEntropyModel, \
    AREntropyModelWithTemporalPrior
from Modules.Compression.NonlinearTransform import ContextualAnalysisTransform, \
    ContextualSynthesisTransform, HyperAnalysisTransform, HyperSynthesisTransform


class FrameCompressionDCVC(AutoregressiveCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = ContextualAnalysisTransform(in_channels=3, ctx_channels=64, internal_channels=64, out_channels=96)
        self.synthesis_transform = ContextualSynthesisTransform(in_channels=96, ctx_channels=64, internal_channels=64, out_channels=3)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=96, internal_channels=64, out_channels=64)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=64, internal_channels=64, out_channels=96)
        self.entropy_bottleneck = FactorizedEntropyModel(channels=64)
        self.entropy_model = AREntropyModelWithTemporalPrior(latent_channels=96, ctx_channels=64)

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> dict:
        assert "ctx" in kwargs, "Context should be provided!"

        latents = self.analysis_transform(inputs, ctx=kwargs["ctx"])

        hyperprior = self.hyper_analysis_transform(latents)
        rec_hyperprior, hyperprior_likelihoods = self.entropy_bottleneck(hyperprior)
        hyperprior = self.hyper_synthesis_transform(rec_hyperprior)

        latents_hat, latents_likelihoods = self.entropy_model(latents, hyperprior=hyperprior, ctx=kwargs["ctx"])
        inputs_hat = self.synthesis_transform(latents_hat, ctx=kwargs["ctx"])
        return {
            "inputs_hat": inputs_hat,
            "likelihoods": {"latents": latents_likelihoods, "hyperprior": hyperprior_likelihoods},
        }
