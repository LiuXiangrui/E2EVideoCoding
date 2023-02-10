from Modules.Compression.CompressionNetwork import AutoregressiveCompression
from Modules.Compression.EntropyModel import AutoRegressiveEntropyModel, FactorizedEntropyModel
from Modules.Compression.NonlinearTransform import AnalysisTransform, \
    SynthesisTransform, HyperAnalysisTransform, HyperSynthesisTransform


class MotionCompression(AutoregressiveCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=2, internal_channels=128, out_channels=128)

        self.synthesis_transform = SynthesisTransform(in_channels=128, internal_channels=128, out_channels=2)

        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=128, internal_channels=64, out_channels=64)
        self.hyper_synthesis_transform = HyperSynthesisTransform(channels=[64, 64, 96, 256])

        self.entropy_model = AutoRegressiveEntropyModel(latent_channels=128)
        self.entropy_bottleneck = FactorizedEntropyModel(channels=64)
