from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Common.Compression import FactorizedCompression, HyperpriorCompression
from Common.Compression import AnalysisTransform, SynthesisTransform
from Common.Compression import HyperAnalysisTransform, HyperSynthesisTransform


class ResiduesCompression(HyperpriorCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=3, internal_channels=128, out_channels=192, kernel_size=5)
        self.synthesis_transform = SynthesisTransform(in_channels=192, internal_channels=128, out_channels=3, kernel_size=5)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=192, internal_channels=128, out_channels=128)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=128, internal_channels=128, out_channels=192)
        self.entropy_bottleneck = EntropyBottleneck(channels=128)
        self.gaussian_conditional = GaussianConditional(None)


class MotionCompression(FactorizedCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=2, internal_channels=128, out_channels=128, kernel_size=3)
        self.synthesis_transform = SynthesisTransform(in_channels=128, internal_channels=128, out_channels=2, kernel_size=3)
        self.entropy_bottleneck = EntropyBottleneck(channels=128)
