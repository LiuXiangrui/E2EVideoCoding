from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Common.Compression import FactorizedCompression, HyperpriorCompression
from Common.Compression import AnalysisTransform, SynthesisTransform
from Common.Compression import HyperAnalysisTransform, HyperSynthesisTransform


class MotionCompression(FactorizedCompression):
    def __init__(self, N: int = 128, M: int = 128):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=2, internal_channels=N, out_channels=M, kernel_size=3)
        self.synthesis_transform = SynthesisTransform(in_channels=M, internal_channels=N, out_channels=2, kernel_size=3)
        self.entropy_bottleneck = EntropyBottleneck(channels=M)


class ResiduesCompression(HyperpriorCompression):
    def __init__(self, N: int = 128, M: int = 192):
        super().__init__()
        self.analysis_transform = AnalysisTransform(in_channels=3, internal_channels=N, out_channels=M, kernel_size=5)
        self.synthesis_transform = SynthesisTransform(in_channels=M, internal_channels=N, out_channels=3, kernel_size=5)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=M, internal_channels=N, out_channels=N)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=N, internal_channels=N, out_channels=M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)
