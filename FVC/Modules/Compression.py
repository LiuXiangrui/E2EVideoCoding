from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Common.Compression import HyperpriorCompression
from Common.Compression import HyperAnalysisTransform, HyperSynthesisTransform
from Common.Compression import AnalysisTransformWithResBlocks, SynthesisTransformWithResBlocks


class MotionCompression(HyperpriorCompression):
    def __init__(self, in_channels: int = 64, out_channels: int = 128, N: int = 128, M: int = 128):
        super().__init__()
        self.analysis_transform = AnalysisTransformWithResBlocks(in_channels=in_channels, internal_channels=N, out_channels=M)
        self.synthesis_transform = SynthesisTransformWithResBlocks(in_channels=M, internal_channels=N, out_channels=out_channels)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=M, internal_channels=N, out_channels=N)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=N, internal_channels=N, out_channels=M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)


class ResiduesCompression(HyperpriorCompression):
    def __init__(self, in_channels: int = 64, N: int = 128, M: int = 128):
        super().__init__()
        self.analysis_transform = AnalysisTransformWithResBlocks(in_channels=in_channels, internal_channels=N, out_channels=M)
        self.synthesis_transform = SynthesisTransformWithResBlocks(in_channels=M, internal_channels=N, out_channels=in_channels)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=M, internal_channels=N, out_channels=N)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=N, internal_channels=N, out_channels=M)
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)
