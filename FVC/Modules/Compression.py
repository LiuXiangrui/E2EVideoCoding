from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from Common.Compression import HyperpriorCompression
from Common.Compression import HyperAnalysisTransform, HyperSynthesisTransform
from Common.Compression import AnalysisTransformWithResBlocks, SynthesisTransformWithResBlocks


class ResiduesCompression(HyperpriorCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransformWithResBlocks(in_channels=64, internal_channels=128, out_channels=128)
        self.synthesis_transform = SynthesisTransformWithResBlocks(in_channels=128, internal_channels=128, out_channels=64)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=128, internal_channels=128, out_channels=128)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=128, internal_channels=128, out_channels=128)
        self.entropy_bottleneck = EntropyBottleneck(channels=128)
        self.gaussian_conditional = GaussianConditional(None)


class MotionCompression(HyperpriorCompression):
    def __init__(self):
        super().__init__()
        self.analysis_transform = AnalysisTransformWithResBlocks(in_channels=64, internal_channels=128, out_channels=128)
        self.synthesis_transform = SynthesisTransformWithResBlocks(in_channels=128, internal_channels=128, out_channels=64)
        self.hyper_analysis_transform = HyperAnalysisTransform(in_channels=128, internal_channels=128, out_channels=128)
        self.hyper_synthesis_transform = HyperSynthesisTransform(in_channels=128, internal_channels=128, out_channels=128)
        self.entropy_bottleneck = EntropyBottleneck(channels=128)
        self.gaussian_conditional = GaussianConditional(None)
