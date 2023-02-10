from Modules.Compression.CompressionNetwork import FactorizedCompression
from Modules.Compression.EntropyModel import FactorizedEntropyModel
from Modules.Compression.NonlinearTransform import AnalysisTransform, SynthesisTransform


class MotionCompression(FactorizedCompression):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = FactorizedEntropyModel(channels=128)

        self.analysis_transform = AnalysisTransform(in_channels=2, internal_channels=128, out_channels=128)
        self.synthesis_transform = SynthesisTransform(in_channels=128, internal_channels=128, out_channels=2)
