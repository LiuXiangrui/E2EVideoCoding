import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models import CompressionModel


class MotionCodecDVC(CompressionModel):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=128)

        self.analysis_transform = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )

        self.synthesis_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, offset: torch.Tensor) -> dict:
        latents = self.analysis_transform(offset)
        rec_latents, likelihoods = self.entropy_bottleneck(latents)
        rec_offset = self.synthesis_transform(rec_latents)

        return {
            "rec_offset": rec_offset,
            "likelihoods":  likelihoods,
        }


