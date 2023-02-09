import torch
import torch.nn as nn

from Modules.Utils import optical_flow_warp


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MotionCompDCVC(nn.Module):
    def __init__(self, feats_channels: int = 64):
        super().__init__()

        self.ref_feats_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResBlock(channels=64)
        )

        self.ctx_refinement = nn.Sequential(
            ResBlock(channels=feats_channels),
            nn.Conv2d(in_channels=feats_channels, out_channels=feats_channels, kernel_size=3, stride=1, padding=1)
        )

        self.offset_refinement = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, ref_frame: torch.Tensor, rec_offset: torch.Tensor) -> torch.Tensor:
        offset_refined = self.offset_refinement(torch.cat([rec_offset, ref_frame], dim=1))
        ref_feats = self.ref_feats_extraction(ref_frame)
        ctx = optical_flow_warp(ref_feats, optical_flow=offset_refined)
        ctx_refined = self.ctx_refinement(ctx)
        return ctx_refined
