import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MotionCompDVC(nn.Module):
    def __init__(self, feats_channels: int = 64):
        super(MotionCompDVC, self).__init__()
        self.scale_levels = 3
        self.head = nn.Conv2d(in_channels=8, out_channels=feats_channels, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(channels=feats_channels) for _ in range(2 * self.scale_levels)])
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=feats_channels, out_channels=feats_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feats_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, ref_frame: torch.Tensor, rec_offset: torch.Tensor) -> torch.Tensor:
        warped_ref = optical_flow_warp(ref_frame, optical_flow=rec_offset)

        feats = self.head(torch.cat([rec_offset, ref_frame, warped_ref], dim=1))
        feats_list = []
        for level in range(self.scale_levels):
            feats = self.res_blocks[level](feats)
            if level < self.scale_levels - 1:
                feats_list.append(feats)
                feats = F.avg_pool2d(feats, kernel_size=(2, 2))
        feats_list.reverse()
        for level in range(self.scale_levels):
            feats = self.res_blocks[self.scale_levels + level](feats)
            if level < self.scale_levels - 1:
                feats = F.interpolate(feats, scale_factor=(2, 2), mode='bilinear', align_corners=True)
                feats = feats + feats_list[level]
        pred_frame = self.tail(feats)
        return pred_frame
