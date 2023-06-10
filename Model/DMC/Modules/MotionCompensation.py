import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Common.Utils import optical_flow_warp


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel: int = 48, aux_feature_num: int = 53, feats_channels: int = 64,
                 offset_num: int = 2, group: int = 16, max_residue_magnitude: int = 40):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group = group
        self.max_residue_magnitude = max_residue_magnitude

        self.offset_prediction = nn.Sequential(
            nn.Conv2d(in_channels=aux_feature_num, out_channels=feats_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feats_channels, out_channels=feats_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=feats_channels, out_channels=3 * group * offset_num, kernel_size=3, stride=1, padding=1)
        )

        self.cross_group_fusion = nn.Conv2d(in_channels=in_channel * offset_num, out_channels=in_channel, kernel_size=1, stride=1, groups=group)

    def forward(self, x: torch.Tensor, aux_feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = self.offset_prediction(aux_feature)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        residual_offsets, mask = torch.split(out, split_size_or_sections=[2 * self.group * self.offset_num, self.group * self.offset_num], dim=1)
        residual_offsets = self.max_residue_magnitude * residual_offsets
        offset = residual_offsets + flow.repeat(1, self.group * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group * self.offset_num, 2, H, W)
        mask = torch.sigmoid(mask).view(B * self.group * self.offset_num, 1, H, W)

        x = x.view(B * self.group, C // self.group, H, W).repeat(self.offset_num, 1, 1, 1)
        x = optical_flow_warp(x, motion_fields=offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)

        x = self.cross_group_fusion(x)
        return x
