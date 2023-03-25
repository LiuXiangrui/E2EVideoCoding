import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Common.Utils import optical_flow_warp
from Model.Common.BasicBlock import ResBlock
from Model.Common.MotionEstimation import SpyNetOpticalFlowEst


class PredRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_levels = 3
        self.in_channels = 6  # channels of current frame added with channels of reference frame
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResBlock(channels=64, activation=nn.ReLU, head_act=True) for _ in
                                         range(2 * self.scale_levels)])
        self.tail = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, aligned_ref: torch.Tensor, ref: torch.Tensor, motion_fields: torch.Tensor) -> torch.Tensor:
        feats = self.head(torch.cat([motion_fields, ref, aligned_ref], dim=1))
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
        pred = self.tail(feats)
        return pred


class MotionCompensation(nn.Module):
    def __init__(self):
        super(MotionCompensation, self).__init__()
        self.pred_refine = PredRefine()

    def forward(self, ref: torch.Tensor, motion_fields: torch.Tensor) -> tuple:
        aligned_ref = optical_flow_warp(ref, motion_fields=motion_fields)
        pred = self.pred_refine(aligned_ref, ref=ref, motion_fields=motion_fields)
        return aligned_ref, pred


MotionEstimation = SpyNetOpticalFlowEst
