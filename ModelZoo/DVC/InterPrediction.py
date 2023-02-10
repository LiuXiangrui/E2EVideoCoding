import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.Utils import OpticalFlowWarpNet, ResBlock
from Modules.InterPrediction.MotionCompensation.MotionCompensation import MotionCompensationABC
from Modules.InterPrediction.MotionEstimation.MotionEstimation import SpyNetOpticalFlowEst

class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_levels = 3
        self.head = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(channels=64, activation=nn.ReLU, head_act=True) for _ in
                                         range(2 * self.scale_levels)])
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, aligned_ref: torch.Tensor, **kwargs) -> torch.Tensor:
        assert "offset" in kwargs.keys() and "ref" in kwargs.keys()
        feats = self.head(torch.cat([kwargs["offset"], kwargs["ref"], aligned_ref], dim=1))
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


class MotionCompensation(MotionCompensationABC):
    def __init__(self):
        super(MotionCompensation, self).__init__()
        self.warp_net = OpticalFlowWarpNet()
        self.refine_net = RefineNet()


MotionEstimation = SpyNetOpticalFlowEst