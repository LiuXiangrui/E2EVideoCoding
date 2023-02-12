import torch
import torch.nn as nn

from Common.Utils import optical_flow_warp
from Common.BasicBlock import ResBlock
from Common.MotionEstimation import SpyNetOpticalFlowEst


class MotionRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, offset: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([offset, ref], dim=1))


class RefRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResBlock(channels=64, activation=nn.ReLU, head_act=True)
        )

    def forward(self, ref: torch.Tensor):
        return self.net(ref)


class PredRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(channels=64, activation=nn.ReLU, head_act=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, pred: torch.Tensor):
        return self.net(pred)


class MotionCompensation(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_refine = MotionRefine()
        self.ref_refine = RefRefine()
        self.pred_refine = PredRefine()

    def forward(self, ref: torch.Tensor, motion_fields: torch.Tensor) -> torch.Tensor:
        pred = optical_flow_warp(self.ref_refine(ref), motion_fields=self.motion_refine(motion_fields, ref=ref))
        pred = self.pred_refine(pred)
        return pred


MotionEstimation = SpyNetOpticalFlowEst
