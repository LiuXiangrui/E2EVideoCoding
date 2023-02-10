import torch
import torch.nn as nn
from Modules.Utils import OpticalFlowWarpNet, ResBlock

from Modules.InterPrediction.MotionCompensation.MotionCompensation import MotionCompensationABC
from Modules.InterPrediction.MotionEstimation.MotionEstimation import DeformableOffsetEst


class OffsetProcess(nn.Module):
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

    def forward(self, offset: torch.Tensor, *args, **kwargs):
        assert "ref" in kwargs.keys()
        return self.net(torch.cat([offset, kwargs["ref"]], dim=1))


class RefProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResBlock(channels=64, activation=nn.ReLU, head_act=True)
        )

    def forward(self, ref: torch.Tensor, *args, **kwargs):
        return self.net(ref)


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(channels=64, activation=nn.ReLU, head_act=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, aligned_ref: torch.Tensor, *args, **kwargs):
        return self.net(aligned_ref)


class MotionCompensation(MotionCompensationABC):
    def __init__(self):
        super().__init__()
        self.offset_process = OffsetProcess()
        self.ref_process = RefProcess()
        self.warp_net = OpticalFlowWarpNet()
        self.refine_net = RefineNet()


MotionEstimation = DeformableOffsetEst
