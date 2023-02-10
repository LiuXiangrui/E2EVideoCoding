import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from Modules.InterPrediction.MotionCompensation.MotionCompensation import MotionCompensationABC
from Modules.InterPrediction.MotionEstimation.MotionEstimation import DeformableOffsetEst

class OffsetProcess(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, offset: torch.Tensor, *args, **kwargs):
        return self.net(offset)


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, aligned_ref: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert "ref" in kwargs

        return self.net(torch.cat([kwargs["ref"], aligned_ref], dim=1))


class MotionCompensation(MotionCompensationABC):
    def __init__(self):
        super().__init__()
        self.group = 8
        self.kernel_size = 3
        self.offset_process = OffsetProcess(in_channels=64, out_channels=2 * self.kernel_size ** 2 * self.group)
        self.warp_net = DeformConv2d(in_channels=64, out_channels=64, groups=self.group, kernel_size=3, stride=1, padding=1)
        self.refine_net = RefineNet()


MotionEstimation = DeformableOffsetEst
