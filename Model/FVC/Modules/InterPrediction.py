import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class MotionEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        offset = self.net(torch.cat([cur, ref], dim=1))
        return offset


class PredRefine(nn.Module):
    def __init__(self, feats_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=feats_channels * 2, out_channels=feats_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feats_channels, out_channels=feats_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, aligned_ref: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return aligned_ref + self.net(torch.cat([ref, aligned_ref], dim=1))


# TODO: 论文图3中offset的通道数为64，但重建offset通道数是128
# 如果采用128作为重建通道数，则图5中deform compensation就无法与inter prediction里deform comp权值共享
# 因此，我在实现中采用了64作为offset重建通道数
class MotionCompensation(nn.Module):
    def __init__(self, feats_channels: int, offset_channels: int, group: int = 8, deform_kernel_size: int = 3):
        super().__init__()
        self.deform_offset_est = nn.Conv2d(in_channels=offset_channels, out_channels=2 * deform_kernel_size ** 2 * group, kernel_size=3, stride=1, padding=1)
        self.deform_conv = DeformConv2d(in_channels=feats_channels, out_channels=feats_channels, groups=group, kernel_size=deform_kernel_size, stride=1, padding=1)
        self.pred_refine = PredRefine(feats_channels=feats_channels)

    def forward(self, ref: torch.Tensor, motion_fields: torch.Tensor) -> torch.Tensor:
        deform_offset = self.deform_offset_est(motion_fields)
        pred = self.deform_conv(ref, offset=deform_offset)
        pred = self.pred_refine(pred, ref=ref)
        return pred
