import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class MotionCompFVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = 8
        self.kernel_size = 3
        self.deform_offset_extraction = nn.Conv2d(in_channels=64, out_channels=2 * self.kernel_size ** 2 * self.group,
                                                  kernel_size=3, stride=1, padding=1)

        self.deform_conv = DeformConv2d(in_channels=64, out_channels=64, groups=self.group, kernel_size=3, stride=1, padding=1)

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, offset: torch.Tensor, ref: torch.Tensor):
        deform_offsets = self.deform_offset_extraction(offset)
        aligned_ref = self.deform_conv(ref, offset=deform_offsets)
        pred = self.refine(torch.cat([ref, aligned_ref], dim=1)) + aligned_ref
        return pred
